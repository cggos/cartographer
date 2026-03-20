/*
 * Cartographer 3D 局部轨迹构建器
 * 
 * 功能：负责前端局部SLAM处理
 * - 接收激光点云、IMU、里程计数据
 * - 实现点云累积、运动补偿、扫描匹配
 * - 管理子图插入
 * 
 * 工作流程：
 * 1. AddImuData: 接收IMU数据，初始化姿态外推器
 * 2. AddRangeData: 接收激光点云，执行扫描匹配
 * 3. InsertIntoSubmap: 将匹配结果插入子图
 * 
 * Copyright 2016 The Cartographer Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cartographer/mapping/internal/3d/local_trajectory_builder_3d.h"

#include <memory>

#include "absl/memory/memory.h"
#include "cartographer/common/time.h"
#include "cartographer/mapping/internal/3d/scan_matching/rotational_scan_matcher.h"
#include "cartographer/mapping/proto/local_trajectory_builder_options_3d.pb.h"
#include "cartographer/mapping/proto/scan_matching/ceres_scan_matcher_options_3d.pb.h"
#include "cartographer/mapping/proto/scan_matching/real_time_correlative_scan_matcher_options.pb.h"
#include "cartographer/mapping/proto/submaps_options_3d.pb.h"
#include "cartographer/transform/timestamped_transform.h"
#include "glog/logging.h"

namespace cartographer {
namespace mapping {

// TODO(spielawa): Adjust metrics for multi-trajectory. So far we assume a
// single trajectory.
static auto* kLocalSlamLatencyMetric = metrics::Gauge::Null();
static auto* kLocalSlamVoxelFilterFraction = metrics::Gauge::Null();
static auto* kLocalSlamScanMatcherFraction = metrics::Gauge::Null();
static auto* kLocalSlamInsertIntoSubmapFraction = metrics::Gauge::Null();
static auto* kLocalSlamRealTimeRatio = metrics::Gauge::Null();
static auto* kLocalSlamCpuRealTimeRatio = metrics::Gauge::Null();
static auto* kRealTimeCorrelativeScanMatcherScoreMetric =
    metrics::Histogram::Null();
static auto* kCeresScanMatcherCostMetric = metrics::Histogram::Null();
static auto* kScanMatcherResidualDistanceMetric = metrics::Histogram::Null();
static auto* kScanMatcherResidualAngleMetric = metrics::Histogram::Null();

LocalTrajectoryBuilder3D::LocalTrajectoryBuilder3D(
    const mapping::proto::LocalTrajectoryBuilderOptions3D& options,
    const std::vector<std::string>& expected_range_sensor_ids)
    : options_(options),
      active_submaps_(options.submaps_options()),
      motion_filter_(options.motion_filter_options()),
      real_time_correlative_scan_matcher_(
          absl::make_unique<scan_matching::RealTimeCorrelativeScanMatcher3D>(
              options_.real_time_correlative_scan_matcher_options())),
      ceres_scan_matcher_(absl::make_unique<scan_matching::CeresScanMatcher3D>(
          options_.ceres_scan_matcher_options())),
      range_data_collator_(expected_range_sensor_ids) {}

LocalTrajectoryBuilder3D::~LocalTrajectoryBuilder3D() {}

/**
 * @brief 扫描匹配 - 核心位姿估计函数
 * 
 * 功能：在子图坐标系下，通过点云匹配估计机器人位姿
 * 
 * 匹配算法（两步）：
 * 1. RealTimeCorrelativeScanMatching (可选): 快速全局搜索，提供初始估计
 * 2. CeresScanMatching: 非线性优化精细化位姿
 * 
 * @param pose_prediction 位姿预测（来自PoseExtrapolator）
 * @param low_resolution_point_cloud_in_tracking 低分辨率点云（粗匹配）
 * @param high_resolution_point_cloud_in_tracking 高分辨率点云（精匹配）
 * @return 估计的机器人位姿 Rigid3d，nullptr表示匹配失败
 */
std::unique_ptr<transform::Rigid3d> LocalTrajectoryBuilder3D::ScanMatch(
    const transform::Rigid3d& pose_prediction,
    const sensor::PointCloud& low_resolution_point_cloud_in_tracking,
    const sensor::PointCloud& high_resolution_point_cloud_in_tracking) {
  if (active_submaps_.submaps().empty()) {
    return absl::make_unique<transform::Rigid3d>(pose_prediction);
  }
  // 获取当前活跃子图（用于匹配）
  std::shared_ptr<const mapping::Submap3D> matching_submap =
      active_submaps_.submaps().front();
  
  // 计算预测位姿在子图坐标系下的初始估计
  // local_pose是子图原点相对于world的变换
  // initial_ceres_pose = submap^-1 * prediction
  transform::Rigid3d initial_ceres_pose =
      matching_submap->local_pose().inverse() * pose_prediction;
      
  // ========== 步骤1: RealTime相关性扫描匹配（可选） ==========
  // 使用分支定界算法快速搜索最佳匹配位置
  // 提供更好的初始估计给Ceres优化器
  if (options_.use_online_correlative_scan_matching()) {
    const transform::Rigid3d initial_pose = initial_ceres_pose;
    // 在高分辨率子图上进行匹配
    const double score = real_time_correlative_scan_matcher_->Match(
        initial_pose, high_resolution_point_cloud_in_tracking,
        matching_submap->high_resolution_hybrid_grid(), &initial_ceres_pose);
    kRealTimeCorrelativeScanMatcherScoreMetric->Observe(score);
  }

  // ========== 步骤2: Ceres非线性优化 ==========
  // 使用Ceres求解器进行精细化位姿估计
  // 成本函数：点云与子图栅格匹配程度
  transform::Rigid3d pose_observation_in_submap;
  ceres::Solver::Summary summary;
  const auto* high_resolution_intensity_hybrid_grid =
      options_.use_intensities()
          ? &matching_submap->high_resolution_intensity_hybrid_grid()
          : nullptr;
  // 双分辨率匹配：高分辨率用于精细匹配，低分辨率用于粗匹配
  ceres_scan_matcher_->Match(
      (matching_submap->local_pose().inverse() * pose_prediction).translation(),
      initial_ceres_pose, {{&high_resolution_point_cloud_in_tracking,
                            &matching_submap->high_resolution_hybrid_grid(),
                            high_resolution_intensity_hybrid_grid},
                           {&low_resolution_point_cloud_in_tracking,
                            &matching_submap->low_resolution_hybrid_grid(),
                            /*intensity_hybrid_grid=*/nullptr}},
      &pose_observation_in_submap, &summary);
  // 记录匹配残差用于调试
  kCeresScanMatcherCostMetric->Observe(summary.final_cost);
  const double residual_distance = (pose_observation_in_submap.translation() -
                                    initial_ceres_pose.translation())
                                       .norm();
  kScanMatcherResidualDistanceMetric->Observe(residual_distance);
  const double residual_angle =
      pose_observation_in_submap.rotation().angularDistance(
          initial_ceres_pose.rotation());
  kScanMatcherResidualAngleMetric->Observe(residual_angle);
  
  // 将子图坐标系下的位姿转换回world坐标系
  // result = submap_local_pose * observation_in_submap
  return absl::make_unique<transform::Rigid3d>(matching_submap->local_pose() *
                                                 pose_observation_in_submap);
}

/**
 * @brief 添加IMU数据
 * 
 * 功能：
 * 1. 如果外推器已初始化，直接将IMU数据添加到外推器
 * 2. 如果外推器未初始化，使用IMU数据初始化外推器
 * 
 * IMU作用：
 * - 提供角速度，用于姿态外推预测
 * - 提供线加速度，用于重力方向估计（roll/pitch不会漂移）
 * 
 * @param imu_data IMU数据，包含：
 *        - time: 时间戳
 *        - linear_acceleration: 线加速度 (ax, ay, az)
 *        - angular_velocity: 角速度 (wx, wy, wz)
 */
void LocalTrajectoryBuilder3D::AddImuData(const sensor::ImuData& imu_data) {
  // 情况1: 外推器已初始化，直接添加IMU数据
  if (extrapolator_ != nullptr) {
    extrapolator_->AddImuData(imu_data);
    return;
  }
  
  // 情况2: 外推器未初始化，使用IMU数据创建新的外推器
  // 从配置中获取初始位姿列表
  std::vector<transform::TimestampedTransform> initial_poses;
  for (const auto& pose_proto : options_.initial_poses()) {
    initial_poses.push_back(transform::FromProto(pose_proto));
  }
  
  // 从配置中获取初始IMU数据序列
  std::vector<sensor::ImuData> initial_imu_data;
  for (const auto& imu : options_.initial_imu_data()) {
    initial_imu_data.push_back(sensor::FromProto(imu));
  }
  // 添加当前接收到的IMU数据
  initial_imu_data.push_back(imu_data);
  
  // 创建并初始化姿态外推器（详见PoseExtrapolator）
  extrapolator_ = mapping::PoseExtrapolatorInterface::CreateWithImuData(
      options_.pose_extrapolator_options(), initial_imu_data, initial_poses);
}

/**
 * @brief 添加激光点云数据 - 局部SLAM核心函数
 * 
 * 功能：处理一帧激光点云数据，执行局部SLAM
 * 
 * 处理流程（共8步）：
 * 1. 数据同步 (RangeDataCollator): 多传感器数据时间对齐
 * 2. IMU初始化检查: 等待IMU数据初始化外推器
 * 3. 点云累积: 累积多帧点云减少噪声 (num_accumulated_range_data帧)
 * 4. 运动补偿: 根据时间戳外推每个点的姿态，校正点云位置
 * 5. VoxelFilter: 降采样，减少计算量
 * 6. AdaptiveVoxelFilter: 生成高/低分辨率点云
 * 7. ScanMatch: 扫描匹配，估计机器人位姿
 * 8. InsertIntoSubmap: 将点云插入子图
 * 
 * @param sensor_id 传感器ID
 * @param unsynchronized_data 未经同步的激光点云数据
 * @return MatchingResult 匹配结果，包含：
 *         - time: 时间戳
 *         - local_pose: 估计的机器人位姿 (Rigid3d: translation + rotation)
 *         - range_data_in_local: 局部坐标系下的点云
 *         - insertion_result: 插入子图的结果
 *         - nullptr: 表示数据不足或匹配失败
 */
std::unique_ptr<LocalTrajectoryBuilder3D::MatchingResult>
LocalTrajectoryBuilder3D::AddRangeData(
    const std::string& sensor_id,
    const sensor::TimedPointCloudData& unsynchronized_data) {
  // ========== 步骤1: 数据同步 ==========
  // 检查强度数据与点云数据数量是否一致
  if (options_.use_intensities()) {
    CHECK_EQ(unsynchronized_data.ranges.size(),
             unsynchronized_data.intensities.size())
        << "Passed point cloud has inconsistent number of intensities and "
           "ranges.";
  }
  // 通过RangeDataCollator进行多传感器数据同步
  auto synchronized_data =
      range_data_collator_.AddRangeData(sensor_id, unsynchronized_data);
  // 数据缓冲区未满，等待更多数据
  if (synchronized_data.ranges.empty()) {
    LOG(INFO) << "Range data collator filling buffer.";
    return nullptr;
  }

  // ========== 步骤2: IMU初始化检查 ==========
  // 必须等到IMU初始化完成才能处理点云（需要IMU提供姿态预测）
  if (extrapolator_ == nullptr) {
    // Until we've initialized the extrapolator with our first IMU message, we
    // cannot compute the orientation of the rangefinder.
    LOG(INFO) << "IMU not yet initialized.";
    return nullptr;
  }

  CHECK(!synchronized_data.ranges.empty());
  CHECK_LE(synchronized_data.ranges.back().point_time.time, 0.f);
  // 计算第一个点的时间戳（绝对时间 = 结束时间 + 相对时间偏移）
  const common::Time time_first_point =
      synchronized_data.time +
      common::FromSeconds(synchronized_data.ranges.front().point_time.time);
  // 如果第一个点的时间早于外推器最后姿态时间，说明外推器还未初始化完成
  if (time_first_point < extrapolator_->GetLastPoseTime()) {
    LOG(INFO) << "Extrapolator is still initializing.";
    return nullptr;
  }

  // ========== 步骤3: 点云累积 ==========
  // num_accumulated_range_data帧累积，减少噪声影响
  if (num_accumulated_ == 0) {
    accumulated_point_cloud_origin_data_.clear();
  }

  // 首次降采样：粗滤波，减少累积数据量
  synchronized_data.ranges = sensor::VoxelFilter(
      synchronized_data.ranges, 0.5f * options_.voxel_filter_size());
  accumulated_point_cloud_origin_data_.emplace_back(
      std::move(synchronized_data));
  ++num_accumulated_;

  // 累积帧数不足，返回继续累积
  if (num_accumulated_ < options_.num_accumulated_range_data()) {
    return nullptr;
  }
  num_accumulated_ = 0;

  // ========== 步骤4: 运动补偿 ==========
  // 激光扫描期间机器人可能移动，根据时间戳外推每个点的姿态
  bool warned = false;
  std::vector<common::Time> hit_times;
  // 获取外推器最后外推时间
  common::Time prev_time_point = extrapolator_->GetLastExtrapolatedTime();
  
  // 收集所有点云中每个点的时间戳
  for (const auto& point_cloud_origin_data :
       accumulated_point_cloud_origin_data_) {
    for (const auto& hit : point_cloud_origin_data.ranges) {
      // 计算每个点的绝对时间 = 扫描结束时间 + 相对时间偏移
      common::Time time_point = point_cloud_origin_data.time +
                                common::FromSeconds(hit.point_time.time);
      // 检查时间戳是否倒流
      if (time_point < prev_time_point) {
        if (!warned) {
          LOG(ERROR) << "Timestamp of individual range data point jumps "
                        "backwards from "
                     << prev_time_point << " to " << time_point;
          warned = true;
        }
        time_point = prev_time_point;
      }

      hit_times.push_back(time_point);
      prev_time_point = time_point;
    }
  }
  // 添加最后一帧的结束时间
  hit_times.push_back(accumulated_point_cloud_origin_data_.back().time);

  // 调用外推器，获取每个点时刻的机器人姿态
  // ExtrapolatePosesWithGravity: 根据IMU/里程计外推姿态
  const PoseExtrapolatorInterface::ExtrapolationResult extrapolation_result =
      extrapolator_->ExtrapolatePosesWithGravity(hit_times);
  
  // 获取外推的姿态序列，用于运动补偿
  std::vector<transform::Rigid3f> hits_poses(
      std::move(extrapolation_result.previous_poses));
  // 添加当前姿态
  hits_poses.push_back(extrapolation_result.current_pose.cast<float>());
  CHECK_EQ(hits_poses.size(), hit_times.size());

  // ========== 运动补偿：校正点云位置 ==========
  // 根据每个点的采集时刻的姿态，将点云转换到同一坐标系
  const size_t max_possible_number_of_accumulated_points = hit_times.size();
  std::vector<sensor::RangefinderPoint> accumulated_points;
  std::vector<float> accumulated_intensities;
  accumulated_points.reserve(max_possible_number_of_accumulated_points);
  if (options_.use_intensities()) {
    accumulated_intensities.reserve(max_possible_number_of_accumulated_points);
  }
  sensor::PointCloud misses;
  std::vector<transform::Rigid3f>::const_iterator hits_poses_it =
      hits_poses.begin();
      
  // 遍历所有累积的点云
  for (const auto& point_cloud_origin_data :
       accumulated_point_cloud_origin_data_) {
    for (const auto& hit : point_cloud_origin_data.ranges) {
      // 将点从采集时的姿态转换到当前姿态
      // hit.point_time.position: 原始点坐标（传感器坐标系）
      // hits_poses_it: 该点采集时刻的机器人姿态
      const Eigen::Vector3f hit_in_local =
          *hits_poses_it * hit.point_time.position;
      const Eigen::Vector3f origin_in_local =
          *hits_poses_it * point_cloud_origin_data.origins.at(hit.origin_index);
      const Eigen::Vector3f delta = hit_in_local - origin_in_local;
      const float range = delta.norm();
      
      // 距离过滤：保留有效距离范围内的点
      if (range >= options_.min_range()) {
        if (range <= options_.max_range()) {
          // 有效命中点
          accumulated_points.push_back(sensor::RangefinderPoint{hit_in_local});
          if (options_.use_intensities()) {
            accumulated_intensities.push_back(hit.intensity);
          }
        } else {
          // 超过最大距离的点作为miss（未命中），用于更新空闲空间
          // 在3D中miss实际上未被使用
          misses.push_back(sensor::RangefinderPoint{
              origin_in_local + options_.max_range() / range * delta});
        }
      }
      ++hits_poses_it;
    }
  }
  CHECK(std::next(hits_poses_it) == hits_poses.end());
  
  // 整理为点云returns
  const sensor::PointCloud returns(std::move(accumulated_points),
                                   std::move(accumulated_intensities));

  // ========== 步骤5: VoxelFilter再次降采样 ==========
  const common::Time current_sensor_time = synchronized_data.time;
  absl::optional<common::Duration> sensor_duration;
  if (last_sensor_time_.has_value()) {
    sensor_duration = current_sensor_time - last_sensor_time_.value();
  }
  last_sensor_time_ = current_sensor_time;

  const common::Time current_time = hit_times.back();
  const auto voxel_filter_start = std::chrono::steady_clock::now();
  // 二次降采样，进一步减少计算量
  const sensor::RangeData filtered_range_data = {
      extrapolation_result.current_pose.translation().cast<float>(),
      sensor::VoxelFilter(returns, options_.voxel_filter_size()),
      sensor::VoxelFilter(misses, options_.voxel_filter_size())};
  const auto voxel_filter_stop = std::chrono::steady_clock::now();
  const auto voxel_filter_duration = voxel_filter_stop - voxel_filter_start;

  if (sensor_duration.has_value()) {
    const double voxel_filter_fraction =
        common::ToSeconds(voxel_filter_duration) /
        common::ToSeconds(sensor_duration.value());
    kLocalSlamVoxelFilterFraction->Set(voxel_filter_fraction);
  }

  // ========== 调用步骤6-8: 自适应滤波、扫描匹配、子图插入 ==========
  return AddAccumulatedRangeData(
      current_time,
      // 将点云从local frame转换到tracking frame
      sensor::TransformRangeData(
          filtered_range_data,
          extrapolation_result.current_pose.inverse().cast<float>()),
      sensor_duration, extrapolation_result.current_pose,
      extrapolation_result.gravity_from_tracking);
}

/**
 * @brief 处理累积的点云数据 - 执行扫描匹配和子图插入
 * 
 * 处理步骤：
 * 1. AdaptiveVoxelFilter: 生成高/低分辨率点云
 * 2. ScanMatch: 扫描匹配，估计机器人位姿
 * 3. InsertIntoSubmap: 将点云插入子图
 * 
 * @param time 当前时间戳
 * @param filtered_range_data_in_tracking tracking坐标系下的点云
 * @param sensor_duration 传感器扫描周期（用于性能监控）
 * @param pose_prediction 位姿预测（来自PoseExtrapolator外推器）
 * @param gravity_alignment 重力对齐四元数（IMU估计的重力方向）
 * @return MatchingResult 匹配结果，包含位姿和插入结果，nullptr表示失败
 */
std::unique_ptr<LocalTrajectoryBuilder3D::MatchingResult>
LocalTrajectoryBuilder3D::AddAccumulatedRangeData(
    const common::Time time,
    const sensor::RangeData& filtered_range_data_in_tracking,
    const absl::optional<common::Duration>& sensor_duration,
    const transform::Rigid3d& pose_prediction,
    const Eigen::Quaterniond& gravity_alignment) {
  // ========== 步骤0: 空数据检查 ==========
  if (filtered_range_data_in_tracking.returns.empty()) {
    LOG(WARNING) << "Dropped empty range data.";
    return nullptr;
  }

  // ========== 步骤1: 自适应VoxelFilter - 生成多分辨率点云 ==========
  const auto scan_matcher_start = std::chrono::steady_clock::now();

  // 高分辨率点云：用于精细匹配
  // - 点数较多，细节保留好
  // - 计算量大，用于Ceres优化
  const sensor::PointCloud high_resolution_point_cloud_in_tracking =
      sensor::AdaptiveVoxelFilter(
          filtered_range_data_in_tracking.returns,
          options_.high_resolution_adaptive_voxel_filter_options());
  if (high_resolution_point_cloud_in_tracking.empty()) {
    LOG(WARNING) << "Dropped empty high resolution point cloud data.";
    return nullptr;
  }
  
  // 低分辨率点云：用于粗匹配和RealTimeCorrelativeScanMatcher
  // - 点数较少，计算速度快
  // - 用于相关性扫描匹配提供初始估计
  const sensor::PointCloud low_resolution_point_cloud_in_tracking =
      sensor::AdaptiveVoxelFilter(
          filtered_range_data_in_tracking.returns,
          options_.low_resolution_adaptive_voxel_filter_options());
  if (low_resolution_point_cloud_in_tracking.empty()) {
    LOG(WARNING) << "Dropped empty low resolution point cloud data.";
    return nullptr;
  }

  // ========== 步骤2: 扫描匹配 - 估计机器人位姿 ==========
  // 使用低分辨率点云做初始估计，高分辨率点云做精细匹配
  // pose_prediction: 来自外推器的预测位姿
  // 返回优化后的精确位姿
  std::unique_ptr<transform::Rigid3d> pose_estimate =
      ScanMatch(pose_prediction, low_resolution_point_cloud_in_tracking,
                high_resolution_point_cloud_in_tracking);
  if (pose_estimate == nullptr) {
    LOG(WARNING) << "Scan matching failed.";
    return nullptr;
  }
  
  // ========== 更新外推器 ==========
  // 将匹配结果反馈给外推器
  // 用于预测下一帧的位姿
  extrapolator_->AddPose(time, *pose_estimate);

  // 记录扫描匹配耗时（用于性能监控）
  const auto scan_matcher_stop = std::chrono::steady_clock::now();
  const auto scan_matcher_duration = scan_matcher_stop - scan_matcher_start;
  if (sensor_duration.has_value()) {
    const double scan_matcher_fraction =
        common::ToSeconds(scan_matcher_duration) /
        common::ToSeconds(sensor_duration.value());
    kLocalSlamScanMatcherFraction->Set(scan_matcher_fraction);
  }

  // ========== 转换点云坐标系 ==========
  // 将点云从tracking frame转换到local frame
  // 用于后续插入到子图
  sensor::RangeData filtered_range_data_in_local = sensor::TransformRangeData(
      filtered_range_data_in_tracking, pose_estimate->cast<float>());

  // ========== 步骤3: 插入子图 ==========
  const auto insert_into_submap_start = std::chrono::steady_clock::now();
  std::unique_ptr<InsertionResult> insertion_result = InsertIntoSubmap(
      time, filtered_range_data_in_local, filtered_range_data_in_tracking,
      high_resolution_point_cloud_in_tracking,
      low_resolution_point_cloud_in_tracking, *pose_estimate,
      gravity_alignment);
  const auto insert_into_submap_stop = std::chrono::steady_clock::now();

  // 记录子图插入耗时（用于性能监控）
  const auto insert_into_submap_duration =
      insert_into_submap_stop - insert_into_submap_start;
  if (sensor_duration.has_value()) {
    const double insert_into_submap_fraction =
        common::ToSeconds(insert_into_submap_duration) /
        common::ToSeconds(sensor_duration.value());
    kLocalSlamInsertIntoSubmapFraction->Set(insert_into_submap_fraction);
  }
  
  // ========== 性能指标记录 ==========
  const auto wall_time = std::chrono::steady_clock::now();
  if (last_wall_time_.has_value()) {
    const auto wall_time_duration = wall_time - last_wall_time_.value();
    kLocalSlamLatencyMetric->Set(common::ToSeconds(wall_time_duration));
    if (sensor_duration.has_value()) {
      // 实时性比率：传感器周期/实际处理时间
      kLocalSlamRealTimeRatio->Set(common::ToSeconds(sensor_duration.value()) /
                                   common::ToSeconds(wall_time_duration));
    }
  }
  const double thread_cpu_time_seconds = common::GetThreadCpuTimeSeconds();
  if (last_thread_cpu_time_seconds_.has_value()) {
    const double thread_cpu_duration_seconds =
        thread_cpu_time_seconds - last_thread_cpu_time_seconds_.value();
    if (sensor_duration.has_value()) {
      kLocalSlamCpuRealTimeRatio->Set(
          common::ToSeconds(sensor_duration.value()) /
          thread_cpu_duration_seconds);
    }
  }
  last_wall_time_ = wall_time;
  last_thread_cpu_time_seconds_ = thread_cpu_time_seconds;
  return absl::make_unique<MatchingResult>(MatchingResult{
      time, *pose_estimate, std::move(filtered_range_data_in_local),
      std::move(insertion_result)});
}

void LocalTrajectoryBuilder3D::AddOdometryData(
    const sensor::OdometryData& odometry_data) {
  if (extrapolator_ == nullptr) {
    // Until we've initialized the extrapolator we cannot add odometry data.
    LOG(INFO) << "Extrapolator not yet initialized.";
    return;
  }
  extrapolator_->AddOdometryData(odometry_data);
}

/**
 * @brief 将点云数据插入子图
 * 
 * 功能：
 * 1. 运动滤波：过滤微小移动
 * 2. 计算旋转直方图：用于方向估计
 * 3. 插入数据到活跃子图
 * 4. 构建插入结果供后端使用
 * 
 * @param time 时间戳
 * @param filtered_range_data_in_local local坐标系下的点云
 * @param filtered_range_data_in_tracking tracking坐标系下的点云
 * @param high_resolution_point_cloud_in_tracking 高分辨率点云
 * @param low_resolution_point_cloud_in_tracking 低分辨率点云
 * @param pose_estimate 估计的机器人位姿
 * @param gravity_alignment 重力对齐四元数
 * @return InsertionResult 插入结果
 */
std::unique_ptr<LocalTrajectoryBuilder3D::InsertionResult>
LocalTrajectoryBuilder3D::InsertIntoSubmap(
    const common::Time time,
    const sensor::RangeData& filtered_range_data_in_local,
    const sensor::RangeData& filtered_range_data_in_tracking,
    const sensor::PointCloud& high_resolution_point_cloud_in_tracking,
    const sensor::PointCloud& low_resolution_point_cloud_in_tracking,
    const transform::Rigid3d& pose_estimate,
    const Eigen::Quaterniond& gravity_alignment) {
  // ========== 步骤1: 运动滤波 ==========
  // 如果位移/旋转小于阈值，跳过本次插入（减少计算）
  if (motion_filter_.IsSimilar(time, pose_estimate)) {
    return nullptr;
  }
  
  // ========== 步骤2: 计算旋转直方图 ==========
  // 用于后续闭环检测时的方向估计
  // 将点云旋转到重力对齐坐标系下
  const Eigen::VectorXf rotational_scan_matcher_histogram_in_gravity =
      scan_matching::RotationalScanMatcher::ComputeHistogram(
          sensor::TransformPointCloud(
              filtered_range_data_in_tracking.returns,
              transform::Rigid3f::Rotation(gravity_alignment.cast<float>())),
          options_.rotational_histogram_size());

  // 计算局部坐标系到重力对齐坐标系的旋转
  // 用于将点云数据插入子图
  const Eigen::Quaterniond local_from_gravity_aligned =
      pose_estimate.rotation() * gravity_alignment.inverse();
      
  // ========== 步骤3: 插入数据到活跃子图 ==========
  // 将点云插入子图的HybridGrid中
  std::vector<std::shared_ptr<const mapping::Submap3D>> insertion_submaps =
      active_submaps_.InsertData(filtered_range_data_in_local,
                                 local_from_gravity_aligned,
                                 rotational_scan_matcher_histogram_in_gravity);
                                 
  // ========== 步骤4: 构建插入结果 ==========
  // 包含节点数据（时间、位姿、点云）和子图引用，供后端使用
  return absl::make_unique<InsertionResult>(
      InsertionResult{std::make_shared<const mapping::TrajectoryNode::Data>(
                          mapping::TrajectoryNode::Data{
                              time,                               // 时间戳
                              gravity_alignment,                 // 重力对齐
                              {},  // 'filtered_point_cloud' is only used in 2D.
                              high_resolution_point_cloud_in_tracking,  // 高分辨点云
                              low_resolution_point_cloud_in_tracking,   // 低分辨点云
                              rotational_scan_matcher_histogram_in_gravity,  // 旋转直方图
                              pose_estimate}),                     // 估计位姿
                      std::move(insertion_submaps)});             // 插入的子图
}

void LocalTrajectoryBuilder3D::RegisterMetrics(
    metrics::FamilyFactory* family_factory) {
  auto* latency = family_factory->NewGaugeFamily(
      "mapping_3d_local_trajectory_builder_latency",
      "Duration from first incoming point cloud in accumulation to local slam "
      "result");
  kLocalSlamLatencyMetric = latency->Add({});

  auto* voxel_filter_fraction = family_factory->NewGaugeFamily(
      "mapping_3d_local_trajectory_builder_voxel_filter_fraction",
      "Fraction of total sensor time taken up by voxel filter.");
  kLocalSlamVoxelFilterFraction = voxel_filter_fraction->Add({});

  auto* scan_matcher_fraction = family_factory->NewGaugeFamily(
      "mapping_3d_local_trajectory_builder_scan_matcher_fraction",
      "Fraction of total sensor time taken up by scan matcher.");
  kLocalSlamScanMatcherFraction = scan_matcher_fraction->Add({});

  auto* insert_into_submap_fraction = family_factory->NewGaugeFamily(
      "mapping_3d_local_trajectory_builder_insert_into_submap_fraction",
      "Fraction of total sensor time taken up by inserting into submap.");
  kLocalSlamInsertIntoSubmapFraction = insert_into_submap_fraction->Add({});

  auto* real_time_ratio = family_factory->NewGaugeFamily(
      "mapping_3d_local_trajectory_builder_real_time_ratio",
      "sensor duration / wall clock duration.");
  kLocalSlamRealTimeRatio = real_time_ratio->Add({});

  auto* cpu_real_time_ratio = family_factory->NewGaugeFamily(
      "mapping_3d_local_trajectory_builder_cpu_real_time_ratio",
      "sensor duration / cpu duration.");
  kLocalSlamCpuRealTimeRatio = cpu_real_time_ratio->Add({});

  auto score_boundaries = metrics::Histogram::FixedWidth(0.05, 20);
  auto* scores = family_factory->NewHistogramFamily(
      "mapping_3d_local_trajectory_builder_scores", "Local scan matcher scores",
      score_boundaries);
  kRealTimeCorrelativeScanMatcherScoreMetric =
      scores->Add({{"scan_matcher", "real_time_correlative"}});
  auto cost_boundaries = metrics::Histogram::ScaledPowersOf(2, 0.01, 100);
  auto* costs = family_factory->NewHistogramFamily(
      "mapping_3d_local_trajectory_builder_costs", "Local scan matcher costs",
      cost_boundaries);
  kCeresScanMatcherCostMetric = costs->Add({{"scan_matcher", "ceres"}});
  auto distance_boundaries = metrics::Histogram::ScaledPowersOf(2, 0.01, 10);
  auto* residuals = family_factory->NewHistogramFamily(
      "mapping_3d_local_trajectory_builder_residuals",
      "Local scan matcher residuals", distance_boundaries);
  kScanMatcherResidualDistanceMetric =
      residuals->Add({{"component", "distance"}});
  kScanMatcherResidualAngleMetric = residuals->Add({{"component", "angle"}});
}

}  // namespace mapping
}  // namespace cartographer
