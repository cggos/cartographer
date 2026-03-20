/*
 * Cartographer 姿态外推器 (Pose Extrapolator)
 * 
 * 功能：根据历史位姿、IMU数据、里程计数据预测机器人未来位姿
 * 
 * 位姿表示：transform::Rigid3d = translation(x,y,z) + rotation(四元数)
 * 
 * 核心数据结构：
 * - timed_pose_queue_: 存储历史位姿的队列，用于计算速度
 * - imu_data_: IMU数据队列
 * - odometry_data_: 里程计数据队列
 * - ImuTracker: IMU姿态跟踪器
 * 
 * 外推原理：
 * 1. 从位姿队列计算线速度和角速度
 * 2. 使用IMU角速度积分估计旋转
 * 3. 使用重力方向估计（roll/pitch不漂移）
 * 4. 预测未来某时刻的位姿
 * 
 * Copyright 2017 The Cartographer Authors
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

#include "cartographer/mapping/pose_extrapolator.h"

#include <algorithm>

#include "absl/memory/memory.h"
#include "cartographer/transform/transform.h"
#include "glog/logging.h"

namespace cartographer {
namespace mapping {

PoseExtrapolator::PoseExtrapolator(const common::Duration pose_queue_duration,
                                   double imu_gravity_time_constant)
    : pose_queue_duration_(pose_queue_duration),
      gravity_time_constant_(imu_gravity_time_constant),
      cached_extrapolated_pose_{common::Time::min(),
                                transform::Rigid3d::Identity()} {}

/**
 * @brief 使用IMU数据初始化姿态外推器
 * 
 * 初始化流程：
 * 1. 创建外推器实例
 * 2. 添加初始IMU数据
 * 3. 创建IMU跟踪器(ImuTracker)
 * 4. 用IMU数据初始化跟踪器
 * 5. 推进跟踪器到IMU时间点
 * 6. 添加初始位姿(仅旋转，基于IMU重力方向)
 * 
 * @param pose_queue_duration 位姿队列持续时间(默认5秒)
 * @param imu_gravity_time_constant IMU重力时间常数
 * @param imu_data 初始IMU数据
 * @return 初始化后的外推器实例
 */
std::unique_ptr<PoseExtrapolator> PoseExtrapolator::InitializeWithImu(
    const common::Duration pose_queue_duration,
    const double imu_gravity_time_constant, const sensor::ImuData& imu_data) {
  // 1. 创建外推器实例
  auto extrapolator = absl::make_unique<PoseExtrapolator>(
      pose_queue_duration, imu_gravity_time_constant);
  // 2. 添加初始IMU数据到队列
  extrapolator->AddImuData(imu_data);
  // 3. 创建IMU跟踪器，用于根据IMU数据估计姿态
  extrapolator->imu_tracker_ =
      absl::make_unique<ImuTracker>(imu_gravity_time_constant, imu_data.time);
  // 4. 用IMU线加速度初始化跟踪器（用于重力方向估计）
  extrapolator->imu_tracker_->AddImuLinearAccelerationObservation(
      imu_data.linear_acceleration);
  // 4. 用IMU角速度初始化跟踪器（用于姿态积分）
  extrapolator->imu_tracker_->AddImuAngularVelocityObservation(
      imu_data.angular_velocity);
  // 5. 推进IMU跟踪器到IMU数据的时间点
  extrapolator->imu_tracker_->Advance(imu_data.time);
  // 6. 添加初始位姿（只有旋转方向，无平移，基于IMU估计的重力方向）
  // 这是一个关键步骤：让外推器从IMU估计的姿态开始
  extrapolator->AddPose(
      imu_data.time,
      transform::Rigid3d::Rotation(extrapolator->imu_tracker_->orientation()));
  return extrapolator;
}

common::Time PoseExtrapolator::GetLastPoseTime() const {
  if (timed_pose_queue_.empty()) {
    return common::Time::min();
  }
  return timed_pose_queue_.back().time;
}

common::Time PoseExtrapolator::GetLastExtrapolatedTime() const {
  if (!extrapolation_imu_tracker_) {
    return common::Time::min();
  }
  return extrapolation_imu_tracker_->time();
}

/**
 * @brief 添加位姿到历史队列
 * 
 * 功能：将扫描匹配产生的位姿添加到队列，用于后续速度计算和外推
 * 
 * 处理步骤：
 * 1. 如果没有IMU跟踪器，创建新的
 * 2. 将新位姿加入队列 (timed_pose_queue_)
 * 3. 清理过期位姿（超出pose_queue_duration_时间的）
 * 4. 根据位姿序列更新速度估计
 * 5. 推进IMU跟踪器
 * 6. 清理过期的IMU和里程计数据
 * 
 * @param time 位姿时间戳
 * @param pose 机器人位姿 (Rigid3d: translation + rotation)
 */
void PoseExtrapolator::AddPose(const common::Time time,
                               const transform::Rigid3d& pose) {
  // 1. 如果没有IMU跟踪器，创建一个
  if (imu_tracker_ == nullptr) {
    common::Time tracker_start = time;
    if (!imu_data_.empty()) {
      tracker_start = std::min(tracker_start, imu_data_.front().time);
    }
    imu_tracker_ =
        absl::make_unique<ImuTracker>(gravity_time_constant_, tracker_start);
  }
  // 2. 将新位姿加入队列
  timed_pose_queue_.push_back(TimedPose{time, pose});
  // 3. 清理过期位姿（保持队列只包含一定时间范围内的位姿）
  while (timed_pose_queue_.size() > 2 &&
         timed_pose_queue_[1].time <= time - pose_queue_duration_) {
    timed_pose_queue_.pop_front();
  }
  // 4. 根据位姿序列计算线速度和角速度
  UpdateVelocitiesFromPoses();
  // 5. 推进IMU跟踪器到当前时间
  AdvanceImuTracker(time, imu_tracker_.get());
  // 6. 清理过期的IMU和里程计数据
  TrimImuData();
  TrimOdometryData();
  // 7. 复制IMU跟踪器用于外推
  odometry_imu_tracker_ = absl::make_unique<ImuTracker>(*imu_tracker_);
  extrapolation_imu_tracker_ = absl::make_unique<ImuTracker>(*imu_tracker_);
}

void PoseExtrapolator::AddImuData(const sensor::ImuData& imu_data) {
  CHECK(timed_pose_queue_.empty() ||
        imu_data.time >= timed_pose_queue_.back().time);
  imu_data_.push_back(imu_data);
  TrimImuData();
}

void PoseExtrapolator::AddOdometryData(
    const sensor::OdometryData& odometry_data) {
  CHECK(timed_pose_queue_.empty() ||
        odometry_data.time >= timed_pose_queue_.back().time);
  odometry_data_.push_back(odometry_data);
  TrimOdometryData();
  if (odometry_data_.size() < 2) {
    return;
  }
  // TODO(whess): Improve by using more than just the last two odometry poses.
  // Compute extrapolation in the tracking frame.
  const sensor::OdometryData& odometry_data_oldest = odometry_data_.front();
  const sensor::OdometryData& odometry_data_newest = odometry_data_.back();
  const double odometry_time_delta =
      common::ToSeconds(odometry_data_oldest.time - odometry_data_newest.time);
  const transform::Rigid3d odometry_pose_delta =
      odometry_data_newest.pose.inverse() * odometry_data_oldest.pose;
  angular_velocity_from_odometry_ =
      transform::RotationQuaternionToAngleAxisVector(
          odometry_pose_delta.rotation()) /
      odometry_time_delta;
  if (timed_pose_queue_.empty()) {
    return;
  }
  const Eigen::Vector3d
      linear_velocity_in_tracking_frame_at_newest_odometry_time =
          odometry_pose_delta.translation() / odometry_time_delta;
  const Eigen::Quaterniond orientation_at_newest_odometry_time =
      timed_pose_queue_.back().pose.rotation() *
      ExtrapolateRotation(odometry_data_newest.time,
                          odometry_imu_tracker_.get());
  linear_velocity_from_odometry_ =
      orientation_at_newest_odometry_time *
      linear_velocity_in_tracking_frame_at_newest_odometry_time;
}

/**
 * @brief 外推位姿
 * 
 * 功能：预测未来某时刻机器人的完整位姿
 * 
 * 算法原理：
 * - 平移预测 = 当前平移 + 外推的平移增量
 * - 旋转预测 = 当前旋转 × 外推的旋转增量
 * 
 * @param time 目标时间
 * @return transform::Rigid3d 外推后的位姿
 */
transform::Rigid3d PoseExtrapolator::ExtrapolatePose(const common::Time time) {
  const TimedPose& newest_timed_pose = timed_pose_queue_.back();
  CHECK_GE(time, newest_timed_pose.time);
  if (cached_extrapolated_pose_.time != time) {
    // 外推平移
    const Eigen::Vector3d translation =
        ExtrapolateTranslation(time) + newest_timed_pose.pose.translation();
    // 外推旋转（基于IMU角速度积分）
    const Eigen::Quaterniond rotation =
        newest_timed_pose.pose.rotation() *
        ExtrapolateRotation(time, extrapolation_imu_tracker_.get());
    cached_extrapolated_pose_ =
        TimedPose{time, transform::Rigid3d{translation, rotation}};
  }
  return cached_extrapolated_pose_.pose;
}

/**
 * @brief 估计重力方向
 * 
 * 功能：根据IMU数据估计重力方向
 * 
 * @param time 目标时间
 * @return Eigen::Quaterniond 从tracking frame到重力对齐frame的旋转
 */
Eigen::Quaterniond PoseExtrapolator::EstimateGravityOrientation(
    const common::Time time) {
  // 创建IMU跟踪器副本，推进到目标时间
  ImuTracker imu_tracker = *imu_tracker_;
  AdvanceImuTracker(time, &imu_tracker);
  return imu_tracker.orientation();
}

/**
 * @brief 外推多个时间点的位姿（带重力方向）
 * 
 * 功能：批量外推一组时间点的位姿，用于运动补偿
 * 
 * @param times 时间点列表
 * @return ExtrapolationResult 包含：
 *         - previous_poses: 外推的位姿序列
 *         - current_pose: 当前位姿
 *         - current_velocity: 当前速度
 *         - gravity_from_tracking: 重力方向
 */
PoseExtrapolator::ExtrapolationResult
PoseExtrapolator::ExtrapolatePosesWithGravity(
    const std::vector<common::Time>& times) {
  // 外推除最后一个时间点之外的所有位姿
  std::vector<transform::Rigid3f> poses;
  for (auto it = times.begin(); it != std::prev(times.end()); ++it) {
    poses.push_back(ExtrapolatePose(*it).cast<float>());
  }

  // 选择速度源：有里程计用里程计速度，否则用位姿速度
  const Eigen::Vector3d current_velocity = odometry_data_.size() < 2
                                               ? linear_velocity_from_poses_
                                               : linear_velocity_from_odometry_;
  return ExtrapolationResult{poses, ExtrapolatePose(times.back()),
                             current_velocity,
                             EstimateGravityOrientation(times.back())};
}

void PoseExtrapolator::TrimImuData() {
  while (imu_data_.size() > 1 && !timed_pose_queue_.empty() &&
         imu_data_[1].time <= timed_pose_queue_.back().time) {
    imu_data_.pop_front();
  }
}

void PoseExtrapolator::TrimOdometryData() {
  while (odometry_data_.size() > 2 && !timed_pose_queue_.empty() &&
         odometry_data_[1].time <= timed_pose_queue_.back().time) {
    odometry_data_.pop_front();
  }
}

void PoseExtrapolator::AdvanceImuTracker(const common::Time time,
                                         ImuTracker* const imu_tracker) const {
  CHECK_GE(time, imu_tracker->time());
  if (imu_data_.empty() || time < imu_data_.front().time) {
    // There is no IMU data until 'time', so we advance the ImuTracker and use
    // the angular velocities from poses and fake gravity to help 2D stability.
    imu_tracker->Advance(time);
    imu_tracker->AddImuLinearAccelerationObservation(Eigen::Vector3d::UnitZ());
    imu_tracker->AddImuAngularVelocityObservation(
        odometry_data_.size() < 2 ? angular_velocity_from_poses_
                                  : angular_velocity_from_odometry_);
    return;
  }
  if (imu_tracker->time() < imu_data_.front().time) {
    // Advance to the beginning of 'imu_data_'.
    imu_tracker->Advance(imu_data_.front().time);
  }
  auto it = std::lower_bound(
      imu_data_.begin(), imu_data_.end(), imu_tracker->time(),
      [](const sensor::ImuData& imu_data, const common::Time& time) {
        return imu_data.time < time;
      });
  while (it != imu_data_.end() && it->time < time) {
    imu_tracker->Advance(it->time);
    imu_tracker->AddImuLinearAccelerationObservation(it->linear_acceleration);
    imu_tracker->AddImuAngularVelocityObservation(it->angular_velocity);
    ++it;
  }
  imu_tracker->Advance(time);
}

/**
 * @brief 外推旋转姿态
 * 
 * 功能：根据IMU角速度积分，外推未来某时刻的旋转
 * 
 * 算法原理：
 * 1. 推进IMU跟踪器到目标时间
 * 2. 计算旋转增量 = last_orientation^-1 * current_orientation
 * 
 * @param time 目标时间
 * @param imu_tracker IMU跟踪器指针
 * @return Eigen::Quaterniond 旋转增量
 */
Eigen::Quaterniond PoseExtrapolator::ExtrapolateRotation(
    const common::Time time, ImuTracker* const imu_tracker) const {
  CHECK_GE(time, imu_tracker->time());
  // 推进IMU跟踪器到目标时间（积分角速度）
  AdvanceImuTracker(time, imu_tracker);
  // 计算从上次位姿到当前姿态的旋转增量
  const Eigen::Quaterniond last_orientation = imu_tracker_->orientation();
  return last_orientation.inverse() * imu_tracker->orientation();
}

/**
 * @brief 外推平移
 * 
 * 功能：根据线速度外推未来某时刻的平移
 * 
 * 算法原理：
 * - 如果没有里程计数据，使用位姿计算的线速度
 * - 如果有里程计数据，使用里程计计算的线速度（更精确）
 * - 平移增量 = 速度 × 时间差
 * 
 * @param time 目标时间
 * @return Eigen::Vector3d 平移增量
 */
Eigen::Vector3d PoseExtrapolator::ExtrapolateTranslation(common::Time time) {
  const TimedPose& newest_timed_pose = timed_pose_queue_.back();
  // 计算时间差
  const double extrapolation_delta =
      common::ToSeconds(time - newest_timed_pose.time);
  // 平移增量 = 速度 × 时间差
  if (odometry_data_.size() < 2) {
    // 使用位姿计算的线速度
    return extrapolation_delta * linear_velocity_from_poses_;
  }
  // 使用里程计计算的线速度（优先）
  return extrapolation_delta * linear_velocity_from_odometry_;
}

}  // namespace mapping
}  // namespace cartographer
