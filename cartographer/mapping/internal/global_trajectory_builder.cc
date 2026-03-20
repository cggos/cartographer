/*
 * GlobalTrajectoryBuilder - 全局轨迹构建器
 * 
 * 功能：连接前端局部SLAM和后端全局SLAM的核心类
 * 
 * 设计模式：模板方法模式
 * - 模板参数 LocalTrajectoryBuilder: LocalTrajectoryBuilder2D 或 LocalTrajectoryBuilder3D
 * - 模板参数 PoseGraph: PoseGraph2D 或 PoseGraph3D
 * 
 * 数据流：
 * ┌─────────────────────────────────────────────────────────────┐
 * │ SensorBridge → CollatedTrajectoryBuilder                   │
 * │                         ↓                                  │
 * │              GlobalTrajectoryBuilder::AddSensorData()     │
 * │                         ↓                                  │
 * │    ┌──────────────────────────────────────────────┐        │
 * │    │     LocalTrajectoryBuilder (前端局部SLAM)    │        │
 * │    │  - AddRangeData(): 点云累积、运动补偿         │        │
 * │    │  - ScanMatch(): 扫描匹配                     │        │
 * │    │  - InsertIntoSubmap(): 子图插入              │        │
 * │    └──────────────────────────────────────────────┘        │
 * │                         ↓                                  │
 * │    ┌──────────────────────────────────────────────┐        │
 * │    │     PoseGraph (后端全局SLAM)                 │        │
 * │    │  - AddNode(): 添加节点                       │        │
 * │    │  - ConstraintBuilder: 闭环检测               │        │
 * │    │  - OptimizationProblem: 图优化               │        │
 * │    └──────────────────────────────────────────────┘        │
 * └─────────────────────────────────────────────────────────────┘
 * 
 * 成员变量：
 * - local_trajectory_builder_: 局部SLAM构建器
 * - pose_graph_: 全局位姿图
 * - local_slam_result_callback_: 局部SLAM结果回调（用于发布TF）
 * 
 * Copyright 2018 The Cartographer Authors
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

#include "cartographer/mapping/internal/global_trajectory_builder.h"

#include <memory>

#include "absl/memory/memory.h"
#include "absl/types/optional.h"
#include "cartographer/common/time.h"
#include "cartographer/mapping/internal/local_slam_result_data.h"
#include "cartographer/mapping/internal/motion_filter.h"
#include "cartographer/metrics/family_factory.h"
#include "glog/logging.h"

namespace cartographer {
namespace mapping {
namespace {

static auto* kLocalSlamMatchingResults = metrics::Counter::Null();
static auto* kLocalSlamInsertionResults = metrics::Counter::Null();

template <typename LocalTrajectoryBuilder, typename PoseGraph>
class GlobalTrajectoryBuilder : public mapping::TrajectoryBuilderInterface {
 public:
  // Passing a 'nullptr' for 'local_trajectory_builder' is acceptable, but no
  // 'TimedPointCloudData' may be added in that case.
  GlobalTrajectoryBuilder(
      std::unique_ptr<LocalTrajectoryBuilder> local_trajectory_builder,
      const int trajectory_id, PoseGraph* const pose_graph,
      const LocalSlamResultCallback& local_slam_result_callback,
      const absl::optional<MotionFilter>& pose_graph_odometry_motion_filter)
      : trajectory_id_(trajectory_id),
        pose_graph_(pose_graph),
        local_trajectory_builder_(std::move(local_trajectory_builder)),
        local_slam_result_callback_(local_slam_result_callback),
        pose_graph_odometry_motion_filter_(pose_graph_odometry_motion_filter) {}
  ~GlobalTrajectoryBuilder() override {}

  GlobalTrajectoryBuilder(const GlobalTrajectoryBuilder&) = delete;
  GlobalTrajectoryBuilder& operator=(const GlobalTrajectoryBuilder&) = delete;

/**
 * @brief 添加激光点云数据 - 核心SLAM处理入口
 * 
 * 功能：
 * 1. 调用 LocalTrajectoryBuilder 进行局部SLAM处理
 * 2. 将结果添加到 PoseGraph 进行全局优化
 * 3. 触发回调发布TF
 * 
 * 调用链：
 * Dispatchable::AddToTrajectoryBuilder()
 *     ↓
 * GlobalTrajectoryBuilder::AddSensorData(TimedPointCloudData)
 *     ↓
 * LocalTrajectoryBuilder::AddRangeData()
 *     ↓
 * 返回 MatchingResult {time, local_pose, range_data, insertion_result}
 *     ↓
 * PoseGraph::AddNode() ← 添加到后端
 *     ↓
 * local_slam_result_callback_() ← 发布TF
 * 
 * @param sensor_id 传感器ID
 * @param timed_point_cloud_data 激光点云数据
 */
void AddSensorData(
    const std::string& sensor_id,
    const sensor::TimedPointCloudData& timed_point_cloud_data) override {
  // ========== 步骤1: 检查LocalTrajectoryBuilder ==========
  CHECK(local_trajectory_builder_)
      << "Cannot add TimedPointCloudData without a LocalTrajectoryBuilder.";
      
  // ========== 步骤2: 调用局部SLAM处理 ==========
  // 这是局部SLAM的核心入口
  // 内部会进行：点云累积、运动补偿、扫描匹配、子图插入
  std::unique_ptr<typename LocalTrajectoryBuilder::MatchingResult>
      matching_result = local_trajectory_builder_->AddRangeData(
          sensor_id, timed_point_cloud_data);
          
  // 数据累积不足，返回继续等待
  if (matching_result == nullptr) {
    // The range data has not been fully accumulated yet.
    return;
  }
  kLocalSlamMatchingResults->Increment();
  
  // ========== 步骤3: 将节点添加到后端位姿图 ==========
  std::unique_ptr<InsertionResult> insertion_result;
  if (matching_result->insertion_result != nullptr) {
    kLocalSlamInsertionResults->Increment();
    // 添加节点到PoseGraph，用于后续闭环检测和优化
    auto node_id = pose_graph_->AddNode(
        matching_result->insertion_result->constant_data, trajectory_id_,
        matching_result->insertion_result->insertion_submaps);
    CHECK_EQ(node_id.trajectory_id, trajectory_id_);
    insertion_result = absl::make_unique<InsertionResult>(InsertionResult{
        node_id, matching_result->insertion_result->constant_data,
        std::vector<std::shared_ptr<const Submap>>(
            matching_result->insertion_result->insertion_submaps.begin(),
            matching_result->insertion_result->insertion_submaps.end())});
  }
  
  // ========== 步骤4: 触发回调发布TF ==========
  // Node类会订阅这个回调，发布 /tf (map → odom)
  if (local_slam_result_callback_) {
    local_slam_result_callback_(
        trajectory_id_, matching_result->time, matching_result->local_pose,
        std::move(matching_result->range_data_in_local),
        std::move(insertion_result));
  }
}

/**
 * @brief 添加IMU数据
 * 
 * 功能：
 * 1. 发送给 LocalTrajectoryBuilder，用于姿态外推
 * 2. 发送给 PoseGraph，用于后端优化
 * 
 * @param sensor_id 传感器ID
 * @param imu_data IMU数据
 */
void AddSensorData(const std::string& sensor_id,
                   const sensor::ImuData& imu_data) override {
  if (local_trajectory_builder_) {
    // 传递给前端：用于PoseExtrapolator姿态预测
    local_trajectory_builder_->AddImuData(imu_data);
  }
  // 传递给后端：IMU数据作为优化的约束
  pose_graph_->AddImuData(trajectory_id_, imu_data);
}

/**
 * @brief 添加里程计数据
 * 
 * 功能：
 * 1. 发送给 LocalTrajectoryBuilder，用于姿态外推
 * 2. 发送给 PoseGraph，用于后端优化
 * 
 * @param sensor_id 传感器ID
 * @param odometry_data 里程计数据
 */
void AddSensorData(const std::string& sensor_id,
                   const sensor::OdometryData& odometry_data) override {
  CHECK(odometry_data.pose.IsValid()) << odometry_data.pose;
  if (local_trajectory_builder_) {
    // 传递给前端：用于PoseExtrapolator姿态预测
    local_trajectory_builder_->AddOdometryData(odometry_data);
  }
    // TODO(MichaelGrupp): Instead of having an optional filter on this level,
    // odometry could be marginalized between nodes in the pose graph.
    // Related issue: cartographer-project/cartographer/#1768
    if (pose_graph_odometry_motion_filter_.has_value() &&
        pose_graph_odometry_motion_filter_.value().IsSimilar(
            odometry_data.time, odometry_data.pose)) {
      return;
    }
    pose_graph_->AddOdometryData(trajectory_id_, odometry_data);
  }

  void AddSensorData(
      const std::string& sensor_id,
      const sensor::FixedFramePoseData& fixed_frame_pose) override {
    if (fixed_frame_pose.pose.has_value()) {
      CHECK(fixed_frame_pose.pose.value().IsValid())
          << fixed_frame_pose.pose.value();
    }
    pose_graph_->AddFixedFramePoseData(trajectory_id_, fixed_frame_pose);
  }

  void AddSensorData(const std::string& sensor_id,
                     const sensor::LandmarkData& landmark_data) override {
    pose_graph_->AddLandmarkData(trajectory_id_, landmark_data);
  }

  void AddLocalSlamResultData(std::unique_ptr<mapping::LocalSlamResultData>
                                  local_slam_result_data) override {
    CHECK(!local_trajectory_builder_) << "Can't add LocalSlamResultData with "
                                         "local_trajectory_builder_ present.";
    local_slam_result_data->AddToPoseGraph(trajectory_id_, pose_graph_);
  }

 private:
  const int trajectory_id_;
  PoseGraph* const pose_graph_;
  std::unique_ptr<LocalTrajectoryBuilder> local_trajectory_builder_;
  LocalSlamResultCallback local_slam_result_callback_;
  absl::optional<MotionFilter> pose_graph_odometry_motion_filter_;
};

}  // namespace

std::unique_ptr<TrajectoryBuilderInterface> CreateGlobalTrajectoryBuilder2D(
    std::unique_ptr<LocalTrajectoryBuilder2D> local_trajectory_builder,
    const int trajectory_id, mapping::PoseGraph2D* const pose_graph,
    const TrajectoryBuilderInterface::LocalSlamResultCallback&
        local_slam_result_callback,
    const absl::optional<MotionFilter>& pose_graph_odometry_motion_filter) {
  return absl::make_unique<
      GlobalTrajectoryBuilder<LocalTrajectoryBuilder2D, mapping::PoseGraph2D>>(
      std::move(local_trajectory_builder), trajectory_id, pose_graph,
      local_slam_result_callback, pose_graph_odometry_motion_filter);
}

std::unique_ptr<TrajectoryBuilderInterface> CreateGlobalTrajectoryBuilder3D(
    std::unique_ptr<LocalTrajectoryBuilder3D> local_trajectory_builder,
    const int trajectory_id, mapping::PoseGraph3D* const pose_graph,
    const TrajectoryBuilderInterface::LocalSlamResultCallback&
        local_slam_result_callback,
    const absl::optional<MotionFilter>& pose_graph_odometry_motion_filter) {
  return absl::make_unique<
      GlobalTrajectoryBuilder<LocalTrajectoryBuilder3D, mapping::PoseGraph3D>>(
      std::move(local_trajectory_builder), trajectory_id, pose_graph,
      local_slam_result_callback, pose_graph_odometry_motion_filter);
}

void GlobalTrajectoryBuilderRegisterMetrics(metrics::FamilyFactory* factory) {
  auto* results = factory->NewCounterFamily(
      "mapping_global_trajectory_builder_local_slam_results",
      "Local SLAM results");
  kLocalSlamMatchingResults = results->Add({{"type", "MatchingResult"}});
  kLocalSlamInsertionResults = results->Add({{"type", "InsertionResult"}});
}

}  // namespace mapping
}  // namespace cartographer
