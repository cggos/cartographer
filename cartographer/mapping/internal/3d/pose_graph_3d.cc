/*
 * PoseGraph3D 实现文件
 * 
 * 核心算法：稀疏位姿调整（SPA）
 * 
 * 主要功能：
 * 1. AddNode: 添加节点到位姿图
 * 2. ComputeConstraintsForNode: 构建约束（内子图+外子图）
 * 3. RunOptimization: 运行Ceres优化
 * 4. HandleWorkQueue: 处理工作队列
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

#include "cartographer/mapping/internal/3d/pose_graph_3d.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>

#include "Eigen/Eigenvalues"
#include "absl/memory/memory.h"
#include "cartographer/common/math.h"
#include "cartographer/mapping/proto/pose_graph/constraint_builder_options.pb.h"
#include "cartographer/sensor/compressed_point_cloud.h"
#include "cartographer/sensor/internal/voxel_filter.h"
#include "cartographer/transform/transform.h"
#include "glog/logging.h"

namespace cartographer {
namespace mapping {

static auto* kWorkQueueDelayMetric = metrics::Gauge::Null();
static auto* kWorkQueueSizeMetric = metrics::Gauge::Null();
static auto* kConstraintsSameTrajectoryMetric = metrics::Gauge::Null();
static auto* kConstraintsDifferentTrajectoryMetric = metrics::Gauge::Null();
static auto* kActiveSubmapsMetric = metrics::Gauge::Null();
static auto* kFrozenSubmapsMetric = metrics::Gauge::Null();
static auto* kDeletedSubmapsMetric = metrics::Gauge::Null();

PoseGraph3D::PoseGraph3D(
    const proto::PoseGraphOptions& options,
    std::unique_ptr<optimization::OptimizationProblem3D> optimization_problem,
    common::ThreadPool* thread_pool)
    : options_(options),
      optimization_problem_(std::move(optimization_problem)),
      constraint_builder_(options_.constraint_builder_options(), thread_pool),
      thread_pool_(thread_pool) {}

PoseGraph3D::~PoseGraph3D() {
  WaitForAllComputations();
  absl::MutexLock locker(&work_queue_mutex_);
  CHECK(work_queue_ == nullptr);
}

/**
 * @brief 初始化全局子图位姿
 * 
 * 功能：为新插入的子图分配全局位姿
 * 
 * 算法逻辑：
 * 1. 如果只有1个子图（首次添加）
 *    - 在优化问题中添加子图
 *    - 从轨迹连接状态推断初始位姿
 * 
 * 2. 如果有2个子图（子图切换时）
 *    - 使用前一个子图的位姿推断新子图位姿
 *    - 公式: new_pose = old_pose * old_local^-1 * new_local
 * 
 * @param trajectory_id 轨迹ID
 * @param time 时间戳
 * @param insertion_submaps 插入的子图列表（1或2个）
 * @return std::vector<SubmapId> 子图ID列表
 */
std::vector<SubmapId> PoseGraph3D::InitializeGlobalSubmapPoses(
    const int trajectory_id, const common::Time time,
    const std::vector<std::shared_ptr<const Submap3D>>& insertion_submaps) {
  CHECK(!insertion_submaps.empty());
  const auto& submap_data = optimization_problem_->submap_data();
  if (insertion_submaps.size() == 1) {
    // If we don't already have an entry for the first submap, add one.
    if (submap_data.SizeOfTrajectoryOrZero(trajectory_id) == 0) {
      if (data_.initial_trajectory_poses.count(trajectory_id) > 0) {
        data_.trajectory_connectivity_state.Connect(
            trajectory_id,
            data_.initial_trajectory_poses.at(trajectory_id).to_trajectory_id,
            time);
      }
      optimization_problem_->AddSubmap(
          trajectory_id, ComputeLocalToGlobalTransform(
                             data_.global_submap_poses_3d, trajectory_id) *
                             insertion_submaps[0]->local_pose());
    }
    CHECK_EQ(1, submap_data.SizeOfTrajectoryOrZero(trajectory_id));
    const SubmapId submap_id{trajectory_id, 0};
    CHECK(data_.submap_data.at(submap_id).submap == insertion_submaps.front());
    return {submap_id};
  }
  CHECK_EQ(2, insertion_submaps.size());
  const auto end_it = submap_data.EndOfTrajectory(trajectory_id);
  CHECK(submap_data.BeginOfTrajectory(trajectory_id) != end_it);
  const SubmapId last_submap_id = std::prev(end_it)->id;
  if (data_.submap_data.at(last_submap_id).submap ==
      insertion_submaps.front()) {
    // In this case, 'last_submap_id' is the ID of 'insertions_submaps.front()'
    // and 'insertions_submaps.back()' is new.
    const auto& first_submap_pose = submap_data.at(last_submap_id).global_pose;
    optimization_problem_->AddSubmap(
        trajectory_id, first_submap_pose *
                           insertion_submaps[0]->local_pose().inverse() *
                           insertion_submaps[1]->local_pose());
    return {last_submap_id,
            SubmapId{trajectory_id, last_submap_id.submap_index + 1}};
  }
  CHECK(data_.submap_data.at(last_submap_id).submap ==
        insertion_submaps.back());
  const SubmapId front_submap_id{trajectory_id,
                                 last_submap_id.submap_index - 1};
  CHECK(data_.submap_data.at(front_submap_id).submap ==
        insertion_submaps.front());
  return {front_submap_id, last_submap_id};
}

NodeId PoseGraph3D::AppendNode(
    std::shared_ptr<const TrajectoryNode::Data> constant_data,
    const int trajectory_id,
    const std::vector<std::shared_ptr<const Submap3D>>& insertion_submaps,
    const transform::Rigid3d& optimized_pose) {
  absl::MutexLock locker(&mutex_);
  AddTrajectoryIfNeeded(trajectory_id);
  if (!CanAddWorkItemModifying(trajectory_id)) {
    LOG(WARNING) << "AddNode was called for finished or deleted trajectory.";
  }
  const NodeId node_id = data_.trajectory_nodes.Append(
      trajectory_id, TrajectoryNode{constant_data, optimized_pose});
  ++data_.num_trajectory_nodes;
  // Test if the 'insertion_submap.back()' is one we never saw before.
  if (data_.submap_data.SizeOfTrajectoryOrZero(trajectory_id) == 0 ||
      std::prev(data_.submap_data.EndOfTrajectory(trajectory_id))
              ->data.submap != insertion_submaps.back()) {
    // We grow 'data_.submap_data' as needed. This code assumes that the first
    // time we see a new submap is as 'insertion_submaps.back()'.
    const SubmapId submap_id =
        data_.submap_data.Append(trajectory_id, InternalSubmapData());
    data_.submap_data.at(submap_id).submap = insertion_submaps.back();
    LOG(INFO) << "Inserted submap " << submap_id << ".";
    kActiveSubmapsMetric->Increment();
  }
  return node_id;
}

/**
 * @brief 添加节点 - 后端SLAM核心入口
 * 
 * 功能：接收前端的节点数据，将其添加到位姿图
 * 
 * 处理流程：
 * 1. AppendNode: 将节点添加到内部数据结构
 * 2. AddWorkItem: 添加异步工作项
 * 3. ComputeConstraintsForNode: （异步）构建约束
 * 
 * @param constant_data 节点的常量数据（点云、位姿、时间）
 * @param trajectory_id 轨迹ID
 * @param insertion_submaps 插入的子图列表
 * @return NodeId 节点ID
 */
NodeId PoseGraph3D::AddNode(
    std::shared_ptr<const TrajectoryNode::Data> constant_data,
    const int trajectory_id,
    const std::vector<std::shared_ptr<const Submap3D>>& insertion_submaps) {
  const transform::Rigid3d optimized_pose(
      GetLocalToGlobalTransform(trajectory_id) * constant_data->local_pose);

  const NodeId node_id = AppendNode(constant_data, trajectory_id,
                                    insertion_submaps, optimized_pose);
  // We have to check this here, because it might have changed by the time we
  // execute the lambda.
  const bool newly_finished_submap =
      insertion_submaps.front()->insertion_finished();
  AddWorkItem([=]() LOCKS_EXCLUDED(mutex_) {
    return ComputeConstraintsForNode(node_id, insertion_submaps,
                                     newly_finished_submap);
  });
  return node_id;
}

/**
 * @brief 添加工作项到队列
 * 
 * 功能：将任务添加到工作队列，由线程池异步执行
 * 
 * 工作流程：
 * 1. 如果没有工作队列，创建一个新的
 * 2. 将任务添加到队列末尾
 * 3. 调度DrainWorkQueue()到线程池执行
 * 
 * @param work_item 工作项函数
 */
void PoseGraph3D::AddWorkItem(
    const std::function<WorkItem::Result()>& work_item) {
  absl::MutexLock locker(&work_queue_mutex_);
  if (work_queue_ == nullptr) {
    work_queue_ = absl::make_unique<WorkQueue>();
    auto task = absl::make_unique<common::Task>();
    task->SetWorkItem([this]() { DrainWorkQueue(); });
    thread_pool_->Schedule(std::move(task));
  }
  const auto now = std::chrono::steady_clock::now();
  work_queue_->push_back({now, work_item});
  kWorkQueueSizeMetric->Set(work_queue_->size());
  kWorkQueueDelayMetric->Set(
      std::chrono::duration_cast<std::chrono::duration<double>>(
          now - work_queue_->front().time)
          .count());
}

void PoseGraph3D::AddTrajectoryIfNeeded(const int trajectory_id) {
  data_.trajectories_state[trajectory_id];
  CHECK(data_.trajectories_state.at(trajectory_id).state !=
        TrajectoryState::FINISHED);
  CHECK(data_.trajectories_state.at(trajectory_id).state !=
        TrajectoryState::DELETED);
  CHECK(data_.trajectories_state.at(trajectory_id).deletion_state ==
        InternalTrajectoryState::DeletionState::NORMAL);
  data_.trajectory_connectivity_state.Add(trajectory_id);
  // Make sure we have a sampler for this trajectory.
  if (!global_localization_samplers_[trajectory_id]) {
    global_localization_samplers_[trajectory_id] =
        absl::make_unique<common::FixedRatioSampler>(
            options_.global_sampling_ratio());
  }
}

void PoseGraph3D::AddImuData(const int trajectory_id,
                             const sensor::ImuData& imu_data) {
  AddWorkItem([=]() LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock locker(&mutex_);
    if (CanAddWorkItemModifying(trajectory_id)) {
      optimization_problem_->AddImuData(trajectory_id, imu_data);
    }
    return WorkItem::Result::kDoNotRunOptimization;
  });
}

void PoseGraph3D::AddOdometryData(const int trajectory_id,
                                  const sensor::OdometryData& odometry_data) {
  AddWorkItem([=]() LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock locker(&mutex_);
    if (CanAddWorkItemModifying(trajectory_id)) {
      optimization_problem_->AddOdometryData(trajectory_id, odometry_data);
    }
    return WorkItem::Result::kDoNotRunOptimization;
  });
}

void PoseGraph3D::AddFixedFramePoseData(
    const int trajectory_id,
    const sensor::FixedFramePoseData& fixed_frame_pose_data) {
  AddWorkItem([=]() LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock locker(&mutex_);
    if (CanAddWorkItemModifying(trajectory_id)) {
      optimization_problem_->AddFixedFramePoseData(trajectory_id,
                                                   fixed_frame_pose_data);
    }
    return WorkItem::Result::kDoNotRunOptimization;
  });
}

void PoseGraph3D::AddLandmarkData(int trajectory_id,
                                  const sensor::LandmarkData& landmark_data) {
  AddWorkItem([=]() LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock locker(&mutex_);
    if (CanAddWorkItemModifying(trajectory_id)) {
      for (const auto& observation : landmark_data.landmark_observations) {
        data_.landmark_nodes[observation.id].landmark_observations.emplace_back(
            PoseGraphInterface::LandmarkNode::LandmarkObservation{
                trajectory_id, landmark_data.time,
                observation.landmark_to_tracking_transform,
                observation.translation_weight, observation.rotation_weight});
      }
    }
    return WorkItem::Result::kDoNotRunOptimization;
  });
}

/**
 * @brief 计算单个节点与子图的约束
 * 
 * 功能：判断是否需要为节点和子图构建约束，并调用约束构建器
 * 
 * 约束类型判断：
 * 1. 本地约束 (maybe_add_local_constraint):
 *    - 同一轨迹内的节点和子图
 *    - 或者轨迹最近有连接
 * 
 * 2. 全局约束 (maybe_add_global_constraint):
 *    - 不同轨迹且距离上次连接超过阈值
 *    - 通过全局定位采样器决定是否添加
 * 
 * @param node_id 节点ID
 * @param submap_id 子图ID
 */
void PoseGraph3D::ComputeConstraint(const NodeId& node_id,
                                    const SubmapId& submap_id) {
  const transform::Rigid3d global_node_pose =
      optimization_problem_->node_data().at(node_id).global_pose;

  const transform::Rigid3d global_submap_pose =
      optimization_problem_->submap_data().at(submap_id).global_pose;

  bool maybe_add_local_constraint = false;
  bool maybe_add_global_constraint = false;
  const TrajectoryNode::Data* constant_data;
  const Submap3D* submap;
  {
    absl::MutexLock locker(&mutex_);
    CHECK(data_.submap_data.at(submap_id).state == SubmapState::kFinished);
    if (!data_.submap_data.at(submap_id).submap->insertion_finished()) {
      // Uplink server only receives grids when they are finished, so skip
      // constraint search before that.
      return;
    }

    const common::Time node_time = GetLatestNodeTime(node_id, submap_id);
    const common::Time last_connection_time =
        data_.trajectory_connectivity_state.LastConnectionTime(
            node_id.trajectory_id, submap_id.trajectory_id);
    if (node_id.trajectory_id == submap_id.trajectory_id ||
        node_time <
            last_connection_time +
                common::FromSeconds(
                    options_.global_constraint_search_after_n_seconds())) {
      // If the node and the submap belong to the same trajectory or if there
      // has been a recent global constraint that ties that node's trajectory to
      // the submap's trajectory, it suffices to do a match constrained to a
      // local search window.
      maybe_add_local_constraint = true;
    } else if (global_localization_samplers_[node_id.trajectory_id]->Pulse()) {
      // In this situation, 'global_node_pose' and 'global_submap_pose' have
      // orientations agreeing on gravity. Their relationship regarding yaw is
      // arbitrary. Finding the correct yaw component will be handled by the
      // matching procedure in the FastCorrelativeScanMatcher, and the given yaw
      // is essentially ignored.
      maybe_add_global_constraint = true;
    }
    constant_data = data_.trajectory_nodes.at(node_id).constant_data.get();
    submap = static_cast<const Submap3D*>(
        data_.submap_data.at(submap_id).submap.get());
  }

  if (maybe_add_local_constraint) {
    constraint_builder_.MaybeAddConstraint(submap_id, submap, node_id,
                                           constant_data, global_node_pose,
                                           global_submap_pose);
  } else if (maybe_add_global_constraint) {
    constraint_builder_.MaybeAddGlobalConstraint(
        submap_id, submap, node_id, constant_data, global_node_pose.rotation(),
        global_submap_pose.rotation());
  }
}

/**
 * @brief 计算节点的约束 - 核心约束构建函数
 * 
 * 功能：为一个节点构建所有可能的约束
 * 
 * 约束类型：
 * 1. INTRA_SUBMAP: 内子图约束（子图内节点与子图的相对位姿）
 * 2. INTER_SUBMAP: 外子图约束（闭环检测）
 * 
 * 处理流程：
 * 1. 初始化子图位姿（InitializeGlobalSubmapPoses）
 * 2. 添加内子图约束
 * 3. 遍历所有完成的子图，添加外子图约束
 * 4. 如果有新完成的子图，为旧节点添加约束
 * 
 * @param node_id 节点ID
 * @param insertion_submaps 插入的子图
 * @param newly_finished_submap 是否有新完成的子图
 * @return WorkItem::Result 是否需要运行优化
 */
WorkItem::Result PoseGraph3D::ComputeConstraintsForNode(
    const NodeId& node_id,
    std::vector<std::shared_ptr<const Submap3D>> insertion_submaps,
    const bool newly_finished_submap) {
  std::vector<SubmapId> submap_ids;
  std::vector<SubmapId> finished_submap_ids;
  std::set<NodeId> newly_finished_submap_node_ids;
  {
    absl::MutexLock locker(&mutex_);
    const auto& constant_data =
        data_.trajectory_nodes.at(node_id).constant_data;
    
    // 1. 初始化插入子图的全局位姿
    submap_ids = InitializeGlobalSubmapPoses(
        node_id.trajectory_id, constant_data->time, insertion_submaps);
    CHECK_EQ(submap_ids.size(), insertion_submaps.size());
    
    // 2. 计算并添加节点到优化问题中
    const SubmapId matching_id = submap_ids.front();
    const transform::Rigid3d& local_pose = constant_data->local_pose;
    // 计算节点的全局位姿：利用子图的全局位姿和节点在子图中的相对位姿
    const transform::Rigid3d global_pose =
        optimization_problem_->submap_data().at(matching_id).global_pose *
        insertion_submaps.front()->local_pose().inverse() * local_pose;
    optimization_problem_->AddTrajectoryNode(
        matching_id.trajectory_id,
        optimization::NodeSpec3D{constant_data->time, local_pose, global_pose});
    
    // 3. 添加内子图约束 (INTRA_SUBMAP)
    // 节点与其插入的子图之间的约束，这些约束是基于局部轨迹推算的，非常可靠
    for (size_t i = 0; i < insertion_submaps.size(); ++i) {
      const SubmapId submap_id = submap_ids[i];
      // 确保子图还处于未搜索约束的状态
      CHECK(data_.submap_data.at(submap_id).state ==
            SubmapState::kNoConstraintSearch);
      data_.submap_data.at(submap_id).node_ids.emplace(node_id);
      // 计算节点相对于子图的位姿变换
      const transform::Rigid3d constraint_transform =
          insertion_submaps[i]->local_pose().inverse() * local_pose;
      data_.constraints.push_back(Constraint{
          submap_id,
          node_id,
          {constraint_transform, options_.matcher_translation_weight(),
           options_.matcher_rotation_weight()},
          Constraint::INTRA_SUBMAP});
    }
    
    // 4. 收集所有已完成的子图 ID，用于后续寻找闭环约束 (INTER_SUBMAP)
    for (const auto& submap_id_data : data_.submap_data) {
      if (submap_id_data.data.state == SubmapState::kFinished) {
        CHECK_EQ(submap_id_data.data.node_ids.count(node_id), 0);
        finished_submap_ids.emplace_back(submap_id_data.id);
      }
    }
    
    // 5. 如果有子图新完成，更新其状态并记录其中的节点
    if (newly_finished_submap) {
      const SubmapId newly_finished_submap_id = submap_ids.front();
      InternalSubmapData& finished_submap_data =
          data_.submap_data.at(newly_finished_submap_id);
      CHECK(finished_submap_data.state == SubmapState::kNoConstraintSearch);
      finished_submap_data.state = SubmapState::kFinished;
      newly_finished_submap_node_ids = finished_submap_data.node_ids;
    }
  }

  // 6. 为当前节点寻找与所有已完成子图之间的约束 (闭环检测)
  for (const auto& submap_id : finished_submap_ids) {
    ComputeConstraint(node_id, submap_id);
  }

  // 7. 如果有新完成的子图，为所有旧节点寻找与该新子图之间的约束
  if (newly_finished_submap) {
    const SubmapId newly_finished_submap_id = submap_ids.front();
    for (const auto& node_id_data : optimization_problem_->node_data()) {
      const NodeId& node_id = node_id_data.id;
      // 排除掉该子图自身的节点，因为内子图约束已经添加过了
      if (newly_finished_submap_node_ids.count(node_id) == 0) {
        ComputeConstraint(node_id, newly_finished_submap_id);
      }
    }
  }
  
  // 通知约束构建器该节点处理完成
  constraint_builder_.NotifyEndOfNode();
  
  // 8. 检查是否达到优化周期
  absl::MutexLock locker(&mutex_);
  ++num_nodes_since_last_loop_closure_;
  if (options_.optimize_every_n_nodes() > 0 &&
      num_nodes_since_last_loop_closure_ > options_.optimize_every_n_nodes()) {
    return WorkItem::Result::kRunOptimization;
  }
  return WorkItem::Result::kDoNotRunOptimization;
}

common::Time PoseGraph3D::GetLatestNodeTime(const NodeId& node_id,
                                            const SubmapId& submap_id) const {
  common::Time time = data_.trajectory_nodes.at(node_id).constant_data->time;
  const InternalSubmapData& submap_data = data_.submap_data.at(submap_id);
  if (!submap_data.node_ids.empty()) {
    const NodeId last_submap_node_id =
        *data_.submap_data.at(submap_id).node_ids.rbegin();
    time = std::max(
        time,
        data_.trajectory_nodes.at(last_submap_node_id).constant_data->time);
  }
  return time;
}

void PoseGraph3D::UpdateTrajectoryConnectivity(const Constraint& constraint) {
  CHECK_EQ(constraint.tag, PoseGraphInterface::Constraint::INTER_SUBMAP);
  const common::Time time =
      GetLatestNodeTime(constraint.node_id, constraint.submap_id);
  data_.trajectory_connectivity_state.Connect(
      constraint.node_id.trajectory_id, constraint.submap_id.trajectory_id,
      time);
}

void PoseGraph3D::DeleteTrajectoriesIfNeeded() {
  TrimmingHandle trimming_handle(this);
  for (auto& it : data_.trajectories_state) {
    if (it.second.deletion_state ==
        InternalTrajectoryState::DeletionState::WAIT_FOR_DELETION) {
      // TODO(gaschler): Consider directly deleting from data_, which may be
      // more complete.
      auto submap_ids = trimming_handle.GetSubmapIds(it.first);
      for (auto& submap_id : submap_ids) {
        trimming_handle.TrimSubmap(submap_id);
      }
      it.second.state = TrajectoryState::DELETED;
      it.second.deletion_state = InternalTrajectoryState::DeletionState::NORMAL;
    }
  }
}

/**
 * @brief 处理工作队列 - 约束构建完成回调
 * 
 * 功能：约束构建完成后，进行优化并处理后续任务
 * 
 * 处理流程：
 * 1. 将新约束添加到data_.constraints
 * 2. 运行位姿图优化
 * 3. 调用全局SLAM回调（如发布TF）
 * 4. 更新轨迹连接状态
 * 5. 执行trimmer（裁剪过期数据）
 * 6. 继续处理工作队列
 * 
 * @param result 约束构建器返回的约束结果
 */
void PoseGraph3D::HandleWorkQueue(
    const constraints::ConstraintBuilder3D::Result& result) {
  // ========== 步骤1: 添加新约束 ==========
  {
    absl::MutexLock locker(&mutex_);
    data_.constraints.insert(data_.constraints.end(), result.begin(),
                             result.end());
  }
  // ========== 步骤2: 运行优化 ==========
  RunOptimization();

  // ========== 步骤3: 调用全局SLAM回调 ==========
  if (global_slam_optimization_callback_) {
    std::map<int, NodeId> trajectory_id_to_last_optimized_node_id;
    std::map<int, SubmapId> trajectory_id_to_last_optimized_submap_id;
    {
      absl::MutexLock locker(&mutex_);
      const auto& submap_data = optimization_problem_->submap_data();
      const auto& node_data = optimization_problem_->node_data();
      for (const int trajectory_id : node_data.trajectory_ids()) {
        if (node_data.SizeOfTrajectoryOrZero(trajectory_id) == 0 ||
            submap_data.SizeOfTrajectoryOrZero(trajectory_id) == 0) {
          continue;
        }
        trajectory_id_to_last_optimized_node_id.emplace(
            trajectory_id,
            std::prev(node_data.EndOfTrajectory(trajectory_id))->id);
        trajectory_id_to_last_optimized_submap_id.emplace(
            trajectory_id,
            std::prev(submap_data.EndOfTrajectory(trajectory_id))->id);
      }
    }
    global_slam_optimization_callback_(
        trajectory_id_to_last_optimized_submap_id,
        trajectory_id_to_last_optimized_node_id);
  }

  {
    absl::MutexLock locker(&mutex_);
    // ========== 步骤4: 更新轨迹连接状态 ==========
    for (const Constraint& constraint : result) {
      UpdateTrajectoryConnectivity(constraint);
    }
    // ========== 步骤5: 删除待删除轨迹 ==========
    DeleteTrajectoriesIfNeeded();
    
    // ========== 步骤6: 执行trimmer（裁剪） ==========
    TrimmingHandle trimming_handle(this);
    for (auto& trimmer : trimmers_) {
      trimmer->Trim(&trimming_handle);
    }
    trimmers_.erase(
        std::remove_if(trimmers_.begin(), trimmers_.end(),
                       [](std::unique_ptr<PoseGraphTrimmer>& trimmer) {
                         return trimmer->IsFinished();
                       }),
        trimmers_.end());

    num_nodes_since_last_loop_closure_ = 0;
  }

  // ========== 步骤7: 继续处理工作队列 ==========
  DrainWorkQueue();
}

/**
 * @brief 排空工作队列 - 循环处理工作项
 * 
 * 功能：从工作队列中取出工作项并执行，直到队列为空或需要优化
 * 
 * 处理流程：
 * 1. 从队列中取出工作项
 * 2. 执行工作项
 * 3. 如果需要优化，调用约束构建器
 * 4. 如果不需要优化，继续处理下一个工作项
 */
void PoseGraph3D::DrainWorkQueue() {
  bool process_work_queue = true;
  size_t work_queue_size;
  while (process_work_queue) {
    std::function<WorkItem::Result()> work_item;
    {
      absl::MutexLock locker(&work_queue_mutex_);
      if (work_queue_->empty()) {
        work_queue_.reset();
        return;
      }
      work_item = work_queue_->front().task;
      work_queue_->pop_front();
      work_queue_size = work_queue_->size();
      kWorkQueueSizeMetric->Set(work_queue_size);
    }
    // 执行工作项，判断是否需要运行优化
    process_work_queue = work_item() == WorkItem::Result::kDoNotRunOptimization;
  }
  LOG(INFO) << "Remaining work items in queue: " << work_queue_size;
  // ========== 需要运行优化 ==========
  // 调用约束构建器WhenDone回调，在约束构建完成后处理工作队列
  constraint_builder_.WhenDone(
      [this](const constraints::ConstraintBuilder3D::Result& result) {
        HandleWorkQueue(result);
      });
}

void PoseGraph3D::WaitForAllComputations() {
  int num_trajectory_nodes;
  {
    absl::MutexLock locker(&mutex_);
    num_trajectory_nodes = data_.num_trajectory_nodes;
  }

  const int num_finished_nodes_at_start =
      constraint_builder_.GetNumFinishedNodes();

  auto report_progress = [this, num_trajectory_nodes,
                          num_finished_nodes_at_start]() {
    // Log progress on nodes only when we are actually processing nodes.
    if (num_trajectory_nodes != num_finished_nodes_at_start) {
      std::ostringstream progress_info;
      progress_info << "Optimizing: " << std::fixed << std::setprecision(1)
                    << 100. *
                           (constraint_builder_.GetNumFinishedNodes() -
                            num_finished_nodes_at_start) /
                           (num_trajectory_nodes - num_finished_nodes_at_start)
                    << "%...";
      std::cout << "\r\x1b[K" << progress_info.str() << std::flush;
    }
  };

  // First wait for the work queue to drain so that it's safe to schedule
  // a WhenDone() callback.
  {
    const auto predicate = [this]()
                               EXCLUSIVE_LOCKS_REQUIRED(work_queue_mutex_) {
                                 return work_queue_ == nullptr;
                               };
    absl::MutexLock locker(&work_queue_mutex_);
    while (!work_queue_mutex_.AwaitWithTimeout(
        absl::Condition(&predicate),
        absl::FromChrono(common::FromSeconds(1.)))) {
      report_progress();
    }
  }

  // Now wait for any pending constraint computations to finish.
  absl::MutexLock locker(&mutex_);
  bool notification = false;
  constraint_builder_.WhenDone(
      [this,
       &notification](const constraints::ConstraintBuilder3D::Result& result)
          LOCKS_EXCLUDED(mutex_) {
            absl::MutexLock locker(&mutex_);
            data_.constraints.insert(data_.constraints.end(), result.begin(),
                                     result.end());
            notification = true;
          });
  const auto predicate = [&notification]() EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    return notification;
  };
  while (!mutex_.AwaitWithTimeout(absl::Condition(&predicate),
                                  absl::FromChrono(common::FromSeconds(1.)))) {
    report_progress();
  }
  CHECK_EQ(constraint_builder_.GetNumFinishedNodes(), num_trajectory_nodes);
  std::cout << "\r\x1b[KOptimizing: Done.     " << std::endl;
}

void PoseGraph3D::DeleteTrajectory(const int trajectory_id) {
  {
    absl::MutexLock locker(&mutex_);
    auto it = data_.trajectories_state.find(trajectory_id);
    if (it == data_.trajectories_state.end()) {
      LOG(WARNING) << "Skipping request to delete non-existing trajectory_id: "
                   << trajectory_id;
      return;
    }
    it->second.deletion_state =
        InternalTrajectoryState::DeletionState::SCHEDULED_FOR_DELETION;
  }
  AddWorkItem([this, trajectory_id]() LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock locker(&mutex_);
    CHECK(data_.trajectories_state.at(trajectory_id).state !=
          TrajectoryState::ACTIVE);
    CHECK(data_.trajectories_state.at(trajectory_id).state !=
          TrajectoryState::DELETED);
    CHECK(data_.trajectories_state.at(trajectory_id).deletion_state ==
          InternalTrajectoryState::DeletionState::SCHEDULED_FOR_DELETION);
    data_.trajectories_state.at(trajectory_id).deletion_state =
        InternalTrajectoryState::DeletionState::WAIT_FOR_DELETION;
    return WorkItem::Result::kDoNotRunOptimization;
  });
}

void PoseGraph3D::FinishTrajectory(const int trajectory_id) {
  AddWorkItem([this, trajectory_id]() LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock locker(&mutex_);
    CHECK(!IsTrajectoryFinished(trajectory_id));
    data_.trajectories_state[trajectory_id].state = TrajectoryState::FINISHED;

    for (const auto& submap : data_.submap_data.trajectory(trajectory_id)) {
      data_.submap_data.at(submap.id).state = SubmapState::kFinished;
    }
    return WorkItem::Result::kRunOptimization;
  });
}

bool PoseGraph3D::IsTrajectoryFinished(const int trajectory_id) const {
  return data_.trajectories_state.count(trajectory_id) != 0 &&
         data_.trajectories_state.at(trajectory_id).state ==
             TrajectoryState::FINISHED;
}

void PoseGraph3D::FreezeTrajectory(const int trajectory_id) {
  {
    absl::MutexLock locker(&mutex_);
    data_.trajectory_connectivity_state.Add(trajectory_id);
  }
  AddWorkItem([this, trajectory_id]() LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock locker(&mutex_);
    CHECK(!IsTrajectoryFrozen(trajectory_id));
    // Connect multiple frozen trajectories among each other.
    // This is required for localization against multiple frozen trajectories
    // because we lose inter-trajectory constraints when freezing.
    for (const auto& entry : data_.trajectories_state) {
      const int other_trajectory_id = entry.first;
      if (!IsTrajectoryFrozen(other_trajectory_id)) {
        continue;
      }
      if (data_.trajectory_connectivity_state.TransitivelyConnected(
              trajectory_id, other_trajectory_id)) {
        // Already connected, nothing to do.
        continue;
      }
      data_.trajectory_connectivity_state.Connect(
          trajectory_id, other_trajectory_id, common::FromUniversal(0));
    }
    data_.trajectories_state[trajectory_id].state = TrajectoryState::FROZEN;
    return WorkItem::Result::kDoNotRunOptimization;
  });
}

bool PoseGraph3D::IsTrajectoryFrozen(const int trajectory_id) const {
  return data_.trajectories_state.count(trajectory_id) != 0 &&
         data_.trajectories_state.at(trajectory_id).state ==
             TrajectoryState::FROZEN;
}

void PoseGraph3D::AddSubmapFromProto(
    const transform::Rigid3d& global_submap_pose, const proto::Submap& submap) {
  if (!submap.has_submap_3d()) {
    return;
  }

  const SubmapId submap_id = {submap.submap_id().trajectory_id(),
                              submap.submap_id().submap_index()};
  std::shared_ptr<const Submap3D> submap_ptr =
      std::make_shared<const Submap3D>(submap.submap_3d());

  {
    absl::MutexLock locker(&mutex_);
    AddTrajectoryIfNeeded(submap_id.trajectory_id);
    if (!CanAddWorkItemModifying(submap_id.trajectory_id)) return;
    data_.submap_data.Insert(submap_id, InternalSubmapData());
    data_.submap_data.at(submap_id).submap = submap_ptr;
    // Immediately show the submap at the 'global_submap_pose'.
    data_.global_submap_poses_3d.Insert(
        submap_id, optimization::SubmapSpec3D{global_submap_pose});
  }

  // TODO(MichaelGrupp): MapBuilder does freezing before deserializing submaps,
  // so this should be fine.
  if (IsTrajectoryFrozen(submap_id.trajectory_id)) {
    kFrozenSubmapsMetric->Increment();
  } else {
    kActiveSubmapsMetric->Increment();
  }

  AddWorkItem([this, submap_id, global_submap_pose]() LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock locker(&mutex_);
    data_.submap_data.at(submap_id).state = SubmapState::kFinished;
    optimization_problem_->InsertSubmap(submap_id, global_submap_pose);
    return WorkItem::Result::kDoNotRunOptimization;
  });
}

void PoseGraph3D::AddNodeFromProto(const transform::Rigid3d& global_pose,
                                   const proto::Node& node) {
  const NodeId node_id = {node.node_id().trajectory_id(),
                          node.node_id().node_index()};
  std::shared_ptr<const TrajectoryNode::Data> constant_data =
      std::make_shared<const TrajectoryNode::Data>(FromProto(node.node_data()));

  {
    absl::MutexLock locker(&mutex_);
    AddTrajectoryIfNeeded(node_id.trajectory_id);
    if (!CanAddWorkItemModifying(node_id.trajectory_id)) return;
    data_.trajectory_nodes.Insert(node_id,
                                  TrajectoryNode{constant_data, global_pose});
  }

  AddWorkItem([this, node_id, global_pose]() LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock locker(&mutex_);
    const auto& constant_data =
        data_.trajectory_nodes.at(node_id).constant_data;
    optimization_problem_->InsertTrajectoryNode(
        node_id,
        optimization::NodeSpec3D{constant_data->time, constant_data->local_pose,
                                 global_pose});
    return WorkItem::Result::kDoNotRunOptimization;
  });
}

void PoseGraph3D::SetTrajectoryDataFromProto(
    const proto::TrajectoryData& data) {
  TrajectoryData trajectory_data;
  trajectory_data.gravity_constant = data.gravity_constant();
  trajectory_data.imu_calibration = {
      {data.imu_calibration().w(), data.imu_calibration().x(),
       data.imu_calibration().y(), data.imu_calibration().z()}};
  if (data.has_fixed_frame_origin_in_map()) {
    trajectory_data.fixed_frame_origin_in_map =
        transform::ToRigid3(data.fixed_frame_origin_in_map());
  }

  const int trajectory_id = data.trajectory_id();
  AddWorkItem([this, trajectory_id, trajectory_data]() LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock locker(&mutex_);
    if (CanAddWorkItemModifying(trajectory_id)) {
      optimization_problem_->SetTrajectoryData(trajectory_id, trajectory_data);
    }
    return WorkItem::Result::kDoNotRunOptimization;
  });
}

void PoseGraph3D::AddNodeToSubmap(const NodeId& node_id,
                                  const SubmapId& submap_id) {
  AddWorkItem([this, node_id, submap_id]() LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock locker(&mutex_);
    if (CanAddWorkItemModifying(submap_id.trajectory_id)) {
      data_.submap_data.at(submap_id).node_ids.insert(node_id);
    }
    return WorkItem::Result::kDoNotRunOptimization;
  });
}

void PoseGraph3D::AddSerializedConstraints(
    const std::vector<Constraint>& constraints) {
  AddWorkItem([this, constraints]() LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock locker(&mutex_);
    for (const auto& constraint : constraints) {
      CHECK(data_.trajectory_nodes.Contains(constraint.node_id));
      CHECK(data_.submap_data.Contains(constraint.submap_id));
      CHECK(data_.trajectory_nodes.at(constraint.node_id).constant_data !=
            nullptr);
      CHECK(data_.submap_data.at(constraint.submap_id).submap != nullptr);
      switch (constraint.tag) {
        case Constraint::Tag::INTRA_SUBMAP:
          CHECK(data_.submap_data.at(constraint.submap_id)
                    .node_ids.emplace(constraint.node_id)
                    .second);
          break;
        case Constraint::Tag::INTER_SUBMAP:
          UpdateTrajectoryConnectivity(constraint);
          break;
      }
      data_.constraints.push_back(constraint);
    }
    LOG(INFO) << "Loaded " << constraints.size() << " constraints.";
    return WorkItem::Result::kDoNotRunOptimization;
  });
}

void PoseGraph3D::AddTrimmer(std::unique_ptr<PoseGraphTrimmer> trimmer) {
  // C++11 does not allow us to move a unique_ptr into a lambda.
  PoseGraphTrimmer* const trimmer_ptr = trimmer.release();
  AddWorkItem([this, trimmer_ptr]() LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock locker(&mutex_);
    trimmers_.emplace_back(trimmer_ptr);
    return WorkItem::Result::kDoNotRunOptimization;
  });
}

void PoseGraph3D::RunFinalOptimization() {
  {
    AddWorkItem([this]() LOCKS_EXCLUDED(mutex_) {
      absl::MutexLock locker(&mutex_);
      optimization_problem_->SetMaxNumIterations(
          options_.max_num_final_iterations());
      return WorkItem::Result::kRunOptimization;
    });
    AddWorkItem([this]() LOCKS_EXCLUDED(mutex_) {
      absl::MutexLock locker(&mutex_);
      optimization_problem_->SetMaxNumIterations(
          options_.optimization_problem_options()
              .ceres_solver_options()
              .max_num_iterations());
      return WorkItem::Result::kDoNotRunOptimization;
    });
  }
  WaitForAllComputations();
}

void PoseGraph3D::LogResidualHistograms() const {
  common::Histogram rotational_residual;
  common::Histogram translational_residual;
  for (const Constraint& constraint : data_.constraints) {
    if (constraint.tag == Constraint::Tag::INTRA_SUBMAP) {
      const cartographer::transform::Rigid3d optimized_node_to_map =
          data_.trajectory_nodes.at(constraint.node_id).global_pose;
      const cartographer::transform::Rigid3d node_to_submap_constraint =
          constraint.pose.zbar_ij;
      const cartographer::transform::Rigid3d optimized_submap_to_map =
          data_.global_submap_poses_3d.at(constraint.submap_id).global_pose;
      const cartographer::transform::Rigid3d optimized_node_to_submap =
          optimized_submap_to_map.inverse() * optimized_node_to_map;
      const cartographer::transform::Rigid3d residual =
          node_to_submap_constraint.inverse() * optimized_node_to_submap;
      rotational_residual.Add(
          common::NormalizeAngleDifference(transform::GetAngle(residual)));
      translational_residual.Add(residual.translation().norm());
    }
  }
  LOG(INFO) << "Translational residuals histogram:\n"
            << translational_residual.ToString(10);
  LOG(INFO) << "Rotational residuals histogram:\n"
            << rotational_residual.ToString(10);
}

/**
 * @brief 运行位姿图优化 - 核心优化函数
 * 
 * 功能：使用Ceres求解器优化位姿图
 * 
 * 优化目标：
 * - 最小化所有约束的残差
 * - 同时考虑IMU、里程计、GPS等约束
 * 
 * 处理流程：
 * 1. 调用optimization_problem_->Solve()进行Ceres优化
 * 2. 更新节点全局位姿
 * 3. 外推未优化节点的位姿
 * 4. 更新路标位姿
 * 5. 更新子图位姿
 */
void PoseGraph3D::RunOptimization() {
  if (optimization_problem_->submap_data().empty()) {
    return;
  }

  // ========== 步骤1: 运行Ceres优化 ==========
  // Solve()是时间消耗最大的操作，所以不持有mutex以避免阻塞前台处理
  optimization_problem_->Solve(data_.constraints, GetTrajectoryStates(),
                               data_.landmark_nodes);
  absl::MutexLock locker(&mutex_);

  // ========== 步骤2: 更新节点全局位姿 ==========
  const auto& submap_data = optimization_problem_->submap_data();
  const auto& node_data = optimization_problem_->node_data();
  for (const int trajectory_id : node_data.trajectory_ids()) {
    // 更新已优化节点的全局位姿
    for (const auto& node : node_data.trajectory(trajectory_id)) {
      data_.trajectory_nodes.at(node.id).global_pose = node.data.global_pose;
    }

    // ========== 步骤3: 外推未优化节点的位姿 ==========
    // 有些节点可能还没被优化，使用变换矩阵外推其位姿
    const auto local_to_new_global =
        ComputeLocalToGlobalTransform(submap_data, trajectory_id);
    const auto local_to_old_global = ComputeLocalToGlobalTransform(
        data_.global_submap_poses_3d, trajectory_id);
    const transform::Rigid3d old_global_to_new_global =
        local_to_new_global * local_to_old_global.inverse();

    const NodeId last_optimized_node_id =
        std::prev(node_data.EndOfTrajectory(trajectory_id))->id;
    auto node_it =
        std::next(data_.trajectory_nodes.find(last_optimized_node_id));
    for (; node_it != data_.trajectory_nodes.EndOfTrajectory(trajectory_id);
         ++node_it) {
      auto& mutable_trajectory_node = data_.trajectory_nodes.at(node_it->id);
      mutable_trajectory_node.global_pose =
          old_global_to_new_global * mutable_trajectory_node.global_pose;
    }
  }
  
  // ========== 步骤4: 更新路标位姿 ==========
  for (const auto& landmark : optimization_problem_->landmark_data()) {
    data_.landmark_nodes[landmark.first].global_landmark_pose = landmark.second;
  }
  
  // ========== 步骤5: 更新子图位姿 ==========
  data_.global_submap_poses_3d = submap_data;

  // Log the histograms for the pose residuals.
  if (options_.log_residual_histograms()) {
    LogResidualHistograms();
  }
}

bool PoseGraph3D::CanAddWorkItemModifying(int trajectory_id) {
  auto it = data_.trajectories_state.find(trajectory_id);
  if (it == data_.trajectories_state.end()) {
    return true;
  }
  if (it->second.state == TrajectoryState::FINISHED) {
    // TODO(gaschler): Replace all FATAL to WARNING after some testing.
    LOG(FATAL) << "trajectory_id " << trajectory_id
               << " has finished "
                  "but modification is requested, skipping.";
    return false;
  }
  if (it->second.deletion_state !=
      InternalTrajectoryState::DeletionState::NORMAL) {
    LOG(FATAL) << "trajectory_id " << trajectory_id
               << " has been scheduled for deletion "
                  "but modification is requested, skipping.";
    return false;
  }
  if (it->second.state == TrajectoryState::DELETED) {
    LOG(FATAL) << "trajectory_id " << trajectory_id
               << " has been deleted "
                  "but modification is requested, skipping.";
    return false;
  }
  return true;
}

MapById<NodeId, TrajectoryNode> PoseGraph3D::GetTrajectoryNodes() const {
  absl::MutexLock locker(&mutex_);
  return data_.trajectory_nodes;
}

MapById<NodeId, TrajectoryNodePose> PoseGraph3D::GetTrajectoryNodePoses()
    const {
  MapById<NodeId, TrajectoryNodePose> node_poses;
  absl::MutexLock locker(&mutex_);
  for (const auto& node_id_data : data_.trajectory_nodes) {
    absl::optional<TrajectoryNodePose::ConstantPoseData> constant_pose_data;
    if (node_id_data.data.constant_data != nullptr) {
      constant_pose_data = TrajectoryNodePose::ConstantPoseData{
          node_id_data.data.constant_data->time,
          node_id_data.data.constant_data->local_pose};
    }
    node_poses.Insert(
        node_id_data.id,
        TrajectoryNodePose{node_id_data.data.global_pose, constant_pose_data});
  }
  return node_poses;
}

std::map<int, PoseGraphInterface::TrajectoryState>
PoseGraph3D::GetTrajectoryStates() const {
  std::map<int, PoseGraphInterface::TrajectoryState> trajectories_state;
  absl::MutexLock locker(&mutex_);
  for (const auto& it : data_.trajectories_state) {
    trajectories_state[it.first] = it.second.state;
  }
  return trajectories_state;
}

std::map<std::string, transform::Rigid3d> PoseGraph3D::GetLandmarkPoses()
    const {
  std::map<std::string, transform::Rigid3d> landmark_poses;
  absl::MutexLock locker(&mutex_);
  for (const auto& landmark : data_.landmark_nodes) {
    // Landmark without value has not been optimized yet.
    if (!landmark.second.global_landmark_pose.has_value()) continue;
    landmark_poses[landmark.first] =
        landmark.second.global_landmark_pose.value();
  }
  return landmark_poses;
}

void PoseGraph3D::SetLandmarkPose(const std::string& landmark_id,
                                  const transform::Rigid3d& global_pose,
                                  const bool frozen) {
  AddWorkItem([=]() LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock locker(&mutex_);
    data_.landmark_nodes[landmark_id].global_landmark_pose = global_pose;
    data_.landmark_nodes[landmark_id].frozen = frozen;
    return WorkItem::Result::kDoNotRunOptimization;
  });
}

sensor::MapByTime<sensor::ImuData> PoseGraph3D::GetImuData() const {
  absl::MutexLock locker(&mutex_);
  return optimization_problem_->imu_data();
}

sensor::MapByTime<sensor::OdometryData> PoseGraph3D::GetOdometryData() const {
  absl::MutexLock locker(&mutex_);
  return optimization_problem_->odometry_data();
}

sensor::MapByTime<sensor::FixedFramePoseData>
PoseGraph3D::GetFixedFramePoseData() const {
  absl::MutexLock locker(&mutex_);
  return optimization_problem_->fixed_frame_pose_data();
}

std::map<std::string /* landmark ID */, PoseGraphInterface::LandmarkNode>
PoseGraph3D::GetLandmarkNodes() const {
  absl::MutexLock locker(&mutex_);
  return data_.landmark_nodes;
}

std::map<int, PoseGraphInterface::TrajectoryData>
PoseGraph3D::GetTrajectoryData() const {
  absl::MutexLock locker(&mutex_);
  return optimization_problem_->trajectory_data();
}

std::vector<PoseGraphInterface::Constraint> PoseGraph3D::constraints() const {
  absl::MutexLock locker(&mutex_);
  return data_.constraints;
}

void PoseGraph3D::SetInitialTrajectoryPose(const int from_trajectory_id,
                                           const int to_trajectory_id,
                                           const transform::Rigid3d& pose,
                                           const common::Time time) {
  absl::MutexLock locker(&mutex_);
  data_.initial_trajectory_poses[from_trajectory_id] =
      InitialTrajectoryPose{to_trajectory_id, pose, time};
}

transform::Rigid3d PoseGraph3D::GetInterpolatedGlobalTrajectoryPose(
    const int trajectory_id, const common::Time time) const {
  CHECK_GT(data_.trajectory_nodes.SizeOfTrajectoryOrZero(trajectory_id), 0);
  const auto it = data_.trajectory_nodes.lower_bound(trajectory_id, time);
  if (it == data_.trajectory_nodes.BeginOfTrajectory(trajectory_id)) {
    return data_.trajectory_nodes.BeginOfTrajectory(trajectory_id)
        ->data.global_pose;
  }
  if (it == data_.trajectory_nodes.EndOfTrajectory(trajectory_id)) {
    return std::prev(data_.trajectory_nodes.EndOfTrajectory(trajectory_id))
        ->data.global_pose;
  }
  return transform::Interpolate(
             transform::TimestampedTransform{std::prev(it)->data.time(),
                                             std::prev(it)->data.global_pose},
             transform::TimestampedTransform{it->data.time(),
                                             it->data.global_pose},
             time)
      .transform;
}

transform::Rigid3d PoseGraph3D::GetLocalToGlobalTransform(
    const int trajectory_id) const {
  absl::MutexLock locker(&mutex_);
  return ComputeLocalToGlobalTransform(data_.global_submap_poses_3d,
                                       trajectory_id);
}

std::vector<std::vector<int>> PoseGraph3D::GetConnectedTrajectories() const {
  absl::MutexLock locker(&mutex_);
  return data_.trajectory_connectivity_state.Components();
}

PoseGraphInterface::SubmapData PoseGraph3D::GetSubmapData(
    const SubmapId& submap_id) const {
  absl::MutexLock locker(&mutex_);
  return GetSubmapDataUnderLock(submap_id);
}

MapById<SubmapId, PoseGraphInterface::SubmapData>
PoseGraph3D::GetAllSubmapData() const {
  absl::MutexLock locker(&mutex_);
  return GetSubmapDataUnderLock();
}

MapById<SubmapId, PoseGraphInterface::SubmapPose>
PoseGraph3D::GetAllSubmapPoses() const {
  absl::MutexLock locker(&mutex_);
  MapById<SubmapId, SubmapPose> submap_poses;
  for (const auto& submap_id_data : data_.submap_data) {
    auto submap_data = GetSubmapDataUnderLock(submap_id_data.id);
    submap_poses.Insert(
        submap_id_data.id,
        PoseGraphInterface::SubmapPose{submap_data.submap->num_range_data(),
                                       submap_data.pose});
  }
  return submap_poses;
}

transform::Rigid3d PoseGraph3D::ComputeLocalToGlobalTransform(
    const MapById<SubmapId, optimization::SubmapSpec3D>& global_submap_poses,
    const int trajectory_id) const {
  auto begin_it = global_submap_poses.BeginOfTrajectory(trajectory_id);
  auto end_it = global_submap_poses.EndOfTrajectory(trajectory_id);
  if (begin_it == end_it) {
    const auto it = data_.initial_trajectory_poses.find(trajectory_id);
    if (it != data_.initial_trajectory_poses.end()) {
      return GetInterpolatedGlobalTrajectoryPose(it->second.to_trajectory_id,
                                                 it->second.time) *
             it->second.relative_pose;
    } else {
      return transform::Rigid3d::Identity();
    }
  }
  const SubmapId last_optimized_submap_id = std::prev(end_it)->id;
  // Accessing 'local_pose' in Submap is okay, since the member is const.
  return global_submap_poses.at(last_optimized_submap_id).global_pose *
         data_.submap_data.at(last_optimized_submap_id)
             .submap->local_pose()
             .inverse();
}

PoseGraphInterface::SubmapData PoseGraph3D::GetSubmapDataUnderLock(
    const SubmapId& submap_id) const {
  const auto it = data_.submap_data.find(submap_id);
  if (it == data_.submap_data.end()) {
    return {};
  }
  auto submap = it->data.submap;
  if (data_.global_submap_poses_3d.Contains(submap_id)) {
    // We already have an optimized pose.
    return {submap, data_.global_submap_poses_3d.at(submap_id).global_pose};
  }
  // We have to extrapolate.
  return {submap, ComputeLocalToGlobalTransform(data_.global_submap_poses_3d,
                                                submap_id.trajectory_id) *
                      submap->local_pose()};
}

PoseGraph3D::TrimmingHandle::TrimmingHandle(PoseGraph3D* const parent)
    : parent_(parent) {}

int PoseGraph3D::TrimmingHandle::num_submaps(const int trajectory_id) const {
  const auto& submap_data = parent_->optimization_problem_->submap_data();
  return submap_data.SizeOfTrajectoryOrZero(trajectory_id);
}

std::vector<SubmapId> PoseGraph3D::TrimmingHandle::GetSubmapIds(
    int trajectory_id) const {
  std::vector<SubmapId> submap_ids;
  const auto& submap_data = parent_->optimization_problem_->submap_data();
  for (const auto& it : submap_data.trajectory(trajectory_id)) {
    submap_ids.push_back(it.id);
  }
  return submap_ids;
}
MapById<SubmapId, PoseGraphInterface::SubmapData>
PoseGraph3D::TrimmingHandle::GetOptimizedSubmapData() const {
  MapById<SubmapId, PoseGraphInterface::SubmapData> submaps;
  for (const auto& submap_id_data : parent_->data_.submap_data) {
    if (submap_id_data.data.state != SubmapState::kFinished ||
        !parent_->data_.global_submap_poses_3d.Contains(submap_id_data.id)) {
      continue;
    }
    submaps.Insert(
        submap_id_data.id,
        SubmapData{submap_id_data.data.submap,
                   parent_->data_.global_submap_poses_3d.at(submap_id_data.id)
                       .global_pose});
  }
  return submaps;
}

const MapById<NodeId, TrajectoryNode>&
PoseGraph3D::TrimmingHandle::GetTrajectoryNodes() const {
  return parent_->data_.trajectory_nodes;
}

const std::vector<PoseGraphInterface::Constraint>&
PoseGraph3D::TrimmingHandle::GetConstraints() const {
  return parent_->data_.constraints;
}

bool PoseGraph3D::TrimmingHandle::IsFinished(const int trajectory_id) const {
  return parent_->IsTrajectoryFinished(trajectory_id);
}

void PoseGraph3D::TrimmingHandle::SetTrajectoryState(int trajectory_id,
                                                     TrajectoryState state) {
  parent_->data_.trajectories_state[trajectory_id].state = state;
}

void PoseGraph3D::TrimmingHandle::TrimSubmap(const SubmapId& submap_id) {
  // TODO(hrapp): We have to make sure that the trajectory has been finished
  // if we want to delete the last submaps.
  CHECK(parent_->data_.submap_data.at(submap_id).state ==
        SubmapState::kFinished);

  // Compile all nodes that are still INTRA_SUBMAP constrained to other submaps
  // once the submap with 'submap_id' is gone.
  // We need to use node_ids instead of constraints here to be also compatible
  // with frozen trajectories that don't have intra-constraints.
  std::set<NodeId> nodes_to_retain;
  for (const auto& submap_data : parent_->data_.submap_data) {
    if (submap_data.id != submap_id) {
      nodes_to_retain.insert(submap_data.data.node_ids.begin(),
                             submap_data.data.node_ids.end());
    }
  }

  // Remove all nodes that are exlusively associated to 'submap_id'.
  std::set<NodeId> nodes_to_remove;
  std::set_difference(parent_->data_.submap_data.at(submap_id).node_ids.begin(),
                      parent_->data_.submap_data.at(submap_id).node_ids.end(),
                      nodes_to_retain.begin(), nodes_to_retain.end(),
                      std::inserter(nodes_to_remove, nodes_to_remove.begin()));

  // Remove all 'data_.constraints' related to 'submap_id'.
  {
    std::vector<Constraint> constraints;
    for (const Constraint& constraint : parent_->data_.constraints) {
      if (constraint.submap_id != submap_id) {
        constraints.push_back(constraint);
      }
    }
    parent_->data_.constraints = std::move(constraints);
  }

  // Remove all 'data_.constraints' related to 'nodes_to_remove'.
  // If the removal lets other submaps lose all their inter-submap constraints,
  // delete their corresponding constraint submap matchers to save memory.
  {
    std::vector<Constraint> constraints;
    std::set<SubmapId> other_submap_ids_losing_constraints;
    for (const Constraint& constraint : parent_->data_.constraints) {
      if (nodes_to_remove.count(constraint.node_id) == 0) {
        constraints.push_back(constraint);
      } else {
        // A constraint to another submap will be removed, mark it as affected.
        other_submap_ids_losing_constraints.insert(constraint.submap_id);
      }
    }
    parent_->data_.constraints = std::move(constraints);
    // Go through the remaining constraints to ensure we only delete scan
    // matchers of other submaps that have no inter-submap constraints left.
    for (const Constraint& constraint : parent_->data_.constraints) {
      if (constraint.tag == Constraint::Tag::INTRA_SUBMAP) {
        continue;
      } else if (other_submap_ids_losing_constraints.count(
                     constraint.submap_id)) {
        // This submap still has inter-submap constraints - ignore it.
        other_submap_ids_losing_constraints.erase(constraint.submap_id);
      }
    }
    // Delete scan matchers of the submaps that lost all constraints.
    // TODO(wohe): An improvement to this implementation would be to add the
    // caching logic at the constraint builder which could keep around only
    // recently used scan matchers.
    for (const SubmapId& submap_id : other_submap_ids_losing_constraints) {
      parent_->constraint_builder_.DeleteScanMatcher(submap_id);
    }
  }

  // Mark the submap with 'submap_id' as trimmed and remove its data.
  CHECK(parent_->data_.submap_data.at(submap_id).state ==
        SubmapState::kFinished);
  parent_->data_.submap_data.Trim(submap_id);
  parent_->constraint_builder_.DeleteScanMatcher(submap_id);
  parent_->optimization_problem_->TrimSubmap(submap_id);

  // We have one submap less, update the gauge metrics.
  kDeletedSubmapsMetric->Increment();
  if (parent_->IsTrajectoryFrozen(submap_id.trajectory_id)) {
    kFrozenSubmapsMetric->Decrement();
  } else {
    kActiveSubmapsMetric->Decrement();
  }

  // Remove the 'nodes_to_remove' from the pose graph and the optimization
  // problem.
  for (const NodeId& node_id : nodes_to_remove) {
    parent_->data_.trajectory_nodes.Trim(node_id);
    parent_->optimization_problem_->TrimTrajectoryNode(node_id);
  }
}

MapById<SubmapId, PoseGraphInterface::SubmapData>
PoseGraph3D::GetSubmapDataUnderLock() const {
  MapById<SubmapId, PoseGraphInterface::SubmapData> submaps;
  for (const auto& submap_id_data : data_.submap_data) {
    submaps.Insert(submap_id_data.id,
                   GetSubmapDataUnderLock(submap_id_data.id));
  }
  return submaps;
}

void PoseGraph3D::SetGlobalSlamOptimizationCallback(
    PoseGraphInterface::GlobalSlamOptimizationCallback callback) {
  global_slam_optimization_callback_ = callback;
}

void PoseGraph3D::RegisterMetrics(metrics::FamilyFactory* family_factory) {
  auto* latency = family_factory->NewGaugeFamily(
      "mapping_3d_pose_graph_work_queue_delay",
      "Age of the oldest entry in the work queue in seconds");
  kWorkQueueDelayMetric = latency->Add({});
  auto* queue_size =
      family_factory->NewGaugeFamily("mapping_3d_pose_graph_work_queue_size",
                                     "Number of items in the work queue");
  kWorkQueueSizeMetric = queue_size->Add({});
  auto* constraints = family_factory->NewGaugeFamily(
      "mapping_3d_pose_graph_constraints",
      "Current number of constraints in the pose graph");
  kConstraintsDifferentTrajectoryMetric =
      constraints->Add({{"tag", "inter_submap"}, {"trajectory", "different"}});
  kConstraintsSameTrajectoryMetric =
      constraints->Add({{"tag", "inter_submap"}, {"trajectory", "same"}});
  auto* submaps = family_factory->NewGaugeFamily(
      "mapping_3d_pose_graph_submaps", "Number of submaps in the pose graph.");
  kActiveSubmapsMetric = submaps->Add({{"state", "active"}});
  kFrozenSubmapsMetric = submaps->Add({{"state", "frozen"}});
  kDeletedSubmapsMetric = submaps->Add({{"state", "deleted"}});
}

}  // namespace mapping
}  // namespace cartographer
