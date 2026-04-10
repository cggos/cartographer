/*
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

#include "cartographer/mapping/internal/optimization/optimization_problem_3d.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "absl/memory/memory.h"
#include "cartographer/common/internal/ceres_solver_options.h"
#include "cartographer/common/math.h"
#include "cartographer/common/time.h"
#include "cartographer/mapping/internal/3d/imu_integration.h"
#include "cartographer/mapping/internal/3d/rotation_parameterization.h"
#include "cartographer/mapping/internal/optimization/ceres_pose.h"
#include "cartographer/mapping/internal/optimization/cost_functions/acceleration_cost_function_3d.h"
#include "cartographer/mapping/internal/optimization/cost_functions/landmark_cost_function_3d.h"
#include "cartographer/mapping/internal/optimization/cost_functions/rotation_cost_function_3d.h"
#include "cartographer/mapping/internal/optimization/cost_functions/spa_cost_function_3d.h"
#include "cartographer/transform/timestamped_transform.h"
#include "cartographer/transform/transform.h"
#include "ceres/ceres.h"
#include "ceres/jet.h"
#include "ceres/rotation.h"
#include "glog/logging.h"

namespace cartographer {
namespace mapping {
namespace optimization {
namespace {

using LandmarkNode = ::cartographer::mapping::PoseGraphInterface::LandmarkNode;
using TrajectoryData =
    ::cartographer::mapping::PoseGraphInterface::TrajectoryData;

// For odometry.
std::unique_ptr<transform::Rigid3d> Interpolate(
    const sensor::MapByTime<sensor::OdometryData>& map_by_time,
    const int trajectory_id, const common::Time time) {
  const auto it = map_by_time.lower_bound(trajectory_id, time);
  if (it == map_by_time.EndOfTrajectory(trajectory_id)) {
    return nullptr;
  }
  if (it == map_by_time.BeginOfTrajectory(trajectory_id)) {
    if (it->time == time) {
      return absl::make_unique<transform::Rigid3d>(it->pose);
    }
    return nullptr;
  }
  const auto prev_it = std::prev(it);
  return absl::make_unique<transform::Rigid3d>(
      Interpolate(transform::TimestampedTransform{prev_it->time, prev_it->pose},
                  transform::TimestampedTransform{it->time, it->pose}, time)
          .transform);
}

// For fixed frame pose.
std::unique_ptr<transform::Rigid3d> Interpolate(
    const sensor::MapByTime<sensor::FixedFramePoseData>& map_by_time,
    const int trajectory_id, const common::Time time) {
  const auto it = map_by_time.lower_bound(trajectory_id, time);
  if (it == map_by_time.EndOfTrajectory(trajectory_id) ||
      !it->pose.has_value()) {
    return nullptr;
  }
  if (it == map_by_time.BeginOfTrajectory(trajectory_id)) {
    if (it->time == time) {
      return absl::make_unique<transform::Rigid3d>(it->pose.value());
    }
    return nullptr;
  }
  const auto prev_it = std::prev(it);
  if (prev_it->pose.has_value()) {
    return absl::make_unique<transform::Rigid3d>(
        Interpolate(transform::TimestampedTransform{prev_it->time,
                                                    prev_it->pose.value()},
                    transform::TimestampedTransform{it->time, it->pose.value()},
                    time)
            .transform);
  }
  return nullptr;
}

// Selects a trajectory node closest in time to the landmark observation and
// applies a relative transform from it.
transform::Rigid3d GetInitialLandmarkPose(
    const LandmarkNode::LandmarkObservation& observation,
    const NodeSpec3D& prev_node, const NodeSpec3D& next_node,
    const CeresPose& prev_node_pose, const CeresPose& next_node_pose) {
  const double interpolation_parameter =
      common::ToSeconds(observation.time - prev_node.time) /
      common::ToSeconds(next_node.time - prev_node.time);

  const std::tuple<std::array<double, 4>, std::array<double, 3>>
      rotation_and_translation = InterpolateNodes3D(
          prev_node_pose.rotation(), prev_node_pose.translation(),
          next_node_pose.rotation(), next_node_pose.translation(),
          interpolation_parameter);
  return transform::Rigid3d::FromArrays(std::get<0>(rotation_and_translation),
                                        std::get<1>(rotation_and_translation)) *
         observation.landmark_to_tracking_transform;
}

void AddLandmarkCostFunctions(
    const std::map<std::string, LandmarkNode>& landmark_nodes,
    const MapById<NodeId, NodeSpec3D>& node_data,
    MapById<NodeId, CeresPose>* C_nodes,
    std::map<std::string, CeresPose>* C_landmarks, ceres::Problem* problem,
    double huber_scale) {
  for (const auto& landmark_node : landmark_nodes) {
    // Do not use landmarks that were not optimized for localization.
    for (const auto& observation : landmark_node.second.landmark_observations) {
      const std::string& landmark_id = landmark_node.first;
      const auto& begin_of_trajectory =
          node_data.BeginOfTrajectory(observation.trajectory_id);
      // The landmark observation was made before the trajectory was created.
      if (observation.time < begin_of_trajectory->data.time) {
        continue;
      }
      // Find the trajectory nodes before and after the landmark observation.
      auto next =
          node_data.lower_bound(observation.trajectory_id, observation.time);
      // The landmark observation was made, but the next trajectory node has
      // not been added yet.
      if (next == node_data.EndOfTrajectory(observation.trajectory_id)) {
        continue;
      }
      if (next == begin_of_trajectory) {
        next = std::next(next);
      }
      auto prev = std::prev(next);
      // Add parameter blocks for the landmark ID if they were not added before.
      CeresPose* prev_node_pose = &C_nodes->at(prev->id);
      CeresPose* next_node_pose = &C_nodes->at(next->id);
      if (!C_landmarks->count(landmark_id)) {
        const transform::Rigid3d starting_point =
            landmark_node.second.global_landmark_pose.has_value()
                ? landmark_node.second.global_landmark_pose.value()
                : GetInitialLandmarkPose(observation, prev->data, next->data,
                                         *prev_node_pose, *next_node_pose);
        C_landmarks->emplace(
            landmark_id,
            CeresPose(starting_point, nullptr /* translation_parametrization */,
                      absl::make_unique<ceres::QuaternionParameterization>(),
                      problem));
        // Set landmark constant if it is frozen.
        if (landmark_node.second.frozen) {
          problem->SetParameterBlockConstant(
              C_landmarks->at(landmark_id).translation());
          problem->SetParameterBlockConstant(
              C_landmarks->at(landmark_id).rotation());
        }
      }
      problem->AddResidualBlock(
          LandmarkCostFunction3D::CreateAutoDiffCostFunction(
              observation, prev->data, next->data),
          new ceres::HuberLoss(huber_scale), prev_node_pose->rotation(),
          prev_node_pose->translation(), next_node_pose->rotation(),
          next_node_pose->translation(),
          C_landmarks->at(landmark_id).rotation(),
          C_landmarks->at(landmark_id).translation());
    }
  }
}

}  // namespace

OptimizationProblem3D::OptimizationProblem3D(
    const optimization::proto::OptimizationProblemOptions& options)
    : options_(options) {}

OptimizationProblem3D::~OptimizationProblem3D() {}

void OptimizationProblem3D::AddImuData(const int trajectory_id,
                                       const sensor::ImuData& imu_data) {
  imu_data_.Append(trajectory_id, imu_data);
}

void OptimizationProblem3D::AddOdometryData(
    const int trajectory_id, const sensor::OdometryData& odometry_data) {
  odometry_data_.Append(trajectory_id, odometry_data);
}

void OptimizationProblem3D::AddFixedFramePoseData(
    const int trajectory_id,
    const sensor::FixedFramePoseData& fixed_frame_pose_data) {
  fixed_frame_pose_data_.Append(trajectory_id, fixed_frame_pose_data);
}

void OptimizationProblem3D::AddTrajectoryNode(const int trajectory_id,
                                              const NodeSpec3D& node_data) {
  node_data_.Append(trajectory_id, node_data);
  trajectory_data_[trajectory_id];
}

void OptimizationProblem3D::SetTrajectoryData(
    int trajectory_id, const TrajectoryData& trajectory_data) {
  trajectory_data_[trajectory_id] = trajectory_data;
}

void OptimizationProblem3D::InsertTrajectoryNode(const NodeId& node_id,
                                                 const NodeSpec3D& node_data) {
  node_data_.Insert(node_id, node_data);
  trajectory_data_[node_id.trajectory_id];
}

void OptimizationProblem3D::TrimTrajectoryNode(const NodeId& node_id) {
  imu_data_.Trim(node_data_, node_id);
  odometry_data_.Trim(node_data_, node_id);
  fixed_frame_pose_data_.Trim(node_data_, node_id);
  node_data_.Trim(node_id);
  if (node_data_.SizeOfTrajectoryOrZero(node_id.trajectory_id) == 0) {
    trajectory_data_.erase(node_id.trajectory_id);
  }
}

void OptimizationProblem3D::AddSubmap(
    const int trajectory_id, const transform::Rigid3d& global_submap_pose) {
  submap_data_.Append(trajectory_id, SubmapSpec3D{global_submap_pose});
}

void OptimizationProblem3D::InsertSubmap(
    const SubmapId& submap_id, const transform::Rigid3d& global_submap_pose) {
  submap_data_.Insert(submap_id, SubmapSpec3D{global_submap_pose});
}

void OptimizationProblem3D::TrimSubmap(const SubmapId& submap_id) {
  submap_data_.Trim(submap_id);
}

void OptimizationProblem3D::SetMaxNumIterations(
    const int32 max_num_iterations) {
  options_.mutable_ceres_solver_options()->set_max_num_iterations(
      max_num_iterations);
}

/**
 * @brief 执行位姿图优化（Ceres 求解核心）
 * 
 * 该方法负责构建并求解 3D 全局 SLAM 的非线性优化问题。它融合了位姿图约束（闭环）、
 * 惯性测量单元 (IMU) 数据、里程计、路标观测以及外部定位数据。
 * 
 * 核心算法逻辑：
 * 1. 状态初始化：将子图、节点和路标的全局位姿映射为 Ceres 参数块，并处理“冻结”轨迹的固定属性。
 * 2. 约束构建：
 *    - SPA 约束：节点与子图间的 6-DoF 相对位姿约束。
 *    - 惯性约束：利用 IMU 预积分数据约束连续节点间的旋转变化和线加速度。
 *    - 辅助约束：集成里程计、局部 SLAM 连续性和固定参考帧（如 GPS）数据。
 * 3. 优化求解：配置并调用 Ceres::Solve 进行稀疏位姿调整 (Sparse Pose Adjustment)。
 * 4. 状态更新：将优化后的全局位姿写回 node_data_ 和 submap_data_。
 * 
 * @param constraints 所有已发现的位姿图约束（包括内子图和闭环约束）。
 * @param trajectories_state 各个轨迹的当前状态（ACTIVE, FINISHED, FROZEN 等）。
 * @param landmark_nodes 路标观测数据及其当前的全局估计。
 */
void OptimizationProblem3D::Solve(
    const std::vector<Constraint>& constraints,
    const std::map<int, PoseGraphInterface::TrajectoryState>&
        trajectories_state,
    const std::map<std::string, LandmarkNode>& landmark_nodes) {
  if (node_data_.empty()) {
    // 没有节点需要优化，直接返回
    return;
  }

  // 1. 识别并记录处于冻结（FROZEN）状态的轨迹。冻结轨迹的位姿在优化中保持不变。
  std::set<int> frozen_trajectories;
  for (const auto& it : trajectories_state) {
    if (it.second == PoseGraphInterface::TrajectoryState::FROZEN) {
      frozen_trajectories.insert(it.first);
    }
  }

  ceres::Problem::Options problem_options;
  ceres::Problem problem(problem_options);

  // 定义平移参数化方式：如果配置了 fix_z_in_3d，则将 Z 轴设为常量（不优化 Z）
  const auto translation_parameterization =
      [this]() -> std::unique_ptr<ceres::LocalParameterization> {
    return options_.fix_z_in_3d()
               ? absl::make_unique<ceres::SubsetParameterization>(
                     3, std::vector<int>{2})
               : nullptr;
  };

  // 2. 初始化子图和节点的 Ceres 参数块 (Parameter Blocks)
  CHECK(!submap_data_.empty());
  MapById<SubmapId, CeresPose> C_submaps;
  MapById<NodeId, CeresPose> C_nodes;
  std::map<std::string, CeresPose> C_landmarks;

  // 初始化所有子图的参数块
  bool first_submap = true;
  for (const auto& submap_id_data : submap_data_) {
    const bool frozen =
        frozen_trajectories.count(submap_id_data.id.trajectory_id) != 0;
    if (first_submap) {
      first_submap = false;
      // 固定第一个轨迹的第一个子图，除了允许重力对齐相关的旋转外。
      C_submaps.Insert(
          submap_id_data.id,
          CeresPose(submap_id_data.data.global_pose,
                    translation_parameterization(),
                    absl::make_unique<ceres::AutoDiffLocalParameterization<
                        ConstantYawQuaternionPlus, 4, 2>>(),
                    &problem));
      problem.SetParameterBlockConstant(
          C_submaps.at(submap_id_data.id).translation());
    } else {
      C_submaps.Insert(
          submap_id_data.id,
          CeresPose(submap_id_data.data.global_pose,
                    translation_parameterization(),
                    absl::make_unique<ceres::QuaternionParameterization>(),
                    &problem));
    }
    // 如果属于冻结轨迹，则将该子图设为常量（不参与优化）
    if (frozen) {
      problem.SetParameterBlockConstant(
          C_submaps.at(submap_id_data.id).rotation());
      problem.SetParameterBlockConstant(
          C_submaps.at(submap_id_data.id).translation());
    }
  }

  // 初始化所有节点的参数块
  for (const auto& node_id_data : node_data_) {
    const bool frozen =
        frozen_trajectories.count(node_id_data.id.trajectory_id) != 0;
    C_nodes.Insert(
        node_id_data.id,
        CeresPose(node_id_data.data.global_pose, translation_parameterization(),
                  absl::make_unique<ceres::QuaternionParameterization>(),
                  &problem));
    // 如果属于冻结轨迹，则将该节点设为常量
    if (frozen) {
      problem.SetParameterBlockConstant(C_nodes.at(node_id_data.id).rotation());
      problem.SetParameterBlockConstant(
          C_nodes.at(node_id_data.id).translation());
    }
  }

  // 3. 添加位姿图约束：包括内子图约束 (Intra-submap) 和外子图/闭环约束 (Inter-submap)
  for (const Constraint& constraint : constraints) {
    problem.AddResidualBlock(
        SpaCostFunction3D::CreateAutoDiffCostFunction(constraint.pose),
        // 闭环约束通常使用 Huber Loss 以提高鲁棒性，减少误匹配带来的影响
        constraint.tag == Constraint::INTER_SUBMAP
            ? new ceres::HuberLoss(options_.huber_scale())
            : nullptr /* loss function */,
        C_submaps.at(constraint.submap_id).rotation(),
        C_submaps.at(constraint.submap_id).translation(),
        C_nodes.at(constraint.node_id).rotation(),
        C_nodes.at(constraint.node_id).translation());
  }

  // 4. 添加路标 (Landmarks) 约束
  AddLandmarkCostFunctions(landmark_nodes, node_data_, &C_nodes, &C_landmarks,
                           &problem, options_.huber_scale());

  // 5. 添加基于 IMU 观测（角速度和线加速度）的约束
  if (!options_.fix_z_in_3d()) {
    for (auto node_it = node_data_.begin(); node_it != node_data_.end();) {
      const int trajectory_id = node_it->id.trajectory_id;
      const auto trajectory_end = node_data_.EndOfTrajectory(trajectory_id);
      if (frozen_trajectories.count(trajectory_id) != 0) {
        // 跳过冻结轨迹
        node_it = trajectory_end;
        continue;
      }
      TrajectoryData& trajectory_data = trajectory_data_.at(trajectory_id);

      // 添加 IMU 外参参数块（IMU 到 tracking frame 的旋转）
      problem.AddParameterBlock(trajectory_data.imu_calibration.data(), 4,
                                new ceres::QuaternionParameterization());
      if (!options_.use_online_imu_extrinsics_in_3d()) {
        // 如果不在线优化外参，则设为常量
        problem.SetParameterBlockConstant(
            trajectory_data.imu_calibration.data());
      }
      CHECK(imu_data_.HasTrajectory(trajectory_id));
      const auto imu_data = imu_data_.trajectory(trajectory_id);
      CHECK(imu_data.begin() != imu_data.end());

      auto imu_it = imu_data.begin();
      auto prev_node_it = node_it;

      bool gravity_block_added = false;
      for (++node_it; node_it != trajectory_end; ++node_it) {
        const NodeId first_node_id = prev_node_it->id;
        const NodeSpec3D& first_node_data = prev_node_it->data;
        prev_node_it = node_it;
        const NodeId second_node_id = node_it->id;
        const NodeSpec3D& second_node_data = node_it->data;

        if (second_node_id.node_index != first_node_id.node_index + 1) {
          continue;
        }

        // 寻找节点间对应的 IMU 数据
        while (std::next(imu_it) != imu_data.end() &&
               std::next(imu_it)->time <= first_node_data.time) {
          ++imu_it;
        }

        auto imu_it2 = imu_it;
        // 集成两个节点之间的 IMU 数据（预积分）
        const IntegrateImuResult<double> result = IntegrateImu(
            imu_data, first_node_data.time, second_node_data.time, &imu_it);
        const auto next_node_it = std::next(node_it);
        const common::Time first_time = first_node_data.time;
        const common::Time second_time = second_node_data.time;
        const common::Duration first_duration = second_time - first_time;

        // 如果存在连续的三个节点，可以添加加速度约束
        if (next_node_it != trajectory_end &&
            next_node_it->id.node_index == second_node_id.node_index + 1) {
          const NodeId third_node_id = next_node_it->id;
          const NodeSpec3D& third_node_data = next_node_it->data;
          const common::Time third_time = third_node_data.time;
          const common::Duration second_duration = third_time - second_time;
          const common::Time first_center = first_time + first_duration / 2;
          const common::Time second_center = second_time + second_duration / 2;
          const IntegrateImuResult<double> result_to_first_center =
              IntegrateImu(imu_data, first_time, first_center, &imu_it2);
          const IntegrateImuResult<double> result_center_to_center =
              IntegrateImu(imu_data, first_center, second_center, &imu_it2);
          
          // 计算速度变化量 delta_velocity
          const Eigen::Vector3d delta_velocity =
              (result.delta_rotation.inverse() *
               result_to_first_center.delta_rotation) *
              result_center_to_center.delta_velocity;
          
          // 添加加速度代价函数残差块
          problem.AddResidualBlock(
              AccelerationCostFunction3D::CreateAutoDiffCostFunction(
                  options_.acceleration_weight() /
                      common::ToSeconds(first_duration + second_duration),
                  delta_velocity, common::ToSeconds(first_duration),
                  common::ToSeconds(second_duration)),
              nullptr /* loss function */,
              C_nodes.at(second_node_id).rotation(),
              C_nodes.at(first_node_id).translation(),
              C_nodes.at(second_node_id).translation(),
              C_nodes.at(third_node_id).translation(),
              &trajectory_data.gravity_constant,
              trajectory_data.imu_calibration.data());
          gravity_block_added = true;
        }
        // 添加旋转代价函数残差块（利用 IMU 积分得到的姿态增量）
        problem.AddResidualBlock(
            RotationCostFunction3D::CreateAutoDiffCostFunction(
                options_.rotation_weight() / common::ToSeconds(first_duration),
                result.delta_rotation),
            nullptr /* loss function */, C_nodes.at(first_node_id).rotation(),
            C_nodes.at(second_node_id).rotation(),
            trajectory_data.imu_calibration.data());
      }

      if (gravity_block_added) {
        // 确保重力常数为正数
        problem.SetParameterLowerBound(&trajectory_data.gravity_constant, 0,
                                       0.0);
      }
    }
  }

  // 6. 如果配置了 fix_z_in_3d，添加基于里程计和局部 SLAM 位姿的连续性约束
  if (options_.fix_z_in_3d()) {
    for (auto node_it = node_data_.begin(); node_it != node_data_.end();) {
      const int trajectory_id = node_it->id.trajectory_id;
      const auto trajectory_end = node_data_.EndOfTrajectory(trajectory_id);
      if (frozen_trajectories.count(trajectory_id) != 0) {
        node_it = trajectory_end;
        continue;
      }

      auto prev_node_it = node_it;
      for (++node_it; node_it != trajectory_end; ++node_it) {
        const NodeId first_node_id = prev_node_it->id;
        const NodeSpec3D& first_node_data = prev_node_it->data;
        prev_node_it = node_it;
        const NodeId second_node_id = node_it->id;
        const NodeSpec3D& second_node_data = node_it->data;

        if (second_node_id.node_index != first_node_id.node_index + 1) {
          continue;
        }

        // 添加基于里程计的相对位姿约束
        const std::unique_ptr<transform::Rigid3d> relative_odometry =
            CalculateOdometryBetweenNodes(trajectory_id, first_node_data,
                                          second_node_data);
        if (relative_odometry != nullptr) {
          problem.AddResidualBlock(
              SpaCostFunction3D::CreateAutoDiffCostFunction(Constraint::Pose{
                  *relative_odometry, options_.odometry_translation_weight(),
                  options_.odometry_rotation_weight()}),
              nullptr /* loss function */, C_nodes.at(first_node_id).rotation(),
              C_nodes.at(first_node_id).translation(),
              C_nodes.at(second_node_id).rotation(),
              C_nodes.at(second_node_id).translation());
        }

        // 添加基于连续局部 SLAM 位姿的相对位姿约束
        const transform::Rigid3d relative_local_slam_pose =
            first_node_data.local_pose.inverse() * second_node_data.local_pose;
        problem.AddResidualBlock(
            SpaCostFunction3D::CreateAutoDiffCostFunction(
                Constraint::Pose{relative_local_slam_pose,
                                 options_.local_slam_pose_translation_weight(),
                                 options_.local_slam_pose_rotation_weight()}),
            nullptr /* loss function */, C_nodes.at(first_node_id).rotation(),
            C_nodes.at(first_node_id).translation(),
            C_nodes.at(second_node_id).rotation(),
            C_nodes.at(second_node_id).translation());
      }
    }
  }

  // 7. 添加固定帧 (Fixed Frame, 如 GPS) 位姿约束
  std::map<int, CeresPose> C_fixed_frames;
  for (auto node_it = node_data_.begin(); node_it != node_data_.end();) {
    const int trajectory_id = node_it->id.trajectory_id;
    const auto trajectory_end = node_data_.EndOfTrajectory(trajectory_id);
    if (!fixed_frame_pose_data_.HasTrajectory(trajectory_id)) {
      node_it = trajectory_end;
      continue;
    }

    const TrajectoryData& trajectory_data = trajectory_data_.at(trajectory_id);
    bool fixed_frame_pose_initialized = false;
    for (; node_it != trajectory_end; ++node_it) {
      const NodeId node_id = node_it->id;
      const NodeSpec3D& node_data = node_it->data;

      const std::unique_ptr<transform::Rigid3d> fixed_frame_pose =
          Interpolate(fixed_frame_pose_data_, trajectory_id, node_data.time);
      if (fixed_frame_pose == nullptr) {
        continue;
      }

      const Constraint::Pose constraint_pose{
          *fixed_frame_pose, options_.fixed_frame_pose_translation_weight(),
          options_.fixed_frame_pose_rotation_weight()};

      if (!fixed_frame_pose_initialized) {
        transform::Rigid3d fixed_frame_pose_in_map;
        if (trajectory_data.fixed_frame_origin_in_map.has_value()) {
          fixed_frame_pose_in_map =
              trajectory_data.fixed_frame_origin_in_map.value();
        } else {
          fixed_frame_pose_in_map =
              node_data.global_pose * constraint_pose.zbar_ij.inverse();
        }
        C_fixed_frames.emplace(
            std::piecewise_construct, std::forward_as_tuple(trajectory_id),
            std::forward_as_tuple(
                transform::Rigid3d(
                    fixed_frame_pose_in_map.translation(),
                    Eigen::AngleAxisd(
                        transform::GetYaw(fixed_frame_pose_in_map.rotation()),
                        Eigen::Vector3d::UnitZ())),
                nullptr,
                absl::make_unique<ceres::AutoDiffLocalParameterization<
                    YawOnlyQuaternionPlus, 4, 1>>(),
                &problem));
        fixed_frame_pose_initialized = true;
      }

      problem.AddResidualBlock(
          SpaCostFunction3D::CreateAutoDiffCostFunction(constraint_pose),
          options_.fixed_frame_pose_use_tolerant_loss() ?
              new ceres::TolerantLoss(
            options_.fixed_frame_pose_tolerant_loss_param_a(),
            options_.fixed_frame_pose_tolerant_loss_param_b()) : nullptr,
          C_fixed_frames.at(trajectory_id).rotation(),
          C_fixed_frames.at(trajectory_id).translation(),
          C_nodes.at(node_id).rotation(), C_nodes.at(node_id).translation());
    }
  }

  // 8. 调用 Ceres Solver 执行优化
  ceres::Solver::Summary summary;
  ceres::Solve(
      common::CreateCeresSolverOptions(options_.ceres_solver_options()),
      &problem, &summary);
  if (options_.log_solver_summary()) {
    LOG(INFO) << summary.FullReport();
    for (const auto& trajectory_id_and_data : trajectory_data_) {
      const int trajectory_id = trajectory_id_and_data.first;
      const TrajectoryData& trajectory_data = trajectory_id_and_data.second;
      if (trajectory_id != 0) {
        LOG(INFO) << "Trajectory " << trajectory_id << ":";
      }
      LOG(INFO) << "Gravity was: " << trajectory_data.gravity_constant;
      const auto& imu_calibration = trajectory_data.imu_calibration;
      LOG(INFO) << "IMU correction was: "
                << common::RadToDeg(2. *
                                    std::acos(std::abs(imu_calibration[0])))
                << " deg (" << imu_calibration[0] << ", " << imu_calibration[1]
                << ", " << imu_calibration[2] << ", " << imu_calibration[3]
                << ")";
    }
  }

  // 9. 将优化后的结果存储回内部数据结构
  for (const auto& C_submap_id_data : C_submaps) {
    submap_data_.at(C_submap_id_data.id).global_pose =
        C_submap_id_data.data.ToRigid();
  }
  for (const auto& C_node_id_data : C_nodes) {
    node_data_.at(C_node_id_data.id).global_pose =
        C_node_id_data.data.ToRigid();
  }
  for (const auto& C_fixed_frame : C_fixed_frames) {
    trajectory_data_.at(C_fixed_frame.first).fixed_frame_origin_in_map =
        C_fixed_frame.second.ToRigid();
  }
  for (const auto& C_landmark : C_landmarks) {
    landmark_data_[C_landmark.first] = C_landmark.second.ToRigid();
  }
}

std::unique_ptr<transform::Rigid3d>
OptimizationProblem3D::CalculateOdometryBetweenNodes(
    const int trajectory_id, const NodeSpec3D& first_node_data,
    const NodeSpec3D& second_node_data) const {
  if (odometry_data_.HasTrajectory(trajectory_id)) {
    const std::unique_ptr<transform::Rigid3d> first_node_odometry =
        Interpolate(odometry_data_, trajectory_id, first_node_data.time);
    const std::unique_ptr<transform::Rigid3d> second_node_odometry =
        Interpolate(odometry_data_, trajectory_id, second_node_data.time);
    if (first_node_odometry != nullptr && second_node_odometry != nullptr) {
      const transform::Rigid3d relative_odometry =
          first_node_odometry->inverse() * (*second_node_odometry);
      return absl::make_unique<transform::Rigid3d>(relative_odometry);
    }
  }
  return nullptr;
}

}  // namespace optimization
}  // namespace mapping
}  // namespace cartographer
