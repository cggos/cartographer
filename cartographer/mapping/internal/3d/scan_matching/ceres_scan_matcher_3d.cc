/*
 * Ceres 3D扫描匹配器
 * 
 * 功能：使用Ceres非线性优化进行精确的激光点云匹配
 * 
 * 算法原理：
 * 将扫描匹配问题转化为非线性最小二乘优化问题
 * 最小化成本函数找到最优机器人位姿
 * 
 * 成本函数组成：
 * 1. OccupiedSpaceCostFunction: 点云与子图栅格匹配成本
 * 2. IntensityCostFunction: 强度匹配成本(可选)
 * 3. TranslationDeltaCostFunctor: 平移约束成本
 * 4. RotationDeltaCostFunctor: 旋转约束成本
 * 
 * 优化变量：6维 (x,y,z, qx,qy,qz,qw)
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

#include "cartographer/mapping/internal/3d/scan_matching/ceres_scan_matcher_3d.h"

#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "cartographer/common/internal/ceres_solver_options.h"
#include "cartographer/mapping/internal/3d/rotation_parameterization.h"
#include "cartographer/mapping/internal/3d/scan_matching/intensity_cost_function_3d.h"
#include "cartographer/mapping/internal/3d/scan_matching/occupied_space_cost_function_3d.h"
#include "cartographer/mapping/internal/3d/scan_matching/rotation_delta_cost_functor_3d.h"
#include "cartographer/mapping/internal/3d/scan_matching/translation_delta_cost_functor_3d.h"
#include "cartographer/mapping/internal/optimization/ceres_pose.h"
#include "cartographer/transform/rigid_transform.h"
#include "cartographer/transform/transform.h"
#include "ceres/ceres.h"
#include "glog/logging.h"

namespace cartographer {
namespace mapping {
namespace scan_matching {

proto::CeresScanMatcherOptions3D CreateCeresScanMatcherOptions3D(
    common::LuaParameterDictionary* const parameter_dictionary) {
  proto::CeresScanMatcherOptions3D options;
  for (int i = 0;; ++i) {
    const std::string lua_identifier =
        "occupied_space_weight_" + std::to_string(i);
    if (!parameter_dictionary->HasKey(lua_identifier)) {
      break;
    }
    options.add_occupied_space_weight(
        parameter_dictionary->GetDouble(lua_identifier));
  }
  for (int i = 0;; ++i) {
    const std::string lua_identifier =
        "intensity_cost_function_options_" + std::to_string(i);
    if (!parameter_dictionary->HasKey(lua_identifier)) {
      break;
    }
    const auto intensity_cost_function_options_dictionary =
        parameter_dictionary->GetDictionary(lua_identifier);
    auto* intensity_cost_function_options =
        options.add_intensity_cost_function_options();
    intensity_cost_function_options->set_weight(
        intensity_cost_function_options_dictionary->GetDouble("weight"));
    intensity_cost_function_options->set_huber_scale(
        intensity_cost_function_options_dictionary->GetDouble("huber_scale"));
    intensity_cost_function_options->set_intensity_threshold(
        intensity_cost_function_options_dictionary->GetDouble(
            "intensity_threshold"));
  }
  options.set_translation_weight(
      parameter_dictionary->GetDouble("translation_weight"));
  options.set_rotation_weight(
      parameter_dictionary->GetDouble("rotation_weight"));
  options.set_only_optimize_yaw(
      parameter_dictionary->GetBool("only_optimize_yaw"));
  *options.mutable_ceres_solver_options() =
      common::CreateCeresSolverOptionsProto(
          parameter_dictionary->GetDictionary("ceres_solver_options").get());
  return options;
}

CeresScanMatcher3D::CeresScanMatcher3D(
    const proto::CeresScanMatcherOptions3D& options)
    : options_(options),
      ceres_solver_options_(
          common::CreateCeresSolverOptions(options.ceres_solver_options())) {
  ceres_solver_options_.linear_solver_type = ceres::DENSE_QR;
}

/**
 * @brief Ceres扫描匹配 - 非线性优化求解最优位姿
 * 
 * 功能：将激光点云与子图进行匹配，通过最小化成本函数找到最优机器人位姿
 * 
 * 优化问题形式：
 * min ||cost_function(pose)||^2
 * 
 * 成本函数组成：
 * 1. OccupiedSpaceCostFunction: 点云占据空间成本（核心）
 *    - 将点云通过当前位姿变换到世界坐标
 *    - 查询子图HybridGrid中的占据概率
 *    - 成本 = 1 - probability (概率越高，成本越低)
 * 
 * 2. IntensityCostFunction: 强度匹配成本（可选）
 *    - 利用激光回波强度进行匹配
 *    - 使用Huber Loss减少异常值影响
 * 
 * 3. TranslationDeltaCostFunctor: 平移约束
 *    - 惩罚与目标平移的偏差
 *    - 防止位姿偏离预测太远
 * 
 * 4. RotationDeltaCostFunctor: 旋转约束
 *    - 惩罚与初始旋转的偏差
 *    - 保持位姿连续性
 * 
 * @param target_translation 目标平移（来自预测）
 * @param initial_pose_estimate 初始位姿估计（来自相关扫描匹配）
 * @param point_clouds_and_hybrid_grids 点云与子图栅格对（高/低分辨率）
 * @param pose_estimate 输出：优化后的位姿估计
 * @param summary 输出：优化结果摘要
 */
void CeresScanMatcher3D::Match(
    const Eigen::Vector3d& target_translation,
    const transform::Rigid3d& initial_pose_estimate,
    const std::vector<PointCloudAndHybridGridsPointers>&
        point_clouds_and_hybrid_grids,
    transform::Rigid3d* const pose_estimate,
    ceres::Solver::Summary* const summary) const {
  ceres::Problem problem;
  optimization::CeresPose ceres_pose(
      initial_pose_estimate, nullptr /* translation_parameterization */,
      options_.only_optimize_yaw()
          ? std::unique_ptr<ceres::LocalParameterization>(
                absl::make_unique<ceres::AutoDiffLocalParameterization<
                    YawOnlyQuaternionPlus, 4, 1>>())
          : std::unique_ptr<ceres::LocalParameterization>(
                absl::make_unique<ceres::QuaternionParameterization>()),
      &problem);

  CHECK_EQ(options_.occupied_space_weight_size(),
           point_clouds_and_hybrid_grids.size());
  for (size_t i = 0; i != point_clouds_and_hybrid_grids.size(); ++i) {
    CHECK_GT(options_.occupied_space_weight(i), 0.);
    const sensor::PointCloud& point_cloud =
        *point_clouds_and_hybrid_grids[i].point_cloud;
    const HybridGrid& hybrid_grid =
        *point_clouds_and_hybrid_grids[i].hybrid_grid;
    problem.AddResidualBlock(
        OccupiedSpaceCostFunction3D::CreateAutoDiffCostFunction(
            options_.occupied_space_weight(i) /
                std::sqrt(static_cast<double>(point_cloud.size())),
            point_cloud, hybrid_grid),
        nullptr /* loss function */, ceres_pose.translation(),
        ceres_pose.rotation());
    if (point_clouds_and_hybrid_grids[i].intensity_hybrid_grid) {
      CHECK_GT(options_.intensity_cost_function_options(i).huber_scale(), 0.);
      CHECK_GT(options_.intensity_cost_function_options(i).weight(), 0.);
      CHECK_GT(
          options_.intensity_cost_function_options(i).intensity_threshold(), 0);
      const IntensityHybridGrid& intensity_hybrid_grid =
          *point_clouds_and_hybrid_grids[i].intensity_hybrid_grid;
      problem.AddResidualBlock(
          IntensityCostFunction3D::CreateAutoDiffCostFunction(
              options_.intensity_cost_function_options(i).weight() /
                  std::sqrt(static_cast<double>(point_cloud.size())),
              options_.intensity_cost_function_options(i).intensity_threshold(),
              point_cloud, intensity_hybrid_grid),
          new ceres::HuberLoss(
              options_.intensity_cost_function_options(i).huber_scale()),
          ceres_pose.translation(), ceres_pose.rotation());
    }
  }

  CHECK_GT(options_.translation_weight(), 0.);
  problem.AddResidualBlock(
      TranslationDeltaCostFunctor3D::CreateAutoDiffCostFunction(
          options_.translation_weight(), target_translation),
      nullptr /* loss function */, ceres_pose.translation());
  CHECK_GT(options_.rotation_weight(), 0.);
  problem.AddResidualBlock(
      RotationDeltaCostFunctor3D::CreateAutoDiffCostFunction(
          options_.rotation_weight(), initial_pose_estimate.rotation()),
      nullptr /* loss function */, ceres_pose.rotation());

  ceres::Solve(ceres_solver_options_, &problem, summary);

  *pose_estimate = ceres_pose.ToRigid();
}

}  // namespace scan_matching
}  // namespace mapping
}  // namespace cartographer
