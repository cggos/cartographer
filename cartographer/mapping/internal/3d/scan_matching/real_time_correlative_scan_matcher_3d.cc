/*
 * RealTimeCorrelativeScanMatcher3D 实现
 * 
 * 算法核心：穷举搜索
 * - 在预定义的搜索窗口内枚举所有可能的变换
 * - 对每个变换计算匹配分数
 * - 选择分数最高的变换
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

#include "cartographer/mapping/internal/3d/scan_matching/real_time_correlative_scan_matcher_3d.h"

#include <cmath>

#include "Eigen/Geometry"
#include "cartographer/common/math.h"
#include "cartographer/transform/transform.h"
#include "glog/logging.h"

namespace cartographer {
namespace mapping {
namespace scan_matching {

/**
 * @brief 构造函数
 */
RealTimeCorrelativeScanMatcher3D::RealTimeCorrelativeScanMatcher3D(
    const proto::RealTimeCorrelativeScanMatcherOptions& options)
    : options_(options) {}

/**
 * @brief 扫描匹配主函数
 * 
 * 功能：使用穷举搜索方法找到点云与栅格地图的最佳匹配位姿
 * 
 * 算法流程：
 * 1. 生成所有候选变换（穷举搜索空间）
 * 2. 对每个候选变换：
 *    - 应用变换到点云
 *    - 计算匹配分数
 *    - 保留最高分数的变换
 * 3. 返回最佳匹配位姿
 * 
 * @param initial_pose_estimate 初始位姿估计（来自PoseExtrapolator的外推位姿）
 * @param point_cloud 输入点云
 * @param hybrid_grid 子图混合栅格（包含占据概率信息）
 * @param pose_estimate 输出：最佳匹配位姿
 * @return float 最佳匹配分数
 */
float RealTimeCorrelativeScanMatcher3D::Match(
    const transform::Rigid3d& initial_pose_estimate,
    const sensor::PointCloud& point_cloud, const HybridGrid& hybrid_grid,
    transform::Rigid3d* pose_estimate) const {
  CHECK(pose_estimate != nullptr);
  float best_score = -1.f;
  
  // ========== 步骤1: 生成所有候选变换 ==========
  // 在搜索窗口内穷举生成所有可能的变换
  for (const transform::Rigid3f& transform : GenerateExhaustiveSearchTransforms(
           hybrid_grid.resolution(), point_cloud)) {
    
    // ========== 步骤2: 计算候选位姿 ==========
    // 候选位姿 = 初始估计 × 变换增量
    // 这里使用左乘，因为transform是相对于初始估计的增量
    const transform::Rigid3f candidate =
        initial_pose_estimate.cast<float>() * transform;
        
    // ========== 步骤3: 评分并选择最佳 ==========
    // 将点云变换到候选位姿
    // 在栅格中查询每个点的占据概率
    // 分数 = 平均概率 × 偏差惩罚因子
    const float score = ScoreCandidate(
        hybrid_grid, sensor::TransformPointCloud(point_cloud, candidate),
        transform);
    if (score > best_score) {
      best_score = score;
      *pose_estimate = candidate.cast<double>();
    }
  }
  return best_score;
}

/**
 * @brief 生成穷举搜索的所有候选变换
 * 
 * 功能：在3D空间内生成所有可能的平移和旋转变换
 * 
 * 算法流程：
 * 1. 计算线性搜索范围（基于linear_search_window和resolution）
 * 2. 计算角度搜索范围（基于angular_search_window和点云最大距离）
 * 3. 三重循环枚举所有变换组合
 * 
 * 角度步长计算原理：
 * - 保证角度分辨率与空间分辨率一致
 * - 角度步长 = (resolution / max_scan_range) 弧度
 * - 这样在大范围内搜索时，角度误差与位置误差相当
 * 
 * 搜索空间大小：
 * - 平移：linear_search_window / resolution 的立方
 * - 旋转：angular_search_window / angular_step_size 的立方
 * - 总计：(2*linear_window_size+1)^3 × (2*angular_window_size+1)^3
 * 
 * @param resolution 搜索分辨率（通常为栅格分辨率）
 * @param point_cloud 输入点云（用于计算最大扫描距离）
 * @return std::vector<transform::Rigid3f> 候选变换列表
 */
std::vector<transform::Rigid3f>
RealTimeCorrelativeScanMatcher3D::GenerateExhaustiveSearchTransforms(
    const float resolution, const sensor::PointCloud& point_cloud) const {
  std::vector<transform::Rigid3f> result;
  
  // ========== 步骤1: 计算线性搜索窗口 ==========
  // 将线性搜索窗口转换为网格单元数
  const int linear_window_size =
      common::RoundToInt(options_.linear_search_window() / resolution);
      
  // ========== 步骤2: 计算最大扫描距离 ==========
  // 计算点云中距离传感器最远的点
  // We set this value to something on the order of resolution to make sure that
  // the std::acos() below is defined.
  float max_scan_range = 3.f * resolution;
  for (const sensor::RangefinderPoint& point : point_cloud) {
    const float range = point.position.norm();
    max_scan_range = std::max(range, max_scan_range);
  }
  
  // ========== 步骤3: 计算角度步长 ==========
  // 保证角度分辨率与空间分辨率匹配
  // 小角度近似: sin(θ) ≈ θ ≈ resolution / max_scan_range
  const float kSafetyMargin = 1.f - 1e-3f;
  const float angular_step_size =
      kSafetyMargin * std::acos(1.f - common::Pow2(resolution) /
                                          (2.f * common::Pow2(max_scan_range)));
  const int angular_window_size =
      common::RoundToInt(options_.angular_search_window() / angular_step_size);
      
  // ========== 步骤4: 穷举枚举所有变换 ==========
  // 六重循环：x, y, z, rx, ry, rz
  for (int z = -linear_window_size; z <= linear_window_size; ++z) {
    for (int y = -linear_window_size; y <= linear_window_size; ++y) {
      for (int x = -linear_window_size; x <= linear_window_size; ++x) {
        for (int rz = -angular_window_size; rz <= angular_window_size; ++rz) {
          for (int ry = -angular_window_size; ry <= angular_window_size; ++ry) {
            for (int rx = -angular_window_size; rx <= angular_window_size;
                 ++rx) {
              // 创建旋转变换（轴角表示）
              const Eigen::Vector3f angle_axis(rx * angular_step_size,
                                               ry * angular_step_size,
                                               rz * angular_step_size);
              // 创建平移变换
              // result.emplace_back包含平移和旋转
              result.emplace_back(
                  Eigen::Vector3f(x * resolution, y * resolution,
                                  z * resolution),
                  transform::AngleAxisVectorToRotationQuaternion(angle_axis));
            }
          }
        }
      }
    }
  }
  return result;
}

/**
 * @brief 计算候选位姿的匹配分数
 * 
 * 功能：评估点云在给定变换下与栅格地图的匹配程度
 * 
 * 分数计算公式：
 * score = (Σ probability) / N × exp(-penalty)
 * 
 * 其中：
 * - probability: 每个点落在栅格中的占据概率
 * - N: 点云中点的数量
 * - penalty: 偏差惩罚项
 * 
 * 偏差惩罚项：
 * penalty = translation_delta² × translation_delta_cost_weight 
 *         + angle² × rotation_delta_cost_weight
 * 
 * 作用：
 * - 鼓励匹配结果接近初始估计
 * - 避免匹配结果偏离太远
 * 
 * @param hybrid_grid 子图混合栅格
 * @param transformed_point_cloud 已经变换到候选位姿的点云
 * @param transform 候选变换（用于计算偏差惩罚）
 * @return float 匹配分数
 */
float RealTimeCorrelativeScanMatcher3D::ScoreCandidate(
    const HybridGrid& hybrid_grid,
    const sensor::PointCloud& transformed_point_cloud,
    const transform::Rigid3f& transform) const {
  float score = 0.f;
  
  // ========== 步骤1: 累加占据概率 ==========
  // 对变换后的每个点，查询其在栅格中的占据概率
  // 概率值范围通常为[0.45, 0.95]，0.5表示未知
  for (const sensor::RangefinderPoint& point : transformed_point_cloud) {
    score +=
        hybrid_grid.GetProbability(hybrid_grid.GetCellIndex(point.position));
  }
  
  // ========== 步骤2: 计算平均概率 ==========
  score /= static_cast<float>(transformed_point_cloud.size());
  
  // ========== 步骤3: 应用偏差惩罚 ==========
  // 变换与初始估计的偏差越大，惩罚越大，分数越低
  const float angle = transform::GetAngle(transform);  // 旋转变换的角度
  score *=
      std::exp(-common::Pow2(transform.translation().norm() *
                                 options_.translation_delta_cost_weight() +
                             angle * options_.rotation_delta_cost_weight()));
  CHECK_GT(score, 0.f);
  return score;
}

}  // namespace scan_matching
}  // namespace mapping
}  // namespace cartographer
