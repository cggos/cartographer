/*
 * RealTimeCorrelativeScanMatcher3D - 实时相关扫描匹配器
 * 
 * 功能：使用穷举搜索方法进行快速的激光点云匹配
 * 
 * 算法原理：
 * 1. 在给定的搜索窗口内，对所有可能的位姿变换进行穷举搜索
 * 2. 对每个候选位姿计算匹配分数
 * 3. 选择分数最高的位姿作为匹配结果
 * 
 * 特点：
 * - 计算速度快，适合实时应用
 * - 提供良好的初始估计给后续的Ceres优化器
 * - 精度受限于搜索分辨率
 * 
 * 搜索空间：
 * - 平移搜索窗口：linear_search_window
 * - 旋转搜索窗口：angular_search_window
 * - 搜索步长：resolution
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

#ifndef CARTOGRAPHER_MAPPING_INTERNAL_3D_SCAN_MATCHING_REAL_TIME_CORRELATIVE_SCAN_MATCHER_3D_H_
#define CARTOGRAPHER_MAPPING_INTERNAL_3D_SCAN_MATCHING_REAL_TIME_CORRELATIVE_SCAN_MATCHER_3D_H_

#include <vector>

#include "Eigen/Core"
#include "cartographer/mapping/3d/hybrid_grid.h"
#include "cartographer/mapping/proto/scan_matching/real_time_correlative_scan_matcher_options.pb.h"
#include "cartographer/sensor/point_cloud.h"

namespace cartographer {
namespace mapping {
namespace scan_matching {

/**
 * @brief 实时相关扫描匹配器
 * 
 * 使用穷举搜索方法进行快速的激光点云匹配
 * 
 * 成员函数：
 * - Match(): 主匹配函数，穷举搜索最佳位姿
 * - GenerateExhaustiveSearchTransforms(): 生成所有候选变换
 * - ScoreCandidate(): 计算候选位姿的匹配分数
 */
class RealTimeCorrelativeScanMatcher3D {
 public:
  /**
   * @brief 构造函数
   * @param options 匹配器选项
   */
  explicit RealTimeCorrelativeScanMatcher3D(
      const scan_matching::proto::RealTimeCorrelativeScanMatcherOptions&
          options);

  RealTimeCorrelativeScanMatcher3D(const RealTimeCorrelativeScanMatcher3D&) =
      delete;
  RealTimeCorrelativeScanMatcher3D& operator=(
      const RealTimeCorrelativeScanMatcher3D&) = delete;

  /**
   * @brief 扫描匹配
   * 
   * 功能：在HybridGrid中搜索与点云最佳匹配的位姿
   * 
   * 算法流程：
   * 1. 生成所有候选变换（穷举搜索）
   * 2. 对每个候选变换计算分数
   * 3. 返回分数最高的位姿
   * 
   * @param initial_pose_estimate 初始位姿估计（来自PoseExtrapolator）
   * @param point_cloud 输入点云
   * @param hybrid_grid 子图混合栅格
   * @param pose_estimate 输出：最佳匹配位姿
   * @return float 最佳匹配分数
   */
  float Match(const transform::Rigid3d& initial_pose_estimate,
              const sensor::PointCloud& point_cloud,
              const HybridGrid& hybrid_grid,
              transform::Rigid3d* pose_estimate) const;

 private:
  /**
   * @brief 生成穷举搜索的所有候选变换
   * 
   * 在给定的搜索窗口内，生成所有可能的平移和旋转变换
   * 
   * @param resolution 搜索分辨率
   * @param point_cloud 输入点云（用于计算角度步长）
   * @return std::vector<transform::Rigid3f> 候选变换列表
   */
  std::vector<transform::Rigid3f> GenerateExhaustiveSearchTransforms(
      float resolution, const sensor::PointCloud& point_cloud) const;
      
  /**
   * @brief 计算候选位姿的匹配分数
   * 
   * 分数计算：
   * 1. 统计变换后点云落在占据栅格中的概率和
   * 2. 考虑位姿与初始估计的偏差惩罚
   * 
   * @param hybrid_grid 子图栅格
   * @param transformed_point_cloud 变换后的点云
   * @param transform 当前变换
   * @return float 匹配分数
   */
  float ScoreCandidate(const HybridGrid& hybrid_grid,
                       const sensor::PointCloud& transformed_point_cloud,
                       const transform::Rigid3f& transform) const;

  const proto::RealTimeCorrelativeScanMatcherOptions options_;
};

}  // namespace scan_matching
}  // namespace mapping
}  // namespace cartographer

#endif  // CARTOGRAPHER_MAPPING_INTERNAL_3D_SCAN_MATCHING_REAL_TIME_CORRELATIVE_SCAN_MATCHER_3D_H_
