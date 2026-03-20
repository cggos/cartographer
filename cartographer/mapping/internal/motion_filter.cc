/*
 * MotionFilter - 运动滤波器
 * 
 * 功能：过滤微小的运动，避免在地图中插入冗余节点
 * 
 * 原理：
 * - 如果当前位姿与上一次位姿的差异小于阈值，则认为是"相似"的运动
 * - 相似运动不插入子图，减少计算量和存储空间
 * 
 * 阈值参数：
 * - max_time_seconds: 最大时间差
 * - max_distance_meters: 最大平移距离
 * - max_angle_radians: 最大旋转角度
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

#include "cartographer/mapping/internal/motion_filter.h"

#include "cartographer/transform/transform.h"
#include "glog/logging.h"

namespace cartographer {
namespace mapping {

proto::MotionFilterOptions CreateMotionFilterOptions(
    common::LuaParameterDictionary* const parameter_dictionary) {
  proto::MotionFilterOptions options;
  options.set_max_time_seconds(
      parameter_dictionary->GetDouble("max_time_seconds"));
  options.set_max_distance_meters(
      parameter_dictionary->GetDouble("max_distance_meters"));
  options.set_max_angle_radians(
      parameter_dictionary->GetDouble("max_angle_radians"));
  return options;
}

MotionFilter::MotionFilter(const proto::MotionFilterOptions& options)
    : options_(options) {}

/**
 * @brief 判断运动是否相似
 * 
 * 功能：如果机器人运动幅度很小（时间、距离、角度），则返回true
 * 
 * 返回true的条件（全部满足）：
 * 1. 不是第一个位姿
 * 2. 时间差 <= max_time_seconds
 * 3. 平移距离 <= max_distance_meters
 * 4. 旋转角度 <= max_angle_radians
 * 
 * @param time 当前位姿时间戳
 * @param pose 当前位姿
 * @return true 运动相似，不需要插入
 * @return false 运动明显，需要插入
 */
bool MotionFilter::IsSimilar(const common::Time time,
                             const transform::Rigid3d& pose) {
  // 打印过滤统计信息
  LOG_IF_EVERY_N(INFO, num_total_ >= 500, 500)
      << "Motion filter reduced the number of nodes to "
      << 100. * num_different_ / num_total_ << "%.";
  ++num_total_;
  
  // ========== 判断是否相似 ==========
  // 时间差检查
  if (num_total_ > 1 &&
      time - last_time_ <= common::FromSeconds(options_.max_time_seconds()) &&
      // 平移距离检查
      (pose.translation() - last_pose_.translation()).norm() <=
          options_.max_distance_meters() &&
      // 旋转角度检查
      transform::GetAngle(pose.inverse() * last_pose_) <=
          options_.max_angle_radians()) {
    return true;  // 运动相似
  }
  
  // 更新历史记录
  last_time_ = time;
  last_pose_ = pose;
  ++num_different_;
  return false;  // 运动明显，需要插入
}

}  // namespace mapping
}  // namespace cartographer
