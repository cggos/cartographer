/*
 * CollatedTrajectoryBuilder - 数据整理轨迹构建器
 * 
 * 功能：包装 TrajectoryBuilderInterface，负责传感器数据排序和分发
 * 
 * 工作流程：
 * 1. 接收 SensorBridge 发送的原始传感器数据
 * 2. 将数据传递给 TrajectoryCollator 进行排序
 * 3. TrajectoryCollator 按时间顺序触发回调
 * 4. 调用 GlobalTrajectoryBuilder 进行实际的SLAM处理
 * 
 * 核心设计：
 * - TrajectoryCollator: 多传感器数据按时间排序
 * - wrapped_trajectory_builder_: 实际处理数据的 TrajectoryBuilder
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

#include "cartographer/mapping/internal/collated_trajectory_builder.h"

#include "cartographer/common/time.h"
#include "glog/logging.h"

namespace cartographer {
namespace mapping {

namespace {

constexpr double kSensorDataRatesLoggingPeriodSeconds = 15.;

}  // namespace

/**
 * @brief 构造函数 - 初始化数据整理器
 * 
 * @param trajectory_options 轨迹构建器选项
 * @param sensor_collator 传感器数据排序器
 * @param trajectory_id 轨迹ID
 * @param expected_sensor_ids 期望的传感器ID集合
 * @param wrapped_trajectory_builder 实际处理数据的TrajectoryBuilder（通常为GlobalTrajectoryBuilder）
 * 
 * 初始化流程：
 * 1. 保存传感器数据排序器
 * 2. 保存实际处理数据的TrajectoryBuilder
 * 3. 向排序器注册回调函数（用于数据排序后触发）
 */
CollatedTrajectoryBuilder::CollatedTrajectoryBuilder(
    const proto::TrajectoryBuilderOptions& trajectory_options,
    sensor::CollatorInterface* const sensor_collator, const int trajectory_id,
    const std::set<SensorId>& expected_sensor_ids,
    std::unique_ptr<TrajectoryBuilderInterface> wrapped_trajectory_builder)
    : sensor_collator_(sensor_collator),
      collate_landmarks_(trajectory_options.collate_landmarks()),
      collate_fixed_frame_(trajectory_options.collate_fixed_frame()),
      trajectory_id_(trajectory_id),
      wrapped_trajectory_builder_(std::move(wrapped_trajectory_builder)),
      last_logging_time_(std::chrono::steady_clock::now()) {
  absl::flat_hash_set<std::string> expected_sensor_id_strings;
  for (const auto& sensor_id : expected_sensor_ids) {
    if (sensor_id.type == SensorId::SensorType::LANDMARK &&
        !collate_landmarks_) {
      continue;
    }
    if (sensor_id.type == SensorId::SensorType::FIXED_FRAME_POSE &&
        !collate_fixed_frame_) {
      continue;
    }
    expected_sensor_id_strings.insert(sensor_id.id);
  }
  // 向传感器数据排序器注册回调
  // 回调会在数据按时间排序后被触发
  sensor_collator_->AddTrajectory(
      trajectory_id, expected_sensor_id_strings,
      [this](const std::string& sensor_id, std::unique_ptr<sensor::Data> data) {
        HandleCollatedSensorData(sensor_id, std::move(data));
      });
}

/**
 * @brief 添加传感器数据
 * 
 * 功能：
 * 1. 接收 SensorBridge 发送的原始传感器数据
 * 2. 将数据传递给 TrajectoryCollator 进行排序
 * 
 * 调用链：
 * SensorBridge → CollatedTrajectoryBuilder::AddData() → TrajectoryCollator::AddSensorData()
 * 
 * @param data 传感器数据（TimedPointCloudData, ImuData, OdometryData等）
 */
void CollatedTrajectoryBuilder::AddData(std::unique_ptr<sensor::Data> data) {
  // 将数据传递给排序器
  sensor_collator_->AddSensorData(trajectory_id_, std::move(data));
}

/**
 * @brief 处理已排序的传感器数据 - 回调函数
 * 
 * 功能：
 * 1. 记录传感器数据速率
 * 2. 将数据传递给 GlobalTrajectoryBuilder 进行SLAM处理
 * 
 * 调用链：
 * TrajectoryCollator (按时间排序) → HandleCollatedSensorData() → GlobalTrajectoryBuilder::AddSensorData()
 * 
 * 数据流向：
 * - data->AddToTrajectoryBuilder() 会调用 Dispatchable::AddToTrajectoryBuilder()
 * - Dispatchable 内部调用 trajectory_builder_->AddSensorData()
 * - 通过多态，实际调用 GlobalTrajectoryBuilder::AddSensorData()
 * 
 * @param sensor_id 传感器ID
 * @param data 已排序的传感器数据
 */
void CollatedTrajectoryBuilder::HandleCollatedSensorData(
    const std::string& sensor_id, std::unique_ptr<sensor::Data> data) {
  // ========== 步骤1: 记录传感器数据速率 ==========
  // 创建或获取该传感器的计时器
  auto it = rate_timers_.find(sensor_id);
  if (it == rate_timers_.end()) {
    it = rate_timers_
             .emplace(
                 std::piecewise_construct, std::forward_as_tuple(sensor_id),
                 std::forward_as_tuple(
                     common::FromSeconds(kSensorDataRatesLoggingPeriodSeconds)))
             .first;
  }
  it->second.Pulse(data->GetTime());  // 记录数据到达时间

  // ========== 步骤2: 定期打印传感器数据速率 ==========
  if (std::chrono::steady_clock::now() - last_logging_time_ >
      common::FromSeconds(kSensorDataRatesLoggingPeriodSeconds)) {
    for (const auto& pair : rate_timers_) {
      LOG(INFO) << pair.first << " rate: " << pair.second.DebugString();
    }
    last_logging_time_ = std::chrono::steady_clock::now();
  }

  // ========== 步骤3: 将数据传递给GlobalTrajectoryBuilder ==========
  // 关键调用：通过 Dispatchable::AddToTrajectoryBuilder 进行多态调用
  // wrapped_trajectory_builder_ 实际指向 GlobalTrajectoryBuilder
  data->AddToTrajectoryBuilder(wrapped_trajectory_builder_.get());
}

}  // namespace mapping
}  // namespace cartographer
