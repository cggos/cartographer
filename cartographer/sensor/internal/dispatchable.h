/*
 * Dispatchable - 可派发的传感器数据封装
 * 
 * 功能：封装传感器数据，实现数据到 TrajectoryBuilder 的分发
 * 
 * 设计模式：适配器模式
 * - 将不同类型的传感器数据（TimedPointCloudData, ImuData, OdometryData等）
 *   统一封装为 Dispatchable
 * - 通过多态调用 AddToTrajectoryBuilder 将数据发送到 TrajectoryBuilder
 * 
 * 数据流：
 * SensorBridge → Dispatchable<DataType> → TrajectoryBuilderInterface → GlobalTrajectoryBuilder
 * 
 * 模板参数：
 * - DataType: 传感器数据类型（如 TimedPointCloudData, ImuData 等）
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

#ifndef CARTOGRAPHER_SENSOR_INTERNAL_DISPATCHABLE_H_
#define CARTOGRAPHER_SENSOR_INTERNAL_DISPATCHABLE_H_

#include "cartographer/mapping/trajectory_builder_interface.h"
#include "cartographer/sensor/data.h"

namespace cartographer {
namespace sensor {

/**
 * @brief 可派发的传感器数据封装类
 * 
 * 模板参数：
 * - DataType: 传感器数据类型
 *   - TimedPointCloudData: 激光点云
 *   - ImuData: IMU数据
 *   - OdometryData: 里程计数据
 *   - LandmarkData: 路标数据
 * 
 * 成员变量：
 * - sensor_id_: 传感器ID
 * - data_: 封装的传感器数据
 */
template <typename DataType>
class Dispatchable : public Data {
 public:
  /**
   * @brief 构造函数
   * @param sensor_id 传感器ID
   * @param data 传感器数据
   */
  Dispatchable(const std::string &sensor_id, const DataType &data)
      : Data(sensor_id), data_(data) {}

  /**
   * @brief 获取数据时间戳
   * @return common::Time 时间戳
   */
  common::Time GetTime() const override { return data_.time; }
  
  /**
   * @brief 将数据添加到轨迹构建器
   * 
   * 核心功能：
   * - 通过多态调用 TrajectoryBuilderInterface::AddSensorData()
   * - 实际调用 GlobalTrajectoryBuilder::AddSensorData()
   * 
   * 多态原理：
   * - trajectory_builder 是 TrajectoryBuilderInterface* 类型
   * - 实际指向 GlobalTrajectoryBuilder 实例
   * - 通过虚函数表调用具体实现
   * 
   * @param trajectory_builder TrajectoryBuilderInterface指针
   */
  void AddToTrajectoryBuilder(
      mapping::TrajectoryBuilderInterface *const trajectory_builder) override {
    trajectory_builder->AddSensorData(sensor_id_, data_);
  }
  
  /**
   * @brief 获取原始数据
   * @return const DataType& 传感器数据引用
   */
  const DataType &data() const { return data_; }

 private:
  const DataType data_;  // 封装的传感器数据
};

/**
 * @brief 创建 Dispatchable 的工厂函数
 * 
 * @param sensor_id 传感器ID
 * @param data 传感器数据
 * @return std::unique_ptr<Dispatchable<DataType>> 智能指针
 */
template <typename DataType>
std::unique_ptr<Dispatchable<DataType>> MakeDispatchable(
    const std::string &sensor_id, const DataType &data) {
  return absl::make_unique<Dispatchable<DataType>>(sensor_id, data);
}

}  // namespace sensor
}  // namespace cartographer

#endif  // CARTOGRAPHER_SENSOR_INTERNAL_DISPATCHABLE_H_
