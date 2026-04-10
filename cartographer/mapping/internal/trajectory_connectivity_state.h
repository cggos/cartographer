/*
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

#ifndef CARTOGRAPHER_MAPPING_INTERNAL_TRAJECTORY_CONNECTIVITY_STATE_H_
#define CARTOGRAPHER_MAPPING_INTERNAL_TRAJECTORY_CONNECTIVITY_STATE_H_

#include "cartographer/common/time.h"
#include "cartographer/mapping/internal/connected_components.h"

namespace cartographer {
namespace mapping {

// 跟踪轨迹之间连通状态的类。与 ConnectedComponents 相比，
// 它还额外跟踪了全局约束最后一次连接各轨迹的时间。
//
// 该类是线程兼容的。
class TrajectoryConnectivityState {
 public:
  TrajectoryConnectivityState() {}

  TrajectoryConnectivityState(const TrajectoryConnectivityState&) = delete;
  TrajectoryConnectivityState& operator=(const TrajectoryConnectivityState&) =
      delete;

  // 添加一条初始时仅与自身连接的轨迹。
  void Add(int trajectory_id);

  // 连接两条轨迹。如果任一轨迹未被跟踪，它将被添加。
  // 该函数与参数顺序无关。多次调用 Connect 会增加连接计数并更新最后连接时间。
  void Connect(int trajectory_id_a, int trajectory_id_b, common::Time time);

  // 确定两条轨迹是否已（传递地）连通。
  // 如果任一轨迹未被跟踪则返回 false，除非是同一条轨迹则返回 true。
  // 该函数与参数顺序无关。
  bool TransitivelyConnected(int trajectory_id_a, int trajectory_id_b) const;

  // 按连通性分组的轨迹 ID 列表。
  std::vector<std::vector<int>> Components() const;

  // 返回两个轨迹之间最后一次建立连接的时间。
  // 如果任一轨迹未被跟踪或从未连接，则返回时间的起点（beginning of time）。
  common::Time LastConnectionTime(int trajectory_id_a, int trajectory_id_b);

 private:
  // 内部使用的连通性管理器，ConnectedComponents 自身是线程安全的。
  mutable ConnectedComponents connected_components_;

  // 跟踪两个轨迹之间最后一次添加直接连接的时间。
  // 特殊情况：当一次连接使两个原本不相连的分量合并时，
  // 两个分量中所有跨分量的轨迹条目都会更新为此次连接的时间。
  std::map<std::pair<int, int>, common::Time> last_connection_time_map_;
};

}  // namespace mapping
}  // namespace cartographer

#endif  // CARTOGRAPHER_MAPPING_INTERNAL_TRAJECTORY_CONNECTIVITY_STATE_H_
