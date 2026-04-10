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

#ifndef CARTOGRAPHER_MAPPING_INTERNAL_CONNECTED_COMPONENTS_H_
#define CARTOGRAPHER_MAPPING_INTERNAL_CONNECTED_COMPONENTS_H_

#include <map>

#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "cartographer/mapping/proto/connected_components.pb.h"
#include "cartographer/mapping/submaps.h"

namespace cartographer {
namespace mapping {

// 跟踪轨迹之间连通结构的类。
//
// 连通性包括计数（“我直接连接轨迹 i 和 j 有多少次？”）和传递连通性（并查集结构）。
//
// 该类是线程安全的。
class ConnectedComponents {
 public:
  ConnectedComponents();

  ConnectedComponents(const ConnectedComponents&) = delete;
  ConnectedComponents& operator=(const ConnectedComponents&) = delete;

  // 添加一条初始时仅与自身连接的轨迹。
  void Add(int trajectory_id) LOCKS_EXCLUDED(lock_);

  // 连接两条轨迹。如果任一轨迹未被跟踪，它将被添加。
  // 该函数与参数顺序无关。多次调用 Connect 会增加直接连接计数。
  void Connect(int trajectory_id_a, int trajectory_id_b) LOCKS_EXCLUDED(lock_);

  // 确定两条轨迹是否已（传递地）连通。
  // 如果任一轨迹未被跟踪则返回 false，除非是同一条轨迹则返回 true。
  // 该函数与参数顺序无关。
  bool TransitivelyConnected(int trajectory_id_a, int trajectory_id_b)
      LOCKS_EXCLUDED(lock_);

  // 返回轨迹 'trajectory_id_a' 和 'trajectory_id_b' 之间的“直接”连接次数。
  // 如果任一轨迹未被跟踪则返回 0。该函数与参数顺序无关。
  int ConnectionCount(int trajectory_id_a, int trajectory_id_b)
      LOCKS_EXCLUDED(lock_);

  // 按连通性分组的轨迹 ID 列表（每个内部 vector 代表一个连通分量）。
  std::vector<std::vector<int>> Components() LOCKS_EXCLUDED(lock_);

  // 返回与 'trajectory_id' 属于同一连通分量的轨迹 ID 列表。
  std::vector<int> GetComponent(int trajectory_id) LOCKS_EXCLUDED(lock_);

 private:
  // 查找代表元素（根节点）并进行路径压缩。
  int FindSet(int trajectory_id) EXCLUSIVE_LOCKS_REQUIRED(lock_);
  // 合并两个轨迹所属的集合。
  void Union(int trajectory_id_a, int trajectory_id_b)
      EXCLUSIVE_LOCKS_REQUIRED(lock_);

  absl::Mutex lock_;
  // 使用并查集森林跟踪传递连通性，即每个条目指向该轨迹的代表元素。
  std::map<int, int> forest_ GUARDED_BY(lock_);
  // 跟踪一对轨迹之间的直接连接次数。
  std::map<std::pair<int, int>, int> connection_map_ GUARDED_BY(lock_);
};

// Returns a proto encoding connected components.
proto::ConnectedComponents ToProto(
    std::vector<std::vector<int>> connected_components);

}  // namespace mapping
}  // namespace cartographer

#endif  // CARTOGRAPHER_MAPPING_INTERNAL_CONNECTED_COMPONENTS_H_
