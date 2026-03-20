/*
 * 体素滤波器 (Voxel Filter)
 * 
 * 功能：对点云进行降采样，减少计算量同时保留关键特征
 * 
 * 滤波器类型：
 * 1. VoxelFilter: 固定分辨率体素滤波
 * 2. AdaptiveVoxelFilter: 自适应体素滤波
 * 
 * 原理：
 * - 将点云划分为三维体素网格
 * - 每个体素只保留一个代表性点
 * - 使用随机采样保证均匀性
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

#include "cartographer/sensor/internal/voxel_filter.h"

#include <cmath>
#include <random>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "cartographer/common/math.h"

namespace cartographer {
namespace sensor {

namespace {

/**
 * @brief 按最大距离过滤点云
 * 
 * @param point_cloud 输入点云
 * @param max_range 最大有效距离
 * @return PointCloud 过滤后的点云
 */
PointCloud FilterByMaxRange(const PointCloud& point_cloud,
                            const float max_range) {
  return point_cloud.copy_if([max_range](const RangefinderPoint& point) {
    return point.position.norm() <= max_range;  // 只保留距离原点<=max_range的点
  });
}

/**
 * @brief 自适应体素滤波
 * 
 * 功能：根据点云密度自适应调整体素大小，保证滤波后点云密度
 * 
 * 算法步骤：
 * 1. 如果点云已经很稀疏（<=min_num_points），直接返回
 * 2. 用 max_length 进行体素滤波
 * 3. 如果结果点数>=min_num_points，返回结果
 * 4. 否则二分搜索找到合适的体素大小
 * 
 * 参数说明：
 * - max_length: 最大体素边长
 * - min_num_points: 每个体素最少保留点数
 * - max_range: 最大有效距离
 * 
 * @param options 自适应滤波选项
 * @param point_cloud 输入点云
 * @return PointCloud 滤波后的点云
 */
PointCloud AdaptivelyVoxelFiltered(
    const proto::AdaptiveVoxelFilterOptions& options,
    const PointCloud& point_cloud) {
  if (point_cloud.size() <= options.min_num_points()) {
    // 'point_cloud' is already sparse enough.
    return point_cloud;
  }
  PointCloud result = VoxelFilter(point_cloud, options.max_length());
  if (result.size() >= options.min_num_points()) {
    // Filtering with 'max_length' resulted in a sufficiently dense point cloud.
    return result;
  }
  // Search for a 'low_length' that is known to result in a sufficiently
  // dense point cloud. We give up and use the full 'point_cloud' if reducing
  // the edge length by a factor of 1e-2 is not enough.
  for (float high_length = options.max_length();
       high_length > 1e-2f * options.max_length(); high_length /= 2.f) {
    float low_length = high_length / 2.f;
    result = VoxelFilter(point_cloud, low_length);
    if (result.size() >= options.min_num_points()) {
      // Binary search to find the right amount of filtering. 'low_length' gave
      // a sufficiently dense 'result', 'high_length' did not. We stop when the
      // edge length is at most 10% off.
      while ((high_length - low_length) / low_length > 1e-1f) {
        const float mid_length = (low_length + high_length) / 2.f;
        const PointCloud candidate = VoxelFilter(point_cloud, mid_length);
        if (candidate.size() >= options.min_num_points()) {
          low_length = mid_length;
          result = candidate;
        } else {
          high_length = mid_length;
        }
      }
      return result;
    }
  }
  return result;
}

using VoxelKeyType = uint64_t;

VoxelKeyType GetVoxelCellIndex(const Eigen::Vector3f& point,
                               const float resolution) {
  const Eigen::Array3f index = point.array() / resolution;
  const uint64_t x = common::RoundToInt(index.x());
  const uint64_t y = common::RoundToInt(index.y());
  const uint64_t z = common::RoundToInt(index.z());
  return (x << 42) + (y << 21) + z;
}

template <class T, class PointFunction>
std::vector<bool> RandomizedVoxelFilterIndices(
    const std::vector<T>& point_cloud, const float resolution,
    PointFunction&& point_function) {
  // According to https://en.wikipedia.org/wiki/Reservoir_sampling
  std::minstd_rand0 generator;
  absl::flat_hash_map<VoxelKeyType, std::pair<int, int>>
      voxel_count_and_point_index;
  for (size_t i = 0; i < point_cloud.size(); i++) {
    auto& voxel = voxel_count_and_point_index[GetVoxelCellIndex(
        point_function(point_cloud[i]), resolution)];
    voxel.first++;
    if (voxel.first == 1) {
      voxel.second = i;
    } else {
      std::uniform_int_distribution<> distribution(1, voxel.first);
      if (distribution(generator) == voxel.first) {
        voxel.second = i;
      }
    }
  }
  std::vector<bool> points_used(point_cloud.size(), false);
  for (const auto& voxel_and_index : voxel_count_and_point_index) {
    points_used[voxel_and_index.second.second] = true;
  }
  return points_used;
}

template <class T, class PointFunction>
std::vector<T> RandomizedVoxelFilter(const std::vector<T>& point_cloud,
                                     const float resolution,
                                     PointFunction&& point_function) {
  const std::vector<bool> points_used =
      RandomizedVoxelFilterIndices(point_cloud, resolution, point_function);

  std::vector<T> results;
  for (size_t i = 0; i < point_cloud.size(); i++) {
    if (points_used[i]) {
      results.push_back(point_cloud[i]);
    }
  }
  return results;
}

}  // namespace

/**
 * @brief 固定分辨率体素滤波
 * 
 * 功能：将点云划分为固定大小的体素网格，每个体素只保留一个点
 * 
 * 算法原理：
 * 1. 将点云按体素分辨率划分为网格
 * 2. 每个体素格子内只随机保留一个点
 * 3. 减少点云密度，保留空间分布特征
 * 
 * @param points 输入点云
 * @param resolution 体素分辨率（边长）
 * @return std::vector<RangefinderPoint> 滤波后的点云
 */
std::vector<RangefinderPoint> VoxelFilter(
    const std::vector<RangefinderPoint>& points, const float resolution) {
  return RandomizedVoxelFilter(
      points, resolution,
      [](const RangefinderPoint& point) { return point.position; });
}

/**
 * @brief 带强度的点云体素滤波
 * 
 * @param point_cloud 输入点云（包含强度信息）
 * @param resolution 体素分辨率
 * @return PointCloud 滤波后的点云
 */
PointCloud VoxelFilter(const PointCloud& point_cloud, const float resolution) {
  const std::vector<bool> points_used = RandomizedVoxelFilterIndices(
      point_cloud.points(), resolution,
      [](const RangefinderPoint& point) { return point.position; });

  // 收集保留下来的点
  std::vector<RangefinderPoint> filtered_points;
  for (size_t i = 0; i < point_cloud.size(); i++) {
    if (points_used[i]) {
      filtered_points.push_back(point_cloud[i]);
    }
  }
  // 收集对应的强度值
  std::vector<float> filtered_intensities;
  CHECK_LE(point_cloud.intensities().size(), point_cloud.points().size());
  for (size_t i = 0; i < point_cloud.intensities().size(); i++) {
    if (points_used[i]) {
      filtered_intensities.push_back(point_cloud.intensities()[i]);
    }
  }
  return PointCloud(std::move(filtered_points),
                    std::move(filtered_intensities));
}

TimedPointCloud VoxelFilter(const TimedPointCloud& timed_point_cloud,
                            const float resolution) {
  return RandomizedVoxelFilter(
      timed_point_cloud, resolution,
      [](const TimedRangefinderPoint& point) { return point.position; });
}

std::vector<sensor::TimedPointCloudOriginData::RangeMeasurement> VoxelFilter(
    const std::vector<sensor::TimedPointCloudOriginData::RangeMeasurement>&
        range_measurements,
    const float resolution) {
  return RandomizedVoxelFilter(
      range_measurements, resolution,
      [](const sensor::TimedPointCloudOriginData::RangeMeasurement&
             range_measurement) {
        return range_measurement.point_time.position;
      });
}

proto::AdaptiveVoxelFilterOptions CreateAdaptiveVoxelFilterOptions(
    common::LuaParameterDictionary* const parameter_dictionary) {
  proto::AdaptiveVoxelFilterOptions options;
  options.set_max_length(parameter_dictionary->GetDouble("max_length"));
  options.set_min_num_points(
      parameter_dictionary->GetNonNegativeInt("min_num_points"));
  options.set_max_range(parameter_dictionary->GetDouble("max_range"));
  return options;
}

PointCloud AdaptiveVoxelFilter(
    const PointCloud& point_cloud,
    const proto::AdaptiveVoxelFilterOptions& options) {
  return AdaptivelyVoxelFiltered(
      options, FilterByMaxRange(point_cloud, options.max_range()));
}

}  // namespace sensor
}  // namespace cartographer
