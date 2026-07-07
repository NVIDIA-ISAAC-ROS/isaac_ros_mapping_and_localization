// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef ISAAC_ROS_OCCUPANCY_GRID_LOCALIZER__OCCUPANCY_GRID_LOCALIZER_GPU_HPP_
#define ISAAC_ROS_OCCUPANCY_GRID_LOCALIZER__OCCUPANCY_GRID_LOCALIZER_GPU_HPP_

#include <cstdint>
#include <vector>

namespace nvidia
{
namespace isaac_ros
{
namespace occupancy_grid_localizer
{

struct GpuBeam
{
  float angle;
  float range;
};

struct GpuPose
{
  float x;
  float y;
  float yaw;
};

/// GPU-accelerated batch pose scorer for occupancy grid localization.
/// Uploads the occupancy and distance maps to CUDA texture memory,
/// then evaluates candidate poses in parallel on the GPU.
class OccupancyGridLocalizerGpu
{
public:
  OccupancyGridLocalizerGpu(int batch_size, int num_beams_gpu);
  ~OccupancyGridLocalizerGpu();

  OccupancyGridLocalizerGpu(const OccupancyGridLocalizerGpu &) = delete;
  OccupancyGridLocalizerGpu & operator=(const OccupancyGridLocalizerGpu &) = delete;

  /// Upload occupancy mask (uint8, 255=occupied) and distance map (float, meters)
  /// to GPU texture memory. Must be called once after the map is loaded.
  void UploadMap(
    const uint8_t * occupancy_data,
    const float * distance_data,
    int width, int height);

  /// Upload beam samples to device memory. Call once per localization request.
  void UploadBeams(const std::vector<GpuBeam> & beams);

  /// Score a batch of candidate poses on the GPU.
  /// Each pose is independently scored by raycasting all beams against the
  /// occupancy map. Infinite scores indicate rejected poses.
  void ScorePoses(
    const std::vector<GpuPose> & poses,
    std::vector<float> & scores,
    float origin_x, float origin_y, float origin_yaw, float cell_size,
    float lidar_offset_x, float lidar_offset_y, float lidar_offset_yaw,
    float robot_radius, float max_beam_error, float out_of_range_threshold,
    int min_valid_hits);

private:
  void FreeDeviceMemory();

  int batch_size_;
  int num_beams_gpu_;
  int num_beams_{0};
  int map_width_{0};
  int map_height_{0};

  // Device memory
  void * d_beams_{nullptr};
  void * d_poses_{nullptr};
  float * d_scores_{nullptr};

  // CUDA arrays and texture objects
  struct cudaArray * d_occupancy_array_{nullptr};
  struct cudaArray * d_distance_array_{nullptr};
  unsigned long long occupancy_tex_{0};   // NOLINT(runtime/int) cudaTextureObject_t
  unsigned long long distance_tex_{0};    // NOLINT(runtime/int) cudaTextureObject_t

  // Host-side scratch buffer
  std::vector<float> h_scores_;
};

}  // namespace occupancy_grid_localizer
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_OCCUPANCY_GRID_LOCALIZER__OCCUPANCY_GRID_LOCALIZER_GPU_HPP_
