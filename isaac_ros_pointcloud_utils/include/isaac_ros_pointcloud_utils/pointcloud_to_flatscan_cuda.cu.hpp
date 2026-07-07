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

#ifndef ISAAC_ROS_POINTCLOUD_UTILS__POINTCLOUD_TO_FLATSCAN_CUDA_CU_HPP_
#define ISAAC_ROS_POINTCLOUD_UTILS__POINTCLOUD_TO_FLATSCAN_CUDA_CU_HPP_

#include <cuda_runtime.h>

#include <cstdint>

namespace nvidia
{
namespace isaac_ros
{
namespace pointcloud_utils
{

struct ThresholdParams
{
  bool threshold_x_axis;
  bool threshold_y_axis;
  float min_x;
  float max_x;
  float min_y;
  float max_y;
  float min_z;
  float max_z;
};

// Filters input points by axis-aligned box and projects survivors to (angle, range).
// Each survivor atomicAdds *counter_device to reserve an output slot, then writes the
// (angle, range) pair into scratch_angles / scratch_ranges only if the reserved index
// is < max_output; survivors that reserve an index >= max_output are silently dropped
// (no out-of-bounds write). Because the atomic increments for every survivor regardless
// of bounds, *counter_device may end up greater than max_output when more points pass
// the filter than fit in the scratch buffers. Caller contract: zero *counter_device
// before launch, then after the kernel completes treat the produced valid count as
// min(*counter_device, max_output) — only that many leading entries of scratch_angles
// and scratch_ranges are populated. scratch_angles and scratch_ranges must each be
// sized to hold at least max_output floats.
void LaunchPointCloudToFlatscan(
  const float * input_points,
  uint32_t num_points,
  uint32_t point_step_floats,
  float * scratch_angles,
  float * scratch_ranges,
  uint32_t max_output,
  uint32_t * counter_device,
  ThresholdParams params,
  cudaStream_t stream);

}  // namespace pointcloud_utils
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_POINTCLOUD_UTILS__POINTCLOUD_TO_FLATSCAN_CUDA_CU_HPP_
