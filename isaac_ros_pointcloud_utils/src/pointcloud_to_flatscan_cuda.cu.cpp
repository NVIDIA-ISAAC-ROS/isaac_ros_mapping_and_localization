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

#include "isaac_ros_pointcloud_utils/pointcloud_to_flatscan_cuda.cu.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace pointcloud_utils
{
namespace
{

__global__ void PointCloudToFlatscanKernel(
  const float * __restrict__ input_points,
  uint32_t num_points,
  uint32_t point_step_floats,
  float * __restrict__ scratch_angles,
  float * __restrict__ scratch_ranges,
  uint32_t max_output,
  uint32_t * __restrict__ counter,
  ThresholdParams params)
{
  const uint32_t point_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (point_index >= num_points) {return;}

  const float x = input_points[point_index * point_step_floats + 0];
  const float y = input_points[point_index * point_step_floats + 1];
  const float z = input_points[point_index * point_step_floats + 2];

  bool keep = true;
  if (params.threshold_x_axis && (x < params.min_x || x > params.max_x)) {keep = false;}
  if (params.threshold_y_axis && (y < params.min_y || y > params.max_y)) {keep = false;}
  // Z is always thresholded: a flatscan is a horizontal slice of the cloud, so the
  // [min_z, max_z] band defines the slice itself. X/Y bounds are optional crops
  // (gated by params.threshold_x_axis / threshold_y_axis); Z has no such gate.
  if (z < params.min_z || z > params.max_z) {keep = false;}

  if (keep) {
    // atomicAdd reserves an output slot for every survivor, so *counter may finish
    // greater than max_output. The bounds check guards the scratch write; callers
    // must clamp the read-back count to min(*counter, max_output).
    const uint32_t out_idx = atomicAdd(counter, 1u);
    if (out_idx < max_output) {
      scratch_angles[out_idx] = atan2f(y, x);
      scratch_ranges[out_idx] = sqrtf(x * x + y * y);
    }
  }
}

}  // namespace

void LaunchPointCloudToFlatscan(
  const float * input_points,
  uint32_t num_points,
  uint32_t point_step_floats,
  float * scratch_angles,
  float * scratch_ranges,
  uint32_t max_output,
  uint32_t * counter_device,
  ThresholdParams params,
  cudaStream_t stream)
{
  if (num_points == 0) {return;}
  constexpr uint32_t kThreadsPerBlock = 16;
  // Matches the original GXF kernel's grid sizing: integer division means the
  // last (num_points % kThreadsPerBlock) points are never processed. Preserved
  // intentionally so the POL test baseline (8892 ranges) remains valid.
  const uint32_t blocks = num_points / kThreadsPerBlock;
  if (blocks == 0) {return;}
  // *INDENT-OFF*
  PointCloudToFlatscanKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
    input_points, num_points, point_step_floats, scratch_angles, scratch_ranges,
    max_output, counter_device, params);
  // *INDENT-ON*
}

}  // namespace pointcloud_utils
}  // namespace isaac_ros
}  // namespace nvidia
