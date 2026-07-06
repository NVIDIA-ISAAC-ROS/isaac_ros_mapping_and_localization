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

#include "isaac_ros_occupancy_grid_localizer/occupancy_grid_localizer_gpu.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
constexpr int kThreadsPerBlock = 256;

#define CUDA_CHECK(call) \
  do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
      throw std::runtime_error( \
        std::string("CUDA error in ") + __FILE__ + ":" + std::to_string(__LINE__) + \
        " - " + cudaGetErrorString(err)); \
    } \
  } while (0)

// Device-side struct matching GpuBeam
struct DeviceBeam
{
  float angle;
  float range;
};

// Device-side struct matching GpuPose
struct DevicePose
{
  float x;
  float y;
  float yaw;
};

// Device function: world coordinates to pixel coordinates
// Applies the inverse rotation of the map origin yaw before scaling.
__device__ bool WorldToPixel(
  float world_x, float world_y,
  float origin_x, float origin_y, float origin_cos, float origin_sin,
  float cell_size,
  int map_width, int map_height,
  int & pixel_x, int & pixel_y)
{
  float tx = world_x - origin_x;
  float ty = world_y - origin_y;
  float map_x = (origin_cos * tx + origin_sin * ty) / cell_size;
  float map_y = (-origin_sin * tx + origin_cos * ty) / cell_size;
  pixel_x = __float2int_rd(map_x);
  pixel_y = map_height - 1 - __float2int_rd(map_y);
  return pixel_x >= 0 && pixel_x < map_width && pixel_y >= 0 && pixel_y < map_height;
}

// Device function: raycast a single beam and return range to first obstacle
__device__ float RaycastBeam(
  float lidar_x, float lidar_y, float beam_yaw, float max_range,
  float origin_x, float origin_y, float origin_cos, float origin_sin,
  float cell_size,
  int map_width, int map_height,
  cudaTextureObject_t occupancy_tex)
{
  const float step = fmaxf(cell_size * 0.5f, 0.01f);
  const float cos_yaw = __cosf(beam_yaw);
  const float sin_yaw = __sinf(beam_yaw);

  for (float range = 0.0f; range <= max_range; range += step) {
    float sample_x = lidar_x + cos_yaw * range;
    float sample_y = lidar_y + sin_yaw * range;
    int px, py;
    if (!WorldToPixel(sample_x, sample_y, origin_x, origin_y, origin_cos, origin_sin,
      cell_size, map_width, map_height, px, py))
    {
      return range;
    }
    // tex2D returns the occupancy value (255 = occupied, 0 = free)
    unsigned char occ = tex2D<unsigned char>(occupancy_tex, px + 0.5f, py + 0.5f);
    if (occ != 0) {
      return range;
    }
  }
  return max_range;
}

// Main kernel: each thread scores one candidate pose
__global__ void ScorePosesKernel(
  const DevicePose * __restrict__ poses,
  const DeviceBeam * __restrict__ beams,
  float * __restrict__ scores,
  int num_poses,
  int num_beams,
  float origin_x, float origin_y, float origin_cos, float origin_sin,
  float cell_size,
  int map_width, int map_height,
  float lidar_offset_x, float lidar_offset_y, float lidar_offset_yaw,
  float robot_radius, float max_beam_error, float out_of_range_threshold,
  int min_valid_hits,
  cudaTextureObject_t occupancy_tex,
  cudaTextureObject_t distance_tex)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_poses) {
    return;
  }

  DevicePose pose = poses[tid];

  // Check robot radius using distance map
  int px, py;
  if (!WorldToPixel(pose.x, pose.y, origin_x, origin_y, origin_cos, origin_sin,
    cell_size, map_width, map_height, px, py))
  {
    scores[tid] = INFINITY;
    return;
  }
  float dist = tex2D<float>(distance_tex, px + 0.5f, py + 0.5f);
  if (dist < robot_radius) {
    scores[tid] = INFINITY;
    return;
  }

  // Compute lidar position from base pose + lidar offset
  float cos_yaw = __cosf(pose.yaw);
  float sin_yaw = __sinf(pose.yaw);
  float lidar_x = pose.x + cos_yaw * lidar_offset_x - sin_yaw * lidar_offset_y;
  float lidar_y = pose.y + sin_yaw * lidar_offset_x + cos_yaw * lidar_offset_y;
  float lidar_yaw = pose.yaw + lidar_offset_yaw;

  float total_error = 0.0f;
  int valid_hits = 0;

  for (int b = 0; b < num_beams; ++b) {
    float beam_yaw = lidar_yaw + beams[b].angle;
    float expected_range = RaycastBeam(
      lidar_x, lidar_y, beam_yaw, out_of_range_threshold,
      origin_x, origin_y, origin_cos, origin_sin,
      cell_size, map_width, map_height,
      occupancy_tex);
    float range_error = fabsf(expected_range - beams[b].range);
    total_error += fminf(range_error, max_beam_error);
    if (range_error <= max_beam_error) {
      ++valid_hits;
    }
  }

  if (num_beams == 0 || valid_hits < min_valid_hits) {
    scores[tid] = INFINITY;
    return;
  }

  scores[tid] = total_error / static_cast<float>(num_beams);
}

}  // namespace

namespace nvidia
{
namespace isaac_ros
{
namespace occupancy_grid_localizer
{

OccupancyGridLocalizerGpu::OccupancyGridLocalizerGpu(int batch_size, int num_beams_gpu)
: batch_size_(batch_size), num_beams_gpu_(num_beams_gpu)
{
}

OccupancyGridLocalizerGpu::~OccupancyGridLocalizerGpu()
{
  FreeDeviceMemory();
}

void OccupancyGridLocalizerGpu::FreeDeviceMemory()
{
  if (occupancy_tex_) {
    cudaDestroyTextureObject(occupancy_tex_);
    occupancy_tex_ = 0;
  }
  if (distance_tex_) {
    cudaDestroyTextureObject(distance_tex_);
    distance_tex_ = 0;
  }
  if (d_occupancy_array_) {
    cudaFreeArray(d_occupancy_array_);
    d_occupancy_array_ = nullptr;
  }
  if (d_distance_array_) {
    cudaFreeArray(d_distance_array_);
    d_distance_array_ = nullptr;
  }
  if (d_beams_) {
    cudaFree(d_beams_);
    d_beams_ = nullptr;
  }
  if (d_poses_) {
    cudaFree(d_poses_);
    d_poses_ = nullptr;
  }
  if (d_scores_) {
    cudaFree(d_scores_);
    d_scores_ = nullptr;
  }
}

void OccupancyGridLocalizerGpu::UploadMap(
  const uint8_t * occupancy_data,
  const float * distance_data,
  int width, int height)
{
  FreeDeviceMemory();

  map_width_ = width;
  map_height_ = height;

  // Create CUDA arrays for texture binding
  cudaChannelFormatDesc occ_desc = cudaCreateChannelDesc<unsigned char>();
  CUDA_CHECK(cudaMallocArray(&d_occupancy_array_, &occ_desc, width, height));
  CUDA_CHECK(cudaMemcpy2DToArray(
    d_occupancy_array_, 0, 0, occupancy_data, width * sizeof(uint8_t),
    width * sizeof(uint8_t), height, cudaMemcpyHostToDevice));

  cudaChannelFormatDesc dist_desc = cudaCreateChannelDesc<float>();
  CUDA_CHECK(cudaMallocArray(&d_distance_array_, &dist_desc, width, height));
  CUDA_CHECK(cudaMemcpy2DToArray(
    d_distance_array_, 0, 0, distance_data, width * sizeof(float),
    width * sizeof(float), height, cudaMemcpyHostToDevice));

  // Create texture objects
  cudaResourceDesc res_desc = {};
  res_desc.resType = cudaResourceTypeArray;

  cudaTextureDesc tex_desc = {};
  tex_desc.addressMode[0] = cudaAddressModeClamp;
  tex_desc.addressMode[1] = cudaAddressModeClamp;
  tex_desc.filterMode = cudaFilterModePoint;
  tex_desc.readMode = cudaReadModeElementType;
  tex_desc.normalizedCoords = 0;

  res_desc.res.array.array = d_occupancy_array_;
  CUDA_CHECK(cudaCreateTextureObject(&occupancy_tex_, &res_desc, &tex_desc, nullptr));

  res_desc.res.array.array = d_distance_array_;
  CUDA_CHECK(cudaCreateTextureObject(&distance_tex_, &res_desc, &tex_desc, nullptr));

  // Pre-allocate device buffers for batch processing
  CUDA_CHECK(cudaMalloc(&d_poses_, batch_size_ * sizeof(GpuPose)));
  CUDA_CHECK(cudaMalloc(&d_scores_, batch_size_ * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_beams_, num_beams_gpu_ * sizeof(GpuBeam)));

  h_scores_.resize(batch_size_);
}

void OccupancyGridLocalizerGpu::UploadBeams(const std::vector<GpuBeam> & beams)
{
  num_beams_ = std::min(static_cast<int>(beams.size()), num_beams_gpu_);
  CUDA_CHECK(cudaMemcpy(
    d_beams_, beams.data(), num_beams_ * sizeof(GpuBeam), cudaMemcpyHostToDevice));
}

void OccupancyGridLocalizerGpu::ScorePoses(
  const std::vector<GpuPose> & poses,
  std::vector<float> & scores,
  float origin_x, float origin_y, float origin_yaw, float cell_size,
  float lidar_offset_x, float lidar_offset_y, float lidar_offset_yaw,
  float robot_radius, float max_beam_error, float out_of_range_threshold,
  int min_valid_hits)
{
  const float origin_cos = std::cos(origin_yaw);
  const float origin_sin = std::sin(origin_yaw);
  const int num_poses = static_cast<int>(poses.size());
  scores.resize(num_poses);

  // Process in batches
  for (int offset = 0; offset < num_poses; offset += batch_size_) {
    const int batch = std::min(batch_size_, num_poses - offset);

    CUDA_CHECK(cudaMemcpy(
      d_poses_, poses.data() + offset, batch * sizeof(GpuPose), cudaMemcpyHostToDevice));

    const int blocks = (batch + kThreadsPerBlock - 1) / kThreadsPerBlock;
    ScorePosesKernel<<<blocks, kThreadsPerBlock>>>(
      reinterpret_cast<const DevicePose *>(d_poses_),
      reinterpret_cast<const DeviceBeam *>(d_beams_),
      d_scores_,
      batch,
      num_beams_,
      origin_x, origin_y, origin_cos, origin_sin,
      cell_size,
      map_width_, map_height_,
      lidar_offset_x, lidar_offset_y, lidar_offset_yaw,
      robot_radius, max_beam_error, out_of_range_threshold,
      min_valid_hits,
      occupancy_tex_, distance_tex_);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
      scores.data() + offset, d_scores_, batch * sizeof(float), cudaMemcpyDeviceToHost));
  }
}

}  // namespace occupancy_grid_localizer
}  // namespace isaac_ros
}  // namespace nvidia
