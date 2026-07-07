// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef ISAAC_ROS_POINTCLOUD_UTILS__POINTCLOUD_TO_FLATSCAN_NODE_HPP_
#define ISAAC_ROS_POINTCLOUD_UTILS__POINTCLOUD_TO_FLATSCAN_NODE_HPP_

#include <cuda_runtime.h>

#include <cstdint>
#include <mutex>

#include "rclcpp/rclcpp.hpp"

#include "isaac_ros_nitros/types/cuda_memory_pool.hpp"
#include "isaac_ros_nitros_flat_scan_type/nitros_flat_scan.hpp"
#include "isaac_ros_nitros_point_cloud_type/nitros_point_cloud.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace pointcloud_utils
{

class PointCloudToFlatScanNode : public rclcpp::Node
{
public:
  explicit PointCloudToFlatScanNode(const rclcpp::NodeOptions & options);
  ~PointCloudToFlatScanNode() override;

  PointCloudToFlatScanNode(const PointCloudToFlatScanNode &) = delete;
  PointCloudToFlatScanNode & operator=(const PointCloudToFlatScanNode &) = delete;

private:
  void PointCloudCallback(
    const nvidia::isaac_ros::nitros::NitrosPointCloud::ConstSharedPtr & point_cloud);

  const double min_x_;
  const double max_x_;
  const double min_y_;
  const double max_y_;
  const double min_z_;
  const double max_z_;
  const int max_points_;
  const bool threshold_x_axis_;
  const bool threshold_y_axis_;

  cudaStream_t cuda_stream_{};
  uint32_t * counter_device_{nullptr};
  uint8_t * scratch_device_{nullptr};

  nvidia::isaac_ros::nitros::CUDAMemoryPool output_pool_;

  std::mutex tick_mutex_;

  rclcpp::Subscription<nvidia::isaac_ros::nitros::NitrosPointCloud>::SharedPtr pc_sub_;
  rclcpp::Publisher<nvidia::isaac_ros::nitros::NitrosFlatScan>::SharedPtr flatscan_pub_;
};

}  // namespace pointcloud_utils
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_POINTCLOUD_UTILS__POINTCLOUD_TO_FLATSCAN_NODE_HPP_
