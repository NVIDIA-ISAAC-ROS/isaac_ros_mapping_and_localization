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

#include "isaac_ros_pointcloud_utils/pointcloud_to_flatscan_node.hpp"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "rcl_interfaces/msg/parameter_descriptor.hpp"
#include "rclcpp_components/register_node_macro.hpp"

#include "cuda_buffer/cuda_buffer_api.hpp"
#include "isaac_ros_common/qos.hpp"
#include "isaac_ros_pointcloud_utils/pointcloud_to_flatscan_cuda.cu.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace pointcloud_utils
{
namespace
{
// Defaults and descriptions match the legacy GXF PointCloudToFlatscan codelet
// (isaac_ros_gxf_extensions/pointcloud) so this node is a drop-in replacement
// for callers that were previously configured against that graph.
rcl_interfaces::msg::ParameterDescriptor MakeDescriptor(const std::string & description)
{
  rcl_interfaces::msg::ParameterDescriptor d;
  d.description = description;
  return d;
}
}  // namespace

PointCloudToFlatScanNode::PointCloudToFlatScanNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("pointcloud_to_flatscan", options),
  min_x_(declare_parameter<double>(
      "min_x", -1.0,
      MakeDescriptor(
        "Minimum value allowed for x axis of 3D points with respect to flatscan frame"))),
  max_x_(declare_parameter<double>(
      "max_x", 1.0,
      MakeDescriptor(
        "Maximum value allowed for x axis of 3D points with respect to flatscan frame"))),
  min_y_(declare_parameter<double>(
      "min_y", -1.0,
      MakeDescriptor(
        "Minimum value allowed for y axis of 3D points with respect to flatscan frame"))),
  max_y_(declare_parameter<double>(
      "max_y", 1.0,
      MakeDescriptor(
        "Maximum value allowed for y axis of 3D points with respect to flatscan frame"))),
  min_z_(declare_parameter<double>(
      "min_z", -1.0,
      MakeDescriptor(
        "Minimum value allowed for z axis of 3D points with respect to flatscan frame"))),
  max_z_(declare_parameter<double>(
      "max_z", 1.0,
      MakeDescriptor(
        "Maximum value allowed for z axis of 3D points with respect to flatscan frame"))),
  max_points_(declare_parameter<int>(
      "max_points", 691200,
      MakeDescriptor(
        "Maximum expected number of beams; used to pre-allocate GPU memory"))),
  threshold_x_axis_(declare_parameter<bool>(
      "threshold_x_axis", false,
      MakeDescriptor("Enable X axis points thresholding"))),
  threshold_y_axis_(declare_parameter<bool>(
      "threshold_y_axis", false,
      MakeDescriptor("Enable Y axis points thresholding")))
{
  if (max_points_ <= 0) {
    throw std::runtime_error("PointCloudToFlatScanNode: max_points must be positive");
  }

  const rclcpp::QoS input_qos = ::isaac_ros::common::AddQosParameter(
    *this, "DEFAULT", "input_qos");
  const rclcpp::QoS output_qos = ::isaac_ros::common::AddQosParameter(
    *this, "DEFAULT", "output_qos");

  // If any step below throws, the destructor will not run (object not fully
  // constructed), so manually release whatever was already allocated and rethrow.
  try {
    cudaError_t err = cudaStreamCreate(&cuda_stream_);
    if (err != cudaSuccess) {
      throw std::runtime_error(
              std::string("cudaStreamCreate failed: ") + cudaGetErrorString(err));
    }

    err = cudaMalloc(&counter_device_, sizeof(uint32_t));
    if (err != cudaSuccess) {
      throw std::runtime_error(
              std::string("cudaMalloc(counter) failed: ") + cudaGetErrorString(err));
    }

    const size_t buffer_bytes =
      2u * static_cast<size_t>(max_points_) * sizeof(float);
    err = cudaMalloc(&scratch_device_, buffer_bytes);
    if (err != cudaSuccess) {
      throw std::runtime_error(
              std::string("cudaMalloc(scratch) failed: ") + cudaGetErrorString(err));
    }

    rclcpp::PublisherOptions pub_options;
    pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
    flatscan_pub_ = create_publisher<isaac_ros_pointcloud_interfaces::msg::FlatScan>(
      "flatscan", output_qos, pub_options);

    rclcpp::SubscriptionOptions sub_options;
    sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
    // Accept GPU-backed point cloud buffers; from_input_buffer promotes CPU buffers as needed.
    sub_options.acceptable_buffer_backends = "any";
    pc_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      "pointcloud", input_qos,
      std::bind(&PointCloudToFlatScanNode::PointCloudCallback, this, std::placeholders::_1),
      sub_options);
  } catch (...) {
    if (scratch_device_ != nullptr) {
      cudaFree(scratch_device_);
      scratch_device_ = nullptr;
    }
    if (counter_device_ != nullptr) {
      cudaFree(counter_device_);
      counter_device_ = nullptr;
    }
    if (cuda_stream_ != nullptr) {
      cudaStreamDestroy(cuda_stream_);
      cuda_stream_ = nullptr;
    }
    throw;
  }
}

PointCloudToFlatScanNode::~PointCloudToFlatScanNode()
{
  if (scratch_device_ != nullptr) {cudaFree(scratch_device_);}
  if (counter_device_ != nullptr) {cudaFree(counter_device_);}
  if (cuda_stream_ != nullptr) {cudaStreamDestroy(cuda_stream_);}
}

void PointCloudToFlatScanNode::PointCloudCallback(
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr & point_cloud)
{
  std::lock_guard<std::mutex> lock(tick_mutex_);

  const uint32_t num_points = point_cloud->width * point_cloud->height;
  if (num_points == 0) {
    RCLCPP_DEBUG(get_logger(), "Empty point cloud received, skipping");
    return;
  }

  const uint32_t max_output = static_cast<uint32_t>(max_points_);
  const uint32_t point_step_floats = point_cloud->point_step / sizeof(float);
  // Kernel reads x/y/z at offsets 0,1,2 within each point; smaller strides would OOB.
  if (point_step_floats < 3) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "Point cloud point_step (%u bytes) is too small for XYZ; skipping message",
      point_cloud->point_step);
    return;
  }

  // A PointCloud2 does not guarantee data covers width*height*point_step; reject a short buffer
  // before the kernel reads it to avoid reading past the end of the allocation.
  const size_t required_bytes =
    static_cast<size_t>(num_points) * point_cloud->point_step;
  if (point_cloud->data.size() < required_bytes) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "Point cloud data buffer (%zu bytes) is smaller than width*height*point_step (%zu bytes); "
      "skipping message",
      point_cloud->data.size(), required_bytes);
    return;
  }

  auto read_handle = cuda_buffer_backend::from_input_buffer(point_cloud->data, cuda_stream_);
  const float * input_points = reinterpret_cast<const float *>(read_handle.get_ptr());

  float * scratch_angles = reinterpret_cast<float *>(scratch_device_);
  float * scratch_ranges = scratch_angles + static_cast<size_t>(max_points_);

  cudaError_t err = cudaMemsetAsync(counter_device_, 0, sizeof(uint32_t), cuda_stream_);
  if (err != cudaSuccess) {
    RCLCPP_ERROR(get_logger(), "cudaMemsetAsync(counter) failed: %s", cudaGetErrorString(err));
    return;
  }

  const ThresholdParams params{
    threshold_x_axis_,
    threshold_y_axis_,
    static_cast<float>(min_x_),
    static_cast<float>(max_x_),
    static_cast<float>(min_y_),
    static_cast<float>(max_y_),
    static_cast<float>(min_z_),
    static_cast<float>(max_z_),
  };
  LaunchPointCloudToFlatscan(
    input_points, num_points, point_step_floats,
    scratch_angles, scratch_ranges, max_output, counter_device_, params, cuda_stream_);

  uint32_t count = 0;
  err = cudaMemcpyAsync(
    &count, counter_device_, sizeof(uint32_t), cudaMemcpyDeviceToHost, cuda_stream_);
  if (err != cudaSuccess) {
    RCLCPP_ERROR(get_logger(), "cudaMemcpyAsync(count) failed: %s", cudaGetErrorString(err));
    return;
  }
  err = cudaStreamSynchronize(cuda_stream_);
  if (err != cudaSuccess) {
    RCLCPP_ERROR(get_logger(), "cudaStreamSynchronize failed: %s", cudaGetErrorString(err));
    return;
  }
  if (count > max_output) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "Thresholded point count (%u) exceeded max_points (%u); trailing points were dropped",
      count, max_output);
    count = max_output;
  }

  auto flatscan = std::make_unique<isaac_ros_pointcloud_interfaces::msg::FlatScan>();
  flatscan->header.frame_id = point_cloud->header.frame_id;
  flatscan->header.stamp = point_cloud->header.stamp;
  flatscan->range_min = 0.0f;
  flatscan->range_max = 0.0f;

  // Copy GPU scratch planes device-to-host into the message vectors, then sync before publish.
  if (count > 0) {
    flatscan->angles.resize(count);
    flatscan->ranges.resize(count);
    const size_t plane_bytes = static_cast<size_t>(count) * sizeof(float);

    err = cudaMemcpyAsync(
      flatscan->angles.data(), scratch_angles, plane_bytes, cudaMemcpyDeviceToHost, cuda_stream_);
    if (err != cudaSuccess) {
      RCLCPP_ERROR(get_logger(), "cudaMemcpyAsync(angles) failed: %s", cudaGetErrorString(err));
      return;
    }
    err = cudaMemcpyAsync(
      flatscan->ranges.data(), scratch_ranges, plane_bytes, cudaMemcpyDeviceToHost, cuda_stream_);
    if (err != cudaSuccess) {
      RCLCPP_ERROR(get_logger(), "cudaMemcpyAsync(ranges) failed: %s", cudaGetErrorString(err));
      return;
    }
    err = cudaStreamSynchronize(cuda_stream_);
    if (err != cudaSuccess) {
      RCLCPP_ERROR(get_logger(), "cudaStreamSynchronize(copy) failed: %s", cudaGetErrorString(err));
      return;
    }
  }

  flatscan_pub_->publish(std::move(flatscan));
}

}  // namespace pointcloud_utils
}  // namespace isaac_ros
}  // namespace nvidia

RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::pointcloud_utils::PointCloudToFlatScanNode)
