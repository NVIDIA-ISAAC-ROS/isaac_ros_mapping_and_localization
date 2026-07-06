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

#ifndef ISAAC_ROS_OCCUPANCY_GRID_LOCALIZER__OCCUPANCY_GRID_LOCALIZER_NODE_HPP_
#define ISAAC_ROS_OCCUPANCY_GRID_LOCALIZER__OCCUPANCY_GRID_LOCALIZER_NODE_HPP_

#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "isaac_ros_nitros_flat_scan_type/nitros_flat_scan.hpp"
#include "isaac_ros_occupancy_grid_localizer/occupancy_grid_localizer_core.hpp"
#include "isaac_ros_pointcloud_interfaces/msg/flat_scan.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_srvs/srv/empty.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"

namespace nvidia
{
namespace isaac_ros
{
namespace occupancy_grid_localizer
{

class OccupancyGridLocalizerNode : public rclcpp::Node
{
public:
  explicit OccupancyGridLocalizerNode(const rclcpp::NodeOptions &);

  ~OccupancyGridLocalizerNode();

  OccupancyGridLocalizerNode(const OccupancyGridLocalizerNode &) = delete;

  OccupancyGridLocalizerNode & operator=(const OccupancyGridLocalizerNode &) = delete;

  void GridSearchLocalizationCallback(
    const std::shared_ptr<std_srvs::srv::Empty::Request>,
    std::shared_ptr<std_srvs::srv::Empty::Response>);

  void BufferedFlatScanCallback(
    const std::shared_ptr<const nvidia::isaac_ros::nitros::NitrosFlatScan> flat_scan);

  void TriggerFlatScanCallback(
    const std::shared_ptr<const nvidia::isaac_ros::nitros::NitrosFlatScan> flat_scan);

private:
  void HandleIncomingFlatScan(
    const nvidia::isaac_ros::nitros::NitrosFlatScan & flat_scan,
    bool should_localize);

  std::optional<geometry_msgs::msg::TransformStamped> LookupBaseLinkToLidarTransform(
    const std::string & frame_id) const;

  void PublishLocalizationResult(const geometry_msgs::msg::PoseWithCovarianceStamped & pose);

  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr
    localization_result_publisher_;

  rclcpp::Subscription<nvidia::isaac_ros::nitros::NitrosFlatScan>::SharedPtr
    buffered_flat_scan_subscriber_;

  rclcpp::Subscription<nvidia::isaac_ros::nitros::NitrosFlatScan>::SharedPtr
    trigger_flat_scan_subscriber_;

  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr grid_search_localize_service_server_;

  OccupancyGridLocalizerCore localizer_;
  std::string loc_result_frame_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  mutable std::mutex buffered_flat_scan_mutex_;
  isaac_ros_pointcloud_interfaces::msg::FlatScan buffered_flat_scan_;
  bool has_buffered_flat_scan_{false};
  std::atomic<bool> trigger_localization_on_next_flatscan_{false};
};

}  // namespace occupancy_grid_localizer
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_OCCUPANCY_GRID_LOCALIZER__OCCUPANCY_GRID_LOCALIZER_NODE_HPP_
