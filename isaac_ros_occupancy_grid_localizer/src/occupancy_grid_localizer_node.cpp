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

#include "isaac_ros_occupancy_grid_localizer/occupancy_grid_localizer_node.hpp"

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "isaac_ros_common/qos.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "tf2/exceptions.hpp"

namespace fs = std::filesystem;

namespace
{
constexpr char kBaseLinkFrameName[] = "base_link";

nvidia::isaac_ros::occupancy_grid_localizer::OccupancyGridLocalizerParameters MakeParameters(
  rclcpp::Node & node)
{
  nvidia::isaac_ros::occupancy_grid_localizer::OccupancyGridLocalizerParameters parameters;
  const std::string map_yaml_path = node.declare_parameter<std::string>("map_yaml_path", "");
  const std::string image_name = node.declare_parameter<std::string>("image", "");

  parameters.cell_size = node.declare_parameter<double>("resolution", 0.05);
  parameters.occupancy_grid_map_threshold = static_cast<int>(255.0 *
    node.declare_parameter<double>("occupied_thresh", 0.65));
  parameters.map_png_path = map_yaml_path.substr(0, map_yaml_path.find_last_of("/\\")) + "/" +
    image_name;
  parameters.map_origin = node.declare_parameter<std::vector<double>>("origin", {0.0, 0.0, 0.0});
  parameters.max_points = node.declare_parameter<int>("max_points", 20000);
  parameters.use_gxf_map_convention =
    node.declare_parameter<bool>("use_gxf_map_convention", false);
  parameters.robot_radius = node.declare_parameter<double>("robot_radius", 0.25);
  parameters.max_beam_error = node.declare_parameter<double>("max_beam_error", 0.5);
  parameters.max_output_error = node.declare_parameter<double>("max_output_error", 0.35);
  parameters.min_output_error = node.declare_parameter<double>("min_output_error", 0.22);
  parameters.num_beams_gpu = node.declare_parameter<int>("num_beams_gpu", 512);
  parameters.batch_size = node.declare_parameter<int>("batch_size", 512);
  parameters.sample_distance = node.declare_parameter<double>("sample_distance", 0.1);
  parameters.out_of_range_threshold =
    node.declare_parameter<double>("out_of_range_threshold", 100.0);
  parameters.invalid_range_threshold =
    node.declare_parameter<double>("invalid_range_threshold", 0.0);
  parameters.min_scan_fov_degrees =
    node.declare_parameter<double>("min_scan_fov_degrees", 270.0);
  parameters.use_closest_beam = node.declare_parameter<bool>("use_closest_beam", true);
  return parameters;
}
}  // namespace

namespace nvidia
{
namespace isaac_ros
{
namespace occupancy_grid_localizer
{

OccupancyGridLocalizerNode::OccupancyGridLocalizerNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("occupancy_grid_localizer", options),
  localizer_(MakeParameters(*this)),
  loc_result_frame_(declare_parameter<std::string>("loc_result_frame", "map"))
{
  const std::string map_yaml_path = get_parameter("map_yaml_path").as_string();
  const std::string map_png_path =
    map_yaml_path.substr(0, map_yaml_path.find_last_of("/\\")) + "/" +
    get_parameter("image").as_string();

  if (!fs::exists(map_yaml_path)) {
    RCLCPP_ERROR(
      get_logger(), "Could not find map YAML at %s. Exiting.", map_yaml_path.c_str());
    throw std::runtime_error("Parameter parsing failure.");
  }

  if (!fs::exists(map_png_path)) {
    RCLCPP_ERROR(get_logger(), "Could not find map at %s. Exiting.", map_png_path.c_str());
    throw std::runtime_error("Parameter parsing failure.");
  }

  localizer_.LoadMap();

  const rclcpp::QoS input_qos =
    ::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "input_qos");
  const rclcpp::QoS output_qos =
    ::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "output_qos");

  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  localization_result_publisher_ =
    create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
    "localization_result", output_qos);

  buffered_flat_scan_subscriber_ =
    create_subscription<isaac_ros_pointcloud_interfaces::msg::FlatScan>(
    "flatscan", input_qos,
    std::bind(&OccupancyGridLocalizerNode::BufferedFlatScanCallback, this, std::placeholders::_1));

  trigger_flat_scan_subscriber_ =
    create_subscription<isaac_ros_pointcloud_interfaces::msg::FlatScan>(
    "flatscan_localization", input_qos,
    std::bind(&OccupancyGridLocalizerNode::TriggerFlatScanCallback, this, std::placeholders::_1));

  grid_search_localize_service_server_ =
    create_service<std_srvs::srv::Empty>(
    "trigger_grid_search_localization",
    std::bind(
      &OccupancyGridLocalizerNode::GridSearchLocalizationCallback, this,
      std::placeholders::_1, std::placeholders::_2));
}

OccupancyGridLocalizerNode::~OccupancyGridLocalizerNode() = default;

void OccupancyGridLocalizerNode::GridSearchLocalizationCallback(
  const std::shared_ptr<std_srvs::srv::Empty::Request>,
  std::shared_ptr<std_srvs::srv::Empty::Response>)
{
  isaac_ros_pointcloud_interfaces::msg::FlatScan buffered_scan;
  {
    std::lock_guard<std::mutex> lock(buffered_flat_scan_mutex_);
    if (!has_buffered_flat_scan_) {
      trigger_localization_on_next_flatscan_ = true;
      RCLCPP_INFO(get_logger(), "Will trigger localization when the next FlatScan is received.");
      return;
    }
    buffered_scan = buffered_flat_scan_;
  }

  const auto transform = LookupBaseLinkToLidarTransform(buffered_scan.header.frame_id);
  const auto localization_result = localizer_.Localize(
    buffered_scan, transform, loc_result_frame_);
  if (!localization_result) {
    RCLCPP_WARN(get_logger(), "Localization failed for buffered FlatScan.");
    return;
  }

  PublishLocalizationResult(*localization_result);
}

void OccupancyGridLocalizerNode::BufferedFlatScanCallback(
  const isaac_ros_pointcloud_interfaces::msg::FlatScan::ConstSharedPtr flat_scan)
{
  HandleIncomingFlatScan(*flat_scan, trigger_localization_on_next_flatscan_.exchange(false));
}

void OccupancyGridLocalizerNode::TriggerFlatScanCallback(
  const isaac_ros_pointcloud_interfaces::msg::FlatScan::ConstSharedPtr flat_scan)
{
  HandleIncomingFlatScan(*flat_scan, true);
}

void OccupancyGridLocalizerNode::HandleIncomingFlatScan(
  const isaac_ros_pointcloud_interfaces::msg::FlatScan & flat_scan,
  bool should_localize)
{
  {
    std::lock_guard<std::mutex> lock(buffered_flat_scan_mutex_);
    buffered_flat_scan_ = flat_scan;
    has_buffered_flat_scan_ = true;
  }

  if (!should_localize) {
    return;
  }

  const auto transform = LookupBaseLinkToLidarTransform(flat_scan.header.frame_id);
  const auto localization_result = localizer_.Localize(
    flat_scan, transform, loc_result_frame_);
  if (!localization_result) {
    RCLCPP_WARN(get_logger(), "Localization failed for FlatScan in frame %s.",
      flat_scan.header.frame_id.c_str());
    return;
  }

  PublishLocalizationResult(*localization_result);
}

std::optional<geometry_msgs::msg::TransformStamped>
OccupancyGridLocalizerNode::LookupBaseLinkToLidarTransform(const std::string & frame_id) const
{
  if (frame_id.empty()) {
    return std::nullopt;
  }

  try {
    return tf_buffer_->lookupTransform(kBaseLinkFrameName, frame_id, tf2::TimePointZero);
  } catch (const tf2::TransformException & ex) {
    RCLCPP_INFO(
      get_logger(), "Could not transform %s to %s: %s. Using identity transform.",
      kBaseLinkFrameName, frame_id.c_str(), ex.what());
    return std::nullopt;
  }
}

void OccupancyGridLocalizerNode::PublishLocalizationResult(
  const geometry_msgs::msg::PoseWithCovarianceStamped & pose)
{
  localization_result_publisher_->publish(pose);
}

}  // namespace occupancy_grid_localizer
}  // namespace isaac_ros
}  // namespace nvidia

RCLCPP_COMPONENTS_REGISTER_NODE(
  nvidia::isaac_ros::occupancy_grid_localizer::OccupancyGridLocalizerNode)
