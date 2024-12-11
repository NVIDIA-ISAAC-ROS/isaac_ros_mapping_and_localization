// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef ISAAC_ROS_CAMERA_LOCALIZATION_APRILTAG_LOCALIZATION_NODE_HPP_
#define ISAAC_ROS_CAMERA_LOCALIZATION_APRILTAG_LOCALIZATION_NODE_HPP_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "isaac_ros_apriltag_interfaces/msg/april_tag_detection_array.hpp"
#include "isaac_ros_visual_global_localization/transform_manager.hpp"
#include "protos/apriltag/apriltag_observation.pb.h"

namespace nvidia
{
namespace isaac_ros
{
namespace visual_global_localization
{

using AprilTagId = std::string;
using AprilTagMsg = isaac_ros_apriltag_interfaces::msg::AprilTagDetectionArray;
struct AprilTagObservation
{
  AprilTagId apriltag_id;
  // all transform needs to be read from right to left. e.g.
  // p_left = left_T_right * p_right;
  // e.g. this is transform that moves a point from camera's coordinate to tag's coordinate
  Transform tag_T_camera;
  Transform map_T_camera;
  Transform map_T_tag;
};
struct AprilTagMap
{
  std::map<AprilTagId, std::vector<AprilTagObservation>> id_to_observations;
};


class AprilTagLocalizationNode : public rclcpp::Node
{
public:
  explicit AprilTagLocalizationNode(
    const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  virtual ~AprilTagLocalizationNode();

  void getParameters();
  void subscribeToTopics();
  void advertiseTopics();
  void AprilTagDetectionCallback(const AprilTagMsg::SharedPtr msg);
  void SlamOdometryCallback(const nav_msgs::msg::Odometry::SharedPtr msg);

protected:
  // Only one of the following two can be turned on
  bool enable_recording_map_ = true;
  bool enable_localizer_ = false;
  // If enable_recording_map_ is true, the observations will be saved to map_dir_
  // If enable_recording_map_ is false, the observations will be loaded from map_dir_
  std::string map_dir_;
  std::string input_apriltag_detection_topic_ = "/tag_detections";
  std::string input_slam_odometry_topic_ = "/visual_slam/vis/slam_odometry";

  // transform of the odom frame to the world frame
  Transform map_T_odom_ = Transform::Identity();

  // transforms receive from cuvslam, changes on every callback
  Transform baselink_T_camera_ = Transform::Identity();
  Transform odom_T_baselink_ = Transform::Identity();

  TransformManager transform_manager_;
  std::string base_frame_ = "base_link";
  std::string map_frame_ = "map";
  std::string odom_frame_ = "odom";
  bool debug_mode_ = false;

  rclcpp::Subscription<AprilTagMsg>::SharedPtr apriltag_odometry_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr slam_odometry_sub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr
    apriltag_localization_pub_;

  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr apriltag_marker_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr detection_tag_pub_;

  // Publishers

  // publish the transform of odom_frame to world_frame
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  // collect the observations of apriltags
  AprilTagMap apriltag_map_;

  // Use to visualize the map tags for debugging
  visualization_msgs::msg::MarkerArray map_marker_array_;

  bool HandleRecording(const AprilTagMsg::SharedPtr msg);
  bool HandleLocalization(const AprilTagMsg::SharedPtr msg);

  const AprilTagObservation FindBestObservation(
    const AprilTagObservation & current_observation,
    const std::vector<AprilTagObservation> & observations) const;

  void publishMessage(const rclcpp::Time & timestamp);

  // load apriltag observation map to map directory
  bool saveObservationsToDisk();
  // load apriltag observation map from map directory
  bool loadObservationsFromDisk();
  // convert the tag detection message to observation
  AprilTagObservation convertTagDetectionToObservation(
    const isaac_ros_apriltag_interfaces::msg::AprilTagDetection & tag_detection) const;
  // set the visualization marker for the tag
  void setMarker(const Transform & transform, visualization_msgs::msg::Marker & marker) const;

};
} // namespace visual_global_localization
} // namespace isaac_ros
} // namespace nvidia

#endif  // ISAAC_ROS_CAMERA_LOCALIZATION_APRILTAG_LOCALIZATION_NODE_HPP_
