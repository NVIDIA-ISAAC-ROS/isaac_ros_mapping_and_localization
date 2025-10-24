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

#include "isaac_ros_visual_global_localization/apriltag_localization_node.hpp"
#include "isaac_ros_visual_global_localization/constants.h"
#include "common/file_utils/file_utils.h"

namespace nvidia
{
namespace isaac_ros
{
namespace visual_global_localization
{
AprilTagLocalizationNode::AprilTagLocalizationNode(const rclcpp::NodeOptions & options)
: Node("apriltag_localization", options), transform_manager_(this)
{
  RCLCPP_INFO(get_logger(), "Initializing AprilTag Localization Node");
  // Get parameters
  getParameters();
  // Subscribe to topics
  advertiseTopics();
  // Advertise topics
  subscribeToTopics();
}

AprilTagLocalizationNode::~AprilTagLocalizationNode()
{
  if (enable_recording_map_) {
    RCLCPP_INFO(get_logger(), "Start to save observations to disk.");
    // Save observations to disk
    if (!saveObservationsToDisk()) {
      RCLCPP_ERROR(get_logger(), "Failed to save observations to disk!");
    } else {
      RCLCPP_INFO(
        get_logger(),
        "Successfully saved observations to disk, shutting down AprilTag Localization Node.");
    }
  }
}

void AprilTagLocalizationNode::getParameters()
{
  RCLCPP_INFO_STREAM(get_logger(), "AprilTagLocalizationNode Getting parameters");
  map_dir_ = declare_parameter<std::string>("map_dir", map_dir_);
  debug_mode_ = declare_parameter<bool>("debug_mode", debug_mode_);
  enable_recording_map_ = declare_parameter<bool>("enable_recording_map", enable_recording_map_);
  enable_localizer_ = declare_parameter<bool>("enable_localizer", enable_localizer_);
  input_apriltag_detection_topic_ =
    declare_parameter<std::string>(
    "input_apriltag_detection_topic",
    input_apriltag_detection_topic_);
  input_slam_odometry_topic_ = declare_parameter<std::string>(
    "input_slam_odometry_topic",
    input_slam_odometry_topic_);

  // Check the parameters
  if (map_dir_.empty()) {
    RCLCPP_ERROR(get_logger(), "map_dir is empty!");
  }
  if (enable_localizer_ && enable_recording_map_) {
    RCLCPP_ERROR(
      get_logger(),
      "enable_localizer and enable_recording_map cannot be true at the same time!");
  }
  if (!enable_localizer_ && !enable_recording_map_) {
    RCLCPP_ERROR(
      get_logger(),
      "enable_localizer and enable_recording_map cannot be false at the same time!");
  }
  if (enable_localizer_) {
    const std::string observations_file =
      isaac::common::file_utils::FileUtils::JoinPath(map_dir_, kObservationsFileName);
    if (!isaac::common::file_utils::FileUtils::FileExists(observations_file)) {
      RCLCPP_ERROR(
        get_logger(),
        "enable_localizer is true but observations_file does not exist!");
    }
    // load apriltag observation map from disk
    if (!loadObservationsFromDisk()) {
      RCLCPP_ERROR(
        get_logger(),
        "enable_localizer is true but loading observation file failed!");
    }
  }
  transform_manager_.set_reference_frame(base_frame_);
}

void AprilTagLocalizationNode::subscribeToTopics()
{
  RCLCPP_INFO_STREAM(get_logger(), "AprilTagLocalizationNode Subscribing to topics");
  apriltag_odometry_sub_ =
    create_subscription<AprilTagMsg>(
    input_apriltag_detection_topic_, 10,
    std::bind(&AprilTagLocalizationNode::AprilTagDetectionCallback, this, std::placeholders::_1));

  slam_odometry_sub_ = create_subscription<nav_msgs::msg::Odometry>(
    input_slam_odometry_topic_, 10,
    std::bind(
      &TransformManager::odomCallback, &transform_manager_, std::placeholders::_1));

}

void AprilTagLocalizationNode::advertiseTopics()
{
  RCLCPP_INFO_STREAM(get_logger(), "AprilTagLocalizationNode Advertising topics");
  apriltag_localization_pub_ = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
    kAprilTagLocalizationTopic, 10);
  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
  apriltag_marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
    "apriltag_map", 10);
  detection_tag_pub_ = create_publisher<visualization_msgs::msg::Marker>(
    "detection_tag", 10);
}

void AprilTagLocalizationNode::AprilTagDetectionCallback(const AprilTagMsg::SharedPtr msg)
{
  RCLCPP_DEBUG_STREAM(get_logger(), "AprilTagLocalizationNode AprilTagDetectionCallback");
  if (msg->detections.size() == 0) {
    RCLCPP_DEBUG_STREAM(get_logger(), "No apriltag Found.");
    return;
  }
  if (debug_mode_) {
    // publish the map tags when detecting tags
    apriltag_marker_pub_->publish(map_marker_array_);
  }
  const std::string frame_id = msg->header.frame_id;
  const rclcpp::Time timestamp = rclcpp::Time(
    msg->header.stamp.sec, msg->header.stamp.nanosec, RCL_ROS_TIME);
  // RCLCPP_INFO_STREAM(get_logger(), "AprilDet timestamp: " << timestamp);
  // Transform odom_T_baselink_;    // from cuVSLAM
  if (!transform_manager_.lookupTransformOfReferenceToOdom(timestamp, &odom_T_baselink_)) {
    RCLCPP_DEBUG(get_logger(), "Failed to lookup transform from baselink to map!");
    return;
  }
  // Transform baselink_T_camera_;
  if (!transform_manager_.lookupTransformToReferenceFrame(frame_id, timestamp, &baselink_T_camera_)) {
    RCLCPP_ERROR(get_logger(), "Failed to get transform for frame_id: %s", frame_id.c_str());
    return;
  }

  // recording mode
  if (enable_recording_map_ && !HandleRecording(msg)) {
    RCLCPP_WARN(get_logger(), "Failed to record apriltag observations.");
    return;
  }

  // localization mode
  if (enable_localizer_) {
    if (HandleLocalization(msg)) {
      // publish the message
      publishMessage(timestamp);
    } else {
      RCLCPP_WARN(get_logger(), "Apriltag Localization Failed.");
    }
  }

}

bool AprilTagLocalizationNode::HandleRecording(const AprilTagMsg::SharedPtr msg)
{
  for (const auto & tag_detection : msg->detections) {
    AprilTagObservation observation = convertTagDetectionToObservation(tag_detection);
    apriltag_map_.id_to_observations[observation.apriltag_id].push_back(observation);
  }
  return true;
}

bool AprilTagLocalizationNode::HandleLocalization(const AprilTagMsg::SharedPtr msg)
{
  // when several apriltags are detected in single frame, inconsistent results may occur.
  // To workaround this issue, we only use first one to localize camera.
  for (const auto & tag_detection : msg->detections) {
    AprilTagObservation observation = convertTagDetectionToObservation(tag_detection);
    if (!apriltag_map_.id_to_observations.count(observation.apriltag_id)) {
      RCLCPP_WARN_STREAM(
        get_logger(), "Can't find this apriltag in map " << observation.apriltag_id);
      continue;
    }

    // localize camera pose
    const AprilTagObservation best_observation = FindBestObservation(
      observation,
      apriltag_map_.id_to_observations[
        observation.apriltag_id]);

    Transform map_T_camera;
    map_T_camera = best_observation.map_T_tag * observation.tag_T_camera;
    map_T_odom_ = map_T_camera * (odom_T_baselink_ * baselink_T_camera_).inverse();

    if (debug_mode_) {
      //visualize the detection
      visualization_msgs::msg::Marker detection_tag_marker;
      setMarker(best_observation.map_T_tag, detection_tag_marker);
      detection_tag_marker.ns = "detection";
      //set green color for best detection tag
      detection_tag_marker.color.g = 1.0f;
      detection_tag_pub_->publish(detection_tag_marker);

      visualization_msgs::msg::Marker visual_global_localization_marker;
      setMarker(map_T_camera, visual_global_localization_marker);
      visual_global_localization_marker.ns = "camera";
      //set blue color for camera localization
      visual_global_localization_marker.color.b = 1.0f;
      detection_tag_pub_->publish(visual_global_localization_marker);
    }
    return true;
  }

  // When no apriltag match found in observation map, return localization fail
  // This will happen when map is out of date.
  RCLCPP_WARN_STREAM(get_logger(), "Can't find match tag in map.");
  return false;
}

const AprilTagObservation AprilTagLocalizationNode::FindBestObservation(
  const AprilTagObservation & current_observation,
  const std::vector<AprilTagObservation> & observations) const
{
  // Find the best observations based on cloest view point
  double min_view_dist = 0;
  const AprilTagObservation * best_observation = nullptr;
  for (const auto & this_observation : observations) {
    double rot_radians, trans_meters;
    // curr_transform.Difference(this_transform, &rot_radians, &trans_meters);

    Transform diff_transform = current_observation.tag_T_camera.inverse() *
      this_observation.tag_T_camera;
    trans_meters = diff_transform.translation().norm();
    Eigen::Quaternionf quaternion(diff_transform.linear());
    rot_radians = 2 * acos(quaternion.w());

    double current_view_dist = rot_radians * kRotationWeight + trans_meters;

    // update best observation and view distance
    if (current_view_dist < min_view_dist || best_observation == nullptr) {
      // best_observation = const_cast<AprilTagObservation *>(&this_observation);
      best_observation = &this_observation;
      min_view_dist = current_view_dist;
    }
  }
  return *best_observation;

}

void AprilTagLocalizationNode::publishMessage(const rclcpp::Time & timestamp)
{
  geometry_msgs::msg::TransformStamped transform_msg;
  transform_msg.header.frame_id = odom_frame_;
  transform_msg.header.stamp = timestamp;
  transform_msg.child_frame_id = map_frame_;
  convertEigenToRosTransform(map_T_odom_, transform_msg.transform);
  tf_broadcaster_->sendTransform(transform_msg);

  geometry_msgs::msg::PoseWithCovarianceStamped pose_msg;
  pose_msg.header.stamp = timestamp;
  pose_msg.header.frame_id = odom_frame_;
  convertEigenToRosPose(map_T_odom_, pose_msg.pose.pose);
  apriltag_localization_pub_->publish(pose_msg);

}

bool AprilTagLocalizationNode::saveObservationsToDisk()
{
  RCLCPP_INFO_STREAM(get_logger(), "AprilTagLocalizationNode Saving observations to disk");
  if (apriltag_map_.id_to_observations.empty()) {
    RCLCPP_ERROR(get_logger(), "No observations to save!");
    return false;
  }
  protos::apriltag::AprilTagObservationMap observation_map;
  for (const auto & id_to_observations : apriltag_map_.id_to_observations) {
    protos::apriltag::AprilTagObservationList observation_list;
    for (const auto & observation : id_to_observations.second) {
      protos::apriltag::AprilTagObservation * observation_proto =
        observation_list.add_observations();
      observation_proto->set_tag_id(observation.apriltag_id);
      convertTransformToProto(observation.tag_T_camera, observation_proto->mutable_tag_t_camera());
      convertTransformToProto(observation.map_T_camera, observation_proto->mutable_map_t_camera());
      convertTransformToProto(observation.map_T_tag, observation_proto->mutable_map_t_tag());
      // *(observation_list.add_observations()) = observation_proto;
    }
    (*observation_map.mutable_tag_observations())[id_to_observations.first] = observation_list;
  }
  const std::string observations_file =
    isaac::common::file_utils::FileUtils::JoinPath(map_dir_, kObservationsFileName);
  if (isaac::common::file_utils::FileUtils::WriteProtoFileByExtension(
      observations_file, observation_map) != absl::OkStatus())
  {
    RCLCPP_ERROR(get_logger(), "Failed to save observations to disk!");
    return false;
  }
  return true;
}

bool AprilTagLocalizationNode::loadObservationsFromDisk()
{
  RCLCPP_INFO_STREAM(get_logger(), "AprilTagLocalizationNode Loading observations from disk");
  const std::string tag_map_path = isaac::common::file_utils::FileUtils::JoinPath(
    map_dir_, kObservationsFileName);

  protos::apriltag::AprilTagObservationMap observation_map;
  if (isaac::common::file_utils::FileUtils::ReadProtoFileByExtension(
      tag_map_path, &observation_map) != absl::OkStatus())
  {
    RCLCPP_ERROR(get_logger(), "Failed to load observations from disk !");
    return false;
  }

  const size_t num_tag = observation_map.tag_observations_size();
  RCLCPP_INFO_STREAM(get_logger(), "AprilTag Map has " << num_tag << " tags.");
  if (num_tag == 0) {
    RCLCPP_ERROR(get_logger(), "No observations in map !");
    return false;
  }

  for (const auto & id_to_observations : observation_map.tag_observations()) {
    const std::string tag_id = id_to_observations.first;
    const protos::apriltag::AprilTagObservationList & observation_list = id_to_observations.second;
    const size_t num_observations = observation_list.observations_size();
    if (apriltag_map_.id_to_observations.find(tag_id) != apriltag_map_.id_to_observations.end()) {
      RCLCPP_ERROR(get_logger(), "Duplicate tag id in map!");
      return false;
    }
    apriltag_map_.id_to_observations[tag_id].reserve(num_observations);
    for (size_t i = 0; i < num_observations; i++) {
      AprilTagObservation tag_observation;
      tag_observation.apriltag_id = tag_id;
      convertProtoToTransform(
        observation_list.observations(
          i).tag_t_camera(), &(tag_observation.tag_T_camera));
      convertProtoToTransform(
        observation_list.observations(
          i).map_t_camera(), &(tag_observation.map_T_camera));
      convertProtoToTransform(
        observation_list.observations(
          i).map_t_tag(), &(tag_observation.map_T_tag));
      apriltag_map_.id_to_observations[tag_id].emplace_back(tag_observation);
      if (debug_mode_) {
        visualization_msgs::msg::Marker marker;
        setMarker(tag_observation.map_T_tag, marker);
        marker.ns = "map";
        marker.id = i;
        // Set red color for map tags
        marker.color.r = 1.0f;
        map_marker_array_.markers.push_back(marker);
      }
    }
  }
  RCLCPP_INFO_STREAM(
    get_logger(),
    "Loaded " << apriltag_map_.id_to_observations.size() << " apriltags from AprilTag Map.");
  return true;
}

AprilTagObservation AprilTagLocalizationNode::convertTagDetectionToObservation(
  const isaac_ros_apriltag_interfaces::msg::AprilTagDetection & tag_detection) const
{
  const geometry_msgs::msg::Pose & tag_pos = tag_detection.pose.pose.pose;
  Transform camera_T_tag(Eigen::Translation3f(
      tag_pos.position.x, tag_pos.position.y,
      tag_pos.position.z) *
    Eigen::Quaternionf(
      tag_pos.orientation.w, tag_pos.orientation.x, tag_pos.orientation.y,
      tag_pos.orientation.z));

  AprilTagObservation observation;
  observation.apriltag_id = tag_detection.family + "_" + std::to_string(tag_detection.id);
  observation.tag_T_camera = camera_T_tag.inverse();
  observation.map_T_camera = odom_T_baselink_ * baselink_T_camera_;
  observation.map_T_tag = observation.map_T_camera * observation.tag_T_camera.inverse();

  return observation;
}

void AprilTagLocalizationNode::setMarker(
  const Transform & transform,
  visualization_msgs::msg::Marker & marker) const
{
  marker.header.frame_id = "vmap";
  marker.header.stamp = this->now();
  marker.ns = "";
  marker.id = 0;
  marker.type = visualization_msgs::msg::Marker::CUBE;
  marker.action = visualization_msgs::msg::Marker::ADD;
  convertEigenToRosPose(transform, marker.pose);
  marker.scale.x = 0.1;
  marker.scale.y = 0.1;
  marker.scale.z = 0.1;
  marker.color.r = 0.0f;
  marker.color.g = 0.0f;
  marker.color.b = 0.0f;
  marker.color.a = 1.0;
  marker.lifetime = rclcpp::Duration::max();
}

} // namespace visual_global_localization
} // namespace isaac_ros
} // namespace nvidia

// Register as a component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(
  nvidia::isaac_ros::visual_global_localization::AprilTagLocalizationNode)
