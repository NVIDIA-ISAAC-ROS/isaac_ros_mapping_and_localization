// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// (TODO)Copy this class from isaac_ros_nvblox

#ifndef ISAAC_ROS_CAMERA_LOCALIZATION_TRANSFORM_MANAGER_HPP_
#define ISAAC_ROS_CAMERA_LOCALIZATION_TRANSFORM_MANAGER_HPP_

#include <map>
#include <memory>
#include <string>
#include <unordered_map>

#include "Eigen/Core"
#include "Eigen/Geometry"

#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"

#include "constants.h"
#include "protos/common/geometry/transform.pb.h"
#include "common/transform/se3_transform.h"

namespace nvidia
{
namespace isaac_ros
{
namespace visual_global_localization
{

typedef Eigen::Isometry3f Transform;
/// Class that binds to either the TF tree or resolves transformations from the
/// ROS parameter server, depending on settings loaded from ROS params.
/// All transform needs to be read from right to left.
class TransformManager
{
public:
  explicit TransformManager(rclcpp::Node * node);

  /// @brief Looks up the transform between the frame with the passed name and the reference frame
  ///        (which is set by the setters below). We either use tf2 or a stored queue of
  ///         transforms from messages.
  /// @param sensor_frame The frame name.
  /// @param timestamp Time of the transform. Passing rclcpp::Time(0) will return the latest
  ///                  transform in the queue.
  /// @param transform The output transform.
  /// @return true if the lookup was successful.
  bool lookupTransformToReferenceFrame(
    const std::string & sensor_frame,
    const rclcpp::Time & timestamp,
    Transform * transform);

  /// @brief Looks up the transform between the reference frame and the odom frame
  /// @param timestamp Time of the transform. Passing rclcpp::Time(0) will return the latest
  ///                  transform in the queue.
  /// @param transform The output transform.
  /// @return true if the lookup was successful.
  bool lookupTransformOfReferenceToOdom(const rclcpp::Time & timestamp, Transform * transform);

  bool waitForTransform(
    const std::string & target_frame,
    const std::string & source_frame,
    const rclcpp::Time & timestamp,
    double wait_seconds, Transform * out_transform);

  bool lookupTransformTf(
    const std::string & target_frame,
    const std::string & source_frame,
    const rclcpp::Time & timestamp, Transform * transform);

  Transform predictTransformByOdom(
    const std::string & target_frame,
    const std::string & source_frame,
    const rclcpp::Time & previous_timestamp, const rclcpp::Time & current_timestamp);

  /// Assumes these transforms are from POSE frame to REFERENCE frame. Ignores
  /// frame_id. e.g. if the POSE frame is "odom" and REFERENCE frame is "baselink", the transform
  /// is baselink_T_odom.
  void transformCallback(
    const geometry_msgs::msg::TransformStamped::ConstSharedPtr transform_msg);

  void poseCallback(
    const geometry_msgs::msg::PoseStamped::ConstSharedPtr transform_msg);

  void odomCallback(
    const nav_msgs::msg::Odometry::ConstSharedPtr odom_msg);

  /// Set the names of the frames.
  void set_reference_frame(const std::string & reference_frame)
  {
    reference_frame_ = reference_frame;
  }

  void set_pose_frame(const std::string & pose_frame)
  {
    pose_frame_ = pose_frame;
  }

private:
  bool lookupTransformQueue(
    const rclcpp::Time & timestamp,
    Transform * transform);

  bool lookupSensorTransform(
    const std::string & sensor_frame,
    Transform * transform);

  Transform transformToEigen(const geometry_msgs::msg::Transform & transform) const;
  Transform poseToEigen(const geometry_msgs::msg::Pose & pose) const;

  /// ROS State
  rclcpp::Node * node_;
  std::shared_ptr<tf2_ros::TransformListener> transform_listener_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;

  /// Reference coordinate frame. Will always look up TF transforms to this
  /// frame.
  std::string reference_frame_;
  /// Use this as the "pose" frame that's coming in. Needs to be set.
  std::string pose_frame_;

  /// Whether to use TF transforms at all.
  /// If set to false, use_topic_transforms_ must be true
  /// and pose_frame *needs* to be set.
  bool use_tf_transforms_ = true;
  /// Whether to listen to topics for transforms.
  /// If set to true, will try to get `reference_frame` to `pose_frame`
  /// transform from the topics. If set to false,
  /// everything will be resolved through TF.
  bool use_topic_transforms_ = false;
  /// Timestamp tolerance to use for transform *topics* only.
  uint64_t timestamp_tolerance_ns_ = 1e8;  // 100 milliseconds

  // This wait time is used to get the transform
  const double wait_seconds_ = 0.01;

  /// Queues and state
  /// Maps timestamp in ns to transform reference -> odom frame.
  std::map<uint64_t, Transform> transform_queue_;
  /// Maps sensor frame to transform pose frame -> sensor frame.
  std::unordered_map<std::string, Transform> sensor_transforms_;
};

/// Convert Eigen transform to SE3 transform
void convertEigenToTransform(
  const Transform & eigen_transform, isaac::common::transform::SE3TransformD & transform);

/// Convert SE3 transform to Eigen transform
void convertTransformToEigen(
  const isaac::common::transform::SE3TransformD & transform,
  Transform & eigen_transform);

/// Convert Eigen transform to ROS transform
void convertEigenToRosTransform(
  const Transform & eigen_transform,
  geometry_msgs::msg::Transform & ros_transform);

/// Convert SE3 transform to ROS transform
void convertSE3ToRosTransform(
  const isaac::common::transform::SE3TransformD & se3_transform,
  geometry_msgs::msg::Transform & ros_transform);

/// Convert Eigen transform to ROS pose
void convertEigenToRosPose(
  const Transform & eigen_transform,
  geometry_msgs::msg::Pose & ros_pose);

/// Convert SE3 transform to ROS pose
void convertSE3ToRosPose(
  const isaac::common::transform::SE3TransformD & se3_transform,
  geometry_msgs::msg::Pose & ros_pose);

/// Convert ROS transform to Eigen transform
void convertTransformToEigen(
  const geometry_msgs::msg::Transform & msg,
  Transform * transform);

/// Convert Eigen transform to transform protobuf
void convertTransformToProto(
  const Transform & transform, protos::common::geometry::Transform3D * transform_proto);

/// Convert transform protobuf to Eigen transform
void convertProtoToTransform(
  const protos::common::geometry::Transform3D & transform_proto,
  Transform * transform);

}  // namespace visual_global_localization
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_CAMERA_LOCALIZATION_TRANSFORM_MANAGER_HPP_
