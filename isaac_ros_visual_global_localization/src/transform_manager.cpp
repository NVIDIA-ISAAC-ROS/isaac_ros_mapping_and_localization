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

#include "isaac_ros_visual_global_localization/transform_manager.hpp"

#include <algorithm>
#include <memory>
#include <string>

#include "tf2_ros/create_timer_ros.h"


namespace nvidia
{
namespace isaac_ros
{
namespace visual_global_localization
{

TransformManager::TransformManager(rclcpp::Node * node)
: node_(node)
{
  // Get params like "use_tf_transforms".
  use_tf_transforms_ =
    node->declare_parameter<bool>("use_tf_transforms", use_tf_transforms_);
  use_topic_transforms_ = node->declare_parameter<bool>(
    "use_topic_transforms",
    use_topic_transforms_);

  // Init the transform listeners if we ARE using TF at all.
  if (use_tf_transforms_) {
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(node->get_clock());
    transform_listener_ =
      std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    auto timer_interface = std::make_shared<tf2_ros::CreateTimerROS>(
      node->get_node_base_interface(),
      node->get_node_timers_interface());

    // Set the timer interface on the tf2_ros::Buffer object
    tf_buffer_->setCreateTimerInterface(timer_interface);
  }
}

bool TransformManager::lookupTransformOfReferenceToOdom(
  const rclcpp::Time & timestamp,
  Transform * transform)
{
  return lookupTransformQueue(timestamp, transform);
}

bool TransformManager::lookupTransformToReferenceFrame(
  const std::string & sensor_frame,
  const rclcpp::Time & timestamp,
  Transform * transform)
{
  if (!use_tf_transforms_ && !use_topic_transforms_) {
    // ERROR HERE, literally can't do anything.
    RCLCPP_ERROR(
      node_->get_logger(),
      "Not using TF OR topic transforms, what do you want us to use?");
    return false;
  }

  // Check if we have a transform queue.
  if (!use_topic_transforms_) {
    // Then I guess we're using TF.
    // Try to look up the pose in TF.
    return lookupTransformTf(reference_frame_, sensor_frame, timestamp, transform);
  } else {
    // We're using topic transforms.
    if (sensor_frame != pose_frame_) {
      // Ok but we also need a pose_frame -> sensor_frame lookup here.
      Transform P_T_S;
      if (!lookupSensorTransform(sensor_frame, &P_T_S)) {
        return false;
      }

      // Now we have the TPS reports, we need T_G_P which is reference to pose
      // frame. This comes from the queue.
      Transform R_T_P;
      if (!lookupTransformQueue(timestamp, &R_T_P)) {
        return false;
      }

      *transform = R_T_P * P_T_S;
      return true;
    } else {
      return lookupTransformQueue(timestamp, transform);
    }
  }
  return false;
}

void TransformManager::transformCallback(
  const geometry_msgs::msg::TransformStamped::ConstSharedPtr transform_msg)
{
  rclcpp::Time timestamp = transform_msg->header.stamp;
  transform_queue_[timestamp.nanoseconds()] =
    transformToEigen(transform_msg->transform);
}

void TransformManager::poseCallback(
  const geometry_msgs::msg::PoseStamped::ConstSharedPtr transform_msg)
{
  rclcpp::Time timestamp = transform_msg->header.stamp;
  transform_queue_[timestamp.nanoseconds()] = poseToEigen(transform_msg->pose);
}

void TransformManager::odomCallback(
  const nav_msgs::msg::Odometry::ConstSharedPtr odom_msg)
{
  rclcpp::Time timestamp = odom_msg->header.stamp;
  transform_queue_[timestamp.nanoseconds()] = poseToEigen(odom_msg->pose.pose);
}

bool TransformManager::waitForTransform(
  const std::string & target_frame,
  const std::string & source_frame,
  const rclcpp::Time & timestamp,
  double wait_seconds, Transform * out_transform)
{
  // Wait for the transform to become available
  try {
    auto future = tf_buffer_->waitForTransform(
      target_frame, source_frame, timestamp,
      rclcpp::Duration::from_seconds(wait_seconds),
      [&out_transform, this](const tf2_ros::TransformStampedFuture & future) {
        try {
          if (future.valid()) {
            *out_transform = transformToEigen(future.get().transform);
          }
          return;
        } catch (const tf2::TransformException & ex) {
          RCLCPP_WARN(node_->get_logger(), "Could not get transform: %s", ex.what());
          return;
        }
      });
    // Optionally, you can wait for the future to complete
    future.wait();
    return future.valid();
  } catch (const tf2::TransformException & ex) {
    RCLCPP_WARN(node_->get_logger(), "Exception: %s", ex.what());
    return false;
  }
}

bool TransformManager::lookupTransformTf(
  const std::string & target_frame,
  const std::string & source_frame,
  const rclcpp::Time & timestamp,
  Transform * transform)
{
  geometry_msgs::msg::TransformStamped target_T_source_msg;
  try {
    target_T_source_msg = tf_buffer_->lookupTransform(target_frame, source_frame, timestamp);
  } catch (tf2::TransformException & e) {
    RCLCPP_WARN_STREAM(
      node_->get_logger(),
      "Failed to look up transform: from:" << source_frame << " to " << target_frame << ". Error: " <<
        e.what());
    return false;
  }

  *transform = transformToEigen(target_T_source_msg.transform);
  return true;
}

bool TransformManager::lookupTransformQueue(
  const rclcpp::Time & timestamp,
  Transform * transform)
{
  // Get latest transform
  if (timestamp == rclcpp::Time(0, 0, RCL_ROS_TIME)) {
    if (transform_queue_.empty()) {
      return false;
    }

    *transform = transform_queue_.rbegin()->second;
    return true;
  } else {
    // Get closest transform
    uint64_t timestamp_ns = timestamp.nanoseconds();

    auto closest_match = transform_queue_.lower_bound(timestamp_ns);
    if (closest_match == transform_queue_.end()) {
      return false;
    }

    // If we're too far off on the timestamp:
    uint64_t distance = std::max(closest_match->first, timestamp_ns) -
      std::min(closest_match->first, timestamp_ns);
    if (distance > timestamp_tolerance_ns_) {
      return false;
    }

    // We just do nearest neighbor here.
    // TODO(gangh): add interpolation!
    *transform = closest_match->second;
    return true;
  }
}

bool TransformManager::lookupSensorTransform(
  const std::string & sensor_frame,
  Transform * transform)
{
  auto it = sensor_transforms_.find(sensor_frame);
  if (it == sensor_transforms_.end()) {
    // Couldn't find sensor transform. Gotta look it up.
    if (!use_tf_transforms_) {
      // Well we're kind out out of options here.
      return false;
    }
    bool success = lookupTransformTf(
      pose_frame_, sensor_frame,
      node_->get_clock()->now(), transform);
    if (success) {
      sensor_transforms_[sensor_frame] = *transform;
    } else {
      RCLCPP_INFO(
        node_->get_logger(),
        "Could not look up transform from %s to %s.", sensor_frame.c_str(), pose_frame_.c_str());
    }
    return success;
  } else {
    *transform = it->second;
    return true;
  }
}

//target frame: map, source frame: vehicle
Transform TransformManager::predictTransformByOdom(
  const std::string & target_frame,
  const std::string & source_frame,
  const rclcpp::Time & previous_timestamp, const rclcpp::Time & current_timestamp)
{

  Transform target_frame_T_previous_source_frame;
  if (!lookupTransformTf(
      target_frame, source_frame, previous_timestamp,
      &target_frame_T_previous_source_frame))
  {
    RCLCPP_WARN(
      node_->get_logger(),
      "Could not find transform from %s to %s at time %f",
      target_frame.c_str(), source_frame.c_str(), previous_timestamp.seconds());
    return Transform::Identity();
  }
  //Get the transform from the previous time to the current time
  Transform odom_T_current_source_frame;
  if (!lookupTransformTf(
      KOdomFrame, source_frame, current_timestamp,
      &odom_T_current_source_frame))
  {
    RCLCPP_WARN(
      node_->get_logger(),
      "Could not find transform from %s to %s at time %f",
      KOdomFrame.c_str(), source_frame.c_str(), current_timestamp.seconds());
    return Transform::Identity();
  }
  Transform odom_T_previous_source_frame;
  if (!lookupTransformTf(
      KOdomFrame, source_frame, previous_timestamp,
      &odom_T_previous_source_frame))
  {
    RCLCPP_WARN(
      node_->get_logger(),
      "Could not find transform from %s to %s at time %f",
      KOdomFrame.c_str(), source_frame.c_str(), previous_timestamp.seconds());
    return Transform::Identity();
  }
  const Transform previous_source_frame_T_current_source_frame =
    odom_T_previous_source_frame.inverse() *
    odom_T_current_source_frame;
  const Transform target_frame_T_current_source_frame = target_frame_T_previous_source_frame *
    previous_source_frame_T_current_source_frame;
  return target_frame_T_current_source_frame;

}

Transform TransformManager::transformToEigen(
  const geometry_msgs::msg::Transform & msg) const
{
  return Transform(
    Eigen::Translation3f(
      msg.translation.x, msg.translation.y,
      msg.translation.z) *
    Eigen::Quaternionf(
      msg.rotation.w, msg.rotation.x,
      msg.rotation.y, msg.rotation.z));
}

Transform TransformManager::poseToEigen(const geometry_msgs::msg::Pose & msg) const
{
  return Transform(
    Eigen::Translation3d(msg.position.x, msg.position.y, msg.position.z) *
    Eigen::Quaterniond(
      msg.orientation.w, msg.orientation.x,
      msg.orientation.y, msg.orientation.z));
}

void convertEigenToTransform(
  const Transform & eigen_transform,
  isaac::common::transform::SE3TransformD & se3_transform)
{
  Eigen::Quaternionf quaternion(eigen_transform.linear());
  se3_transform.set_translation(eigen_transform.translation().cast<double>());
  se3_transform.set_rotation(quaternion.cast<double>());
}

void convertTransformToEigen(
  const isaac::common::transform::SE3TransformD & se3_transform,
  Transform & eigen_transform)
{
  eigen_transform.translation() = se3_transform.translation3f();
  eigen_transform.linear() = Eigen::Quaternionf(
    se3_transform.rotation().w(), se3_transform.rotation().x(),
    se3_transform.rotation().y(), se3_transform.rotation().z()).toRotationMatrix();
}

void convertEigenToRosTransform(
  const Transform & eigen_transform,
  geometry_msgs::msg::Transform & ros_transform)
{
  ros_transform.translation.x = eigen_transform.translation().x();
  ros_transform.translation.y = eigen_transform.translation().y();
  ros_transform.translation.z = eigen_transform.translation().z();
  Eigen::Quaternionf quaternion(eigen_transform.linear());
  ros_transform.rotation.x = quaternion.x();
  ros_transform.rotation.y = quaternion.y();
  ros_transform.rotation.z = quaternion.z();
  ros_transform.rotation.w = quaternion.w();
}

void convertSE3ToRosTransform(
  const isaac::common::transform::SE3TransformD & se3_transform,
  geometry_msgs::msg::Transform & ros_transform)
{
  ros_transform.translation.x = se3_transform.translation().x();
  ros_transform.translation.y = se3_transform.translation().y();
  ros_transform.translation.z = se3_transform.translation().z();
  ros_transform.rotation.x = se3_transform.rotation().x();
  ros_transform.rotation.y = se3_transform.rotation().y();
  ros_transform.rotation.z = se3_transform.rotation().z();
  ros_transform.rotation.w = se3_transform.rotation().w();
}

void convertEigenToRosPose(const Transform & eigen_transform, geometry_msgs::msg::Pose & ros_pose)
{
  ros_pose.position.x = eigen_transform.translation().x();
  ros_pose.position.y = eigen_transform.translation().y();
  ros_pose.position.z = eigen_transform.translation().z();
  Eigen::Quaternionf quaternion(eigen_transform.linear());
  ros_pose.orientation.x = quaternion.x();
  ros_pose.orientation.y = quaternion.y();
  ros_pose.orientation.z = quaternion.z();
  ros_pose.orientation.w = quaternion.w();
}

void convertSE3ToRosPose(
  const isaac::common::transform::SE3TransformD & se3_transform,
  geometry_msgs::msg::Pose & ros_pose)
{
  ros_pose.position.x = se3_transform.translation().x();
  ros_pose.position.y = se3_transform.translation().y();
  ros_pose.position.z = se3_transform.translation().z();
  ros_pose.orientation.x = se3_transform.rotation().x();
  ros_pose.orientation.y = se3_transform.rotation().y();
  ros_pose.orientation.z = se3_transform.rotation().z();
  ros_pose.orientation.w = se3_transform.rotation().w();
}

void convertTransformToEigen(const geometry_msgs::msg::Transform & msg, Transform * transform)
{
  Eigen::Translation3f trans(msg.translation.x, msg.translation.y,
    msg.translation.z);
  Eigen::Quaternionf quat(msg.rotation.w, msg.rotation.x,
    msg.rotation.y, msg.rotation.z);
  *transform = Transform(trans * quat);
}

void convertTransformToProto(
  const Transform & transform,
  protos::common::geometry::Transform3D * transform_proto)
{
  if (transform_proto == nullptr) {
    return;
  }
  transform_proto->mutable_translation()->set_x(transform.translation().x());
  transform_proto->mutable_translation()->set_y(transform.translation().y());
  transform_proto->mutable_translation()->set_z(transform.translation().z());
  Eigen::Quaternionf quaternion(transform.linear());
  transform_proto->mutable_rotation()->set_x(quaternion.x());
  transform_proto->mutable_rotation()->set_y(quaternion.y());
  transform_proto->mutable_rotation()->set_z(quaternion.z());
  transform_proto->mutable_rotation()->set_w(quaternion.w());
}

void convertProtoToTransform(
  const protos::common::geometry::Transform3D & transform_proto, Transform * transform)
{
  Eigen::Translation3f trans(transform_proto.translation().x(), transform_proto.translation().y(),
    transform_proto.translation().z());
  Eigen::Quaternionf quat(transform_proto.rotation().w(), transform_proto.rotation().x(),
    transform_proto.rotation().y(), transform_proto.rotation().z());
  *transform = Transform(trans * quat);
}

}  // namespace visual_global_localization
}  // namespace isaac_ros
}  // namespace nvidia
