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

#ifndef ISAAC_ROS_CAMERA_LOCALIZATION_POINT_CLOUD_FILTER_NODE_HPP_
#define ISAAC_ROS_CAMERA_LOCALIZATION_POINT_CLOUD_FILTER_NODE_HPP_

#include "rclcpp/rclcpp.hpp"

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"

namespace nvidia
{
namespace isaac_ros
{
namespace visual_global_localization
{
//
// PointCloudFilterNode subsribes to a point cloud topic, a pose topic and /tf.
// When receiving a pose message, it finds cloud messages close to the pose timestamp from the queue,
// converts the cloud to a target frame using tf, and publishes the cloud.
//
class PointCloudFilterNode : public rclcpp::Node
{
public:
  explicit PointCloudFilterNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  virtual ~PointCloudFilterNode() = default;

  void getParameters();
  void subscribeToTopics();
  void advertiseTopics();
  void pointCloudCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg);
  void poseCallback(const geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr msg);
  // Transform cloud to target_frame_ using tf at pose_stamp.
  // Return false if fails to query tf from cloud_frame to target_frame at pose_stamp.
  bool transformCloud(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr & in_cloud,
    const builtin_interfaces::msg::Time & pose_stamp,
    sensor_msgs::msg::PointCloud2 & out_cloud) const;

private:
  // the target frame that the output cloud is transformed to.
  std::string target_frame_;
  // queue size for input cloud messages.
  // When input cloud is 10hz, using queue_size=100 guarantees the associated cloud be queued when pose has 10 seconds delay.
  size_t cloud_queue_size_;
  // max time difference between cloud and pose message, to transform the cloud with the pose.
  float time_thresh_sec_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filtered_cloud_pub_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::list<sensor_msgs::msg::PointCloud2::ConstSharedPtr> cloud_queue_;

};
}  // namespace visual_global_localization
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_CAMERA_LOCALIZATION_POINT_CLOUD_FILTER_NODE_HPP_
