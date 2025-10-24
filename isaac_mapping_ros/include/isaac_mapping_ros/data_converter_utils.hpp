// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include <opencv2/opencv.hpp>
#include <rosbag2_storage/bag_metadata.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <urdf/model.h>
#include <tf2_ros/buffer.h>

#include "isaac_mapping_ros/camera_metadata.hpp"

#include "protos/common/sensor/camera_sensor.pb.h"
#include "common/transform/transform_interpolator.h"
#include "common/image/image_rectifier.h"


namespace isaac_ros
{
namespace isaac_mapping_ros
{

namespace data_converter_utils
{

// Feature distance evaluation functor
struct ImageFrame
{
  cv::Mat image;
  std::string camera_name;
  uint32_t camera_sensor_id = 0;
  // sample id within this camera
  uint64_t sample_id = 0;
  // synced sample id across all cameras
  uint64_t synced_sample_id = 0;
  uint64_t timestamp_nanoseconds = 0;
  nvidia::isaac::common::transform::SE3TransformD camera_to_world;
  bool has_camera_to_world = false;
  bool is_rectified = false;
  bool is_depth_image = false;
};

typedef std::function<bool (ImageFrame & image)>
  WritImageFunctor;

enum class PoseMessageType : uint8_t
{
  kOdometry = 0,
  kPoseStamped = 1,
  kPoseWithCovarianceStamped = 2,
  kPath = 3,
  kUnknown = 4
};

// Stores the camera topic name and the corresponding camera info and image topics
struct CameraTopicConfig
{
  std::string name;
  std::string camera_info_topic;
  std::string image_topic;
  bool is_depth_image = false;
  std::string paired_camera_name;
  std::string frame_id_name;
  bool is_camera_rectified = false;  // If true, images are already rectified
  bool swap_rb_channels = false;     // If true, swap R and B channels (RGB<->BGR)
};

// Returns the mapping from camera info topic name to camera topic config based on topic name
std::map<std::string,
  data_converter_utils::CameraTopicConfig> GetTopicConfigByTopicName(
  const std::map<std::string, std::string> & topic_name_to_message_type);

// Read the topic config file and return the mapping from camera_topic_name to camera topic config
bool ReadCameraTopicConfig(
  const std::string & topic_config_file, std::map<std::string,
  CameraTopicConfig> & camera_info_topic_to_config);

std::map<std::string, std::string> GetTopicNameToMessageTypeMap(
  const rosbag2_storage::BagMetadata & metadata);

bool AddStaticTransformToBuffer(
  const std::string & sensor_data_bag, tf2_ros::Buffer & tf_buffer);

std::optional<nvidia::isaac::common::transform::SE3TransformD> GetRelativeTransformFromTFBuffer(
  const tf2_ros::Buffer & tf_buffer, const std::string & source_frame,
  const std::string & target_frame);

std::map<std::string, CameraMetadata> ExtractCameraMetadata(
  const std::string & sensor_data_bag_file,
  const std::string & output_folder,
  const std::string & base_link_name = "",
  const std::string & topic_yaml_file = "",
  bool do_rectify_images = false);

std::map<std::string, protos::common::sensor::CameraSensor> ExtractCameraSensors(
  const std::string & sensor_data_bag,
  const std::string & base_link_name, std::map<std::string,
  CameraTopicConfig> & camera_info_topic_to_config,
  bool do_rectify_images = false);

std::map<std::string, protos::common::sensor::CameraSensor> ExtractCameraSensors(
  const tf2_ros::Buffer & tf_buffer,
  const std::string & sensor_data_bag,
  const std::string & base_link_name, std::map<std::string,
  CameraTopicConfig> & camera_info_topic_to_config,
  bool do_rectify_images = false);

bool ExtractCameraImagesFromRosbag(
  const std::string & sensor_data_bag,
  std::map<std::string, CameraMetadata> & camera_metadata,
  const data_converter_utils::WritImageFunctor write_image_functor,
  const std::string & codec_name = "h264", bool dry_run = false, bool rectify_images = true);

// Extract images from extracted image dir
bool ExtractImagesFromExtractedDir(
  const std::string & root_dir,
  const std::map<std::string, uint64_t> & camera_name_to_sensor_id,
  const data_converter_utils::WritImageFunctor write_image_functor);

void FindSyncedTimestamps(
  const std::map<std::string, isaac_ros::isaac_mapping_ros::CameraMetadata> & video_writers,
  std::vector<std::vector<uint64_t>> & synced_timestamps,
  const int64_t time_sync_threshold = 100);

bool PopulateCameraParams(
  std::map<std::string, CameraMetadata> & camera_metadata,
  const std::map<std::string, protos::common::sensor::CameraSensor> & camera_params);

bool ExtractOdometryMsgFromPoseBag(
  const std::string & pose_bag_file, const std::string & pose_topic_name,
  std::vector<nav_msgs::msg::Odometry> & odometry_msgs,
  const std::string & expected_child_frame_id = "");

nvidia::isaac::common::transform::SE3PoseLinearInterpolator ExtractPosesFromBag(
  const std::string & pose_bag_file, const std::string & pose_topic_name,
  const std::string & expected_child_frame_id = "");

// if bag does not exists, it will crash, usually used at main function
void CheckBagExists(const std::string & bag_file);

std::map<uint64_t, nvidia::isaac::common::transform::SE3TransformD>
GetPosesForTimestamps(
  const nvidia::isaac::common::transform::SE3PoseLinearInterpolator &
  pose_interpolator,
  const std::vector<uint64_t> & timestamps);

bool GetFrameSyncAndPoseMap(
  const nvidia::isaac::common::transform::SE3PoseLinearInterpolator &
  pose_interpolator,
  const std::vector<uint64_t> & all_timestamps_nanoseconds,
  const std::vector<uint64_t> & select_timestamps_nanoseconds,
  std::map<uint64_t, uint64_t> & sample_id_to_synced_sample_id,
  std::map<uint64_t, nvidia::isaac::common::transform::SE3TransformD> & sample_id_to_pose);

bool ReadTimestampFile(
  const std::string & timestamp_file,
  std::vector<uint64_t> & timestamps);

bool WriteTimestampFile(
  const std::string & timestamp_file,
  const std::vector<uint64_t> & timestamps);

double ROSTimestampToSeconds(
  const builtin_interfaces::msg::Time & stamp);

uint64_t ROSTimestampToMicroseconds(
  const builtin_interfaces::msg::Time & stamp);

uint64_t ROSTimestampToNanoseconds(
  const builtin_interfaces::msg::Time & stamp);

}  // namespace data_converter_utils
}  // namespace isaac_mapping_ros
}  // namespace isaac_ros
