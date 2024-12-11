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

#ifndef ISAAC_ROS_CAMERA_LOCALIZATION_GLOBAL_LOCALIZATION_MAPPER_NODE_HPP_
#define ISAAC_ROS_CAMERA_LOCALIZATION_GLOBAL_LOCALIZATION_MAPPER_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>

#include "isaac_ros_visual_global_localization/transform_manager.hpp"
#include "isaac_common/messaging/message_stream_synchronizer.hpp"
#include "isaac_ros_managed_nitros/managed_nitros_subscriber.hpp"
#include "isaac_ros_nitros_image_type/nitros_image_view.hpp"

#include "visual/general/keyframe.h"
#include "common/image/keypoint_detector.h"
#include "common/image/image_rectifier.h"
#include "visual/loop_closing/loop_closure_index.h"
#include "visual/cusfm/feature_extractor.h"

namespace nvidia
{
namespace isaac_ros
{
namespace visual_global_localization
{

using ImageType = nvidia::isaac_ros::nitros::NitrosImageView;
using CameraInfoType = sensor_msgs::msg::CameraInfo;

class GlobalLocalizationMapperNode : public rclcpp::Node
{
public:
  explicit GlobalLocalizationMapperNode(
    const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  virtual ~GlobalLocalizationMapperNode();

  // Setup functions, called by constructor
  bool getParameters();

  // Load vocabulary and initialize keypoint detector and loop closure index
  bool InitGlobalMapper();

  void advertiseTopics();

  void subscribeToTopics();

  void setupTimers();

  // Callback function for nitros images
  void inputImageCallback(const ImageType & image_msg, int camera_id);

  // Callback function for camera info
  void inputCameraInfoCallback(
    const CameraInfoType::ConstSharedPtr & camera_info_msg,
    uint32_t camera_id);

  void callbackSynchronizedImages(
    const std::vector<std::pair<int, ImageType>> & idx_and_image_mgs);

  void tick();

  bool checkImageSync(
    const std::unordered_map<std::string, std::shared_ptr<ImageType>>
    & images);

  bool processImages(
    const std::unordered_map<std::string, std::shared_ptr<ImageType>>
    & images);

  bool CheckKeyframe(
    const std::string & frame_id,
    const rclcpp::Time & timestamp);

  bool keyframeExtractAndMapping(
    const std::shared_ptr<ImageType> image,
    const Transform & pose);

  bool saveKeyframeToDisk(
    const std::string & frame_id,
    size_t sample_id,
    const protos::visual::general::Keyframe & keyframe);

  nvidia::isaac::common::image::MonoCameraCalibrationParams
  cameraInfoToMonoParams(
    const CameraInfoType & camera_info_msg);

  bool setCameraParams();

protected:
  // ROS publishers and subscribers
  // nitros image subscribers
  std::vector<std::shared_ptr<
      nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<ImageType>>>
  image_subs_;
  // camera info subscriber
  std::vector<rclcpp::Subscription<CameraInfoType>::SharedPtr> camera_info_subs_;

private:
  // Parameters
  std::string input_image_topic_ = "global_localization_mapper_node/image";
  std::string input_camera_info_topic_name_ = "global_localization_mapper_node/camera_info";
  std::string config_dir_;
  // This is the directory where the map is saved
  std::string map_dir_;
  std::string map_frame_ = "map";
  std::string base_frame_ = "base_link";
  // the number of cameras
  int num_cameras_ = 2;
  // the rate of the processing timer
  int tick_period_ms_ = 10;
  int num_thread_ = 20;
  double min_inter_frame_distance_ = 0.05;
  double min_inter_frame_rotation_degrees_ = 1;
  // A incrementing index for the keyframe metadata.
  uint64_t keyframe_id_ = 0;

  // The image buffer size
  int image_buffer_size_;

  TransformManager transform_manager_;
  // Timers.
  rclcpp::TimerBase::SharedPtr processing_timer_;

  std::unique_ptr<isaac::visual::cusfm::FeatureExtractor> feature_extractor_;
  std::unique_ptr<isaac::visual::loop_closing::LoopClosureIndex>
  loop_closure_index_;

  // Synchronizer used to sync all image messages.
  std::unique_ptr<isaac_common::messaging::MessageStreamSynchronizer<ImageType>> sync_;

  std::unordered_map<std::string, int> processed_image_indices_;
  // The last keyframe pose of first camera
  isaac::common::transform::SE3TransformD previous_keyframe_pose_;
  // The current keyframe pose of first camera
  isaac::common::transform::SE3TransformD current_keyframe_pose_;
  // output frames metadata
  protos::visual::general::KeyframesMetadataCollection output_frames_meta_;

  // camera frame-id string -> camera-id
  std::unordered_map<std::string, size_t> camera_frame_id_to_camera_id_;
  std::unordered_map<size_t, std::string> camera_id_to_camera_frame_id_;
  std::unordered_map<std::string,
    nvidia::isaac::common::image::MonoCameraCalibrationParams> camera_params_;
  std::unordered_map<std::string, Transform> camera_transforms_;
};

} // namespace visual_global_localization
} // namespace isaac_ros
} // namespace nvidia

#endif // ISAAC_ROS_CAMERA_LOCALIZATION_GLOBAL_LOCALIZATION_MAPPER_NODE_HPP_
