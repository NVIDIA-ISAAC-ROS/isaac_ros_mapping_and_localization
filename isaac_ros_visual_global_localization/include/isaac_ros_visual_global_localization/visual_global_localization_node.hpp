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

#ifndef ISAAC_ROS_CAMERA_LOCALIZATION_VISUAL_LOCALIZATION_NODE_HPP_
#define ISAAC_ROS_CAMERA_LOCALIZATION_VISUAL_LOCALIZATION_NODE_HPP_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2_ros/transform_broadcaster.h"

#include "isaac_common/messaging/message_stream_synchronizer.hpp"
#include "isaac_ros_managed_nitros/managed_nitros_subscriber.hpp"
#include "isaac_ros_nitros_image_type/nitros_image_view.hpp"
#include "isaac_ros_visual_global_localization/transform_manager.hpp"

#include "common/image/image_rectifier.h"
#include "visual/cuvgl/cuvgl.h"
#include "std_srvs/srv/trigger.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace visual_global_localization
{

using ImageType = nvidia::isaac_ros::nitros::NitrosImageView;
using CameraInfoType = sensor_msgs::msg::CameraInfo;
using SrvTriggerGlobalLocalization = std_srvs::srv::Trigger;
using MsgTriggerGlobalLocalization = geometry_msgs::msg::PoseWithCovarianceStamped;
using PoseType = geometry_msgs::msg::PoseWithCovarianceStamped;
using OdomType = nav_msgs::msg::Odometry;

using DiagnosticArrayType = diagnostic_msgs::msg::DiagnosticArray;
using DiagnosticStatusType = diagnostic_msgs::msg::DiagnosticStatus;

class VisualGlobalLocalizationNode : public rclcpp::Node
{
public:
  explicit VisualGlobalLocalizationNode(
    const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  virtual ~VisualGlobalLocalizationNode() = default;

  // Setup functions, called by constructor
  void getParameters();
  // Init localizer api
  void initLocalizerApi();
  void subscribeToTopics();
  void advertiseTopics();
  void advertiseServices();

  // check if image rectifier are initialized for input images
  bool checkImageRectifier(
    const std::unordered_map<std::string,
    std::shared_ptr<ImageType>> & images) const;

  // Callback function for nitros images
  void inputImageCallback(const ImageType & image_msg, int camera_id);
  // Callback function for camera info
  void inputCameraInfoCallback(
    const CameraInfoType::ConstSharedPtr & camera_info_msg,
    uint32_t camera_id);
  // Callback function for service trigger global localization
  void callbackSrvTriggerGlobalLocalization(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
    std::shared_ptr<std_srvs::srv::Trigger::Response> res);
  // Callback function for topic trigger global localization
  void callbackTopicTriggerGlobalLocalization(const MsgTriggerGlobalLocalization::SharedPtr msg);


  // Process image data
  virtual bool processImages(
    const std::unordered_map<std::string,
    std::shared_ptr<ImageType>> & images,
    rclcpp::Time & timestamp,
    isaac::common::transform::SE3TransformD & localization_pose);

protected:
  bool convertImageMessage(
    const std::shared_ptr<ImageType> & image,
    uint32_t sensor_id,
    isaac::visual::cuvgl::CameraImage & camera_image);

  nvidia::isaac::common::image::MonoCameraCalibrationParams
  cameraInfoToMonoParams(
    const CameraInfoType & camera_info_msg);

  bool checkImageSync(
    const std::unordered_map<std::string,
    std::shared_ptr<ImageType>> images);

  void callbackSynchronizedImages(
    const std::vector<std::pair<int, ImageType>> & idx_and_image_mgs);

  // Publish TF message
  void publishTF(
    const rclcpp::Time & timestamp,
    const isaac::common::transform::SE3TransformD & localization_pose);
  // Publish visual localization pose
  void publishPose(
    const rclcpp::Time & timestamp,
    const isaac::common::transform::SE3TransformD & localization_pose);
  // Publish diagnostics message
  void publishDiagnostics(const rclcpp::Time & timestamp, bool status, double execution_time_sec);

  TransformManager transform_manager_;
  std::unique_ptr<isaac::visual::cuvgl::cuVGL> cuvgl_;

  // ROS publishers and subscribers
  // nitros image subscribers
  std::vector<std::shared_ptr<nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<ImageType>>>
  image_subs_;
  // camera info subscriber
  std::vector<rclcpp::Subscription<CameraInfoType>::SharedPtr> camera_info_subs_;
  // Trigger global localization by topic
  rclcpp::Subscription<MsgTriggerGlobalLocalization>::SharedPtr trigger_global_localization_sub_;

  // Publishers
  rclcpp::Publisher<PoseType>::SharedPtr pose_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_image_pub_;
  rclcpp::Publisher<DiagnosticArrayType>::SharedPtr diagnostics_pub_;
  std::vector<rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr> rectified_image_pubs_;
  std::vector<rclcpp::Publisher<CameraInfoType>::SharedPtr> rectified_camera_info_pubs_;
  // Transform broadcaster
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  // global localization service
  rclcpp::Service<SrvTriggerGlobalLocalization>::SharedPtr global_localization_srv_;

  // Parameters

  // Enable to rectify the images
  bool enable_rectify_images_ = false;
  bool publish_rectified_images_ = false;
  // the number of cameras
  int num_cameras_ = 2;
  // Whether or not trigger localization
  bool trigger_localization_ = true;

  // Whether or not always do global localization
  // If enabled, we run localization for every frame,
  // if not, we only run localization at the beginning or per-request
  bool enable_continuous_localization_ = true;

  // The threshold for image sync in milliseconds
  double image_sync_match_threshold_ms_;

  // The image buffer size
  int image_buffer_size_;

  bool publish_map_to_base_tf_ = false;
  bool invert_map_to_base_tf_ = false;

  // the QoS settings for the image input topics
  std::string image_qos_profile_;

  std::string input_image_topic_name_ = "visual_localization/image";
  std::string input_camera_info_topic_name_ = "visual_localization/camera_info";

  // stereo camera ids for stereo localizer, separated by comma.
  std::string stereo_localizer_cam_ids_ = "";
  // camera frame-id string -> camera-id
  std::map<std::string, size_t> camera_frame_id_to_camera_ids_;

  // if use_initial_guess is true, the localizer will do cuSFM-based localization
  // If it is false, the localizer will do global localization
  bool use_initial_guess_ = false;
  // if bootstrap_localization is true, the localizer will do global localization
  bool bootstrap_localization_ = true;

  // the directory of the map file
  std::string map_dir_;
  // the directory of the config file
  std::string config_dir_;
  // the directory of the localizer debug dir
  std::string debug_dir_;
  // the directory of the model file, it is optional.
  // If it is not set, it will use the model that is defined in the config_dir.
  // If it is set, it will overwrite the model that is defined in the config_dir.
  std::string model_dir_;
  // the directory of the raw data for debugging.
  // It is optional.
  // If it is set and debug_dir is set, the localizer will output the debug image for debugging.
  std::string debug_map_raw_dir_;

  // Output frames hierarchy:
  std::string map_frame_ = "map";
  std::string base_frame_ = "base_link";

  // If print more verbose ROS logs.
  bool verbose_logging_ = false;
  // If initialize glog. It can be called only once per process, otherwise it reports error.
  bool init_glog_ = false;
  // Glog level
  int glog_v_ = 0;

  // Determines the precision level of localization, where:
  // 2 = highest precision (most restrictive, produces less results, potentially more accurate)
  // 1 = medium precision
  // 0 = lowest precision (allows more results, potentially less accurate)
  int localization_precision_level_ = 2;

  // the map of image frame id to sequence number
  std::unordered_map<std::string, uint32_t> image_frame_id_to_sequence_number_;

  // Synchronizer used to sync all image messages.
  std::unique_ptr<isaac_common::messaging::MessageStreamSynchronizer<ImageType>> sync_;

  // Image rectifier and camera params
  std::unordered_map<std::string,
    std::unique_ptr<nvidia::isaac::common::image::ImageRectifier>> image_rectifiers_;
  std::unordered_map<std::string,
    nvidia::isaac::common::image::MonoCameraCalibrationParams> camera_params_;
  std::unordered_map<std::string, CameraInfoType> rectified_camera_infos_;

  // The previous processed image time
  rclcpp::Time previous_processed_image_time_;

  // The published localization pose count
  int num_published_poses_ = 0;
};
}  // namespace visual_global_localization
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_CAMERA_LOCALIZATION_VISUAL_LOCALIZATION_NODE_HPP_
