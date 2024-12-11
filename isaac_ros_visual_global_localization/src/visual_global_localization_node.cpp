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

#include "isaac_ros_visual_global_localization/visual_global_localization_node.hpp"

#include <chrono>
#include <functional>
#include <thread>

#include "rclcpp/serialization.hpp"
#include <cv_bridge/cv_bridge.h>
#include <cuda_runtime.h>

#include "isaac_ros_common/qos.hpp"
#include "common/datetime/timer.h"
#include "common/file_utils/dm_file_utils.h"
#include "common/image/image_calibration_params.h"
#include "isaac_ros_visual_global_localization/constants.h"
#include "isaac_ros_nitros_image_type/nitros_image_builder.hpp"
#include "isaac_ros_nitros/types/nitros_type_message_filter_traits.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace visual_global_localization
{
#define INFO_STREAM(VERBOSE, MESSAGE) \
  if (VERBOSE) { \
    RCLCPP_INFO_STREAM(get_logger(), MESSAGE); \
  }

using NitrosTimeStamp =
  message_filters::message_traits::TimeStamp<ImageType::BaseType>;

uint64_t GetImageTimestampInMicros(const ImageType & image)
{
  return image.GetTimestampSeconds() * kSecondsToMicroseconds +
         image.GetTimestampNanoseconds() / kMicrosecondsToNanoseconds;
}

sensor_msgs::msg::CameraInfo GetRectifiedCameraInfo(
  const sensor_msgs::msg::CameraInfo & raw_camera_info)
{
  sensor_msgs::msg::CameraInfo rectified_camera_info = raw_camera_info;
  rectified_camera_info.distortion_model = "plumb_bob";
  // set distortion to zero
  rectified_camera_info.d = {0, 0, 0, 0, 0};
  // set camera matrix to desired projection matrix
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      rectified_camera_info.k[i * 3 + j] = rectified_camera_info.p[i * 4 + j];
    }
  }
  return rectified_camera_info;
}

VisualGlobalLocalizationNode::VisualGlobalLocalizationNode(const rclcpp::NodeOptions & options)
: Node("visual_localization", options),
  transform_manager_(this)
{
  getParameters();
  // Initialize the localizer API
  initLocalizerApi();
  // Subscribe to the topics
  subscribeToTopics();
  // Advertise the topics
  advertiseTopics();
  // Advertise the services
  advertiseServices();
}

void VisualGlobalLocalizationNode::getParameters()
{
  // Declare & initialize the parameters.
  RCLCPP_INFO_STREAM(get_logger(), "VisualGlobalLocalizationNode::getParameters()");
  // Node params
  map_dir_ = declare_parameter<std::string>("map_dir", "");
  config_dir_ = declare_parameter<std::string>("config_dir", "");
  debug_dir_ = declare_parameter<std::string>("debug_dir", "");
  model_dir_ = declare_parameter<std::string>("model_dir", "");
  debug_map_raw_dir_ = declare_parameter<std::string>("debug_map_raw_dir", "");
  num_cameras_ = declare_parameter<int>("num_cameras", num_cameras_);
  stereo_localizer_cam_ids_ = declare_parameter<std::string>("stereo_localizer_cam_ids", "0,1");
  enable_rectify_images_ = declare_parameter<bool>("enable_rectify_images", enable_rectify_images_);
  publish_rectified_images_ = declare_parameter<bool>(
    "publish_rectified_images",
    publish_rectified_images_);
  enable_continuous_localization_ = declare_parameter<bool>(
    "enable_continuous_localization",
    enable_continuous_localization_);
  use_initial_guess_ = declare_parameter<bool>("use_initial_guess", use_initial_guess_);
  image_qos_profile_ = declare_parameter<std::string>("image_qos_profile", kImageQosProfile);
  image_sync_match_threshold_ms_ = declare_parameter<double>(
    "image_sync_match_threshold_ms",
    kImageSyncMatchThresholdMs);
  image_buffer_size_ = declare_parameter<int>("image_buffer_size", 10);
  input_image_topic_name_ =
    declare_parameter<std::string>("input_image_topic_name", input_image_topic_name_);
  input_camera_info_topic_name_ = declare_parameter<std::string>(
    "input_camera_info_topic_name",
    input_camera_info_topic_name_);
  base_frame_ = declare_parameter<std::string>("base_frame", base_frame_);
  map_frame_ = declare_parameter<std::string>("map_frame", map_frame_);
  publish_map_to_base_tf_ = declare_parameter<bool>(
    "publish_map_to_base_tf",
    publish_map_to_base_tf_);
  invert_map_to_base_tf_ = declare_parameter<bool>("invert_map_to_base_tf", invert_map_to_base_tf_);
  verbose_logging_ = declare_parameter<bool>("verbose_logging", verbose_logging_);
  init_glog_ = declare_parameter<bool>("init_glog", init_glog_);
  glog_v_ = declare_parameter<int>("glog_v", glog_v_);
  localization_precision_level_ = declare_parameter<int>("localization_precision_level", localization_precision_level_);
  transform_manager_.set_reference_frame(base_frame_);
  // Check if the parameters are set correctly
  if (config_dir_.empty()) {
    RCLCPP_ERROR(get_logger(), "Config directory is not set");
  }
  if (map_dir_.empty()) {
    RCLCPP_ERROR(get_logger(), "Map directory is not set");
  }
  if (num_cameras_ < 1) {
    RCLCPP_ERROR(get_logger(), "Invalid num_cameras: %d", num_cameras_);
  }

  if (init_glog_) {
    google::InitGoogleLogging(this->get_name());
    google::InstallFailureSignalHandler();
    FLAGS_v = glog_v_;
    FLAGS_colorlogtostderr = 1;
    FLAGS_alsologtostderr = 1;
  }

  RCLCPP_INFO_STREAM(
    get_logger(),
    "num_cameras for camera localization: " << num_cameras_);

  sync_.reset(
    new isaac_common::messaging::MessageStreamSynchronizer<ImageType>(
      num_cameras_,
      image_sync_match_threshold_ms_ * kMillisecondsToNanoseconds, num_cameras_,
      image_buffer_size_));
  sync_->RegisterCallback(
    std::bind(
      &VisualGlobalLocalizationNode::callbackSynchronizedImages, this,
      std::placeholders::_2));
}

void VisualGlobalLocalizationNode::initLocalizerApi()
{
  // Check if the map and config directories exist
  if (!isaac::common::file_utils::DMFileUtils::Get().DirectoryExists(map_dir_)) {
    RCLCPP_ERROR_STREAM(get_logger(), "Map directory does not exist: " << map_dir_);
  }
  if (!isaac::common::file_utils::DMFileUtils::Get().DirectoryExists(config_dir_)) {
    RCLCPP_ERROR_STREAM(get_logger(), "Config directory does not exist: " << config_dir_);
  }

  // If the model directory is set, check if it exists
  if (!model_dir_.empty()) {
    if (!isaac::common::file_utils::DMFileUtils::Get().DirectoryExists(model_dir_)) {
      RCLCPP_ERROR_STREAM(get_logger(), "Model directory does not exist: " << model_dir_);
    }
  }

  cuvgl_ = std::make_unique<isaac::visual::cuvgl::cuVGL>();

  STATUS_CHECK(cuvgl_->Init(map_dir_, config_dir_, model_dir_));
  if (!stereo_localizer_cam_ids_.empty()) {
    auto status =
      cuvgl_->SetStereoCameraParamsIDFromString(stereo_localizer_cam_ids_);
    if (!status.ok()) {
      RCLCPP_ERROR_STREAM(
        get_logger(), "Can't set stereo-camera-params-id from " << stereo_localizer_cam_ids_);
    }
  }
  if (!debug_dir_.empty()) {
    RCLCPP_INFO_STREAM(get_logger(), "Set Debug Dir to" << debug_dir_);
    auto status = cuvgl_->set_debug_dir(debug_dir_);
    if (!status.ok()) {
      RCLCPP_ERROR_STREAM(
        get_logger(), "Can't set debug_dir to " << debug_dir_);
    }
  }
  if (!debug_map_raw_dir_.empty()) {
    RCLCPP_INFO_STREAM(get_logger(), "Set Debug Map Raw Dir to" << debug_map_raw_dir_);
    auto status = cuvgl_->set_debug_map_raw_dir(debug_map_raw_dir_);
    if (!status.ok()) {
      RCLCPP_ERROR_STREAM(
        get_logger(), "Can't set debug_map_raw_dir to " << debug_map_raw_dir_);
    }
  }
  cuvgl_->set_localization_precision_level(localization_precision_level_);
}

void VisualGlobalLocalizationNode::subscribeToTopics()
{
  RCLCPP_INFO_STREAM(get_logger(), "VisualGlobalLocalizationNode::subscribeToTopics()");
  const rclcpp::QoS input_qos = ::isaac_ros::common::AddQosParameter(
    *this, image_qos_profile_,
    "input_qos");
  for (int i = 0; i < num_cameras_; ++i) {
    image_subs_.emplace_back(
      std::make_shared<nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<ImageType>>(
        this,
        input_image_topic_name_ + "_" + std::to_string(i),
        nvidia::isaac_ros::nitros::nitros_image_rgb8_t::supported_type_name,
        std::bind(
          &VisualGlobalLocalizationNode::inputImageCallback, this, std::placeholders::_1, i),
        nvidia::isaac_ros::nitros::NitrosDiagnosticsConfig(), input_qos));
    camera_info_subs_.emplace_back(
      create_subscription<CameraInfoType>(
        input_camera_info_topic_name_ + "_" + std::to_string(i), input_qos,
        [this, i](const CameraInfoType::ConstSharedPtr & msg) {
          return inputCameraInfoCallback(msg, i);
        }));
  }

  trigger_global_localization_sub_ = create_subscription<MsgTriggerGlobalLocalization>(
    "visual_localization/trigger_localization", rclcpp::QoS(100),
    std::bind(
      &VisualGlobalLocalizationNode::callbackTopicTriggerGlobalLocalization, this,
      std::placeholders::_1));
}

void VisualGlobalLocalizationNode::advertiseServices()
{
  RCLCPP_INFO_STREAM(get_logger(), "VisualGlobalLocalizationNode::advertiseServices()");
  global_localization_srv_ = create_service<SrvTriggerGlobalLocalization>(
    "visual_localization/trigger_localization",
    std::bind(
      &VisualGlobalLocalizationNode::callbackSrvTriggerGlobalLocalization, this,
      std::placeholders::_1,
      std::placeholders::_2));
}

void VisualGlobalLocalizationNode::advertiseTopics()
{
  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
  pose_pub_ = create_publisher<PoseType>(
    "visual_localization/pose", ::isaac_ros::common::ParseQosString(
      "DEFAULT"));
  debug_image_pub_ =
    create_publisher<sensor_msgs::msg::Image>(
    "visual_localization/debug_image", ::isaac_ros::common::ParseQosString(
      "DEFAULT"));
  diagnostics_pub_ = create_publisher<DiagnosticArrayType>(
    "/diagnostics", ::isaac_ros::common::ParseQosString("DEFAULT"));

  if (enable_rectify_images_ && publish_rectified_images_) {
    for (int i = 0; i < num_cameras_; ++i) {
      rectified_image_pubs_.emplace_back(
        create_publisher<sensor_msgs::msg::Image>(
          "visual_localization/camera_" + std::to_string(i) + "/image_rect",
          ::isaac_ros::common::ParseQosString("DEFAULT")
        )
      );
      rectified_camera_info_pubs_.emplace_back(
        create_publisher<sensor_msgs::msg::CameraInfo>(
          "visual_localization/camera_" + std::to_string(i) + "/camera_info_rect",
          ::isaac_ros::common::ParseQosString("DEFAULT")
        )
      );
    }
  }

  // establish a initial link between base to map
  if (publish_map_to_base_tf_) {
    geometry_msgs::msg::TransformStamped transform_msg;
    transform_msg.header.frame_id = invert_map_to_base_tf_ ? base_frame_ : map_frame_;
    transform_msg.child_frame_id = invert_map_to_base_tf_ ? map_frame_ : base_frame_;
    tf_broadcaster_->sendTransform(transform_msg);
  }
}

void VisualGlobalLocalizationNode::callbackSrvTriggerGlobalLocalization(
  const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
  std::shared_ptr<std_srvs::srv::Trigger::Response> res)
{
  (void)req;
  RCLCPP_INFO_STREAM(
    get_logger(), "VisualGlobalLocalizationNode::callbackTriggerGlobalLocalization()");
  trigger_localization_ = true;
  res->success = true;
}

void VisualGlobalLocalizationNode::callbackTopicTriggerGlobalLocalization(
  const MsgTriggerGlobalLocalization::SharedPtr msg)
{
  (void)msg;
  RCLCPP_INFO_STREAM(
    get_logger(),
    "VisualGlobalLocalizationNode::callbackTopicTriggerGlobalLocalization()");
  trigger_localization_ = true;
}

void VisualGlobalLocalizationNode::callbackSynchronizedImages(
  const std::vector<std::pair<int,
  ImageType>> & idx_and_image_msgs)
{
  if (!trigger_localization_) {
    return;
  }

  std::unordered_map<std::string, std::shared_ptr<ImageType>> images;
  for (const auto & [idx, image] : idx_and_image_msgs) {
    images[image.GetFrameId()] = std::make_shared<ImageType>(image);
  }

  rclcpp::Time timestamp;
  isaac::common::transform::SE3TransformD localization_pose;
  isaac::common::datetime::Timer timer;
  bool succeed = processImages(images, timestamp, localization_pose);
  double execution_time_sec = (double)timer.ElapsedMilliseconds() * kMillisecondsToSeconds;

  if (succeed) {
    // Publish the TF base_link_T_map
    if (publish_map_to_base_tf_) {
      publishTF(timestamp, localization_pose);
    }
    // Publish the pose
    publishPose(timestamp, localization_pose);
    num_published_poses_++;

    RCLCPP_INFO_STREAM(
      get_logger(),
      "Publish global localization pose, translation: ["
        << std::fixed << std::setprecision(5)
        << localization_pose.translation().x() << ", "
        << localization_pose.translation().y() << ", "
        << localization_pose.translation().z() << "], rotation: ["
        << localization_pose.rotation().w() << ", "
        << localization_pose.rotation().x() << ", "
        << localization_pose.rotation().y() << ", "
        << localization_pose.rotation().z() << "], execution time: " << execution_time_sec <<
        " seconds");
  }

  // Publish diagnostic message
  publishDiagnostics(timestamp, succeed, execution_time_sec);
}

bool VisualGlobalLocalizationNode::processImages(
  const std::unordered_map<std::string,
  std::shared_ptr<ImageType>> & images, rclcpp::Time & timestamp,
  isaac::common::transform::SE3TransformD & localization_pose)
{
  // Initialize the image rectifier
  if (enable_rectify_images_ && !checkImageRectifier(images)) {
    RCLCPP_ERROR(get_logger(), "Failed to initialize image rectifier");
    return false;
  }

  // Use the timestamp of the first image
  timestamp = rclcpp::Time(
    images.begin()->second->GetTimestampSeconds(),
    images.begin()->second->GetTimestampNanoseconds(), RCL_ROS_TIME);

  // Set the camera images
  std::vector<isaac::visual::cuvgl::CameraImage> camera_images;
  for (const auto & image : images) {
    const std::string frame_id = image.first;
    if (camera_frame_id_to_camera_ids_.count(frame_id) == 0) {
      RCLCPP_ERROR(
        get_logger(),
        "Camera info for : %s is not loaded yet", frame_id.c_str());
      return false;
    }

    image_frame_id_to_sequence_number_[frame_id]++;

    camera_images.emplace_back();
    if (!convertImageMessage(
        image.second, camera_frame_id_to_camera_ids_[frame_id],
        camera_images.back()))
    {
      RCLCPP_ERROR(
        get_logger(),
        "Failed to convert image message from frame_id: %s", frame_id.c_str());
      return false;
    }
  }

  std::vector<isaac::common::transform::SE3TransformD> initial_world_T_cameras;
  if (use_initial_guess_ && !bootstrap_localization_) {
    INFO_STREAM(verbose_logging_, "Predicting the transform by odometry");
    for (const auto & image: images) {
      const std::string frame_id = image.first;
      const Transform map_T_camera_eigen = transform_manager_.predictTransformByOdom(
        map_frame_, frame_id,
        previous_processed_image_time_,
        timestamp);
      if (!map_T_camera_eigen.isApprox(Eigen::Isometry3f::Identity())) {
        isaac::common::transform::SE3TransformD map_T_camera;
        convertEigenToTransform(map_T_camera_eigen, map_T_camera);
        initial_world_T_cameras.emplace_back(map_T_camera);
      } else {
        RCLCPP_WARN_STREAM(
          get_logger(),
          "Failed to predict transform for frame_id: " << frame_id <<
            ", it will use global localization");
        initial_world_T_cameras.clear();
        break;
      }
    }
  }

  // Call the localizer api to get the map tf.
  if (initial_world_T_cameras.empty()) {
    INFO_STREAM(verbose_logging_, "Global localization");
  } else {
    for (size_t image_idx = 0; image_idx < camera_images.size(); ++image_idx) {
      camera_images[image_idx].initial_camera_to_map = initial_world_T_cameras[image_idx];
      camera_images[image_idx].has_initial_camera_to_map = true;
    }
    INFO_STREAM(verbose_logging_, "cuSFM-based localization");
  }
  const auto status = cuvgl_->Localize(camera_images, localization_pose);
  // If debug_dir_ is set, publish the debug image
  if (!use_initial_guess_ && !debug_dir_.empty() && !debug_map_raw_dir_.empty()) {
    cv::Mat debug_image;
    auto debug_status = cuvgl_->GetDebugImage(debug_image);
    if (!debug_status.ok()) {
      RCLCPP_WARN_STREAM(get_logger(), "Failed to get debug images: " << debug_status.message());
    }
    if (!debug_image.empty()) {
      cv_bridge::CvImage cv_image;
      cv_image.image = debug_image;
      cv_image.encoding = sensor_msgs::image_encodings::RGB8;
      cv_image.header.stamp = timestamp;
      cv_image.header.frame_id = base_frame_;
      sensor_msgs::msg::Image::SharedPtr debug_image_msg = cv_image.toImageMsg();
      debug_image_pub_->publish(*debug_image_msg);
    }
  }

  if (!status.ok()) {
    RCLCPP_WARN_STREAM(get_logger(), "Localization failed: " << status.message());
    bootstrap_localization_ = true;
    return false;
  }

  INFO_STREAM(
    verbose_logging_,
    "Frame: " << image_frame_id_to_sequence_number_[images.begin()->first]
              << " localization succeeded.");

  // record the timestamp of the processed image if the localization is successful
  previous_processed_image_time_ = timestamp;
  bootstrap_localization_ = false;

  // if continuous localization is not enabled, then we do not need to trigger localization
  if (!enable_continuous_localization_) {
    trigger_localization_ = false;
    RCLCPP_INFO_STREAM(get_logger(), "Global localization sent a pose. Stopping localization.");
  }
  return true;
}

void VisualGlobalLocalizationNode::inputImageCallback(const ImageType & image_msg, int camera_id)
{
  if (!trigger_localization_) {
    // clear buffers just in case if previous run didn't clear it
    // we can't call this in processImages as message buffer is not thread safe
    sync_->ClearBuffers();
    return;
  }

  const rclcpp::Time timestamp = NitrosTimeStamp::value(image_msg.GetMessage());
  INFO_STREAM(
    verbose_logging_,
    "[" << std::this_thread::get_id() << "] inputImageCallback " << camera_id << ", sec: "
        << std::fixed << std::setprecision(3) << (double)timestamp.nanoseconds() / 1e9);
  sync_->AddMessage(camera_id, timestamp.nanoseconds(), image_msg, true /* trigger callback */);
}

nvidia::isaac::common::image::MonoCameraCalibrationParams VisualGlobalLocalizationNode::
cameraInfoToMonoParams(
  const CameraInfoType & camera_info)
{
  nvidia::isaac::common::image::MonoCameraCalibrationParams params;
  params.image_width_ = camera_info.width;
  params.image_height_ = camera_info.height;
  cv::Mat camera_matrix(3, 3, CV_64F, const_cast<double *>(camera_info.k.data()));
  cv::Mat distortion_coefficients(
    camera_info.d.size(), 1, CV_64F,
    const_cast<double *>(camera_info.d.data()));
  cv::Mat rectification_matrix(3, 3, CV_64F, const_cast<double *>(camera_info.r.data()));
  cv::Mat projection_matrix(3, 4, CV_64F, const_cast<double *>(camera_info.p.data()));
  params.camera_matrix_ = camera_matrix;
  params.distortion_coefficients_ = distortion_coefficients;
  params.rectification_matrix_ = rectification_matrix;
  params.projection_matrix_ = projection_matrix;
  return params;
}

void VisualGlobalLocalizationNode::inputCameraInfoCallback(
  const CameraInfoType::ConstSharedPtr & camera_info_ptr,
  uint32_t camera_id)
{
  const auto & camera_info_msg = *camera_info_ptr;
  if (camera_params_.count(camera_info_msg.header.frame_id) == 1) {
    return;
  }
  const auto params = cameraInfoToMonoParams(camera_info_msg);
  if (!params.IsValid()) {
    RCLCPP_ERROR(get_logger(), "Invalid camera calibration parameters");
    return;
  }

  if (enable_rectify_images_) {
    // TODO: there seems still a bug, somehow need to move rectifier initialization to camera-info callback,
    // or add copy constructor to MonoCameraCalibrationParams to get correct camera-params (but the unit test
    // passes w/ or wo/ the copy constructor).
    auto image_rectifier = std::make_unique<nvidia::isaac::common::image::ImageRectifier>();
    if (!image_rectifier->Init(params)) {
      RCLCPP_ERROR_STREAM(
        get_logger(),
        "Failed to initialize image rectifier for " << camera_info_msg.header.frame_id);
      return;
    }
    image_rectifiers_[camera_info_msg.header.frame_id] = std::move(image_rectifier);

    // save the rectified camera info to be published along with rectified image
    if (publish_rectified_images_) {
      rectified_camera_infos_[camera_info_msg.header.frame_id] = GetRectifiedCameraInfo(
        camera_info_msg);
    }
  }

  Transform baselink_T_camera_eigen;
  // Get the latest transform from the camera frame to the base frame
  if (!transform_manager_.lookupTransformToReferenceFrame(
      camera_info_msg.header.frame_id, rclcpp::Time(0),
      &baselink_T_camera_eigen))
  {
    RCLCPP_ERROR(
      get_logger(), "Failed to get transform for frame_id: %s",
      camera_info_msg.header.frame_id.c_str());
    return;
  }
  isaac::common::transform::SE3TransformD baselink_T_camera;
  convertEigenToTransform(baselink_T_camera_eigen, baselink_T_camera);

  const auto status = cuvgl_->AddCamera(camera_id, baselink_T_camera, params);
  if (!status.ok()) {
    RCLCPP_ERROR_STREAM(
      get_logger(), "Failed to add camera due to " << status.message());
    return;
  }

  camera_params_[camera_info_msg.header.frame_id] = params;
  camera_frame_id_to_camera_ids_[camera_info_msg.header.frame_id] = camera_id;
}

bool VisualGlobalLocalizationNode::checkImageRectifier(
  const std::unordered_map<std::string,
  std::shared_ptr<ImageType>> & images) const
{
  for (auto & [frame_id, image]: images) {
    if (image_rectifiers_.count(frame_id) == 0 || image_rectifiers_.at(frame_id) == nullptr) {
      return false;
    }
  }
  return true;
}

bool VisualGlobalLocalizationNode::convertImageMessage(
  const std::shared_ptr<ImageType> & image,
  uint32_t sensor_id,
  isaac::visual::cuvgl::CameraImage & camera_image)
{
  // Convert the NitrosImage to sensor_msgs::msg::Image
  sensor_msgs::msg::Image img_msg;
  img_msg.header.frame_id = image->GetFrameId();
  img_msg.header.stamp.sec = image->GetTimestampSeconds();
  img_msg.header.stamp.nanosec = image->GetTimestampNanoseconds();
  img_msg.height = image->GetHeight();
  img_msg.width = image->GetWidth();
  img_msg.encoding = image->GetEncoding();
  img_msg.step = image->GetSizeInBytes() / image->GetHeight();
  img_msg.data.resize(image->GetSizeInBytes());
  cudaMemcpy(
    img_msg.data.data(), image->GetGpuData(),
    image->GetSizeInBytes(), cudaMemcpyDefault);

  // Note: the image is only support in RGB8/BGR8 format. Check CameraFrameToMat in camera_data_utils.cc
  cv_bridge::CvImagePtr rgb_cv_ptr =
    cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);
  if (enable_rectify_images_) {
    if (image_rectifiers_[image->GetFrameId()] == nullptr) {
      RCLCPP_ERROR(get_logger(), "Image rectifier is not initialized");
      return false;
    }
    // Rectify the image
    cv::Mat rectified_image;
    image_rectifiers_[image->GetFrameId()]->Rectify(rgb_cv_ptr->image, rectified_image);
    rgb_cv_ptr->image = rectified_image;

    if (publish_rectified_images_) {
      rectified_image_pubs_[sensor_id]->publish(*(rgb_cv_ptr->toImageMsg()));
      rectified_camera_info_pubs_[sensor_id]->publish(rectified_camera_infos_[image->GetFrameId()]);
    }
  }

  camera_image.camera_id = sensor_id;
  camera_image.image = rgb_cv_ptr->image;
  camera_image.timestamp_microseconds = GetImageTimestampInMicros(*image);
  return true;
}

void VisualGlobalLocalizationNode::publishTF(
  const rclcpp::Time & timestamp,
  const isaac::common::transform::SE3TransformD & localization_pose)
{
  geometry_msgs::msg::TransformStamped transform_msg;
  transform_msg.header.stamp = timestamp;
  if (invert_map_to_base_tf_) {
    transform_msg.header.frame_id = base_frame_;
    transform_msg.child_frame_id = map_frame_;
    const auto & base_link_T_map = localization_pose.Inverse();
    convertSE3ToRosTransform(base_link_T_map, transform_msg.transform);
  } else {
    transform_msg.header.frame_id = map_frame_;
    transform_msg.child_frame_id = base_frame_;
    convertSE3ToRosTransform(localization_pose, transform_msg.transform);
  }
  tf_broadcaster_->sendTransform(transform_msg);
}

void VisualGlobalLocalizationNode::publishPose(
  const rclcpp::Time & timestamp,
  const isaac::common::transform::SE3TransformD & localization_pose)
{
  PoseType pose_msg;
  pose_msg.header.frame_id = map_frame_;
  pose_msg.header.stamp = timestamp;
  convertSE3ToRosPose(localization_pose, pose_msg.pose.pose);
  pose_pub_->publish(pose_msg);
}

void VisualGlobalLocalizationNode::publishDiagnostics(
  const rclcpp::Time & timestamp, bool loc_succeed,
  double execution_time_sec)
{
  DiagnosticArrayType diagnostics;
  diagnostics.header.stamp = timestamp;
  diagnostics.header.frame_id = map_frame_;
  DiagnosticStatusType & status = diagnostics.status.emplace_back();
  if (loc_succeed) {
    status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
  } else {
    status.level = diagnostic_msgs::msg::DiagnosticStatus::WARN;
  }
  status.name = "Visual Global Localization Diagnostics";
  status.message = "Localization state and execution time measurements";
  status.hardware_id = "visual_global_localization";
  status.values.emplace_back();
  status.values.back().key = "trigger_next_localization";
  status.values.back().value = trigger_localization_ ? "Yes" : "No";

  status.values.emplace_back();
  status.values.back().key = "localization_execution_time";
  status.values.back().value = std::to_string(execution_time_sec);

  status.values.emplace_back();
  status.values.back().key = "num_published_poses";
  status.values.back().value = std::to_string(num_published_poses_);

  diagnostics_pub_->publish(diagnostics);
}

}  // namespace visual_global_localization
}  // namespace isaac_ros
}  // namespace nvidia

// Register as a component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(
  nvidia::isaac_ros::visual_global_localization::VisualGlobalLocalizationNode)
