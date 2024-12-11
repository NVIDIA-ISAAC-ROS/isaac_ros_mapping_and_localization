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

#include "isaac_ros_visual_global_localization/global_localization_mapper_node.hpp"

#include <chrono>
#include <thread>

#include <cv_bridge/cv_bridge.h>

#include "common/file_utils/dm_file_utils.h"
#include "common/macros/macros.h"
#include "protos/visual/loop_closing/image_retrieval_config.pb.h"
#include "visual/utils/constants.h"
#include "isaac_ros_common/qos.hpp"
#include "isaac_ros_visual_global_localization/constants.h"
#include "isaac_ros_nitros/types/nitros_type_message_filter_traits.hpp"


namespace nvidia
{
namespace isaac_ros
{
namespace visual_global_localization
{

using isaac::common::file_utils::DMFileUtils;
using NitrosTimeStamp =
  message_filters::message_traits::TimeStamp<ImageType::BaseType>;

double GetImageTimestampInMillis(const ImageType & image)
{
  return image.GetTimestampSeconds() * kSecondsToMilliseconds +
         image.GetTimestampNanoseconds() / kMillisecondsToNanoseconds;
}

GlobalLocalizationMapperNode::GlobalLocalizationMapperNode(
  const rclcpp::NodeOptions & options)
: Node("global_localization_mapper_node", options), transform_manager_(this)
{
  if (!getParameters()) {
    RCLCPP_FATAL(get_logger(), "Failed to get parameters.");
    exit(EXIT_FAILURE);
  }
  if (!InitGlobalMapper()) {
    RCLCPP_FATAL(get_logger(), "Failed to initialize global mapper.");
    exit(EXIT_FAILURE);
  }
  // Subscribe to the topics
  subscribeToTopics();
  // Setup wall timer to tick at a fixed rate
  setupTimers();
}

GlobalLocalizationMapperNode::~GlobalLocalizationMapperNode()
{
  // Save db file to disk
  RCLCPP_INFO(get_logger(), "Saving database to disk...");
  const std::string db_filename = DMFileUtils::Get().JoinPath(
    map_dir_, isaac::visual::kImageRetrievalDatabaseFileName);
  if (!loop_closure_index_->Save(db_filename)) {
    RCLCPP_ERROR(get_logger(), "Failed to save the BoW map file.");
    return;
  }
  // Set camera parameters for frame metadata
  if (!setCameraParams()) {
    RCLCPP_ERROR(get_logger(), "Failed to set camera parameters.");
    return;
  }
  // Save frame metadata to disk
  RCLCPP_INFO(get_logger(), "Saving frame metadata to disk...");
  const auto & frame_meta_file = DMFileUtils::Get().JoinPath(
    map_dir_, "keyframes",
    isaac::visual::kFramesMetaFileName);
  if (!DMFileUtils::Get().WriteProtoFileByExtension(frame_meta_file, output_frames_meta_).ok()) {
    RCLCPP_ERROR_STREAM(get_logger(), "Can not save frames meta file to: " << frame_meta_file);
  }
}

bool GlobalLocalizationMapperNode::getParameters()
{
  RCLCPP_INFO(get_logger(), "Loading parameters...");
  // Load parameters
  config_dir_ = declare_parameter<std::string>("config_dir", "");
  input_image_topic_ = declare_parameter<std::string>("input_image_topic", input_image_topic_);
  input_camera_info_topic_name_ = declare_parameter<std::string>(
    "input_camera_info_topic_name",
    input_camera_info_topic_name_);
  map_dir_ = declare_parameter<std::string>("map_dir", "");
  map_frame_ = declare_parameter<std::string>("map_frame", map_frame_);
  base_frame_ = declare_parameter<std::string>("base_frame", base_frame_);
  num_cameras_ = declare_parameter<int>("num_cameras", num_cameras_);
  tick_period_ms_ = declare_parameter<int>("tick_period_ms", tick_period_ms_);
  image_buffer_size_ = declare_parameter<int>("image_buffer_size", 100);
  min_inter_frame_distance_ = declare_parameter<double>(
    "min_inter_frame_distance",
    min_inter_frame_distance_);
  min_inter_frame_rotation_degrees_ = declare_parameter<double>(
    "min_inter_frame_rotation_degrees",
    min_inter_frame_rotation_degrees_);

  // Check if parameters are set
  if (config_dir_.empty()) {
    RCLCPP_ERROR(get_logger(), "Parameter 'config_dir' is not set.");
    return false;
  }

  if (map_dir_.empty()) {
    RCLCPP_ERROR(get_logger(), "Parameter 'map_dir' is not set.");
    return false;
  }

  if (num_cameras_ <= 0) {
    RCLCPP_ERROR(
      get_logger(),
      "Parameter 'num_cameras' can't be set less than one.");
    return false;
  }

  sync_.reset(
    new isaac_common::messaging::MessageStreamSynchronizer<ImageType>(
      num_cameras_,
      kImageSyncMatchThresholdMs * kMillisecondsToNanoseconds, num_cameras_,
      image_buffer_size_));
  sync_->RegisterCallback(
    std::bind(
      &GlobalLocalizationMapperNode::callbackSynchronizedImages, this,
      std::placeholders::_2));

  transform_manager_.set_reference_frame(base_frame_);
  return true;
}

bool GlobalLocalizationMapperNode::InitGlobalMapper()
{
  RCLCPP_INFO(get_logger(), "Loading vocabulary...");
  const std::string vocabulary_dir_path = map_dir_;
  if (!DMFileUtils::Get().DirectoryExists(vocabulary_dir_path)) {
    RCLCPP_ERROR_STREAM(
      get_logger(),
      "Vocabulary file: " << vocabulary_dir_path
                          << " does not exist.");
    return false;
  }

  const std::string image_retrieval_config_file_path =
    DMFileUtils::Get().JoinPath(
    config_dir_,
    isaac::visual::kImageRetrievalConfigName);

  // Initialize the loop closure index
  protos::visual::loop_closing::ImageRetrievalConfig image_retrieval_config;
  STATUS_OK_OR_RETURN_FALSE(
    DMFileUtils::Get().ReadProtoFileByExtension(
      image_retrieval_config_file_path,
      &image_retrieval_config), "Failed to read config file: " + image_retrieval_config_file_path);

  loop_closure_index_ =
    std::make_unique<isaac::visual::loop_closing::LoopClosureIndex>();
  if (!loop_closure_index_->Init(
      vocabulary_dir_path, image_retrieval_config.shared_node_levelsup()))
  {
    LOG(ERROR) << "Failed to init loop closure index";
    return false;
  }

  // Initialize keypoint detector
  const std::string creation_params_file_path = DMFileUtils::Get().JoinPath(
    config_dir_, isaac::visual::kKeypointCreationConfigFilePath);
  protos::common::image::KeypointCreationParams keypoint_creation_params;
  STATUS_OK_OR_RETURN_FALSE(
    DMFileUtils::Get().ReadProtoFileByExtension(
      creation_params_file_path,
      &keypoint_creation_params), "Failed to read param file: " + creation_params_file_path);

  keypoint_creation_params.set_descriptor_type(loop_closure_index_->descriptor_type());

  switch (keypoint_creation_params.descriptor_type()) {
    case protos::common::image::SIFT_CV_CUDA_DESCRIPTOR:
      keypoint_creation_params.set_detector_type(protos::common::image::SIFT_CV_CUDA_DETECTOR);
      break;
    case protos::common::image::SUPER_POINT_DESCRIPTOR:
      keypoint_creation_params.set_detector_type(protos::common::image::SUPER_POINT_DETECTOR);
      break;
    case protos::common::image::ORB_DESCRIPTOR:
      RCLCPP_ERROR(get_logger(), "Don't support ORB descriptor yet.");
      return false;
    default:
      RCLCPP_ERROR(get_logger(), "Unknown descriptor type.");
      return false;
  }

  feature_extractor_ =
    std::make_unique<isaac::visual::cusfm::FeatureExtractor>(
    keypoint_creation_params);
  feature_extractor_->set_num_thread(num_thread_);

  if (loop_closure_index_ == nullptr || feature_extractor_ == nullptr) {
    RCLCPP_ERROR(get_logger(), "Failed to initialize loop closure index or feature extractor.");
    return false;
  }

  output_frames_meta_.clear_keyframes_metadata();
  output_frames_meta_.set_descriptor_type(keypoint_creation_params.descriptor_type());
  output_frames_meta_.set_detector_type(keypoint_creation_params.detector_type());
  return true;
}

void GlobalLocalizationMapperNode::advertiseTopics()
{
  RCLCPP_INFO(get_logger(), "Advertising topics...");
}

void GlobalLocalizationMapperNode::subscribeToTopics()
{
  RCLCPP_INFO(get_logger(), "Subscribing to topics...");
  // Subscribe to the image topic
  const rclcpp::QoS input_qos = ::isaac_ros::common::AddQosParameter(
    *this, kImageQosProfile, "input_qos");
  for (int i = 0; i < num_cameras_; i++) {
    image_subs_.emplace_back(
      std::make_shared<
        nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<ImageType>>(
        this, input_image_topic_ + "_" + std::to_string(i),
        nvidia::isaac_ros::nitros::nitros_image_rgb8_t::supported_type_name,
        std::bind(
          &GlobalLocalizationMapperNode::inputImageCallback, this,
          std::placeholders::_1, i),
        nvidia::isaac_ros::nitros::NitrosDiagnosticsConfig(), input_qos));
    camera_info_subs_.emplace_back(
      create_subscription<CameraInfoType>(
        input_camera_info_topic_name_ + "_" + std::to_string(i), input_qos,
        [this, i](const CameraInfoType::ConstSharedPtr & msg) {
          return inputCameraInfoCallback(msg, i);
        }));
  }
}

void GlobalLocalizationMapperNode::inputImageCallback(const ImageType & image_msg, int camera_id)
{
  const rclcpp::Time timestamp = NitrosTimeStamp::value(image_msg.GetMessage());
  RCLCPP_DEBUG_STREAM(
    get_logger(),
    "[" << std::this_thread::get_id() << "] inputImageCallback " << camera_id << ", sec: "
        << std::fixed << std::setprecision(3) << (double)timestamp.nanoseconds() / 1e9);
  sync_->AddMessage(camera_id, timestamp.nanoseconds(), image_msg, false);
}

void GlobalLocalizationMapperNode::inputCameraInfoCallback(
  const CameraInfoType::ConstSharedPtr & camera_info_ptr,
  uint32_t camera_id)
{
  const auto & camera_info_msg = *camera_info_ptr;
  if (!camera_params_.count(camera_info_msg.header.frame_id)) {
    const auto params = cameraInfoToMonoParams(camera_info_msg);
    Transform baselink_T_camera;
    if (!transform_manager_.lookupTransformToReferenceFrame(
        camera_info_msg.header.frame_id, camera_info_msg.header.stamp,
        &baselink_T_camera))
    {
      RCLCPP_ERROR(
        get_logger(), "Failed to get transform for frame_id: %s",
        camera_info_msg.header.frame_id.c_str());
      return;
    }
    camera_transforms_[camera_info_msg.header.frame_id] = baselink_T_camera;
    camera_params_[camera_info_msg.header.frame_id] = params;
    camera_id_to_camera_frame_id_[camera_id] = camera_info_msg.header.frame_id;
    camera_frame_id_to_camera_id_[camera_info_msg.header.frame_id] = camera_id;
  }
}

nvidia::isaac::common::image::MonoCameraCalibrationParams GlobalLocalizationMapperNode::
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

void GlobalLocalizationMapperNode::setupTimers()
{
  RCLCPP_INFO(get_logger(), "Setting up timers...");
  // Setup the timer to do global localization
  processing_timer_ = create_wall_timer(
    std::chrono::duration<double>(tick_period_ms_ * kMillisecondsToSeconds),
    std::bind(&GlobalLocalizationMapperNode::tick, this));
}

void GlobalLocalizationMapperNode::tick()
{
  // Get the latest image data in the queue
  RCLCPP_DEBUG(get_logger(), "GlobalLocalizationMapperNode::tick()");

  if (camera_params_.empty()) {
    RCLCPP_DEBUG_STREAM(get_logger(), "Waiting for camera info");
    return;
  }
  sync_->PopBuffersAndTriggerCallback();
}

void GlobalLocalizationMapperNode::callbackSynchronizedImages(
  const std::vector<std::pair<int,
  ImageType>> & idx_and_image_msgs)
{
  std::unordered_map<std::string, std::shared_ptr<ImageType>> images;
  for (const auto & [idx, image] : idx_and_image_msgs) {
    images[image.GetFrameId()] = std::make_shared<ImageType>(image);
  }
  processImages(images);
}

bool GlobalLocalizationMapperNode::processImages(
  const std::unordered_map<std::string, std::shared_ptr<ImageType>> & images)
{
  // Check if the image is a keyframe, only check one image as they are in sync
  const rclcpp::Time first_image_timestamp = rclcpp::Time(
    images.begin()->second->GetTimestampSeconds(),
    images.begin()->second->GetTimestampNanoseconds(), RCL_ROS_TIME);
  const std::string first_image_frame_id = images.begin()->first;
  if (!CheckKeyframe(first_image_frame_id, first_image_timestamp)) {
    RCLCPP_INFO(get_logger(), "This is not a keyframe, skipping...");
    return true;
  }
  for (const auto & image : images) {
    const std::string & frame_id = image.first;
    const auto & nitros_image = image.second;
    const rclcpp::Time timestamp =
      rclcpp::Time(
      nitros_image->GetTimestampSeconds(),
      nitros_image->GetTimestampNanoseconds(), RCL_ROS_TIME);
    Transform map_T_camera; // from cuVSLAM
    if (!transform_manager_.lookupTransformTf(map_frame_, frame_id, timestamp, &map_T_camera)) {
      RCLCPP_ERROR(
        get_logger(), "Failed to get transform for frame_id: %s to map frame",
        frame_id.c_str());
      continue;
    }
    if (!keyframeExtractAndMapping(nitros_image, map_T_camera)) {
      RCLCPP_ERROR(get_logger(), "Failed to extract keyframes");
      return false;
    }
  }
  // Update the previous keyframe pose
  previous_keyframe_pose_ = current_keyframe_pose_;
  return true;
}

bool GlobalLocalizationMapperNode::CheckKeyframe(
  const std::string & frame_id,
  const rclcpp::Time & timestamp)
{
  Transform map_T_camera;
  if (!transform_manager_.lookupTransformTf(map_frame_, frame_id, timestamp, &map_T_camera)) {
    RCLCPP_ERROR(
      get_logger(), "Failed to get transform for frame_id: %s to map frame",
      frame_id.c_str());
    return false;
  }
  isaac::common::transform::SE3TransformD current_pose(map_T_camera);
  // Check if the image is a keyframe
  double translation_difference_meters;
  double rotation_difference_radians;
  previous_keyframe_pose_.Difference(
    current_pose,
    &translation_difference_meters,
    &rotation_difference_radians);
  if (translation_difference_meters > min_inter_frame_distance_ ||
    rotation_difference_radians >
    min_inter_frame_rotation_degrees_ * isaac::common::kDegreesToRadians)
  {
    current_keyframe_pose_ = current_pose;
    return true;
  }
  return false;
}

bool GlobalLocalizationMapperNode::keyframeExtractAndMapping(
  const std::shared_ptr<ImageType> image, const Transform & pose)
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
    img_msg.data.data(), image->GetGpuData(), image->GetSizeInBytes(),
    cudaMemcpyDefault);

  // Note: the image is only support in RGB8/BGR8 format
  cv_bridge::CvImagePtr rgb_cv_ptr =
    cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::RGB8);
  cv::Mat mask;
  isaac::common::image::KeypointVector keypoint_vector;
  if (!feature_extractor_->ExtractImage(
      rgb_cv_ptr->image, mask,
      keypoint_vector))
  {
    RCLCPP_ERROR(get_logger(), "Failed to extract keyframes");
    return false;
  }

  protos::visual::general::Keyframe keyframe_proto;
  keypoint_vector.ToProto(keyframe_proto.mutable_keypoints());

  // get the camera_to_world proto
  protos::common::geometry::RigidTransform3d camera_to_world;
  Eigen::AngleAxisf angle_axis(pose.rotation());
  camera_to_world.mutable_translation()->set_x(pose.translation().x());
  camera_to_world.mutable_translation()->set_y(pose.translation().y());
  camera_to_world.mutable_translation()->set_z(pose.translation().z());
  camera_to_world.mutable_axis_angle()->set_angle_degrees(
    angle_axis.angle() *
    180.0 / M_PI);
  camera_to_world.mutable_axis_angle()->set_x(angle_axis.axis().x());
  camera_to_world.mutable_axis_angle()->set_y(angle_axis.axis().y());
  camera_to_world.mutable_axis_angle()->set_z(angle_axis.axis().z());

  // Set metadata for the keyframe
  protos::visual::general::KeyframeMetaData keyframe_metadata;
  keyframe_metadata.mutable_camera_to_world()->CopyFrom(camera_to_world);
  const int sample_id = processed_image_indices_[img_msg.header.frame_id]++;
  keyframe_metadata.set_sample_id(sample_id);
  const std::string image_name = img_msg.header.frame_id + "/" + std::to_string(sample_id) + ".jpg";
  keyframe_metadata.set_image_name(image_name);
  keyframe_metadata.set_id(keyframe_id_++);
  keyframe_metadata.set_camera_params_id(camera_frame_id_to_camera_id_[img_msg.header.frame_id]);
  keyframe_metadata.set_timestamp_microseconds(
    img_msg.header.stamp.sec * kSecondsToMicroseconds +
    img_msg.header.stamp.nanosec * kNanosecondsToMicroseconds);

  // Add the keyframe to the build BoW map
  const auto keyframe_ptr =
    std::make_shared<isaac::visual::general::Keyframe>(keyframe_proto, keyframe_metadata);
  loop_closure_index_->Add(keyframe_ptr);

  // Add this keyframe to the metadata collection
  *output_frames_meta_.add_keyframes_metadata() = keyframe_metadata;

  // Save the keyframe proto to disk
  if (!saveKeyframeToDisk(img_msg.header.frame_id, keyframe_metadata.sample_id(), keyframe_proto)) {
    RCLCPP_ERROR(get_logger(), "Failed to save keyframe to disk");
    return false;
  }
  // Save image to disk
  DMFileUtils::Get().EnsureDirectoryExists(
    DMFileUtils::Get().JoinPath(map_dir_, "raw", img_msg.header.frame_id));
  const std::string image_file_name = DMFileUtils::Get().JoinPath(
    map_dir_, "raw", img_msg.header.frame_id, std::to_string(
      keyframe_metadata.sample_id()) + isaac::common::kCameraImageFileExtension);
  if (!cv::imwrite(image_file_name, rgb_cv_ptr->image)) {
    RCLCPP_ERROR(get_logger(), "Failed to save image to disk");
    return false;
  }
  return true;
}

bool GlobalLocalizationMapperNode::saveKeyframeToDisk(
  const std::string & frame_id,
  size_t sample_id,
  const protos::visual::general::Keyframe & keyframe)
{
  const std::string keyframe_path = DMFileUtils::Get().JoinPath(
    map_dir_, "keyframes", frame_id,
    std::to_string(sample_id) + isaac::common::kProtoFileExtension);
  const std::string keyframe_directory =
    DMFileUtils::Get().GetFileDirectory(keyframe_path);
  if (!DMFileUtils::Get().EnsureDirectoryExists(
      keyframe_directory))
  {
    RCLCPP_ERROR(get_logger(), "Failed to create directory: %s", keyframe_directory.c_str());
    return false;
  }

  RCLCPP_INFO_STREAM(
    get_logger(),
    "Writing key frame: "
      << sample_id << " to " << keyframe_path
      << " total number of key points on this frame: "
      << keyframe.keypoints().size());
  STATUS_OK_OR_RETURN_FALSE(
    DMFileUtils::Get().WriteProtoFileByExtension(
      keyframe_path,
      keyframe),
    "Failed to write proto file: " + keyframe_path);
  return true;
}

bool GlobalLocalizationMapperNode::setCameraParams()
{
  auto & camera_param_to_session_name =
    (*output_frames_meta_.mutable_camera_params_id_to_session_name());
  auto & camera_param_id_to_camera_param =
    (*output_frames_meta_.mutable_camera_params_id_to_camera_params());
  for (int i = 0; i < num_cameras_; i++) {
    // For the live node, don't have the session name, so just use camera
    camera_param_to_session_name[i] = "camera";
    protos::common::sensor::CameraSensor camera_sensor;
    auto & sensor_meta_data = (*camera_sensor.mutable_sensor_meta_data());
    sensor_meta_data.set_sensor_type(protos::common::sensor::SensorMetaData::CAMERA);
    sensor_meta_data.set_sensor_name(camera_id_to_camera_frame_id_[i]);
    sensor_meta_data.set_frequency(kCameraFrequency);
    protos::common::geometry::RigidTransform3d sensor_to_vehicle_transform;
    const auto & sensor_translation =
      camera_transforms_[camera_id_to_camera_frame_id_[i]].translation();
    const auto & sensor_rotation = camera_transforms_[camera_id_to_camera_frame_id_[i]].rotation();
    const Eigen::AngleAxisf angle_axis(sensor_rotation);
    sensor_to_vehicle_transform.mutable_translation()->set_x(sensor_translation.x());
    sensor_to_vehicle_transform.mutable_translation()->set_y(sensor_translation.y());
    sensor_to_vehicle_transform.mutable_translation()->set_z(sensor_translation.z());
    sensor_to_vehicle_transform.mutable_axis_angle()->set_angle_degrees(
      angle_axis.angle() *
      180.0 / M_PI);
    sensor_to_vehicle_transform.mutable_axis_angle()->set_x(angle_axis.axis().x());
    sensor_to_vehicle_transform.mutable_axis_angle()->set_y(angle_axis.axis().y());
    sensor_to_vehicle_transform.mutable_axis_angle()->set_z(angle_axis.axis().z());
    sensor_meta_data.mutable_sensor_to_vehicle_transform()->CopyFrom(sensor_to_vehicle_transform);
    camera_sensor.mutable_calibration_parameters()->CopyFrom(
      camera_params_[camera_id_to_camera_frame_id_[i]].ToProto());
    camera_param_id_to_camera_param[i] = camera_sensor;
  }
  return true;
}

} // namespace visual_global_localization
} // namespace isaac_ros
} // namespace nvidia

// Register as a component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(
  nvidia::isaac_ros::visual_global_localization::GlobalLocalizationMapperNode)
