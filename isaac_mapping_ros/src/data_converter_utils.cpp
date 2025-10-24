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

#include "isaac_mapping_ros/data_converter_utils.hpp"
#include "isaac_mapping_ros/video_decoder.hpp"

#include <algorithm>
#include <sstream>
#include <glog/logging.h>
#include <omp.h>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/serialization.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <tf2_msgs/msg/tf_message.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <yaml-cpp/yaml.h>

#include "common/datetime/scoped_timer.h"
#include "common/file_utils/file_utils.h"

#include "common/image/image_calibration_params.h"
#include "common/image/image_rectifier.h"
#include "common/transform/se3_transform.h"
#include "common/strings/split.h"
#include "common/transform/transform_interpolator.h"

#include "visual/general/keyframe_metadata.h"
#include "visual/utils/keyframe_utils.h"


namespace isaac_ros
{
namespace isaac_mapping_ros
{
using nvidia::isaac::common::file_utils::FileUtils;

// Helper struct to store parsed topic information
struct TopicInfo
{
  std::string camera_name_;      // First token (e.g., "front_camera")
  std::string sub_camera_name_;  // Second token (e.g., "left", "depth", "color")
  std::string full_topic_;       // Full topic name
  std::string message_type_;     // Message type

  // Pre-computed sensor type flags
  bool is_left_camera = false;
  bool is_right_camera = false;
  bool is_rectified = false;
  bool is_depth_camera = false;
  bool is_rgb_camera = false;
};

// Helper function to parse topic tokens and create TopicInfo
std::vector<TopicInfo> ParseAllTopics(
  const std::map<std::string, std::string> & topic_name_to_message_type)
{
  std::vector<TopicInfo> parsed_topics;

  for (const auto & [topic_name, message_type] : topic_name_to_message_type) {
    std::vector<std::string> tokens;
    std::stringstream ss(topic_name);
    std::string token;

    while (std::getline(ss, token, '/')) {
      if (!token.empty()) {
        tokens.push_back(token);
      }
    }

    TopicInfo topic_info;
    topic_info.full_topic_ = topic_name;
    topic_info.message_type_ = message_type;

    if (tokens.size() >= 1) {
      topic_info.camera_name_ = tokens[0];
    }
    if (tokens.size() >= 2) {
      topic_info.sub_camera_name_ = tokens[1];
    }

    // Analyze sensor type if we have at least the sub_camera_name_
    if (!topic_info.sub_camera_name_.empty()) {
      // Convert topic name and sensor type to lowercase for case-insensitive comparison
      std::string topic_name_lower = topic_name;
      std::string sensor_type_lower = topic_info.sub_camera_name_;
      std::transform(
        topic_name_lower.begin(), topic_name_lower.end(),
        topic_name_lower.begin(), ::tolower);
      std::transform(
        sensor_type_lower.begin(), sensor_type_lower.end(),
        sensor_type_lower.begin(), ::tolower);

      // Detect camera type from second token and topic name
      topic_info.is_left_camera = (sensor_type_lower.find("left") != std::string::npos) ||
        (sensor_type_lower.find("infra1") != std::string::npos);
      topic_info.is_right_camera = (sensor_type_lower.find("right") != std::string::npos) ||
        (sensor_type_lower.find("infra2") != std::string::npos);

      // If camera_info has "rect" it's rectified camera
      topic_info.is_rectified = (topic_name_lower.find("rect") != std::string::npos);

      // If has "depth" it's depth camera
      topic_info.is_depth_camera = (topic_name_lower.find("depth") != std::string::npos) ||
        (sensor_type_lower.find("depth") != std::string::npos);

      // If has "rgb" or "color" it's rgb camera
      topic_info.is_rgb_camera = (topic_name_lower.find("rgb") != std::string::npos) ||
        (topic_name_lower.find("color") != std::string::npos);
    }

    parsed_topics.push_back(topic_info);
  }

  return parsed_topics;
}

const std::string kCompressedImageMessageType = "sensor_msgs/msg/CompressedImage";
const std::string kRawImageMessageType = "sensor_msgs/msg/Image";
const std::string kCameraInfoMessageType = "sensor_msgs/msg/CameraInfo";
const std::string kOdometryMessageType = "nav_msgs/msg/Odometry";
const std::string kPoseStampedMessageType = "geometry_msgs/msg/PoseStamped";
const std::string kPoseWithCovarianceStampedMessageType =
  "geometry_msgs/msg/PoseWithCovarianceStamped";
const std::string kPathMessageType = "nav_msgs/msg/Path";
const std::string kCameraOpticalSuffix = "_optical";
const std::string kBagMetadataFile = "metadata.yaml";

std::unordered_map<std::string, data_converter_utils::PoseMessageType> poseMessageTypeMap = {
  {kOdometryMessageType, data_converter_utils::PoseMessageType::kOdometry},
  {kPoseStampedMessageType, data_converter_utils::PoseMessageType::kPoseStamped},
  {kPoseWithCovarianceStampedMessageType,
    data_converter_utils::PoseMessageType::kPoseWithCovarianceStamped},
  {kPathMessageType, data_converter_utils::PoseMessageType::kPath}
};

// private namespace for local utility
namespace
{

void PrintBagMetadata(const rosbag2_storage::BagMetadata & bag)
{
  std::stringstream ss;
  ss << "Version: " << bag.version << "\n";
  ss << "Bag Size: " << bag.bag_size << "\n";
  ss << "Storage Identifier: " << bag.storage_identifier << "\n";
  ss << "Relative File Paths: ";
  for (const auto & path : bag.relative_file_paths) {
    ss << path << " ";
  }
  ss << "\n";
  ss << "Files: ";
  for (const auto & file : bag.files) {
    ss << "Path: " << file.path << ", Starting Time: " <<
      std::chrono::duration_cast<std::chrono::seconds>(
      file.starting_time.time_since_epoch())
      .count() <<
      " seconds since epoch, Duration: " << file.duration.count() <<
      " nanoseconds, Message Count: " << file.message_count << "\n";
  }
  ss << "Duration: " << bag.duration.count() << " nanoseconds\n";
  ss << "Starting Time: " <<
    std::chrono::duration_cast<std::chrono::seconds>(
    bag.starting_time.time_since_epoch())
    .count() <<
    " seconds since epoch\n";
  ss << "Message Count: " << bag.message_count << "\n";
  ss << "Topics with Message Count: ";
  for (const auto & topic : bag.topics_with_message_count) {
    ss << "Name: " << topic.topic_metadata.name <<
      ", Type: " << topic.topic_metadata.type << ", Serialization Format: " <<
      topic.topic_metadata.serialization_format;

    // Print QoS profiles
    ss << ", Offered QoS Profiles: [";
    for (size_t i = 0; i < topic.topic_metadata.offered_qos_profiles.size(); ++i) {
      if (i > 0) {
        ss << ", ";
      }
      const auto & qos = topic.topic_metadata.offered_qos_profiles[i].get_rmw_qos_profile();
      ss << "{reliability: " << qos.reliability <<
        ", durability: " << qos.durability <<
        ", history: " << qos.history <<
        ", depth: " << qos.depth << "}";
    }
    ss << "]";

    ss << ", Message Count: " << topic.message_count << "\n";
  }
  ss << "Compression Format: " << bag.compression_format << "\n";
  ss << "Compression Mode: " << bag.compression_mode << "\n";
  LOG(INFO) << ss.str();
}

void ReplaceFirst(
  std::string & input, const std::string & to_replace,
  const std::string & replace_with)
{
  std::size_t pos = input.find(to_replace);
  if (pos == std::string::npos) {
    return;
  }
  input.replace(pos, to_replace.length(), replace_with);
}

std::string TopicNameToCameraName(const std::string & topic_name)
{
  std::string camera_name = FileUtils::GetParentDirectory(topic_name);
  if (!camera_name.empty()) {
    // remove the first /
    camera_name.erase(0, 1);
  }

  // replace all / to _
  std::replace(camera_name.begin(), camera_name.end(), '/', '_');

  // Uncomment the next line when using old rosbags.
  // ReplaceFirst(camera_name, "back", "rear");

  return camera_name;
}

nvidia::isaac::common::transform::SE3TransformD JointToSE3Transform(
  const urdf::Joint & joint)
{
  const auto & transform = joint.parent_to_joint_origin_transform;
  return nvidia::isaac::common::transform::SE3TransformD(
    Eigen::Vector3d(
      transform.position.x, transform.position.y,
      transform.position.z),
    Eigen::Quaterniond(
      transform.rotation.w, transform.rotation.x,
      transform.rotation.y, transform.rotation.z));
}

nvidia::isaac::common::transform::SE3TransformD ROSTransformToSE3Transform(
  const geometry_msgs::msg::Transform & transform)
{
  return nvidia::isaac::common::transform::SE3TransformD(
    Eigen::Vector3d(
      transform.translation.x, transform.translation.y,
      transform.translation.z),
    Eigen::Quaterniond(
      transform.rotation.w, transform.rotation.x,
      transform.rotation.y, transform.rotation.z));
}

// Get the joint that has the child as the input link
urdf::JointConstSharedPtr GetParentJoint(
  const urdf::Model & urdf_model,
  const std::string & link_name)
{
  for (const auto & joint_name_to_joint : urdf_model.joints_) {
    const auto & joint = joint_name_to_joint.second;
    if (joint->child_link_name == link_name) {
      return joint;
    }
  }

  return nullptr;
}

uint32_t GetTopicMessageCount(
  const rosbag2_storage::BagMetadata & metadata, const std::string & topic_name)
{
  for (const auto & topic : metadata.topics_with_message_count) {
    if (topic.topic_metadata.name == topic_name) {
      return topic.message_count;
    }
  }
  return 0;
}

template<typename VectorType>
protos::common::geometry::MatrixD ConvertMatrix(
  int rows, int cols,
  const VectorType & data)
{
  CHECK_EQ(data.size(), (size_t)rows * cols);
  protos::common::geometry::MatrixD out_matrix;
  *out_matrix.mutable_data() = {data.begin(), data.end()};
  out_matrix.set_row_count(rows);
  out_matrix.set_column_count(cols);
  return out_matrix;
}

void ConvertRepeatedFieldToMat(
  const google::protobuf::RepeatedField<double> & repeated_field,
  int row_count, int column_count, cv::Mat & matrix)
{
  CHECK_EQ(repeated_field.size(), row_count * column_count);
  matrix.create(row_count, column_count, CV_64F);
  int index = 0;
  for (int i = 0; i < row_count; ++i) {
    for (int j = 0; j < column_count; ++j) {
      matrix.at<double>(i, j) = repeated_field.Get(index++);
    }
  }
}

void ConvertMatrixProtoToMat(
  const protos::common::geometry::MatrixD & matrix_proto,
  cv::Mat & matrix)
{
  ConvertRepeatedFieldToMat(
    matrix_proto.data(), matrix_proto.row_count(),
    matrix_proto.column_count(), matrix);
}

nvidia::isaac::common::image::MonoCameraCalibrationParams
ConvertCameraParmsProto(const protos::common::sensor::MonoCalibrationParameters & proto)
{
  nvidia::isaac::common::image::MonoCameraCalibrationParams params;
  params.image_height_ = proto.image_height();
  params.image_width_ = proto.image_width();

  if (proto.has_camera_matrix()) {
    ConvertMatrixProtoToMat(proto.camera_matrix(), params.camera_matrix_);
  } else {
    LOG(WARNING) << "Mono proto didn't specify camera_matrix, filling zeros";
    params.camera_matrix_ = cv::Mat(3, 3, CV_64F, double(0));
  }

  if (proto.has_distortion_coefficients()) {
    ConvertMatrixProtoToMat(
      proto.distortion_coefficients(),
      params.distortion_coefficients_);
  } else {
    LOG(WARNING) << "Mono proto didn't specify distortion_coefficients, "
      "filling empty matrix";
    params.distortion_coefficients_ = cv::Mat();
  }

  if (proto.has_rectification_matrix()) {
    ConvertMatrixProtoToMat(
      proto.rectification_matrix(),
      params.rectification_matrix_);
  } else {
    LOG(WARNING) <<
      "Mono proto didn't specify rectification_matrix, filling zeros";
    params.rectification_matrix_ = cv::Mat(3, 3, CV_64F, double(0));
  }

  if (proto.has_projection_matrix()) {
    ConvertMatrixProtoToMat(
      proto.projection_matrix(),
      params.projection_matrix_);
  } else {
    LOG(WARNING) <<
      "Mono proto didn't specify projection_matrix, filling zeros";
    params.projection_matrix_ = cv::Mat(3, 4, CV_64F, double(0));
  }

  return params;
}

protos::common::sensor::CameraSensor ConvertCameraInfoToSensor(
  const sensor_msgs::msg::CameraInfo & camera_info,
  const std::string & sensor_name,
  const nvidia::isaac::common::transform::SE3TransformD & sensor_to_vehicle_transform,
  bool is_camera_rectified)
{
  CHECK_EQ(camera_info.k.size(), 9);
  CHECK_EQ(camera_info.r.size(), 9);
  CHECK_EQ(camera_info.p.size(), 12);

  protos::common::sensor::CameraSensor camera_sensor;
  auto metadata = camera_sensor.mutable_sensor_meta_data();
  metadata->set_sensor_type(protos::common::sensor::SensorMetaData_SensorType_CAMERA);
  metadata->set_sensor_name(sensor_name);

  // Only apply the rectification matrix rotation to the sensor to vehicle transform
  // if camera images are raw (not already rectified) and rectification will be performed
  if (is_camera_rectified) {
    // apply the rectification matrix rotation part to the sensor to vehicle transform
    // Map raw R (row-major) into an Eigen matrix then invert
    const Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> R(&camera_info.r[0]);
    // check if rectification matrix is identity or zeros
    if (!R.isIdentity() && !R.isZero()) {
      nvidia::isaac::common::transform::SE3TransformD rectified_camera_to_raw(
        Eigen::Vector3d::Zero(),
        Eigen::Quaterniond(R.inverse()));
      *metadata->mutable_sensor_to_vehicle_transform() =
        (sensor_to_vehicle_transform * rectified_camera_to_raw).ToProto();
    } else {
      LOG(WARNING) <<
        "Rectification matrix is identity or zeros, "
        "skipping applying rectification matrix to sensor to vehicle transform for sensor: " <<
        sensor_name;
      *metadata->mutable_sensor_to_vehicle_transform() =
        sensor_to_vehicle_transform.ToProto();
    }
  } else {
    // Use the original sensor to vehicle transform without rectification adjustments
    // (images are already rectified)
    *metadata->mutable_sensor_to_vehicle_transform() =
      sensor_to_vehicle_transform.ToProto();
  }

  metadata->set_frequency(30);

  auto calibration_params = camera_sensor.mutable_calibration_parameters();
  calibration_params->set_image_width(camera_info.width);
  calibration_params->set_image_height(camera_info.height);
  *calibration_params->mutable_projection_matrix() =
    ConvertMatrix(3, 4, camera_info.p);

  // set the last col to all 0, due to we are treating each camera as individual
  // camera so the stereo extrinsics is 0
  CHECK_EQ(calibration_params->projection_matrix().data_size(), 12);
  calibration_params->mutable_projection_matrix()->set_data(3, 0.0);
  calibration_params->mutable_projection_matrix()->set_data(7, 0.0);
  calibration_params->mutable_projection_matrix()->set_data(11, 0.0);

  *calibration_params->mutable_rectification_matrix() =
    ConvertMatrix(3, 3, camera_info.r);
  *calibration_params->mutable_camera_matrix() =
    ConvertMatrix(3, 3, camera_info.k);
  *calibration_params->mutable_distortion_coefficients() =
    ConvertMatrix(1, camera_info.d.size(), camera_info.d);
  // TODO(dizeng) add ftheta param conversion here
  // calibration_params->mutable_ftheta_parameters();
  // we don't have rolling_shutter_delay_microseconds settings

  // Set camera projection model type based on rectification status
  if (is_camera_rectified) {
    camera_sensor.set_camera_projection_model_type(
      protos::common::sensor::CameraProjectionModelType::RECTIFIED);
  } else {
    camera_sensor.set_camera_projection_model_type(
      protos::common::sensor::CameraProjectionModelType::PINHOLE);
  }

  return camera_sensor;
}

}  // namespace

std::map<std::string, std::string> data_converter_utils::GetTopicNameToMessageTypeMap(
  const rosbag2_storage::BagMetadata & metadata)
{
  std::map<std::string, std::string> topic_name_to_message_type;
  for (const auto & topic : metadata.topics_with_message_count) {
    const std::string & topic_name = topic.topic_metadata.name;
    topic_name_to_message_type[topic_name] = topic.topic_metadata.type;
  }
  return topic_name_to_message_type;
}

bool data_converter_utils::AddStaticTransformToBuffer(
  const std::string & sensor_data_bag, tf2_ros::Buffer & tf_buffer)
{
  try {
    // Set up the bag reader
    rosbag2_cpp::Reader reader;
    reader.open(sensor_data_bag);

    // Create a serialization object for deserialization
    rclcpp::Serialization<tf2_msgs::msg::TFMessage> serialization;

    // Read messages from the bag
    while (reader.has_next()) {
      auto bag_message = reader.read_next();
      if (bag_message->topic_name == "/tf_static") {
        tf2_msgs::msg::TFMessage::SharedPtr tf_msg =
          std::make_shared<tf2_msgs::msg::TFMessage>();
        // Deserialize the message
        rclcpp::SerializedMessage serialized_msg(*bag_message->serialized_data);
        serialization.deserialize_message(&serialized_msg, tf_msg.get());
        // Add each transform to the buffer
        for (const auto & transform : tf_msg->transforms) {
          tf_buffer.setTransform(transform, "default_authority", true);
        }
      }
    }

    return true;
  } catch (const std::exception & e) {
    LOG(ERROR) << "Failed to read tf static transform due to: " << e.what();
    return false;
  }
}

std::optional<nvidia::isaac::common::transform::SE3TransformD> data_converter_utils::
GetRelativeTransformFromTFBuffer(
  const tf2_ros::Buffer & tf_buffer, const std::string & source_frame,
  const std::string & target_frame)
{
  try {
    geometry_msgs::msg::TransformStamped
      source_to_target = tf_buffer.lookupTransform(target_frame, source_frame, tf2::TimePointZero);
    return ROSTransformToSE3Transform(source_to_target.transform);
  } catch (tf2::TransformException & ex) {
    return {};
  }
}

// Convert ROS Header timestamp to timestamp seconds
double data_converter_utils::ROSTimestampToSeconds(
  const builtin_interfaces::msg::Time & stamp)
{
  rclcpp::Time timestamp(stamp.sec, stamp.nanosec);

  // Convert the timestamp to microseconds
  return static_cast<double>(timestamp.nanoseconds()) / 1e9;
}

// Convert ROS Header timestamp to timestamp microseconds
uint64_t data_converter_utils::ROSTimestampToMicroseconds(
  const builtin_interfaces::msg::Time & stamp)
{
  rclcpp::Time timestamp(stamp.sec, stamp.nanosec);

  // Convert the timestamp to microseconds
  return static_cast<uint64_t>(timestamp.nanoseconds()) / 1000;
}

uint64_t data_converter_utils::ROSTimestampToNanoseconds(
  const builtin_interfaces::msg::Time & stamp)
{
  rclcpp::Time timestamp(stamp.sec, stamp.nanosec);

  // Convert the timestamp to nanoseconds
  return timestamp.nanoseconds();
}

std::map<uint64_t, nvidia::isaac::common::transform::SE3TransformD>
data_converter_utils::GetPosesForTimestamps(
  const nvidia::isaac::common::transform::SE3PoseLinearInterpolator &
  pose_interpolator,
  const std::vector<uint64_t> & timestamps)
{
  std::map<uint64_t, nvidia::isaac::common::transform::SE3TransformD>
  frames_has_pose;
  for (size_t i = 0; i < timestamps.size(); ++i) {
    auto pose = pose_interpolator.GetInterpolatedPose(timestamps[i]);
    if (pose) {
      frames_has_pose[i] = pose.value();
    }
  }

  return frames_has_pose;
}


bool data_converter_utils::GetFrameSyncAndPoseMap(
  const nvidia::isaac::common::transform::SE3PoseLinearInterpolator &
  pose_interpolator,
  const std::vector<uint64_t> & all_timestamps_nanoseconds,
  const std::vector<uint64_t> & synced_timestamps_nanoseconds,
  std::map<uint64_t, uint64_t> & sample_id_to_synced_sample_id,
  std::map<uint64_t, nvidia::isaac::common::transform::SE3TransformD> & sample_id_to_pose)
{

  if (all_timestamps_nanoseconds.empty()) {
    LOG(ERROR) << "Got empty all_timestamps_nanoseconds";
    return false;
  }

  if (synced_timestamps_nanoseconds.empty()) {
    LOG(ERROR) << "Got empty synced_timestamps_nanoseconds";
    return false;
  }

  if (!pose_interpolator.timestamps().empty()) {
    if (pose_interpolator.timestamps().front() > all_timestamps_nanoseconds.back() / 1000) {
      LOG(ERROR) << "Front pose timestamp " << pose_interpolator.timestamps().front() <<
        " is larger than back of the input timestamps: " <<
        all_timestamps_nanoseconds.back() / 1000;
      return false;
    }

    if (pose_interpolator.timestamps().back() < all_timestamps_nanoseconds.front() / 1000) {
      LOG(ERROR) << "Back pose timestamp " << pose_interpolator.timestamps().back() <<
        " is smaller than front of the input timestamps: " <<
        all_timestamps_nanoseconds.front() / 1000;
      return false;
    }
  }

  // for timestamp it's actually faster to use std::map as it's sorted
  std::map<uint64_t, size_t> timestamp_to_sample_id;
  for (size_t i = 0; i < all_timestamps_nanoseconds.size(); ++i) {
    timestamp_to_sample_id[all_timestamps_nanoseconds[i]] = i;
  }

  for (size_t i = 0; i < synced_timestamps_nanoseconds.size(); ++i) {
    uint64_t synced_timestamp = synced_timestamps_nanoseconds[i];
    if (synced_timestamp == 0) {
      continue;
    }

    auto iter = timestamp_to_sample_id.find(synced_timestamp);
    if (iter == timestamp_to_sample_id.end()) {
      LOG(ERROR) << "Can't find this timestamp: " << synced_timestamp;
      continue;
    }

    // synced sample id is 1 based, 0 means invalid
    sample_id_to_synced_sample_id[iter->second] = i + 1;

    if (!pose_interpolator.timestamps().empty()) {
      auto pose = pose_interpolator.GetInterpolatedPose(synced_timestamp / 1000);
      if (!pose) {
        continue;
      }
      sample_id_to_pose[iter->second] = pose.value();
    }
  }

  return true;
}

std::map<std::string, CameraMetadata>
data_converter_utils::ExtractCameraMetadata(
  const std::string & sensor_data_bag_file,
  const std::string & output_folder,
  const std::string & base_link_name,
  const std::string & topic_yaml_file,
  bool do_rectify_images)
{
  nvidia::isaac::common::datetime::ScopedTimer timer("ExtractCameraMetadata");

  rosbag2_cpp::Reader reader;
  reader.open(sensor_data_bag_file);

  CHECK(reader.has_next()) << "Sensor data bag file: " << sensor_data_bag_file <<
    " does not have any message";

  if (VLOG_IS_ON(1)) {
    PrintBagMetadata(reader.get_metadata());
  }

  const std::map<std::string, std::string> topic_name_to_message_type =
    GetTopicNameToMessageTypeMap(reader.get_metadata());

  // Add debug logging to show all available topics
  if (!topic_yaml_file.empty()) {
    LOG(INFO) << "Using YAML config file: " << topic_yaml_file;
    LOG(INFO) << "Available topics in bag:";
    for (const auto & [topic_name, message_type] : topic_name_to_message_type) {
      LOG(INFO) << "  Topic: " << topic_name << " -> Type: " << message_type;
    }
  }

  CHECK(!topic_name_to_message_type.empty()) <<
    "Does not have any topic in bag: " << sensor_data_bag_file;

  std::map<std::string, CameraTopicConfig> camera_info_topic_to_config;
  if (!topic_yaml_file.empty()) {
    if (!ReadCameraTopicConfig(topic_yaml_file, camera_info_topic_to_config)) {
      LOG(ERROR) << "Failed to read topic yaml file: " << topic_yaml_file;
      return {};
    }
    if (camera_info_topic_to_config.empty()) {
      LOG(ERROR) << "Didn't get any camera info topic config in file: " << topic_yaml_file;
      return {};
    }
  }

  // If no config file provided, auto-generate config from bag topics
  if (camera_info_topic_to_config.empty()) {
    LOG(INFO) << "No camera info topic config, auto-generating from bag topic names";
    camera_info_topic_to_config = GetTopicConfigByTopicName(topic_name_to_message_type);
  }


  std::map<std::string,
    protos::common::sensor::CameraSensor> camera_name_to_camera_params = ExtractCameraSensors(
    sensor_data_bag_file, base_link_name, camera_info_topic_to_config, do_rectify_images);
  if (camera_name_to_camera_params.empty()) {
    LOG(ERROR) <<
      "Didn't get any camera params, please double check if you pass correct rosbag";
    return {};
  }


  std::map<std::string, std::string> image_topic_to_camera_name;
  std::map<std::string, CameraMetadata> camera_name_to_metadata;
  for (const auto & [camera_topic, config] : camera_info_topic_to_config) {
    const std::string camera_name = config.name;
    if (topic_name_to_message_type.count(config.image_topic) == 0) {
      LOG(ERROR) << "Can't find this topic in bag: " << config.image_topic;
      // continue;
      return {};
    }

    // Add debug logging to see what message type is detected
    const std::string detected_message_type = topic_name_to_message_type.at(config.image_topic);
    LOG(INFO) << "Camera " << camera_name << " image topic: " << config.image_topic
              << " has message type: " << detected_message_type;

    // Check if the message type is supported
    if (detected_message_type != kCompressedImageMessageType &&
      detected_message_type != kRawImageMessageType)
    {
      LOG(ERROR) << "Unsupported message type for image topic " << config.image_topic
                 << ": " << detected_message_type
                 << ". Expected: " << kRawImageMessageType << " or " << kCompressedImageMessageType
                 << " (or older format: sensor_msgs/Image or sensor_msgs/CompressedImage)";
      return {};
    }

    if (!camera_name_to_camera_params.count(camera_name)) {
      LOG(ERROR) << "Can't find camera params for: " << camera_name;
      continue;
    }

    auto & metadata = camera_name_to_metadata[camera_name];
    metadata.set_camera_topic_name(config.image_topic);
    metadata.set_message_type(topic_name_to_message_type.at(config.image_topic));
    metadata.set_paired_camera_name(config.paired_camera_name);
    metadata.set_is_depth_image(config.is_depth_image);
    metadata.set_is_camera_rectified(config.is_camera_rectified);
    metadata.set_swap_rb_channels(config.swap_rb_channels);
    camera_name_to_metadata[camera_name].set_camera_params(
      camera_name_to_camera_params.at(
        camera_name));
    image_topic_to_camera_name[config.image_topic] = camera_name;
  }


  CHECK(!camera_name_to_metadata.empty()) <<
    "Didn't find any camera topics, please check if you passed the right "
    "sensor data bag, or maybe you passed pose bag as sensor bag?";

  std::unordered_map<std::string, uint64_t> per_camera_latest_timestamp;
  while (reader.has_next()) {
    rosbag2_storage::SerializedBagMessageSharedPtr msg = reader.read_next();
    const std::string message_type =
      topic_name_to_message_type.at(msg->topic_name);

    if (image_topic_to_camera_name.count(msg->topic_name) == 0) {
      continue;
    }

    if (message_type != kCompressedImageMessageType &&
      message_type != kRawImageMessageType)
    {
      LOG(ERROR) << "Got a unsupported image message type: " << message_type;
      continue;
    }

    const std::string camera_name = image_topic_to_camera_name[msg->topic_name];
    CHECK(camera_name_to_metadata.count(camera_name)) <<
      "Does not found this camera: " << camera_name;

    // we use get parent directory to get the camera name
    if (camera_name.empty()) {
      LOG(ERROR) << "Got empty camera name: " << msg->topic_name;
      continue;
    }

    rclcpp::SerializedMessage serialized_msg(*msg->serialized_data);
    uint64_t current_time_nanoseconds = 0;
    // TODO(dizeng) in theory camera info and camera message should have the same timestamp,
    // so we could decode all camera info once to get timestamp instead.
    // however nova data recorder does not guarantee this, so we have to decode each message
    // to get timestamp
    if (message_type == kCompressedImageMessageType) {
      sensor_msgs::msg::CompressedImage image;
      rclcpp::Serialization<sensor_msgs::msg::CompressedImage>()
      .deserialize_message(&serialized_msg, &image);
      current_time_nanoseconds = ROSTimestampToNanoseconds(image.header.stamp);
    } else if (message_type == kRawImageMessageType) {
      sensor_msgs::msg::Image image;
      rclcpp::Serialization<sensor_msgs::msg::Image>()
      .deserialize_message(&serialized_msg, &image);
      current_time_nanoseconds = ROSTimestampToNanoseconds(image.header.stamp);
    } else {
      LOG(ERROR) << "Unknown message type: " << message_type;
      continue;
    }

    if (current_time_nanoseconds == 0) {
      LOG(ERROR) << "Got timestamp 0 in topic: " << msg->topic_name;
    }

    // TODO(dizeng) this should be moved to internal function fixes
    // Make sure the timestamp is increasing monotonically
    auto iter = per_camera_latest_timestamp.find(camera_name);
    if (iter != per_camera_latest_timestamp.end() && (iter->second >= current_time_nanoseconds)) {
      LOG(WARNING) <<
        "Camera per_camera_latest_timestamp not increasing. Using previous time + 1 for this frame";
      current_time_nanoseconds = iter->second + 1;
    }
    per_camera_latest_timestamp[camera_name] = current_time_nanoseconds;

    camera_name_to_metadata[camera_name].AddTimestamp(
      current_time_nanoseconds);
  }

  // Done with rosbag reading
  reader.close();

  return camera_name_to_metadata;
}

std::map<std::string, protos::common::sensor::CameraSensor>
data_converter_utils::ExtractCameraSensors(
  const std::string & sensor_data_bag,
  const std::string & base_link_name,
  std::map<std::string,
  data_converter_utils::CameraTopicConfig> & camera_info_topic_to_config,
  bool do_rectify_images)
{

  auto clock = std::make_shared<rclcpp::Clock>(RCL_SYSTEM_TIME);
  tf2_ros::Buffer tf_buffer(clock);

  if (!AddStaticTransformToBuffer(
      sensor_data_bag, tf_buffer))
  {
    LOG(ERROR) << "Failed to read tf static transform";
    return {};
  }

  return ExtractCameraSensors(
    tf_buffer, sensor_data_bag, base_link_name,
    camera_info_topic_to_config, do_rectify_images);
}

std::map<std::string,
  data_converter_utils::CameraTopicConfig> data_converter_utils::GetTopicConfigByTopicName(
  const std::map<std::string, std::string> & topic_name_to_message_type)
{
  std::map<std::string,
    data_converter_utils::CameraTopicConfig> camera_info_topic_to_config;

  // Step 1: Parse all topics once to extract camera_name_ and sub_camera_name_
  std::vector<TopicInfo> all_topics = ParseAllTopics(topic_name_to_message_type);

  // Step 2: Find all camera info topics and process them
  for (const auto & topic_info : all_topics) {
    if (topic_info.message_type_ != kCameraInfoMessageType) {
      continue;
    }

    // Skip topics with empty camera_name_ or sub_camera_name_
    if (topic_info.camera_name_.empty() || topic_info.sub_camera_name_.empty()) {
      LOG(WARNING) << "Skipping camera info topic with incomplete tokens: " <<
        topic_info.full_topic_
                   << " (camera_name: '" << topic_info.camera_name_
                   << "', sub_camera_name: '" << topic_info.sub_camera_name_ << "')";
      continue;
    }

    std::string camera_name = TopicNameToCameraName(topic_info.full_topic_);
    if (camera_name.empty()) {
      LOG(ERROR) << "Failed to get camera name from topic name: " << topic_info.full_topic_;
      continue;
    }

    CameraTopicConfig camera_config;
    camera_config.camera_info_topic = topic_info.full_topic_;

    // Step 3: Use pre-computed sensor type information from topic_info
    std::string base_camera_name = topic_info.camera_name_;
    bool is_left_camera = topic_info.is_left_camera;
    bool is_right_camera = topic_info.is_right_camera;
    bool is_rectified = topic_info.is_rectified;
    bool is_depth_camera = topic_info.is_depth_camera;
    bool is_rgb_camera = topic_info.is_rgb_camera;

    // Set camera properties based on pre-computed detection
    camera_config.is_depth_image = is_depth_camera;
    camera_config.is_camera_rectified = is_rectified;

    // Set camera name with appropriate suffix
    std::string final_camera_name = base_camera_name;

    if (is_left_camera) {
      final_camera_name += "_left";
    } else if (is_right_camera) {
      final_camera_name += "_right";
    }

    if (is_depth_camera && final_camera_name.find("_depth") == std::string::npos) {
      final_camera_name += "_depth";
    } else if (is_rgb_camera && !is_depth_camera &&
      final_camera_name.find("_color") == std::string::npos)
    {
      final_camera_name += "_color";
    }

    camera_config.name = final_camera_name;

    // Step 8: Find matching image topic using pre-parsed topic information
    std::string image_topic_found;

    // Look for image topics that match the first two tokens (camera_name_ and sub_camera_name_)
    for (const auto & candidate_topic : all_topics) {
      // Only consider Image or CompressedImage message types
      if (candidate_topic.message_type_ != kRawImageMessageType &&
        candidate_topic.message_type_ != kCompressedImageMessageType)
      {
        continue;
      }

      // Check if first two tokens match
      if (!candidate_topic.camera_name_.empty() && !candidate_topic.sub_camera_name_.empty() &&
        candidate_topic.camera_name_ == topic_info.camera_name_ &&
        candidate_topic.sub_camera_name_ == topic_info.sub_camera_name_)
      {

        // For depth cameras, prefer topics with "depth" in the name
        if (is_depth_camera) {
          if (candidate_topic.is_depth_camera) {
            image_topic_found = candidate_topic.full_topic_;
            break;  // Found depth image topic, use it
          } else if (image_topic_found.empty()) {
            image_topic_found = candidate_topic.full_topic_;  // Keep as fallback
          }
        } else {
          // For RGB/color cameras, prefer topics with "image", "color", or "rgb"
          std::string candidate_topic_lower = candidate_topic.full_topic_;
          std::transform(
            candidate_topic_lower.begin(), candidate_topic_lower.end(),
            candidate_topic_lower.begin(), ::tolower);

          if (candidate_topic_lower.find("image") != std::string::npos ||
            candidate_topic.is_rgb_camera)
          {
            image_topic_found = candidate_topic.full_topic_;
            break;  // Found color/RGB image topic, use it
          } else if (image_topic_found.empty()) {
            image_topic_found = candidate_topic.full_topic_;  // Keep as fallback
          }
        }
      }
    }

    if (!image_topic_found.empty()) {
      camera_config.image_topic = image_topic_found;
      LOG(INFO) << "Found matching image topic: " << image_topic_found
                << " for camera info: " << topic_info.full_topic_;
    } else {
      LOG(ERROR) << "No matching image topic found for camera info: " << topic_info.full_topic_
                 << " (camera_name: " << topic_info.camera_name_
                 << ", sub_camera_name: " << topic_info.sub_camera_name_ << ")";
      continue;  // Skip this camera info topic since we can't find its image topic
    }

    // Step 9: Handle stereo camera pairing using camera_name_
    if (is_left_camera) {
      // Look for corresponding right camera using the same base camera_name_
      std::string right_camera_name = base_camera_name;
      if (is_depth_camera) {
        right_camera_name += "_right_depth";
      } else if (is_rgb_camera) {
        right_camera_name += "_right_color";
      } else {
        right_camera_name += "_right";
      }
      camera_config.paired_camera_name = right_camera_name;
      LOG(INFO) << "Add " << final_camera_name << " and " << right_camera_name <<
        " to a stereo pair";
    }

    camera_info_topic_to_config[camera_config.camera_info_topic] = camera_config;

    std::string camera_type_str = "";
    if (is_depth_camera) {camera_type_str += "depth ";}
    if (is_rgb_camera) {camera_type_str += "RGB ";}
    if (is_rectified) {camera_type_str += "rectified ";}
    if (is_left_camera) {camera_type_str += "left ";}
    if (is_right_camera) {camera_type_str += "right ";}
    if (camera_type_str.empty()) {camera_type_str = "generic ";}

    LOG(INFO) << "Add camera info topic: " << camera_config.camera_info_topic <<
      " camera image topic: " << camera_config.image_topic <<
      " (camera: " << camera_config.name << ", type: " << camera_type_str << "camera)";
  }

  return camera_info_topic_to_config;
}

bool data_converter_utils::ReadCameraTopicConfig(
  const std::string & topic_config_file, std::map<std::string,
  data_converter_utils::CameraTopicConfig> & camera_info_topic_to_config)
{
  try {
    // Load the YAML file
    YAML::Node config = YAML::LoadFile(topic_config_file);

    // Access data from the YAML file
    if (config["depth_cameras"]) {
      for (const auto & camera : config["depth_cameras"]) {
        if (!camera["name"]) {
          LOG(ERROR) << "No camera name for camera: " << camera.as<std::string>();
          continue;
        }

        if (camera["depth"]) {
          if (!camera["color"]) {
            LOG(ERROR) << "Expecting the color topic for depth camera: " <<
              camera["name"].as<std::string>();
            continue;
          }

          CameraTopicConfig depth_camera_config;
          depth_camera_config.name = camera["name"].as<std::string>() + "_depth";
          depth_camera_config.camera_info_topic = camera["depth_camera_info"].as<std::string>();
          depth_camera_config.image_topic = camera["depth"].as<std::string>();
          depth_camera_config.is_depth_image = true;
          if (camera["depth_frame_id"]) {
            depth_camera_config.frame_id_name = camera["depth_frame_id"].as<std::string>();
          }
          if (camera["is_camera_rectified"]) {
            depth_camera_config.is_camera_rectified = camera["is_camera_rectified"].as<bool>();
          }
          if (camera["swap_rb_channels"]) {
            depth_camera_config.swap_rb_channels = camera["swap_rb_channels"].as<bool>();
          }


          CameraTopicConfig color_camera_config;
          color_camera_config.name = camera["name"].as<std::string>() + "_color";
          color_camera_config.camera_info_topic = camera["color_camera_info"].as<std::string>();
          color_camera_config.image_topic = camera["color"].as<std::string>();
          color_camera_config.is_depth_image = false;
          if (camera["color_frame_id"]) {
            color_camera_config.frame_id_name = camera["color_frame_id"].as<std::string>();
          }
          if (camera["is_camera_rectified"]) {
            color_camera_config.is_camera_rectified = camera["is_camera_rectified"].as<bool>();
          }
          if (camera["swap_rb_channels"]) {
            color_camera_config.swap_rb_channels = camera["swap_rb_channels"].as<bool>();
          }

          // pair depth camera with color camera
          depth_camera_config.paired_camera_name = color_camera_config.name;

          camera_info_topic_to_config[depth_camera_config.camera_info_topic] =
            depth_camera_config;
          camera_info_topic_to_config[color_camera_config.camera_info_topic] =
            color_camera_config;
        }
      }
    }

    if (config["stereo_cameras"]) {
      for (const auto & camera : config["stereo_cameras"]) {
        if (!camera["name"]) {
          LOG(ERROR) << "No camera name for camera: " << camera.as<std::string>();
          continue;
        }

        if (camera["left"]) {
          if (!camera["right"]) {
            LOG(ERROR) << "Expecting the color topic for depth camera: " <<
              camera["name"].as<std::string>();
            continue;
          }

          CameraTopicConfig left_camera;
          left_camera.name = camera["name"].as<std::string>() + "_left";
          left_camera.camera_info_topic = camera["left_camera_info"].as<std::string>();
          left_camera.image_topic = camera["left"].as<std::string>();
          left_camera.is_depth_image = false;
          if (camera["left_frame_id"]) {
            left_camera.frame_id_name = camera["left_frame_id"].as<std::string>();
          }
          if (camera["is_camera_rectified"]) {
            left_camera.is_camera_rectified = camera["is_camera_rectified"].as<bool>();
          }
          if (camera["swap_rb_channels"]) {
            left_camera.swap_rb_channels = camera["swap_rb_channels"].as<bool>();
          }

          CameraTopicConfig right_camera;
          right_camera.name = camera["name"].as<std::string>() + "_right";
          right_camera.camera_info_topic = camera["right_camera_info"].as<std::string>();
          right_camera.image_topic = camera["right"].as<std::string>();
          right_camera.is_depth_image = false;
          if (camera["right_frame_id"]) {
            right_camera.frame_id_name = camera["right_frame_id"].as<std::string>();
          }
          if (camera["is_camera_rectified"]) {
            right_camera.is_camera_rectified = camera["is_camera_rectified"].as<bool>();
          }
          if (camera["swap_rb_channels"]) {
            right_camera.swap_rb_channels = camera["swap_rb_channels"].as<bool>();
          }

          // pair depth camera with color camera
          left_camera.paired_camera_name = right_camera.name;

          camera_info_topic_to_config[left_camera.camera_info_topic] =
            left_camera;
          camera_info_topic_to_config[right_camera.camera_info_topic] =
            right_camera;
        }
      }
    }

    if (camera_info_topic_to_config.empty()) {
      LOG(ERROR) << "No valid camera specified in file: " << topic_config_file;
      return false;
    }

  } catch (const YAML::Exception & e) {
    LOG(ERROR) << "Error parsing YAML file: " << e.what();
    return false;
  }

  return true;
}

std::map<std::string, protos::common::sensor::CameraSensor>
data_converter_utils::ExtractCameraSensors(
  const tf2_ros::Buffer & tf_buffer,
  const std::string & sensor_data_bag,
  const std::string & base_link_name,
  std::map<std::string,
  data_converter_utils::CameraTopicConfig> & camera_info_topic_to_config,
  bool do_rectify_images)
{

  rosbag2_cpp::Reader reader;
  reader.open(sensor_data_bag);

  CHECK(reader.has_next()) << "Sensor data bag file: " << sensor_data_bag <<
    " does not have any message";

  if (VLOG_IS_ON(1)) {
    PrintBagMetadata(reader.get_metadata());
  }

  const std::map<std::string, std::string> topic_name_to_message_type =
    GetTopicNameToMessageTypeMap(reader.get_metadata());

  if (camera_info_topic_to_config.empty()) {
    LOG(INFO) << "No camera info topic to config mapping, try to infer it from bag topic name";
    camera_info_topic_to_config = GetTopicConfigByTopicName(topic_name_to_message_type);
  }

  if (camera_info_topic_to_config.empty()) {
    LOG(ERROR) << "No camera info topic to config mapping";
    return {};
  }

  CHECK(!topic_name_to_message_type.empty()) <<
    "Does not have any topic in bag: " << sensor_data_bag;

  std::map<std::string, protos::common::sensor::CameraSensor> camera_name_to_sensors;
  std::set<std::string> camera_info_topics_to_find;
  for (const auto & [camera_topic, config] : camera_info_topic_to_config) {
    camera_info_topics_to_find.insert(config.camera_info_topic);
  }

  while (reader.has_next()) {
    rosbag2_storage::SerializedBagMessageSharedPtr msg = reader.read_next();
    const std::string message_type =
      topic_name_to_message_type.at(msg->topic_name);

    if (message_type != kCameraInfoMessageType) {
      continue;
    }

    if (camera_info_topic_to_config.count(msg->topic_name) == 0) {
      continue;
    }

    const auto & camera_config = camera_info_topic_to_config.at(msg->topic_name);

    const std::string & camera_name = camera_config.name;
    if (camera_name_to_sensors.count(camera_name)) {
      continue;
    }

    camera_info_topics_to_find.erase(msg->topic_name);

    protos::common::sensor::CameraSensor camera_sensor;

    rclcpp::SerializedMessage serialized_msg(*msg->serialized_data);
    sensor_msgs::msg::CameraInfo camera_info;
    rclcpp::Serialization<sensor_msgs::msg::CameraInfo>().deserialize_message(
      &serialized_msg, &camera_info);
    // get the transform from urdf
    std::string camera_frame_id =
      camera_config.frame_id_name.empty() ? camera_info.header.frame_id : camera_config.
      frame_id_name;

    nvidia::isaac::common::transform::SE3TransformD camera_to_base_link_transform;

    if (base_link_name.empty()) {
      // If base_link_name is not provided, use identity transform and print warning
      LOG(WARNING) << "base_link_name not provided for camera " << camera_name
                   << ", using identity transform (camera frame assumed to be vehicle frame)";
      camera_to_base_link_transform = nvidia::isaac::common::transform::SE3TransformD();
    } else {
      // Try to get transform from TF buffer
      auto transform_opt = GetRelativeTransformFromTFBuffer(
        tf_buffer, camera_frame_id, base_link_name);

      if (!transform_opt) {
        // if it's back stereo camera we replace it to rear
        if (camera_name.find("back_stereo_camera") != std::string::npos) {
          std::string new_camera_name = camera_frame_id;
          ReplaceFirst(new_camera_name, "back", "rear");
          transform_opt = GetRelativeTransformFromTFBuffer(
            tf_buffer, new_camera_name, base_link_name);
        }
      }

      if (!transform_opt) {
        LOG(ERROR) <<
          "Can't find " << camera_frame_id << " to " << base_link_name << " transform in TF";
        return {};
      }

      camera_to_base_link_transform = *transform_opt;
    }

    // If do_rectify_images is true, the output images will be rectified regardless of input state
    bool output_will_be_rectified = do_rectify_images || camera_config.is_camera_rectified;
    camera_name_to_sensors[camera_name] = ConvertCameraInfoToSensor(
      camera_info, camera_name, camera_to_base_link_transform, output_will_be_rectified);
  }

  // Done with rosbag reading
  reader.close();

  if (!camera_info_topics_to_find.empty()) {
    for (const auto & topic : camera_info_topics_to_find) {
      LOG(ERROR) << "Can't find camera info topic: " << topic;
    }
    return {};
  }

  // update sensor id based on order
  uint32_t sensor_id = 0;
  for (auto & camera_to_sensor : camera_name_to_sensors) {
    camera_to_sensor.second.mutable_sensor_meta_data()->set_sensor_id(
      sensor_id);
    ++sensor_id;
  }

  return camera_name_to_sensors;
}
struct CameraProcessingData
{
  std::unique_ptr<VideoDecoder> decoder;
  std::unique_ptr<nvidia::isaac::common::image::ImageRectifier> rectifier;
  std::atomic<size_t> message_count{0};
  std::atomic<size_t> decoded_count{0};
  std::atomic<size_t> write_frame_count{0};

  std::queue<rosbag2_storage::SerializedBagMessageSharedPtr> frame_queue;
  std::mutex queue_mutex;
  std::condition_variable queue_cv;
  std::thread decoding_thread;
  bool stop_decoding{false};

  const size_t max_queue_size = 10;
  std::condition_variable queue_not_full_cv;
};

void DecodingThread(
  CameraProcessingData & processing_data,
  const std::string & camera_name,
  const CameraMetadata & metadata,
  bool rectify_images,
  bool swap_rb_channels,
  const data_converter_utils::WritImageFunctor & write_image_functor, bool dry_run)
{
  while (true) {
    rosbag2_storage::SerializedBagMessageSharedPtr msg;
    {
      std::unique_lock<std::mutex> lock(processing_data.queue_mutex);
      processing_data.queue_cv.wait(
        lock, [&]() {
          return !processing_data.frame_queue.empty() || processing_data.stop_decoding;
        });

      if (processing_data.stop_decoding && processing_data.frame_queue.empty()) {
        break;
      }

      msg = processing_data.frame_queue.front();
      processing_data.frame_queue.pop();
      processing_data.queue_not_full_cv.notify_one();
    }

    const std::set<uint64_t> & selected_frames = metadata.get_selected_frames();

    size_t sample_id = processing_data.message_count;
    processing_data.message_count++;

    rclcpp::SerializedMessage serialized_msg(*msg->serialized_data);

    data_converter_utils::ImageFrame image_frame;

    if (metadata.message_type() == kCompressedImageMessageType) {
      sensor_msgs::msg::CompressedImage image;
      {
        nvidia::isaac::common::datetime::ScopedTimer timer("DeserializeMessage");
        rclcpp::Serialization<sensor_msgs::msg::CompressedImage>()
        .deserialize_message(&serialized_msg, &image);
      }

      if (!dry_run) {
        // Check if the image format contains JPEG/JPG - support both common variants
        // JPEG images can be decoded directly with OpenCV, avoiding the need for codec_name
        std::string format_lower = image.format;
        std::transform(format_lower.begin(), format_lower.end(), format_lower.begin(), ::tolower);
        bool is_jpeg_format = (format_lower.find("jpeg") != std::string::npos ||
          format_lower.find("jpg") != std::string::npos);

        if (is_jpeg_format) {
          // Use OpenCV to decode JPEG images directly
          nvidia::isaac::common::datetime::ScopedTimer timer("DecodeJPEG");
          cv::Mat jpeg_image = cv::imdecode(image.data, cv::IMREAD_COLOR);
          if (jpeg_image.empty()) {
            LOG(WARNING) << "Failed to decode JPEG image for camera: " << camera_name
                         << " (format: " << image.format << ")";
            continue;
          }

          image_frame.image = jpeg_image;
          LOG_EVERY_N(INFO, 100) << "Decoded JPEG image for camera: " << camera_name
                                 << " (format: " << image.format << ")";
        } else {
          // Use VideoDecoder for other formats (e.g., H.264)
          if (!processing_data.decoder) {
            LOG(ERROR) << "Decoder is not initialized for format '" << image.format
                       << "', please pass --codec_name for non-JPEG formats";
            continue;
          }

          nvidia::isaac::common::datetime::ScopedTimer timer("DecodePacket");
          std::vector<cv::Mat> frames;
          if (!processing_data.decoder->DecodePacket(image.data, frames) || frames.empty()) {
            LOG(WARNING) << "Failed to decode frames for format '" << image.format
                         << "', No frames decoded";
            continue;
          }

          if (frames.size() > 1) {
            LOG(ERROR) << "total frames decoded from one message is larger than 1";
            continue;
          }

          image_frame.image = frames[0];
          LOG_EVERY_N(INFO, 100) << "Decoded " << image.format << " image for camera: "
                                 << camera_name;
        }
      }

      image_frame.timestamp_nanoseconds = data_converter_utils::ROSTimestampToNanoseconds(
        image.header.stamp);
      processing_data.decoded_count++;

      // for video decoding, we can only skip this frame after we decoded it
      if (!selected_frames.empty() && !selected_frames.count(sample_id)) {
        continue;
      }

    } else if (metadata.message_type() == kRawImageMessageType) {
      // for raw images, we can skip converting this frame if it's not selected
      if (!selected_frames.empty() && !selected_frames.count(sample_id)) {
        continue;
      }

      sensor_msgs::msg::Image image;
      {
        nvidia::isaac::common::datetime::ScopedTimer timer("DeserializeMessage");
        rclcpp::Serialization<sensor_msgs::msg::Image>()
        .deserialize_message(&serialized_msg, &image);
      }

      if (!dry_run) {
        nvidia::isaac::common::datetime::ScopedTimer timer("ConvertImageToCvMat");
        image_frame.image = cv_bridge::toCvCopy(image, image.encoding)->image;
        // Convert RGB to BGR for OpenCV compatibility (only for color images)
        if (image.encoding == "rgb8") {
          LOG_EVERY_N(INFO, 100) << "Converting RGB to BGR for camera: " << camera_name;
          cv::cvtColor(image_frame.image, image_frame.image, cv::COLOR_RGB2BGR);
        }
      }

      image_frame.timestamp_nanoseconds = data_converter_utils::ROSTimestampToNanoseconds(
        image.header.stamp);
      processing_data.decoded_count++;
    } else {
      LOG(ERROR) << "Unknown message type: " << metadata.message_type();
      continue;
    }


    // Check if timestamp is the same as the one in metadata
    uint64_t metadata_timestamp = metadata.timestamp_nanoseconds()[sample_id];
    if (image_frame.timestamp_nanoseconds != metadata_timestamp) {
      LOG(WARNING) << "Timestamp of message changed from metadata, msg says"
                   << image_frame.timestamp_nanoseconds << " metadata says:" << metadata_timestamp;
    }


    const std::map<uint64_t, nvidia::isaac::common::transform::SE3TransformD> &
    vehicle_to_world_poses = metadata.get_vehicle_to_world_poses();
    bool populate_pose = !vehicle_to_world_poses.empty();

    if (populate_pose && !vehicle_to_world_poses.count(sample_id)) {
      LOG(ERROR) << "Can't find poses of this frame: " << sample_id <<
        " for camera: " << camera_name;
      continue;
    }


    if (populate_pose) {
      image_frame.camera_to_world =
        vehicle_to_world_poses.at(sample_id) * metadata.camera_to_vehicle();
      image_frame.has_camera_to_world = true;
    }

    // Force Swap R and B channels if needed (RGB <-> BGR)
    if (!dry_run && swap_rb_channels && !image_frame.image.empty()) {
      nvidia::isaac::common::datetime::ScopedTimer timer("swap_rb_channels");
      cv::cvtColor(image_frame.image, image_frame.image, cv::COLOR_RGB2BGR);
    }

    if (!dry_run && rectify_images && processing_data.rectifier) {
      nvidia::isaac::common::datetime::ScopedTimer timer("rectify");
      cv::Mat rectified_image;
      processing_data.rectifier->Rectify(image_frame.image, rectified_image);
      image_frame.image = rectified_image;
    }

    image_frame.camera_name = camera_name;
    image_frame.camera_sensor_id = metadata.get_camera_sensor_id();
    image_frame.sample_id = sample_id;
    image_frame.is_rectified = rectify_images;
    image_frame.is_depth_image = metadata.is_depth_image();

    const std::map<uint64_t, uint64_t> & sample_id_to_synced_sample_id =
      metadata.get_sample_id_to_synced_sample_id();
    auto synced_sample_id_iter = sample_id_to_synced_sample_id.find(sample_id);
    if (synced_sample_id_iter != sample_id_to_synced_sample_id.end()) {
      image_frame.synced_sample_id = synced_sample_id_iter->second;
    } else {
      // 0 synced sample id means unsynced
      image_frame.synced_sample_id = 0;
    }

    if (!write_image_functor(image_frame)) {
      LOG(ERROR) << "Failed to write image: " << sample_id <<
        " for camera: " << camera_name;
      continue;
    }

    size_t write_frame_count = ++processing_data.write_frame_count;
    if (write_frame_count % 500 == 0) {
      LOG(INFO) << "For camera " << camera_name << ": Write: " << write_frame_count
                << " frames, decoded: " << processing_data.decoded_count
                << " frames";
    }
  }
}

bool data_converter_utils::ExtractCameraImagesFromRosbag(
  const std::string & sensor_data_bag,
  std::map<std::string, CameraMetadata> & camera_to_metadata,
  const data_converter_utils::WritImageFunctor write_image_functor,
  const std::string & codec_name,
  bool dry_run,
  bool rectify_images)
{
  nvidia::isaac::common::datetime::ScopedTimer timer("ExtractCameraImagesFromRosbag");

  rosbag2_cpp::Reader reader;
  reader.open(sensor_data_bag);

  std::map<std::string, CameraProcessingData> camera_name_to_processing_data;
  std::map<std::string, std::string> camera_topics_to_camera_name;

  // Initialize CameraProcessingData for each camera
  for (auto & [camera_name, metadata] : camera_to_metadata) {
    auto & processing_data = camera_name_to_processing_data[camera_name];

    if (!dry_run) {
      if (!codec_name.empty()) {
        processing_data.decoder = std::make_unique<VideoDecoder>();
        CHECK(processing_data.decoder->Init(codec_name)) <<
          "Failed to initialize decoder for camera: " <<
          camera_name;
      }

      if (rectify_images && !metadata.get_is_camera_rectified()) {
        processing_data.rectifier =
          std::make_unique<nvidia::isaac::common::image::ImageRectifier>();
        if (!processing_data.rectifier->Init(
            ConvertCameraParmsProto(
              metadata.get_camera_params().
              calibration_parameters())))
        {
          LOG(ERROR) << "Failed to initialize image rectifier for camera: " << camera_name;
          return false;
        }
      }
    }


    camera_topics_to_camera_name[metadata.camera_topic_name()] = camera_name;

    // Start the decoding thread for this camera
    processing_data.decoding_thread = std::thread(
      DecodingThread,
      std::ref(processing_data), camera_name, std::cref(metadata),
      rectify_images, metadata.get_swap_rb_channels(), std::cref(write_image_functor), dry_run);
  }

  while (reader.has_next()) {
    rosbag2_storage::SerializedBagMessageSharedPtr msg;
    {
      nvidia::isaac::common::datetime::ScopedTimer timer("ReadMessage");
      msg = reader.read_next();
    }

    if (!camera_topics_to_camera_name.count(msg->topic_name)) {
      continue;
    }

    std::string camera_name = camera_topics_to_camera_name[msg->topic_name];
    CHECK(camera_name_to_processing_data.count(camera_name));

    auto & processing_data = camera_name_to_processing_data[camera_name];
    {
      std::unique_lock<std::mutex> lock(processing_data.queue_mutex);
      // Wait until the queue has space
      processing_data.queue_not_full_cv.wait(
        lock, [&]() {
          return processing_data.frame_queue.size() < processing_data.max_queue_size ||
          processing_data.stop_decoding;
        });
      processing_data.frame_queue.push(msg);
    }
    processing_data.queue_cv.notify_one();
  }

  // Stop all decoding threads and wait for them to finish
  for (auto & [camera_name, processing_data] : camera_name_to_processing_data) {
    processing_data.stop_decoding = true;
    processing_data.queue_cv.notify_one();
    if (processing_data.decoding_thread.joinable()) {
      processing_data.decoding_thread.join();
    }
  }

  for (const auto & [camera_name, metadata] : camera_to_metadata) {
    const auto & processing_data = camera_name_to_processing_data.at(camera_name);
    CHECK_EQ(
      processing_data.message_count,
      metadata.timestamp_nanoseconds().size()) <<
      "Frame id and total timestamp does not match for camera: " << camera_name;
    LOG(INFO) << "For camera " << camera_name << ": got message " << processing_data.message_count
              << ", decoded: " << processing_data.decoded_count
              << ", total image written: " << processing_data.write_frame_count;
  }

  return true;
}

void data_converter_utils::FindSyncedTimestamps(
  const std::map<std::string, isaac_ros::isaac_mapping_ros::CameraMetadata> & metadatas,
  std::vector<std::vector<uint64_t>> & synced_timestamps_per_camera,
  const int64_t extrinsic_timestamp_threshold_nanoseconds)
{
  std::vector<std::vector<uint64_t>> all_timestamps;
  for (const auto & video_pair : metadatas) {
    all_timestamps.push_back(video_pair.second.timestamp_nanoseconds());
  }

  nvidia::isaac::visual::FindSyncedTimestamps(
    all_timestamps, synced_timestamps_per_camera,
    extrinsic_timestamp_threshold_nanoseconds);
}

nvidia::isaac::common::transform::SE3PoseLinearInterpolator
data_converter_utils::ExtractPosesFromBag(
  const std::string & pose_bag_file, const std::string & pose_topic_name,
  const std::string & expected_child_frame_id)
{
  rosbag2_cpp::Reader pose_reader;
  pose_reader.open(pose_bag_file);
  nvidia::isaac::common::transform::SE3PoseLinearInterpolator pose_interpolator;

  if (GetTopicMessageCount(pose_reader.get_metadata(), pose_topic_name) == 0) {
    LOG(ERROR) << "No message in topic: " << pose_topic_name;
    return pose_interpolator;
  }

  const std::map<std::string, std::string> topic_name_to_message_type =
    GetTopicNameToMessageTypeMap(pose_reader.get_metadata());

  // read messages
  while (pose_reader.has_next()) {
    rosbag2_storage::SerializedBagMessageSharedPtr msg =
      pose_reader.read_next();
    if (msg->topic_name != pose_topic_name) {
      continue;
    }

    const std::string message_type =
      topic_name_to_message_type.at(msg->topic_name);
    if (poseMessageTypeMap.count(message_type) == 0) {
      LOG(ERROR) << "Unsupported message type: " << message_type;
      return pose_interpolator;
    }

    nvidia::isaac::common::transform::SE3TransformD se3_transform;
    builtin_interfaces::msg::Time stamp;
    auto set_transform = [&](const auto & position, const auto & orientation) {
        se3_transform.set_translation(Eigen::Vector3d(position.x, position.y, position.z));
        se3_transform.set_rotation(
          Eigen::Quaterniond(
            orientation.w, orientation.x, orientation.y,
            orientation.z));
      };
    rclcpp::SerializedMessage serialized_msg(*msg->serialized_data);
    switch (poseMessageTypeMap.at(message_type)) {
      case PoseMessageType::kOdometry: {
          nav_msgs::msg::Odometry odometry_msg;
          rclcpp::Serialization<nav_msgs::msg::Odometry>().deserialize_message(
            &serialized_msg, &odometry_msg);
          if (!expected_child_frame_id.empty() &&
            expected_child_frame_id != odometry_msg.child_frame_id)
          {
            LOG(ERROR) << "Expected child frame id: " << expected_child_frame_id <<
              " but got: " << odometry_msg.child_frame_id;
            break;
          }
          set_transform(odometry_msg.pose.pose.position, odometry_msg.pose.pose.orientation);
          stamp = odometry_msg.header.stamp;
          break;
        }
      case PoseMessageType::kPoseStamped: {
          geometry_msgs::msg::PoseStamped pose_stamped_msg;
          rclcpp::Serialization<geometry_msgs::msg::PoseStamped>().deserialize_message(
            &serialized_msg, &pose_stamped_msg);
          set_transform(pose_stamped_msg.pose.position, pose_stamped_msg.pose.orientation);
          stamp = pose_stamped_msg.header.stamp;
          break;
        }
      case PoseMessageType::kPoseWithCovarianceStamped: {
          geometry_msgs::msg::PoseWithCovarianceStamped pose_with_covariance_stamped_msg;
          rclcpp::Serialization<geometry_msgs::msg::PoseWithCovarianceStamped>().deserialize_message(
            &serialized_msg, &pose_with_covariance_stamped_msg);
          set_transform(
            pose_with_covariance_stamped_msg.pose.pose.position,
            pose_with_covariance_stamped_msg.pose.pose.orientation);
          stamp = pose_with_covariance_stamped_msg.header.stamp;
          break;
        }
      case PoseMessageType::kPath: {
          nav_msgs::msg::Path path_msg;
          rclcpp::Serialization<nav_msgs::msg::Path>().deserialize_message(
            &serialized_msg, &path_msg);
          if (path_msg.poses.empty()) {
            LOG(ERROR) << "Empty path message";
            return pose_interpolator;
          }
          for (const auto pose : path_msg.poses) {
            set_transform(pose.pose.position, pose.pose.orientation);
            stamp = pose.header.stamp;
            pose_interpolator.AddNextPose(ROSTimestampToMicroseconds(stamp), se3_transform);
          }
          break;
        }
      default:
        LOG(ERROR) << "Unsupported message type: " << message_type;
        return pose_interpolator;
    }
    if (poseMessageTypeMap.at(message_type) != PoseMessageType::kPath) {
      pose_interpolator.AddNextPose(ROSTimestampToMicroseconds(stamp), se3_transform);
    }
  }
  return pose_interpolator;
}

bool data_converter_utils::ExtractOdometryMsgFromPoseBag(
  const std::string & pose_bag_file, const std::string & pose_topic_name,
  std::vector<nav_msgs::msg::Odometry> & odometry_msgs,
  const std::string & expected_child_frame_id)
{
  rosbag2_cpp::Reader pose_reader;
  pose_reader.open(pose_bag_file);

  if (GetTopicMessageCount(pose_reader.get_metadata(), pose_topic_name) == 0) {
    LOG(ERROR) << "No message in topic: " << pose_topic_name;
    return false;
  }

  const std::map<std::string, std::string> topic_name_to_message_type =
    GetTopicNameToMessageTypeMap(pose_reader.get_metadata());

  // read pose messages
  while (pose_reader.has_next()) {
    rosbag2_storage::SerializedBagMessageSharedPtr msg =
      pose_reader.read_next();
    if (msg->topic_name != pose_topic_name) {
      continue;
    }

    const std::string message_type =
      topic_name_to_message_type.at(msg->topic_name);
    if (message_type != kOdometryMessageType) {
      continue;
    }

    rclcpp::SerializedMessage serialized_msg(*msg->serialized_data);
    nav_msgs::msg::Odometry odometry_msg;
    rclcpp::Serialization<nav_msgs::msg::Odometry>().deserialize_message(
      &serialized_msg, &odometry_msg);

    if (!expected_child_frame_id.empty() &&
      expected_child_frame_id != odometry_msg.child_frame_id)
    {
      return false;
    }
    odometry_msgs.push_back(odometry_msg);
  }

  // Done with rosbag reading
  pose_reader.close();

  return true;
}

void data_converter_utils::CheckBagExists(const std::string & bag_file)
{
  CHECK(
    FileUtils::FileExists(bag_file) ||
    FileUtils::FileExists(
      FileUtils::JoinPath(bag_file, kBagMetadataFile))) <<
    "Bag file does not exists: " << bag_file;
}

bool data_converter_utils::ReadTimestampFile(
  const std::string & timestamp_file_path,
  std::vector<uint64_t> & timestamps)
{
  timestamps.clear();

  std::ifstream timestamp_file(timestamp_file_path);
  if (!timestamp_file.is_open()) {
    LOG(ERROR) << "Failed to open timestamp file: " << timestamp_file_path;
    return false;
  }

  size_t line_index = 0;
  while (timestamp_file.good()) {
    std::string timestamp_string;
    timestamp_file >> timestamp_string;
    line_index++;
    // skip empty lines
    if (timestamp_string.empty()) {
      continue;
    }

    uint64_t timestamp_nanoseconds = 0;
    try {
      timestamp_nanoseconds = std::stoull(timestamp_string);
    } catch (const std::exception & e) {
      LOG(WARNING) << "Invalid characters: " <<
        timestamp_string << " at line: " << line_index;
      return false;
    }
    timestamps.push_back(timestamp_nanoseconds);
  }

  return true;
}


bool data_converter_utils::WriteTimestampFile(
  const std::string & timestamp_file_path,
  const std::vector<uint64_t> & timestamps)
{
  std::ofstream timestamp_file(timestamp_file_path);
  if (!timestamp_file.is_open()) {
    LOG(ERROR) << "Failed to open timestamp file: " << timestamp_file_path;
    return false;
  }

  for (uint64_t timestamp : timestamps) {
    timestamp_file << timestamp << std::endl;
  }

  return true;
}

bool data_converter_utils::ExtractImagesFromExtractedDir(
  const std::string & root_dir,
  const std::map<std::string, uint64_t> & camera_name_to_sensor_id,
  const data_converter_utils::WritImageFunctor write_image_functor)
{
  const std::vector<std::string> camera_names =
    FileUtils::GetSubDirectoriesFromDirectory(root_dir);
  for (const std::string & camera_name : camera_names) {
    if (!camera_name_to_sensor_id.count(camera_name)) {
      LOG(ERROR) << "Can't find this camera: " << camera_name <<
        " in input camera_name_to_sensor_id";
      return false;
    }

    uint64_t sensor_id = camera_name_to_sensor_id.at(camera_name);
    const std::string image_dir = FileUtils::JoinPath(
      root_dir,
      camera_name);

    const std::string timestamp_file_path = FileUtils::JoinPath(
      root_dir, camera_name + "_timestamp.txt");
    if (!FileUtils::FileExists(timestamp_file_path)) {
      LOG(ERROR) << "Timestamp file: " << timestamp_file_path << " does not exists";
      return false;
    }

    std::vector<uint64_t> timestamps_nanoseconds;
    if (!ReadTimestampFile(timestamp_file_path, timestamps_nanoseconds)) {
      LOG(ERROR) << "Failed to read timestamp file: " << timestamp_file_path;
      return false;
    }

    std::vector<std::string> file_names =
      FileUtils::GetFileNamesFromDirectory(image_dir);

    if (file_names.size() != timestamps_nanoseconds.size()) {
      LOG(ERROR) << "Number of images under: " << image_dir << " is " << file_names.size() <<
        " does not meet with the number of timestamps: " << timestamps_nanoseconds.size();
      return false;
    }

    LOG(INFO) << "Start processing images under: " << image_dir << " total number of images: " <<
      file_names.size();

    uint64_t last_timestamp_nanoseconds = 0;

    for (size_t i = 0; i < timestamps_nanoseconds.size(); ++i) {
      uint64_t timestamp_nanoseconds = timestamps_nanoseconds[i];
      if (timestamp_nanoseconds == 0) {
        LOG(WARNING) << "Got timestamp 0 at index: " << i;
        continue;
      }

      if (timestamp_nanoseconds <= last_timestamp_nanoseconds) {
        LOG(ERROR) << "Current frame timestamp: " << timestamp_nanoseconds <<
          " is smaller than last timestamp: " << last_timestamp_nanoseconds <<
          " skipping this frame";
        continue;
      }

      last_timestamp_nanoseconds = timestamp_nanoseconds;

      const std::string image_file = FileUtils::JoinPath(
        image_dir,
        file_names[i]);

      if (!FileUtils::FileExists(image_file)) {
        LOG(ERROR) << "Image file does not exist: " << image_file << " for timestamp: " <<
          timestamp_nanoseconds << " at index: " << i;
        continue;
      }

      ImageFrame image_frame;
      image_frame.image = cv::imread(image_file);

      if (image_frame.image.empty()) {
        LOG(ERROR) << "Failed to read file: " << image_file;
        continue;
      }

      image_frame.camera_name = camera_name;
      image_frame.camera_sensor_id = sensor_id;
      image_frame.sample_id = i;
      image_frame.timestamp_nanoseconds = timestamp_nanoseconds;
      image_frame.is_rectified = true;
      image_frame.is_depth_image = (camera_name.find("_depth") != std::string::npos);
      if (!write_image_functor(image_frame)) {
        LOG(ERROR) << "Failed to write image frame: " << image_file;
        return false;
      }
    }   // for each timestamp
  } // for each camera name

  return true;
}

}  // namespace isaac_mapping_ros
}  // namespace isaac_ros
