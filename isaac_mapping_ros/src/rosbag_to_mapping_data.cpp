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

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <chrono>
#include <functional>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>

#include "isaac_mapping_ros/data_converter_utils.hpp"

#include "common/datetime/scoped_timer.h"
#include "common/file_utils/file_utils.h"
#include "protos/visual/general/keyframe_metadata.pb.h"
#include "visual/cusfm/data_selector.h"
#include "visual/general/keyframe_metadata.h"
#include "visual/utils/keyframe_edex_utils.h"
#include "common/transform/pose_serializer.h"

using namespace nvidia::isaac;
using nvidia::isaac::common::file_utils::FileUtils;
namespace ConverterUtil =
  isaac_ros::isaac_mapping_ros::data_converter_utils;
using common::transform::SE3TransformD;

DEFINE_string(
  sensor_data_bag_file, "",
  "[REQUIRED] The path to the bag file that contains sensor data");
DEFINE_string(
  pose_bag_file, "",
  "[OPTIONAL] The path to the bag file that contains poses");
DEFINE_string(tum_pose_file, "", "[OPTIONAL] TUM pose file for the poses");
DEFINE_string(
  output_folder_path, "",
  "[REQUIRED] The path to the converted folder");
DEFINE_string(
  pose_topic_name, "/visual_slam/vis/slam_odometry",
  "[OPTIONAL] The topic name of the pose to use");
DEFINE_double(
  min_inter_frame_distance, 0.05,
  "[OPTIONAL] The minimal inter-frame distance between two key frames");
DEFINE_double(
  min_inter_frame_rotation_degrees, 1,
  "[OPTIONAL] The minimal inter-frame rotation between two key frames");
DEFINE_int64(
  sample_sync_threshold_microseconds, 50,
  "[OPTIONAL] Timestamp differences for the same sample");
DEFINE_string(
  base_link_name, "base_link",
  "The frame name of the base link");
DEFINE_bool(rectify_images, true, "If true will rectify images");
DEFINE_string(
  image_extension, ".jpg",
  "[OPTIONAL] The image extension used to save images");
DEFINE_string(video_codec_name, "h264", "Codec used to decode compressed images");
DEFINE_bool(generate_edex, true, "[OPTIONAL] Generate edex files for the extracted images");
DEFINE_string(camera_topic_config, "", "[OPTIONAL] The camera topic config file");
DEFINE_bool(
  dry_run, false,
  "In dry run mode it will only generate metadata instead of the real camera images");

const char kEdexFrameMetaFileName[] = "frame_metadata.jsonl";
const char kEdexStereoFileName[] = "stereo.edex";

bool WriteImage(
  uint64_t & unique_id,
  protos::visual::general::KeyframesMetadataCollection & metadata_collection,
  std::mutex & mutex,
  ConverterUtil::ImageFrame & image_frame)
{
  common::datetime::ScopedTimer timer("rosbag_converter::WriteImage");

  const std::string image_relative_path =
    FileUtils::JoinPath(
    image_frame.camera_name,
    std::to_string(image_frame.timestamp_nanoseconds) +
    (image_frame.is_depth_image ? ".png" : FLAGS_image_extension));

  if (!FLAGS_dry_run) {
    const std::string image_full_path =
      FileUtils::JoinPath(
      FLAGS_output_folder_path,
      image_relative_path);

    CHECK(
      FileUtils::EnsureDirectoryExists(
        FileUtils::JoinPath(
          FLAGS_output_folder_path, image_frame.camera_name)));

    {
      common::datetime::ScopedTimer timer("rosbag_converter::cv::imwrite");

      // Check if this is a depth image by looking at the camera name or metadata
      bool is_depth_image = image_frame.is_depth_image;

      // Also check if we can get this information from the metadata
      // This would require passing the metadata to this function, but for now we'll use the camera name

      if (is_depth_image) {
        // For depth images, write as uint16 PNG without normalization
        // Convert to uint16 if not already, and clip values > 65535 to 0
        // if it's a float image, convert it to uint16 by using scale 1000
        cv::Mat depth_image;
        if (image_frame.image.type() == CV_16UC1) {
          depth_image = image_frame.image;
        } else if (image_frame.image.type() == CV_32FC1) {
          LOG_EVERY_N(INFO, 100) << "Converting depth image to uint16 from float with scale 1000";
          image_frame.image.convertTo(depth_image, CV_16UC1, 1000);
        } else {
          LOG(ERROR) << "Unsupported depth image type: " << image_frame.image.type();
          return false;
        }

        // Clip values > 65535 to 0 (max value for uint16)
        cv::threshold(depth_image, depth_image, 65535, 0, cv::THRESH_TOZERO_INV);

        // Write as PNG with uint16 format
        std::vector<int> compression_params;
        compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
        compression_params.push_back(9);  // Maximum compression

        if (!cv::imwrite(image_full_path, depth_image, compression_params)) {
          LOG(ERROR) << "Failed to write depth image to file:" << image_full_path;
          return false;
        }
      } else {
        // For color images, use the original approach
        if (!cv::imwrite(image_full_path, image_frame.image)) {
          LOG(ERROR) << "Failed to write image to file:" << image_full_path;
          return false;
        }
      }
    }
  }

  // Depth image does not need to be added to the metadata collection as we assume it's synced with the color image
  if (image_frame.is_depth_image) {
    return true;
  }

  {
    std::lock_guard<std::mutex> lock(mutex);
    auto metadata = metadata_collection.add_keyframes_metadata();
    // TODO(dizeng) id can be constructed by camera_id + sample_id
    metadata->set_id(unique_id);
    metadata->set_camera_params_id(image_frame.camera_sensor_id);
    metadata->set_synced_sample_id(image_frame.synced_sample_id);
    *metadata->mutable_image_name() = image_relative_path;
    if (image_frame.has_camera_to_world) {
      *metadata->mutable_camera_to_world() =
        image_frame.camera_to_world.ToProto();
    }
    metadata->set_timestamp_microseconds(image_frame.timestamp_nanoseconds / 1000);
    // TODO(dizeng) populate pose_covariance here
    ++unique_id;
  }

  return true;
}

void AddStereoPair(
  const std::map<std::string,
  isaac_ros::isaac_mapping_ros::CameraMetadata> & camera_name_to_camera_metadata,
  protos::visual::general::KeyframesMetadataCollection & metadata_collection)
{

  const float kNonBaselineEps = 0.001;
  const float kMinBaselineThreshold = 0.01;  // 1 cm

  for (const auto & [camera_name, camera_metadata] : camera_name_to_camera_metadata) {
    if (camera_metadata.get_paired_camera_name().empty()) {
      continue;
    }

    if (camera_name_to_camera_metadata.count(camera_metadata.get_paired_camera_name()) == 0) {
      LOG(ERROR) << "Paired camera not found: " << camera_metadata.get_paired_camera_name();
      continue;
    }

    const auto & paired_camera_metadata =
      camera_name_to_camera_metadata.at(camera_metadata.get_paired_camera_name());

    if (paired_camera_metadata.is_depth_image() || camera_metadata.is_depth_image()) {
      LOG(WARNING) << "Skipping stereo pair: [" << camera_name << "," <<
        camera_metadata.get_paired_camera_name() << "] because one of the cameras is a depth image";
      continue;
    }

    SE3TransformD left_transform = camera_metadata.camera_to_vehicle_se3();
    SE3TransformD right_transform = paired_camera_metadata.camera_to_vehicle_se3();

    SE3TransformD relative_pose =
      left_transform.Inverse() * right_transform;

    if (std::fabs(relative_pose.translation().y()) > kNonBaselineEps ||
      std::fabs(relative_pose.translation().z()) > kNonBaselineEps ||
      std::fabs(relative_pose.translation().x()) < kMinBaselineThreshold)
    {

      LOG(WARNING) << "Input stereo pair: [" << camera_name << "," <<
        camera_metadata.get_paired_camera_name() << ": has invalid transform:" <<
        relative_pose.ToString();
    }

    auto stereo_pair = metadata_collection.add_stereo_pair();
    stereo_pair->set_left_camera_param_id(camera_metadata.get_camera_sensor_id());
    stereo_pair->set_right_camera_param_id(paired_camera_metadata.get_camera_sensor_id());
    stereo_pair->set_baseline_meters(std::fabs(relative_pose.translation().x()));
  }
}

void ExtractFromRosBag()
{
  common::datetime::ScopedTimer timer("rosbag_converter::ExtractFromRosBag");

  auto camera_name_to_camera_metadata =
    isaac_ros::isaac_mapping_ros::data_converter_utils::ExtractCameraMetadata(
    FLAGS_sensor_data_bag_file,
    FLAGS_output_folder_path,
    FLAGS_base_link_name,
    FLAGS_camera_topic_config,
    FLAGS_rectify_images);
  CHECK(!camera_name_to_camera_metadata.empty()) << "Failed to extract camera metadata";

  // When --rectify_images=False, ensure camera metadata reflects that images will be raw
  // so that distortion coefficients are preserved for BROWN model in EDEX files
  if (!FLAGS_rectify_images) {
    LOG(INFO) << "rectify_images=false: configuring cameras for raw image processing with distortion preservation";
    for (auto & [camera_name, camera_metadata] : camera_name_to_camera_metadata) {
      if (camera_metadata.get_is_camera_rectified()){
        LOG(FATAL) << "Camera " << camera_name << " is already rectified, can't set --rectify_images=False";
        return;
      }
    }
  }

  // Get synced timestamps vector per video
  std::vector<std::vector<uint64_t>> synced_timestamps;
  ConverterUtil::FindSyncedTimestamps(
    camera_name_to_camera_metadata, synced_timestamps,
    FLAGS_sample_sync_threshold_microseconds * 1000);

  // Selected by pose
  common::transform::SE3PoseLinearInterpolator pose_interpolator;
  if (!FLAGS_pose_bag_file.empty()) {
    pose_interpolator = ConverterUtil::ExtractPosesFromBag(
      FLAGS_pose_bag_file,
      FLAGS_pose_topic_name,
      FLAGS_base_link_name);
    CHECK(!pose_interpolator.timestamps().empty()) << "Did not get any poses in file: " <<
      FLAGS_pose_bag_file;
    LOG(INFO) << "Pose Timestamps: " << pose_interpolator.timestamps().front() <<
      ", " << pose_interpolator.timestamps().back();
  } else if (!FLAGS_tum_pose_file.empty()) {
    CHECK(
      nvidia::isaac::common::transform::PoseSerializer::GetPoseInterpolatorFromTumFile(
        FLAGS_tum_pose_file,
        pose_interpolator)) << "Failed to read pose from file: " << FLAGS_tum_pose_file;
  } else {
    LOG(INFO) << "Not using any pose to select frames";
  }

  protos::visual::general::KeyframesMetadataCollection metadata_collection;
  metadata_collection.set_initial_pose_type(protos::visual::general::EGO_MOTION);

  const std::string session_name =
    FileUtils::GetFileStemName(
    FLAGS_sensor_data_bag_file);

  auto & camera_param_to_session_name =
    (*metadata_collection.mutable_camera_params_id_to_session_name());

  auto & camera_param_id_to_camera_param =
    (*metadata_collection.mutable_camera_params_id_to_camera_params());

  int camera_index = 0;
  for (auto & camera_name_metadata_pair : camera_name_to_camera_metadata) {
    auto & camera_metadata = camera_name_metadata_pair.second;
    if (camera_metadata.is_depth_image()) {
      LOG(INFO) << "Skipping depth camera: " << camera_name_metadata_pair.first;
      continue;
    }

    const auto & synced_camera_timestamps = synced_timestamps[camera_index];
    std::map<uint64_t, uint64_t> sample_id_to_synced_sample_id;
    std::map<uint64_t, common::transform::SE3TransformD> sample_id_to_poses;

    CHECK(
      ConverterUtil::GetFrameSyncAndPoseMap(
        pose_interpolator,
        camera_metadata.timestamp_nanoseconds(),
        synced_camera_timestamps,
        sample_id_to_synced_sample_id,
        sample_id_to_poses)) << "Failed to get frame id and sync info for: " <<
      camera_name_metadata_pair.first;

    std::vector<uint64_t> selected_frames = visual::cusfm::DataSelector::SelectKeyFramesByPose(
      sample_id_to_poses, FLAGS_min_inter_frame_distance, FLAGS_min_inter_frame_rotation_degrees);

    camera_metadata.set_selected_frames(
      std::set<uint64_t>(
        selected_frames.begin(),
        selected_frames.end()));

    camera_metadata.set_vehicle_to_world_poses(sample_id_to_poses);
    camera_metadata.set_sample_id_to_synced_sample_id(sample_id_to_synced_sample_id);

    camera_param_id_to_camera_param[camera_metadata.get_camera_sensor_id()] =
      camera_metadata.get_camera_params();
    camera_param_to_session_name[camera_metadata.get_camera_sensor_id()] =
      session_name;

    camera_index++;
  }

  // Save to file
  uint64_t unique_id = 0;
  std::mutex mutex;
  CHECK(
    ConverterUtil::ExtractCameraImagesFromRosbag(
      FLAGS_sensor_data_bag_file,
      camera_name_to_camera_metadata,
      std::bind(
        &WriteImage, std::ref(unique_id), std::ref(metadata_collection),
        std::ref(mutex), std::placeholders::_1),
      FLAGS_video_codec_name, FLAGS_dry_run, FLAGS_rectify_images)) <<
    "Failed to extract images from video";

  AddStereoPair(camera_name_to_camera_metadata, metadata_collection);

  FileUtils::WriteProtoFileByExtensionOrDie(
    FileUtils::JoinPath(
      FLAGS_output_folder_path,
      nvidia::isaac::visual::kFramesMetaFileName),
    metadata_collection);

  if (FLAGS_generate_edex) {
    LOG(INFO) << "Converting edex files";
    std::string frames_meta_str;
    std::string stereo_edx_str;

    // Extract frames metadata
    frames_meta_str = visual::utils::ExtractFramesMeta(metadata_collection);

    // Extract stereo edx - BROWN distortion params are automatically populated
    // when camera model is PINHOLE and distortion coefficients are present
    stereo_edx_str = visual::utils::ExtractStereoEdex(
      metadata_collection,
      frames_meta_str);

    std::string frame_meta_file = FileUtils::JoinPath(
      FLAGS_output_folder_path,
      kEdexFrameMetaFileName);
    auto status = FileUtils::WriteTextFile(frame_meta_file, frames_meta_str);
    CHECK(status.ok()) << status.message();
    LOG(INFO) << "frame_meta jsonl  saved: " << frame_meta_file;

    std::string stereo_edex_file = FileUtils::JoinPath(
      FLAGS_output_folder_path,
      kEdexStereoFileName);
    status = FileUtils::WriteTextFile(stereo_edex_file, stereo_edx_str);
    CHECK(status.ok()) << status.message();
    LOG(INFO) << "stereo edex file saved: " << stereo_edex_file;
  }
}


int main(int argc, char ** argv)
{
  common::datetime::ScopedTimer timer("main");
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InstallFailureSignalHandler();

  FLAGS_alsologtostderr = true;
  FLAGS_colorlogtostderr = true;

  CHECK(!FLAGS_sensor_data_bag_file.empty()) <<
    "Please provide --sensor_data_bag_file";
  CHECK(!FLAGS_output_folder_path.empty()) <<
    "Please provide an output folder path.";
  CHECK(!FLAGS_pose_topic_name.empty()) << "Please provide --pose_topic_name";

  CHECK(
    FileUtils::EnsureDirectoryExists(
      FLAGS_output_folder_path)) <<
    "Failed to create output folder: " + FLAGS_output_folder_path;

  ConverterUtil::CheckBagExists(FLAGS_sensor_data_bag_file);
  if (!FLAGS_pose_bag_file.empty()) {
    ConverterUtil::CheckBagExists(FLAGS_pose_bag_file);
  } else {
    if (FLAGS_min_inter_frame_distance > 0 || FLAGS_min_inter_frame_rotation_degrees > 0) {
      LOG(ERROR) <<
        "--pose_bag_file is not provided, ignore --min_inter_frame_distance and --min_inter_frame_rotation_degrees";
    }
  }

  if (!FLAGS_tum_pose_file.empty()) {
    if (!FileUtils::FileExists(FLAGS_tum_pose_file)) {
      LOG(FATAL) << "TUM pose file does not exist: " << FLAGS_tum_pose_file;
    }
  }

  ExtractFromRosBag();

  return 0;
}
