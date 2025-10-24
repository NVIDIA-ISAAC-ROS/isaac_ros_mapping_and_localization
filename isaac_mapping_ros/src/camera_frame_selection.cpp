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


#include "isaac_mapping_ros/data_converter_utils.hpp"

#include "common/datetime/scoped_timer.h"
#include "common/datetime/time_utils.h"
#include "common/file_utils/file_utils.h"

#include "visual/cusfm/data_selector.h"
#include "visual/general/keyframe_metadata.h"
#include "visual/utils/keyframe_utils.h"

using namespace nvidia::isaac;

namespace ConverterUtil =
  isaac_ros::isaac_mapping_ros::data_converter_utils;

DEFINE_string(
  sensor_data_bag_file, "",
  "[REQUIRED] The path to the bag file that contains sensor data");
DEFINE_string(
  pose_bag_file, "",
  "[REQUIRED] The path to the bag file that contains poses");
DEFINE_string(
  pose_topic_name, "/visual_slam/vis/slam_odometry",
  "[OPTIONAL] The topic name of the pose to use");
DEFINE_string(
  output_folder_path, "",
  "[REQUIRED] The path to the converted folder");
DEFINE_string(
  expected_pose_child_frame_name, "base_link",
  "The expected pose message's child frame name, if name doesn't "
  "match, it raises error");
DEFINE_int64(
  sample_sync_threshold_microseconds, 50,
  "Timestamp differences for the same sample");
DEFINE_double(
  min_inter_frame_distance, 0.05,
  "The minimal inter-frame distance between two key frames");
DEFINE_double(
  min_inter_frame_rotation_degrees, 1,
  "The minimal inter-frame rotation between two key frames");

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
  CHECK(!FLAGS_pose_bag_file.empty()) <<
    "Please provide --pose_bag_file";
  CHECK(!FLAGS_output_folder_path.empty()) <<
    "Please provide an output folder path.";
  CHECK(!FLAGS_pose_topic_name.empty()) << "Please provide --pose_topic_name";

  CHECK(
    common::file_utils::FileUtils::EnsureDirectoryExists(
      FLAGS_output_folder_path)) <<
    "Can not create directory: " << FLAGS_output_folder_path;

  ConverterUtil::CheckBagExists(FLAGS_sensor_data_bag_file);
  ConverterUtil::CheckBagExists(FLAGS_pose_bag_file);

  const common::transform::SE3PoseLinearInterpolator pose_interpolator =
    ConverterUtil::ExtractPosesFromBag(
    FLAGS_pose_bag_file,
    FLAGS_pose_topic_name,
    FLAGS_expected_pose_child_frame_name);

  LOG(INFO) << "Pose Timestamps: " << pose_interpolator.timestamps().front() <<
    ", " << pose_interpolator.timestamps().back();

  auto camera_metadata_map = ConverterUtil::ExtractCameraMetadata(
    FLAGS_sensor_data_bag_file, FLAGS_output_folder_path, "", "");

  std::vector<std::vector<uint64_t>> all_camera_timestamps;
  std::vector<std::string> all_camera_names;
  for (const auto & camera_metadata: camera_metadata_map) {
    all_camera_timestamps.push_back(camera_metadata.second.timestamp_nanoseconds());
    all_camera_names.push_back(camera_metadata.first);
  }

  std::vector<std::vector<uint64_t>> synced_timestamps_nanoseconds;
  visual::FindSyncedTimestamps(
    all_camera_timestamps, synced_timestamps_nanoseconds,
    FLAGS_sample_sync_threshold_microseconds * 1000);


  for (size_t i = 0; i < all_camera_timestamps.size(); ++i) {
    std::map<uint64_t, uint64_t> sample_id_to_synced_sample_id;
    std::map<uint64_t, common::transform::SE3TransformD> sample_id_to_pose;

    ConverterUtil::GetFrameSyncAndPoseMap(
      pose_interpolator,
      all_camera_timestamps[i],
      synced_timestamps_nanoseconds[i],
      sample_id_to_synced_sample_id,
      sample_id_to_pose);

    std::vector<uint64_t> selected_frames = visual::cusfm::DataSelector::SelectKeyFramesByPose(
      sample_id_to_pose, FLAGS_min_inter_frame_distance, FLAGS_min_inter_frame_rotation_degrees);

    std::vector<uint64_t> selected_timestamps;
    for (uint64_t frame_id : selected_frames) {
      selected_timestamps.push_back(all_camera_timestamps[i][frame_id]);
    }

    const std::string timestamp_file = common::file_utils::FileUtils::JoinPath(
      FLAGS_output_folder_path, all_camera_names[i] + "_selected_timestamps.txt"
    );

    CHECK(
      ConverterUtil::WriteTimestampFile(
        timestamp_file, selected_timestamps
    ));

    LOG(INFO) << "Write timestamps file " << timestamp_file << " in total: " <<
      selected_timestamps.size() << " frames selected";
  }
}
