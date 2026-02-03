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

#include "common/file_utils/file_utils.h"
#include "visual/general/data_selector.h"
#include "visual/utils/types.h"
#include "protos/visual/general/keyframe_metadata.pb.h"

using namespace nvidia::isaac;

DEFINE_string(
  input_frames_meta_file, "", "[REQUIRED] The input frames meta file");
DEFINE_string(
  output_frames_meta_file, "", "[REQUIRED] The output frames meta file");
DEFINE_double(
  min_inter_frame_distance, 0.05,
  "The minimal inter-frame distance between two key frames");
DEFINE_double(
  min_inter_frame_rotation_degrees, 1,
  "The minimal inter-frame rotation between two key frames");

using common::file_utils::FileUtils;

int main(int argc, char ** argv)
{
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InstallFailureSignalHandler();

  FLAGS_alsologtostderr = true;
  FLAGS_colorlogtostderr = true;

  protos::visual::general::KeyframesMetadataCollection frames_meta;
  FileUtils::ReadProtoFileByExtensionOrDie(FLAGS_input_frames_meta_file, &frames_meta);

  LOG(INFO) << "Loaded " << frames_meta.keyframes_metadata_size() << " frames metadata";

  if (frames_meta.keyframes_metadata_size() == 0) {
    LOG(ERROR) << "Empty keyframes in metadata file";
    return EXIT_FAILURE;
  }

  // Gets synced-sample-id to frame-id map
  std::vector<uint64_t> synced_sample_id;
  std::vector<std::pair<uint64_t, common::transform::SE3TransformD>> ref_cam_timestamp_pose_pairs;

  visual::camera_params_id_t ref_cam_id = frames_meta.keyframes_metadata(0).camera_params_id();
  common::transform::SE3TransformD vehicle_to_ref_cam;
  bool find_ref_cam_cal = false;
  for (auto & [cam_id, cam_sensor] : frames_meta.camera_params_id_to_camera_params()) {
    if (cam_id == ref_cam_id) {
      vehicle_to_ref_cam = common::transform::SE3TransformD(
        cam_sensor.sensor_meta_data().sensor_to_vehicle_transform()).Inverse();
      find_ref_cam_cal = true;
    }
  }

  if (!find_ref_cam_cal) {
    LOG(ERROR) << "Didn't find camera_to_vehicle calbiration for camera id: " << ref_cam_id;
    return EXIT_FAILURE;
  }

  std::map<uint64_t, common::transform::SE3TransformD> timestamp_to_pose;
  std::map<uint64_t, uint64_t> timestamp_to_sample_id;

  for (const auto & frame_meta : frames_meta.keyframes_metadata()) {
    if (frame_meta.camera_params_id() == ref_cam_id) {
      timestamp_to_pose[frame_meta.timestamp_microseconds()] = common::transform::SE3TransformD(
        frame_meta.camera_to_world()) * vehicle_to_ref_cam;
      timestamp_to_sample_id[frame_meta.timestamp_microseconds()] = frame_meta.synced_sample_id();
    }
  }

  const auto & selected_timestamps = visual::general::DataSelector::SelectKeyFramesByPose(
    timestamp_to_pose, FLAGS_min_inter_frame_distance, FLAGS_min_inter_frame_rotation_degrees);

  std::set<uint64_t> selected_sample_ids;
  for (auto timestamp : selected_timestamps) {
    selected_sample_ids.insert(timestamp_to_sample_id.at(timestamp));
  }

  // Pick all keyframes at selected_sample_ids
  protos::visual::general::KeyframesMetadataCollection output_frames_meta(frames_meta);
  output_frames_meta.mutable_keyframes_metadata()->Clear();

  for (const auto & frame_meta : frames_meta.keyframes_metadata()) {
    if (selected_sample_ids.count(frame_meta.synced_sample_id()) == 1) {
      *output_frames_meta.add_keyframes_metadata() = frame_meta;
    }
  }

  FileUtils::WriteProtoFileByExtensionOrDie(FLAGS_output_frames_meta_file, output_frames_meta);
  LOG(INFO) << "Write " << output_frames_meta.keyframes_metadata_size() << " frames metadata to " <<
    FLAGS_output_frames_meta_file;
  return EXIT_SUCCESS;
}
