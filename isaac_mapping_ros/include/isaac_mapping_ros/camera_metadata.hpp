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

#include <string>
#include <vector>

#include <glog/logging.h>

#include "protos/common/sensor/camera_sensor.pb.h"
#include "common/transform/se3_transform.h"


namespace isaac_ros
{
namespace isaac_mapping_ros
{


// A helper class to store all the camera metadata
class CameraMetadata
{
public:
  CameraMetadata() {}

  ~CameraMetadata() {}

  void AddTimestamp(uint64_t timestamp_nanoseconds)
  {
    timestamp_nanoseconds_.push_back(timestamp_nanoseconds);

  }

  void set_camera_topic_name(const std::string & camera_topic_name)
  {
    camera_topic_name_ = camera_topic_name;
  }

  const std::string & camera_topic_name() const {return camera_topic_name_;}

  // get and set for message type
  void set_message_type(const std::string & message_type)
  {
    message_type_ = message_type;
  }

  const std::string & message_type() const {return message_type_;}

  const std::vector<uint64_t> & timestamp_nanoseconds() const
  {
    return timestamp_nanoseconds_;
  }

  void set_camera_params(const protos::common::sensor::CameraSensor & camera_params)
  {
    camera_params_ = camera_params;
    has_camera_params_ = true;
  }

  const protos::common::sensor::CameraSensor & get_camera_params() const
  {
    return camera_params_;
  }

  bool has_camera_params() const {return has_camera_params_;}

  void set_camera_sensor_id(int sensor_id)
  {
    camera_params_.mutable_sensor_meta_data()->set_sensor_id(sensor_id);
  }

  uint32_t get_camera_sensor_id()const
  {
    return camera_params_.sensor_meta_data().sensor_id();
  }

  const protos::common::geometry::RigidTransform3d & camera_to_vehicle() const
  {
    return camera_params_.sensor_meta_data().sensor_to_vehicle_transform();
  }

  nvidia::isaac::common::transform::SE3TransformD camera_to_vehicle_se3() const
  {
    return
      camera_params_.sensor_meta_data().sensor_to_vehicle_transform();
  }

  void set_selected_frames(const std::set<uint64_t> & selected_frames)
  {
    selected_frames_ = selected_frames;
  }

  const std::set<uint64_t> & get_selected_frames()const
  {
    return selected_frames_;
  }

  void set_vehicle_to_world_poses(
    const std::map<uint64_t,
    nvidia::isaac::common::transform::SE3TransformD> & vehicle_to_world_poses)
  {
    vehicle_to_world_poses_ = vehicle_to_world_poses;
  }

  const std::map<uint64_t,
    nvidia::isaac::common::transform::SE3TransformD> & get_vehicle_to_world_poses()const
  {
    return vehicle_to_world_poses_;
  }

  void set_sample_id_to_synced_sample_id(
    const std::map<uint64_t, uint64_t> & sample_id_to_synced_sample_id)
  {
    sample_id_to_synced_sample_id_ = sample_id_to_synced_sample_id;
  }

  const std::map<uint64_t, uint64_t> & get_sample_id_to_synced_sample_id()const
  {
    return sample_id_to_synced_sample_id_;
  }

  void set_paired_camera_name(const std::string & paired_camera_name)
  {
    paired_camera_name_ = paired_camera_name;
  }

  std::string get_paired_camera_name() const
  {
    return paired_camera_name_;
  }

  void set_is_depth_image(bool is_depth_image)
  {
    is_depth_image_ = is_depth_image;
  }

  bool is_depth_image() const
  {
    return is_depth_image_;
  }

  void set_is_camera_rectified(bool is_camera_rectified)
  {
    is_camera_rectified_ = is_camera_rectified;
  }

  bool get_is_camera_rectified() const
  {
    return is_camera_rectified_;
  }

  void set_swap_rb_channels(bool swap_rb_channels)
  {
    swap_rb_channels_ = swap_rb_channels;
  }

  bool get_swap_rb_channels() const
  {
    return swap_rb_channels_;
  }

private:
  std::string camera_topic_name_;
  std::string message_type_;
  std::vector<uint64_t> timestamp_nanoseconds_;
  std::string paired_camera_name_;

  protos::common::sensor::CameraSensor camera_params_;

  bool has_camera_params_ = false;
  bool is_depth_image_ = false;
  bool is_camera_rectified_ = false;
  bool swap_rb_channels_ = false;

  // if populated we only want to use frames set by this
  std::set<uint64_t> selected_frames_;

  std::map<uint64_t, nvidia::isaac::common::transform::SE3TransformD> vehicle_to_world_poses_;

  std::map<uint64_t, uint64_t> sample_id_to_synced_sample_id_;
};

} // namespace isaac_mapping_ros
} // namespace isaac_ros
