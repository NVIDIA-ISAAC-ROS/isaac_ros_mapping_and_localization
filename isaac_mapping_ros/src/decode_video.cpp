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

#include <rosbag2_cpp/reader.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>

#include "isaac_mapping_ros/video_decoder.hpp"

#include "common/file_utils/file_utils.h"
#include "common/strings/stringprintf.h"

DEFINE_string(
  sensor_data_bag_file, "",
  "[REQUIRED] The path to the bag file that contains sensor data");
DEFINE_string(topic_name, "", "The topic name to decode");
DEFINE_string(
  output_folder_path, "",
  "[REQUIRED] The path to the converted folder");

const std::string kImageMessageType = "sensor_msgs/msg/CompressedImage";


int main(int argc, char ** argv)
{
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InstallFailureSignalHandler();

  FLAGS_alsologtostderr = true;
  FLAGS_colorlogtostderr = true;

  CHECK(!FLAGS_sensor_data_bag_file.empty()) <<
    "Please provide --sensor_data_bag_file";
  CHECK(!FLAGS_output_folder_path.empty()) <<
    "Please provide --output_folder_path";
  CHECK(!FLAGS_topic_name.empty()) <<
    "Please provide --topic_name";

  nvidia::isaac::common::file_utils::FileUtils::EnsureDirectoryExists(
    FLAGS_output_folder_path);

  rosbag2_cpp::Reader reader;
  reader.open(FLAGS_sensor_data_bag_file);

  size_t frame_id = 0;

  isaac_ros::isaac_mapping_ros::VideoDecoder decoder;
  CHECK(decoder.Init("h264")) << "Failed to initialize decoder";

  while (reader.has_next()) {
    rosbag2_storage::SerializedBagMessageSharedPtr msg = reader.read_next();
    if (msg->topic_name != FLAGS_topic_name) {
      continue;
    }

    rclcpp::SerializedMessage serialized_msg(*msg->serialized_data);
    sensor_msgs::msg::CompressedImage image;
    rclcpp::Serialization<sensor_msgs::msg::CompressedImage>()
    .deserialize_message(&serialized_msg, &image);

    // Decode the packet
    std::vector<cv::Mat> frames;
    if (!decoder.DecodePacket(image.data, frames) || frames.empty()) {
      LOG(ERROR) << "Failed to decode frames, No frames decoded";
      continue;
    }

    // Process the frames (e.g., display them)
    for (const auto & frame : frames) {
      const std::string image_file =
        nvidia::isaac::common::file_utils::FileUtils::JoinPath(
        FLAGS_output_folder_path,
        nvidia::isaac::common::strings::StringPrintf("%010zu.jpg", frame_id++));

      if (!cv::imwrite(image_file, frame)) {
        LOG(ERROR) << "Failed to write image to file:" << image_file;
        return EXIT_FAILURE;
      }

      if (frame_id % 500 == 0) {
        LOG(INFO) << "Decoded: " << frame_id << " number of frames";
      }
    }
  }


  LOG(INFO) << "In total converted: " << frame_id << " images, and write to directory: " <<
    FLAGS_output_folder_path;

  return EXIT_SUCCESS;
}
