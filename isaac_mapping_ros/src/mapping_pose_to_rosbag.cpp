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

// A tool to convert deepmap mapping poses
// into ROS bag

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_msgs/msg/tf_message.hpp>
#include <rosbag2_cpp/writer.hpp>
#include <rosbag2_cpp/writers/sequential_writer.hpp>

#include "common/file_utils/csv_file_utils.h"
#include "common/transform/se3_transform.h"
#include "protos/visual/general/keyframe_metadata.pb.h"

using namespace nvidia::isaac;

DEFINE_string(
  output_pose_bag_file, "",
  "[REQUIRED] The path to the bag file that contains poses");
DEFINE_string(
  input_pose_file, "",
  "[REQUIRED] The input pose file of tum format");
DEFINE_string(
  pose_topic_name, "/sfm/odometry",
  "[OPTIONAL] The topic name used to write pose");
DEFINE_string(
  storage_id, "mcap",
  "[OPTIONAL] The output storage id of the bag file");
DEFINE_string(
  pose_child_frame_id, "base_link",
  "[OPTIONAL] the child frame id of the output pose");
DEFINE_string(
  pose_header_frame_id, "map",
  "[OPTIONAL] the header frame id of the output pose");

const uint64_t kSecondsToNanoseconds = 1000000000;
const std::string kTransformTopicName = "/tf";
// const std::string kTransformMessageType = "tf2_msgs/msg/TFMessage";

int main(int argc, char ** argv)
{
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InstallFailureSignalHandler();

  FLAGS_alsologtostderr = true;
  FLAGS_colorlogtostderr = true;

  CHECK(!FLAGS_input_pose_file.empty()) << "Please provide --input_pose_file";
  CHECK(!FLAGS_output_pose_bag_file.empty())
    << "Please provide --output_pose_bag_file";

  // Read the pose file, those poses are baselink to map
  std::vector<std::vector<std::string>> content;
  auto status =
    common::file_utils::ReadCsvFromFile(FLAGS_input_pose_file, " ", &content);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to read tum-pose-file from: "
               << FLAGS_input_pose_file;
    return EXIT_FAILURE;
  }
  if (content.empty()) {
    LOG(ERROR) << "file has empty row";
    return EXIT_FAILURE;
  }
  if (content[0].size() != 8) {
    LOG(ERROR) << "tum-pose-file should have 8 rows, actual: "
               << content[0].size();
    return EXIT_FAILURE;
  }

  rosbag2_cpp::Writer writer;

  rosbag2_storage::StorageOptions storage_options;
  storage_options.uri = FLAGS_output_pose_bag_file;
  storage_options.storage_id = "mcap";
  rosbag2_cpp::ConverterOptions converter_options;
  converter_options.input_serialization_format = "cdr";
  converter_options.output_serialization_format = "cdr";

  try {
    writer.open(storage_options, converter_options);
  } catch (const std::exception & e) {
    LOG(ERROR) << "Failed to open: " << FLAGS_output_pose_bag_file
               << " due to: " << e.what();
    return EXIT_FAILURE;
  }
  // Write the pose messages to the bag file
  try {
    for (size_t row_count = 0; row_count < content.size(); row_count++) {
      const auto & row = content[row_count];
      double timestamp_sec = std::stod(row[0]);
      double x = std::stod(row[1]);
      double y = std::stod(row[2]);
      double z = std::stod(row[3]);
      double qx = std::stod(row[4]);
      double qy = std::stod(row[5]);
      double qz = std::stod(row[6]);
      double qw = std::stod(row[7]);

      rclcpp::Time ros_time(timestamp_sec * kSecondsToNanoseconds,
        RCL_SYSTEM_TIME);
      nav_msgs::msg::Odometry odometry_msg;
      odometry_msg.header.stamp = ros_time;
      odometry_msg.header.frame_id = FLAGS_pose_header_frame_id;
      odometry_msg.child_frame_id = FLAGS_pose_child_frame_id;
      odometry_msg.pose.pose.position.x = x;
      odometry_msg.pose.pose.position.y = y;
      odometry_msg.pose.pose.position.z = z;
      odometry_msg.pose.pose.orientation.w = qw;
      odometry_msg.pose.pose.orientation.x = qx;
      odometry_msg.pose.pose.orientation.y = qy;
      odometry_msg.pose.pose.orientation.z = qz;
      writer.write(odometry_msg, FLAGS_pose_topic_name, ros_time);

      // Create TF message
      tf2_msgs::msg::TFMessage tf_msgs;
      geometry_msgs::msg::TransformStamped tf_msg;
      tf_msg.header.stamp = ros_time;
      tf_msg.header.frame_id = FLAGS_pose_header_frame_id;
      tf_msg.child_frame_id = FLAGS_pose_child_frame_id;
      tf_msg.transform.translation.x = x;
      tf_msg.transform.translation.y = y;
      tf_msg.transform.translation.z = z;
      tf_msg.transform.rotation.w = qw;
      tf_msg.transform.rotation.x = qx;
      tf_msg.transform.rotation.y = qy;
      tf_msg.transform.rotation.z = qz;
      tf_msgs.transforms.emplace_back(tf_msg);
      writer.write(tf_msgs, kTransformTopicName, ros_time);

    }
  } catch (const std::exception & e) {
    LOG(ERROR) << "Failed to read tum file row: " << std::string(e.what());
    LOG(ERROR) << "Failed to write message to bag file: "
               << FLAGS_output_pose_bag_file;
    return EXIT_FAILURE;
  }

  LOG(INFO) << "Successfully output: " << content.size()
            << " number of poses to file: " << FLAGS_output_pose_bag_file;

  return EXIT_SUCCESS;
}
