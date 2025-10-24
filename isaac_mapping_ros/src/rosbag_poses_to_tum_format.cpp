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
#include <fstream>
#include <memory>
#include <string>
#include <thread>

#include "isaac_mapping_ros/data_converter_utils.hpp"


using namespace nvidia::isaac;
namespace ConverterUtil =
  isaac_ros::isaac_mapping_ros::data_converter_utils;

DEFINE_string(
  pose_bag_file, "",
  "[REQUIRED] The path to the bag file that contains sensor data");
DEFINE_string(
  pose_topic_name, "/visual_slam/vis/slam_odometry",
  "[OPTIONAL] The topic name of the pose to use");
DEFINE_string(
  base_link_name, "base_link",
  "The name of the base link, or in other words the vehicle coordinate");
DEFINE_string(
  output_tum_file, "",
  "[REQUIRED] The file path where poses will be written");


bool ExtraAndExportPosesFromPoseBag()
{
  std::vector<nav_msgs::msg::Odometry> odometry_msgs;
  CHECK(
    ConverterUtil::ExtractOdometryMsgFromPoseBag(
      FLAGS_pose_bag_file, FLAGS_pose_topic_name, odometry_msgs,
      FLAGS_base_link_name));

  std::ofstream file(FLAGS_output_tum_file);
  if (!file.is_open()) {
    LOG(ERROR) << "Error: Could not open file " << FLAGS_output_tum_file <<
      " for writing.";

    return false;
  }

  for (const auto & odom_msg : odometry_msgs) {
    file << std::fixed << std::setprecision(16) <<
      ConverterUtil::ROSTimestampToSeconds(
      odom_msg.header.stamp) << " " <<
      odom_msg.pose.pose.position.x << " " <<
      odom_msg.pose.pose.position.y << " " <<
      odom_msg.pose.pose.position.z << " " <<
      odom_msg.pose.pose.orientation.x << " " <<
      odom_msg.pose.pose.orientation.y << " " <<
      odom_msg.pose.pose.orientation.z << " " <<
      odom_msg.pose.pose.orientation.w << std::endl;
  }

  file.close();

  return true;
}

int main(int argc, char ** argv)
{
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InstallFailureSignalHandler();

  FLAGS_alsologtostderr = true;
  FLAGS_colorlogtostderr = true;

  CHECK(!FLAGS_pose_bag_file.empty()) << "Please provide --pose_bag_file";
  CHECK(!FLAGS_output_tum_file.empty()) <<
    "Please provide an output folder path.";
  CHECK(!FLAGS_pose_topic_name.empty()) << "Please provide --pose_topic_name";

  ConverterUtil::CheckBagExists(FLAGS_pose_bag_file);

  ExtraAndExportPosesFromPoseBag();

  return 0;
}
