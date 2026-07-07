// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "isaac_ros_occupancy_grid_localizer/occupancy_grid_localizer_core.hpp"

namespace fs = std::filesystem;

namespace nvidia
{
namespace isaac_ros
{
namespace occupancy_grid_localizer
{
namespace
{
constexpr double kPi = 3.14159265358979323846;

cv::Mat CreateTestMap()
{
  // Non-square map (8m x 5m) eliminates 90-degree rotational ambiguity.
  // A wall protrusion from the bottom border adds further asymmetry.
  cv::Mat image(100, 160, CV_8UC1, cv::Scalar(254));
  // Border walls (4 pixels thick) drawn with setTo to avoid cv::rectangle
  // assertion issues on OpenCV 4.5.4.
  image(cv::Rect(0, 0, 160, 4)).setTo(cv::Scalar(0));     // top
  image(cv::Rect(0, 96, 160, 4)).setTo(cv::Scalar(0));    // bottom
  image(cv::Rect(0, 0, 4, 100)).setTo(cv::Scalar(0));     // left
  image(cv::Rect(156, 0, 4, 100)).setTo(cv::Scalar(0));   // right
  // Vertical wall protrusion from the bottom border (1.5m tall, 0.4m thick)
  // at x = 5.0m, breaking left-right symmetry as well.
  image(cv::Rect(100, 70, 8, 30)).setTo(cv::Scalar(0));
  return image;
}

bool IsOccupied(const cv::Mat & map, double resolution, double x, double y)
{
  const int pixel_x = static_cast<int>(std::floor(x / resolution));
  const int pixel_y = map.rows - 1 - static_cast<int>(std::floor(y / resolution));
  if (pixel_x < 0 || pixel_x >= map.cols || pixel_y < 0 || pixel_y >= map.rows) {
    return true;
  }
  return map.at<uint8_t>(pixel_y, pixel_x) < 128;
}

isaac_ros_pointcloud_interfaces::msg::FlatScan GenerateScan(
  const cv::Mat & map,
  double resolution,
  double pose_x,
  double pose_y,
  double pose_yaw)
{
  isaac_ros_pointcloud_interfaces::msg::FlatScan scan;
  scan.header.frame_id = "base_link";
  scan.range_min = 0.05F;
  scan.range_max = 12.0F;

  const double start_angle = -3.0 * kPi / 4.0;
  const double end_angle = 3.0 * kPi / 4.0;
  const int beam_count = 181;
  const double delta_angle = (end_angle - start_angle) / static_cast<double>(beam_count - 1);

  for (int beam_idx = 0; beam_idx < beam_count; ++beam_idx) {
    const double angle = start_angle + delta_angle * static_cast<double>(beam_idx);
    const double world_angle = pose_yaw + angle;
    double range = scan.range_max;
    for (double step = scan.range_min; step <= scan.range_max; step += resolution / 2.0) {
      const double sample_x = pose_x + std::cos(world_angle) * step;
      const double sample_y = pose_y + std::sin(world_angle) * step;
      if (IsOccupied(map, resolution, sample_x, sample_y)) {
        range = step;
        break;
      }
    }
    scan.angles.push_back(static_cast<float>(angle));
    scan.ranges.push_back(static_cast<float>(range));
  }

  return scan;
}

TEST(OccupancyGridLocalizerCore, LocalizesSyntheticFlatScan)
{
  const cv::Mat map = CreateTestMap();
  const fs::path map_path = fs::temp_directory_path() /
    "occupancy_grid_localizer_core_test_map.png";
  ASSERT_TRUE(cv::imwrite(map_path.string(), map));

  OccupancyGridLocalizerParameters parameters;
  parameters.cell_size = 0.05;
  parameters.occupancy_grid_map_threshold = 166;
  parameters.map_png_path = map_path.string();
  parameters.map_origin = {0.0, 0.0, 0.0};
  parameters.max_points = 1000;
  parameters.robot_radius = 0.2;
  parameters.max_beam_error = 0.25;
  parameters.max_output_error = 0.3;
  parameters.min_output_error = 0.1;
  parameters.sample_distance = 0.1;
  parameters.out_of_range_threshold = 12.0;
  parameters.invalid_range_threshold = 0.05;
  parameters.min_scan_fov_degrees = 250.0;

  OccupancyGridLocalizerCore localizer(parameters);
  ASSERT_NO_THROW(localizer.LoadMap());
  ASSERT_TRUE(localizer.IsMapLoaded());

  const double expected_x = 3.0;
  const double expected_y = 2.5;
  const double expected_yaw = 0.3;
  const auto scan = GenerateScan(map, parameters.cell_size, expected_x, expected_y, expected_yaw);

  const auto result = localizer.Localize(scan, std::nullopt, "map");
  ASSERT_TRUE(result.has_value());
  EXPECT_NEAR(result->pose.pose.position.x, expected_x, 0.2);
  EXPECT_NEAR(result->pose.pose.position.y, expected_y, 0.2);
  EXPECT_NEAR(
    2.0 * std::atan2(result->pose.pose.orientation.z, result->pose.pose.orientation.w),
    expected_yaw,
    0.12);
}

TEST(OccupancyGridLocalizerCore, ConvertsRosPoseToGxfConvention)
{
  OccupancyGridLocalizerParameters parameters;
  parameters.cell_size = 0.1;
  parameters.map_png_path = (fs::temp_directory_path() /
    "occupancy_grid_localizer_conversion_test.png").string();
  parameters.map_origin = {1.0, 2.0, 0.0};

  cv::Mat map(100, 50, CV_8UC1, cv::Scalar(254));
  ASSERT_TRUE(cv::imwrite(parameters.map_png_path, map));

  OccupancyGridLocalizerCore localizer(parameters);
  ASSERT_NO_THROW(localizer.LoadMap());

  Pose2D ros_pose;
  ros_pose.x = 2.0;
  ros_pose.y = 3.0;
  ros_pose.yaw = 0.0;
  const Pose2D gxf_pose = localizer.ConvertRosPoseToGxfMapPose(ros_pose);
  EXPECT_NEAR(gxf_pose.x, 9.0, 1e-6);
  EXPECT_NEAR(gxf_pose.y, 1.0, 1e-6);
  EXPECT_NEAR(gxf_pose.yaw, kPi / 2.0, 1e-6);
}

}  // namespace
}  // namespace occupancy_grid_localizer
}  // namespace isaac_ros
}  // namespace nvidia
