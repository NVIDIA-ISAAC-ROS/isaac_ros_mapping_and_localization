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

#ifndef ISAAC_ROS_OCCUPANCY_GRID_LOCALIZER__OCCUPANCY_GRID_LOCALIZER_CORE_HPP_
#define ISAAC_ROS_OCCUPANCY_GRID_LOCALIZER__OCCUPANCY_GRID_LOCALIZER_CORE_HPP_

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "isaac_ros_pointcloud_interfaces/msg/flat_scan.hpp"

// Forward declaration — avoids pulling CUDA headers into every consumer
namespace nvidia
{
namespace isaac_ros
{
namespace occupancy_grid_localizer
{
class OccupancyGridLocalizerGpu;
}  // namespace occupancy_grid_localizer
}  // namespace isaac_ros
}  // namespace nvidia

namespace nvidia
{
namespace isaac_ros
{
namespace occupancy_grid_localizer
{

struct Pose2D
{
  double x{0.0};
  double y{0.0};
  double yaw{0.0};
};

struct OccupancyGridLocalizerParameters
{
  double cell_size{0.05};
  int occupancy_grid_map_threshold{166};
  std::string map_png_path;
  std::vector<double> map_origin{0.0, 0.0, 0.0};
  int max_points{20000};
  bool use_gxf_map_convention{false};
  double robot_radius{0.25};
  double max_beam_error{0.5};
  double max_output_error{0.35};
  double min_output_error{0.22};
  double sample_distance{0.1};
  double out_of_range_threshold{100.0};
  double invalid_range_threshold{0.0};
  double min_scan_fov_degrees{270.0};
  bool use_closest_beam{true};
  int num_beams_gpu{512};
  int batch_size{512};
};

class OccupancyGridLocalizerCore
{
public:
  explicit OccupancyGridLocalizerCore(const OccupancyGridLocalizerParameters & parameters);

  ~OccupancyGridLocalizerCore();

  void LoadMap();

  bool IsMapLoaded() const;

  std::optional<geometry_msgs::msg::PoseWithCovarianceStamped> Localize(
    const isaac_ros_pointcloud_interfaces::msg::FlatScan & flat_scan,
    const std::optional<geometry_msgs::msg::TransformStamped> & base_link_to_lidar,
    const std::string & loc_result_frame) const;

  Pose2D ConvertRosPoseToGxfMapPose(const Pose2D & ros_pose) const;

private:
  struct BeamSample
  {
    double angle{0.0};
    double range{0.0};
  };

  struct CandidateScore
  {
    Pose2D pose;
    double score{0.0};
  };

  static double NormalizeAngle(double angle);

  bool IsInsideMap(double world_x, double world_y) const;

  bool WorldToPixel(double world_x, double world_y, int & pixel_x, int & pixel_y) const;

  double DistanceToNearestObstacle(double world_x, double world_y) const;

  double RaycastRange(
    double world_x,
    double world_y,
    double yaw,
    double max_range) const;

  std::vector<BeamSample> BuildBeamSamples(
    const isaac_ros_pointcloud_interfaces::msg::FlatScan & flat_scan) const;

  double ScorePose(
    const Pose2D & base_pose,
    const std::vector<BeamSample> & beams,
    const Pose2D & base_link_to_lidar) const;

  std::vector<CandidateScore> SearchCandidates(
    const std::vector<BeamSample> & beams,
    const Pose2D & base_link_to_lidar,
    double position_step,
    double yaw_step,
    std::optional<Pose2D> center,
    double position_radius,
    double yaw_radius,
    size_t top_k) const;

  std::optional<CandidateScore> FindBestPose(
    const isaac_ros_pointcloud_interfaces::msg::FlatScan & flat_scan,
    const std::optional<geometry_msgs::msg::TransformStamped> & base_link_to_lidar) const;

  std::vector<CandidateScore> SearchCandidatesGpu(
    const std::vector<BeamSample> & beams,
    const Pose2D & base_link_to_lidar,
    double position_step,
    double yaw_step,
    std::optional<Pose2D> center,
    double position_radius,
    double yaw_radius,
    size_t top_k) const;

  OccupancyGridLocalizerParameters parameters_;
  cv::Mat occupancy_mask_;
  cv::Mat distance_map_meters_;
  int map_width_{0};
  int map_height_{0};
  double origin_cos_{1.0};
  double origin_sin_{0.0};
  mutable std::unique_ptr<OccupancyGridLocalizerGpu> gpu_;
};

}  // namespace occupancy_grid_localizer
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_OCCUPANCY_GRID_LOCALIZER__OCCUPANCY_GRID_LOCALIZER_CORE_HPP_
