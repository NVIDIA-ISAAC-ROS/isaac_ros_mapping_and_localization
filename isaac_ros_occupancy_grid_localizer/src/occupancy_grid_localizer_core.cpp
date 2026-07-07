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

#include "isaac_ros_occupancy_grid_localizer/occupancy_grid_localizer_core.hpp"
#include "isaac_ros_occupancy_grid_localizer/occupancy_grid_localizer_gpu.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace
{
constexpr double kPi = 3.14159265358979323846;
constexpr double kCoarsePositionStepMeters = 0.5;
constexpr double kMediumPositionStepMeters = 0.2;
constexpr double kFineYawStepRadians = kPi / 720.0;
constexpr double kMediumYawStepRadians = kPi / 180.0;
constexpr double kCoarseYawStepRadians = kPi / 36.0;
constexpr double kCoarseYawRadiusRadians = kPi;
constexpr double kMediumYawRadiusRadians = kPi / 18.0;
constexpr double kFineYawRadiusRadians = kPi / 90.0;
constexpr size_t kCoarseTopK = 8;
constexpr size_t kMediumTopK = 4;
constexpr size_t kFineTopK = 1;
constexpr size_t kMaxBeamSamples = 128;
}  // namespace

namespace nvidia
{
namespace isaac_ros
{
namespace occupancy_grid_localizer
{

OccupancyGridLocalizerCore::OccupancyGridLocalizerCore(
  const OccupancyGridLocalizerParameters & parameters)
: parameters_(parameters)
{
}

OccupancyGridLocalizerCore::~OccupancyGridLocalizerCore() = default;

void OccupancyGridLocalizerCore::LoadMap()
{
  constexpr char kMapLoadError[] = "Could not load occupancy grid map image: ";
  cv::Mat map_image = cv::imread(parameters_.map_png_path, cv::IMREAD_GRAYSCALE);
  if (map_image.empty()) {
    throw std::runtime_error(std::string(kMapLoadError) + parameters_.map_png_path);
  }

  map_width_ = map_image.cols;
  map_height_ = map_image.rows;
  origin_cos_ = std::cos(parameters_.map_origin[2]);
  origin_sin_ = std::sin(parameters_.map_origin[2]);

  occupancy_mask_ = cv::Mat::zeros(map_height_, map_width_, CV_8UC1);
  for (int row = 0; row < map_height_; ++row) {
    for (int col = 0; col < map_width_; ++col) {
      occupancy_mask_.at<uint8_t>(row, col) =
        map_image.at<uint8_t>(row, col) <= parameters_.occupancy_grid_map_threshold ? 255 : 0;
    }
  }

  cv::Mat free_mask;
  cv::bitwise_not(occupancy_mask_, free_mask);

  cv::Mat distance_pixels;
  cv::distanceTransform(free_mask, distance_pixels, cv::DIST_L2, 3);
  distance_map_meters_ = distance_pixels * parameters_.cell_size;

  // Upload map to GPU
  try {
    gpu_ = std::make_unique<OccupancyGridLocalizerGpu>(
      parameters_.batch_size, parameters_.num_beams_gpu);
    gpu_->UploadMap(
      occupancy_mask_.data,
      reinterpret_cast<const float *>(distance_map_meters_.data),
      map_width_, map_height_);
  } catch (const std::runtime_error &) {
    gpu_.reset();  // GPU unavailable, will fall back to CPU
  }
}

bool OccupancyGridLocalizerCore::IsMapLoaded() const
{
  return !distance_map_meters_.empty();
}

std::optional<geometry_msgs::msg::PoseWithCovarianceStamped>
OccupancyGridLocalizerCore::Localize(
  const isaac_ros_pointcloud_interfaces::msg::FlatScan & flat_scan,
  const std::optional<geometry_msgs::msg::TransformStamped> & base_link_to_lidar,
  const std::string & loc_result_frame) const
{
  std::optional<CandidateScore> best = FindBestPose(flat_scan, base_link_to_lidar);
  if (!best) {
    return std::nullopt;
  }

  Pose2D pose = best->pose;
  if (parameters_.use_gxf_map_convention) {
    pose = ConvertRosPoseToGxfMapPose(pose);
  }

  // Compute confidence from the best score, linearly mapping
  // [min_output_error, max_output_error] -> [1.0, 0.0].
  const double error_range = parameters_.max_output_error - parameters_.min_output_error;
  double confidence = 1.0;
  if (error_range > 0.0) {
    confidence = 1.0 - std::clamp(
      (best->score - parameters_.min_output_error) / error_range, 0.0, 1.0);
  }

  geometry_msgs::msg::PoseWithCovarianceStamped result;
  result.header = flat_scan.header;
  result.header.frame_id = loc_result_frame;
  result.pose.pose.position.x = pose.x;
  result.pose.pose.position.y = pose.y;
  result.pose.pose.position.z = 0.0;
  result.pose.pose.orientation.x = 0.0;
  result.pose.pose.orientation.y = 0.0;
  result.pose.pose.orientation.z = std::sin(pose.yaw / 2.0);
  result.pose.pose.orientation.w = std::cos(pose.yaw / 2.0);
  result.pose.covariance.fill(0.0);
  // Encode confidence on the x/y/yaw diagonal elements (indices 0, 7, 35).
  // Lower confidence → higher variance.
  const double variance = (confidence > 0.0) ? (1.0 - confidence) : 1e6;
  result.pose.covariance[0] = variance;
  result.pose.covariance[7] = variance;
  result.pose.covariance[35] = variance;
  return result;
}

Pose2D OccupancyGridLocalizerCore::ConvertRosPoseToGxfMapPose(const Pose2D & ros_pose) const
{
  const double map_height_m = static_cast<double>(map_height_) * parameters_.cell_size;
  const double cos_origin = std::cos(parameters_.map_origin[2]);
  const double sin_origin = std::sin(parameters_.map_origin[2]);

  const double translated_x = ros_pose.x - parameters_.map_origin[0];
  const double translated_y = ros_pose.y - parameters_.map_origin[1];

  const double local_x = cos_origin * translated_x + sin_origin * translated_y;
  const double local_y = -sin_origin * translated_x + cos_origin * translated_y;

  Pose2D gxf_pose;
  gxf_pose.x = map_height_m - local_y;
  gxf_pose.y = local_x;
  gxf_pose.yaw = NormalizeAngle(ros_pose.yaw + kPi / 2.0 - parameters_.map_origin[2]);
  return gxf_pose;
}

double OccupancyGridLocalizerCore::NormalizeAngle(double angle)
{
  while (angle > kPi) {
    angle -= 2.0 * kPi;
  }
  while (angle < -kPi) {
    angle += 2.0 * kPi;
  }
  return angle;
}

bool OccupancyGridLocalizerCore::IsInsideMap(double world_x, double world_y) const
{
  int pixel_x = 0;
  int pixel_y = 0;
  return WorldToPixel(world_x, world_y, pixel_x, pixel_y);
}

bool OccupancyGridLocalizerCore::WorldToPixel(
  double world_x, double world_y, int & pixel_x, int & pixel_y) const
{
  // Translate then rotate by the inverse of the map origin yaw.
  const double tx = world_x - parameters_.map_origin[0];
  const double ty = world_y - parameters_.map_origin[1];
  const double map_x = (origin_cos_ * tx + origin_sin_ * ty) / parameters_.cell_size;
  const double map_y = (-origin_sin_ * tx + origin_cos_ * ty) / parameters_.cell_size;

  pixel_x = static_cast<int>(std::floor(map_x));
  pixel_y = map_height_ - 1 - static_cast<int>(std::floor(map_y));

  return pixel_x >= 0 && pixel_x < map_width_ && pixel_y >= 0 && pixel_y < map_height_;
}

double OccupancyGridLocalizerCore::DistanceToNearestObstacle(double world_x, double world_y) const
{
  int pixel_x = 0;
  int pixel_y = 0;
  if (!WorldToPixel(world_x, world_y, pixel_x, pixel_y)) {
    return parameters_.max_beam_error;
  }
  return distance_map_meters_.at<float>(pixel_y, pixel_x);
}

double OccupancyGridLocalizerCore::RaycastRange(
  double world_x,
  double world_y,
  double yaw,
  double max_range) const
{
  const double step = std::max(parameters_.cell_size / 2.0, 0.01);
  for (double range = 0.0; range <= max_range; range += step) {
    const double sample_x = world_x + std::cos(yaw) * range;
    const double sample_y = world_y + std::sin(yaw) * range;
    int pixel_x = 0;
    int pixel_y = 0;
    if (!WorldToPixel(sample_x, sample_y, pixel_x, pixel_y)) {
      return range;
    }
    if (occupancy_mask_.at<uint8_t>(pixel_y, pixel_x) != 0) {
      return range;
    }
  }
  return max_range;
}

std::vector<OccupancyGridLocalizerCore::BeamSample>
OccupancyGridLocalizerCore::BuildBeamSamples(
  const isaac_ros_pointcloud_interfaces::msg::FlatScan & flat_scan) const
{
  std::vector<BeamSample> beams;
  const size_t sample_count = std::min(flat_scan.angles.size(), flat_scan.ranges.size());
  if (sample_count == 0) {
    return beams;
  }

  const size_t max_points = static_cast<size_t>(parameters_.max_points);
  const size_t effective_max = std::min<size_t>(kMaxBeamSamples, max_points);

  // Helper to check if a range reading is valid.
  const auto is_valid_range = [this](double range) {
      return std::isfinite(range) &&
             range > parameters_.invalid_range_threshold &&
             range < parameters_.out_of_range_threshold;
    };

  const auto clamp_range = [this, &flat_scan](double range) {
      return std::min(
        range,
        std::max(static_cast<double>(flat_scan.range_max), parameters_.out_of_range_threshold));
    };

  if (parameters_.use_closest_beam && sample_count > effective_max) {
    // Angular bucketing: divide the scan FOV into effective_max uniform buckets
    // and pick the beam closest to each bucket center angle.
    float min_angle = flat_scan.angles[0];
    float max_angle = flat_scan.angles[0];
    for (size_t i = 1; i < sample_count; ++i) {
      min_angle = std::min(min_angle, flat_scan.angles[i]);
      max_angle = std::max(max_angle, flat_scan.angles[i]);
    }
    const double bucket_width =
      (static_cast<double>(max_angle) - static_cast<double>(min_angle)) /
      static_cast<double>(effective_max);

    beams.reserve(effective_max);
    for (size_t bucket = 0; bucket < effective_max; ++bucket) {
      const double target_angle =
        static_cast<double>(min_angle) + (static_cast<double>(bucket) + 0.5) * bucket_width;
      double best_delta = std::numeric_limits<double>::max();
      size_t best_idx = sample_count;  // sentinel
      for (size_t i = 0; i < sample_count; ++i) {
        if (!is_valid_range(flat_scan.ranges[i])) {
          continue;
        }
        const double delta = std::abs(static_cast<double>(flat_scan.angles[i]) - target_angle);
        if (delta < best_delta) {
          best_delta = delta;
          best_idx = i;
        }
      }
      if (best_idx < sample_count) {
        BeamSample beam;
        beam.angle = flat_scan.angles[best_idx];
        beam.range = clamp_range(flat_scan.ranges[best_idx]);
        beams.push_back(beam);
      }
    }
  } else {
    // Uniform stride subsampling.
    const size_t stride = std::max<size_t>(1, sample_count / effective_max);
    beams.reserve(std::min<size_t>(sample_count, effective_max));
    for (size_t idx = 0; idx < sample_count && beams.size() < max_points; idx += stride) {
      if (!is_valid_range(flat_scan.ranges[idx])) {
        continue;
      }
      BeamSample beam;
      beam.angle = flat_scan.angles[idx];
      beam.range = clamp_range(flat_scan.ranges[idx]);
      beams.push_back(beam);
    }
  }

  return beams;
}

double OccupancyGridLocalizerCore::ScorePose(
  const Pose2D & base_pose,
  const std::vector<BeamSample> & beams,
  const Pose2D & base_link_to_lidar) const
{
  if (DistanceToNearestObstacle(base_pose.x, base_pose.y) < parameters_.robot_radius) {
    return std::numeric_limits<double>::infinity();
  }

  const double cos_yaw = std::cos(base_pose.yaw);
  const double sin_yaw = std::sin(base_pose.yaw);
  const double lidar_x =
    base_pose.x + cos_yaw * base_link_to_lidar.x - sin_yaw * base_link_to_lidar.y;
  const double lidar_y =
    base_pose.y + sin_yaw * base_link_to_lidar.x + cos_yaw * base_link_to_lidar.y;
  const double lidar_yaw = base_pose.yaw + base_link_to_lidar.yaw;

  double total_error = 0.0;
  size_t valid_hits = 0;

  for (const auto & beam : beams) {
    const double beam_yaw = lidar_yaw + beam.angle;
    const double expected_range = RaycastRange(
      lidar_x, lidar_y, beam_yaw, parameters_.out_of_range_threshold);
    const double range_error = std::abs(expected_range - beam.range);
    total_error += std::min(range_error, parameters_.max_beam_error);
    if (range_error <= parameters_.max_beam_error) {
      ++valid_hits;
    }
  }

  if (beams.empty() || valid_hits < std::max<size_t>(4, beams.size() / 12)) {
    return std::numeric_limits<double>::infinity();
  }

  return total_error / static_cast<double>(beams.size());
}

std::vector<OccupancyGridLocalizerCore::CandidateScore>
OccupancyGridLocalizerCore::SearchCandidates(
  const std::vector<BeamSample> & beams,
  const Pose2D & base_link_to_lidar,
  double position_step,
  double yaw_step,
  std::optional<Pose2D> center,
  double position_radius,
  double yaw_radius,
  size_t top_k) const
{
  std::vector<CandidateScore> candidates;

  const auto append_candidate = [&](const Pose2D & pose) {
      const double score = ScorePose(pose, beams, base_link_to_lidar);
      if (!std::isfinite(score)) {
        return;
      }
      CandidateScore candidate;
      candidate.pose = pose;
      candidate.score = score;
      candidates.push_back(candidate);
    };

  if (center) {
    for (double x = center->x - position_radius; x <= center->x + position_radius + 1e-6;
      x += position_step)
    {
      for (double y = center->y - position_radius; y <= center->y + position_radius + 1e-6;
        y += position_step)
      {
        for (double yaw = center->yaw - yaw_radius; yaw <= center->yaw + yaw_radius + 1e-6;
          yaw += yaw_step)
        {
          Pose2D pose;
          pose.x = x;
          pose.y = y;
          pose.yaw = NormalizeAngle(yaw);
          append_candidate(pose);
        }
      }
    }
  } else {
    const double map_max_x =
      parameters_.map_origin[0] + static_cast<double>(map_width_) * parameters_.cell_size;
    const double map_max_y =
      parameters_.map_origin[1] + static_cast<double>(map_height_) * parameters_.cell_size;

    for (double x = parameters_.map_origin[0]; x <= map_max_x + 1e-6; x += position_step) {
      for (double y = parameters_.map_origin[1]; y <= map_max_y + 1e-6; y += position_step) {
        for (double yaw = -kPi; yaw <= kPi + 1e-6; yaw += yaw_step) {
          Pose2D pose;
          pose.x = x;
          pose.y = y;
          pose.yaw = NormalizeAngle(yaw);
          append_candidate(pose);
        }
      }
    }
  }

  std::sort(
    candidates.begin(), candidates.end(),
    [](const CandidateScore & lhs, const CandidateScore & rhs) {
      return lhs.score < rhs.score;
    });
  if (candidates.size() > top_k) {
    candidates.resize(top_k);
  }
  return candidates;
}

std::optional<OccupancyGridLocalizerCore::CandidateScore>
OccupancyGridLocalizerCore::FindBestPose(
  const isaac_ros_pointcloud_interfaces::msg::FlatScan & flat_scan,
  const std::optional<geometry_msgs::msg::TransformStamped> & base_link_to_lidar) const
{
  if (!IsMapLoaded()) {
    throw std::runtime_error("Occupancy grid localizer map was not loaded before localization");
  }

  std::vector<BeamSample> beams = BuildBeamSamples(flat_scan);
  if (beams.empty()) {
    return std::nullopt;
  }

  const auto angle_bounds = std::minmax_element(
    beams.begin(), beams.end(),
    [](const BeamSample & lhs, const BeamSample & rhs) {
      return lhs.angle < rhs.angle;
    });
  const double scan_fov_deg =
    (angle_bounds.second->angle - angle_bounds.first->angle) * 180.0 / kPi;
  if (scan_fov_deg < parameters_.min_scan_fov_degrees) {
    return std::nullopt;
  }

  Pose2D lidar_offset;
  if (base_link_to_lidar) {
    const auto & tf = *base_link_to_lidar;
    lidar_offset.x = tf.transform.translation.x;
    lidar_offset.y = tf.transform.translation.y;
    const auto & q = tf.transform.rotation;
    lidar_offset.yaw = std::atan2(
      2.0 * (q.w * q.z + q.x * q.y),
      1.0 - 2.0 * (q.y * q.y + q.z * q.z));
  }

  const double fine_position_step = std::max(parameters_.sample_distance, parameters_.cell_size);

  // Select GPU or CPU search path
  const auto search = [this](
    const std::vector<BeamSample> & b, const Pose2D & lidar,
    double pos_step, double y_step, std::optional<Pose2D> ctr,
    double pos_radius, double yaw_radius, size_t top_k)
    {
      if (gpu_) {
        return SearchCandidatesGpu(b, lidar, pos_step, y_step, ctr, pos_radius, yaw_radius, top_k);
      }
      return SearchCandidates(b, lidar, pos_step, y_step, ctr, pos_radius, yaw_radius, top_k);
    };

  const std::vector<CandidateScore> coarse_candidates = search(
    beams, lidar_offset,
    std::max(kCoarsePositionStepMeters, fine_position_step * 4.0),
    kCoarseYawStepRadians,
    std::nullopt,
    0.0,
    kCoarseYawRadiusRadians,
    kCoarseTopK);
  if (coarse_candidates.empty()) {
    return std::nullopt;
  }

  std::vector<CandidateScore> medium_candidates;
  for (const auto & candidate : coarse_candidates) {
    std::vector<CandidateScore> refined = search(
      beams, lidar_offset,
      std::max(kMediumPositionStepMeters, fine_position_step * 2.0),
      kMediumYawStepRadians,
      candidate.pose,
      std::max(kCoarsePositionStepMeters, fine_position_step * 4.0),
      kMediumYawRadiusRadians,
      kMediumTopK);
    medium_candidates.insert(medium_candidates.end(), refined.begin(), refined.end());
  }
  if (medium_candidates.empty()) {
    return coarse_candidates.front();
  }

  std::sort(
    medium_candidates.begin(), medium_candidates.end(),
    [](const CandidateScore & lhs, const CandidateScore & rhs) {
      return lhs.score < rhs.score;
    });
  if (medium_candidates.size() > kMediumTopK) {
    medium_candidates.resize(kMediumTopK);
  }

  std::vector<CandidateScore> fine_candidates;
  for (const auto & candidate : medium_candidates) {
    std::vector<CandidateScore> refined = search(
      beams, lidar_offset,
      fine_position_step,
      kFineYawStepRadians,
      candidate.pose,
      std::max(kMediumPositionStepMeters, fine_position_step * 2.0),
      kFineYawRadiusRadians,
      kFineTopK);
    fine_candidates.insert(fine_candidates.end(), refined.begin(), refined.end());
  }
  if (fine_candidates.empty()) {
    return medium_candidates.front();
  }

  const auto best_it = std::min_element(
    fine_candidates.begin(), fine_candidates.end(),
    [](const CandidateScore & lhs, const CandidateScore & rhs) {
      return lhs.score < rhs.score;
    });
  if (best_it == fine_candidates.end() || best_it->score > parameters_.max_output_error) {
    return std::nullopt;
  }
  return *best_it;
}

std::vector<OccupancyGridLocalizerCore::CandidateScore>
OccupancyGridLocalizerCore::SearchCandidatesGpu(
  const std::vector<BeamSample> & beams,
  const Pose2D & base_link_to_lidar,
  double position_step,
  double yaw_step,
  std::optional<Pose2D> center,
  double position_radius,
  double yaw_radius,
  size_t top_k) const
{
  // Build candidate pose list on CPU
  std::vector<GpuPose> gpu_poses;

  if (center) {
    for (double x = center->x - position_radius; x <= center->x + position_radius + 1e-6;
      x += position_step)
    {
      for (double y = center->y - position_radius; y <= center->y + position_radius + 1e-6;
        y += position_step)
      {
        for (double yaw = center->yaw - yaw_radius; yaw <= center->yaw + yaw_radius + 1e-6;
          yaw += yaw_step)
        {
          GpuPose p;
          p.x = static_cast<float>(x);
          p.y = static_cast<float>(y);
          p.yaw = static_cast<float>(NormalizeAngle(yaw));
          gpu_poses.push_back(p);
        }
      }
    }
  } else {
    const double map_max_x =
      parameters_.map_origin[0] + static_cast<double>(map_width_) * parameters_.cell_size;
    const double map_max_y =
      parameters_.map_origin[1] + static_cast<double>(map_height_) * parameters_.cell_size;

    for (double x = parameters_.map_origin[0]; x <= map_max_x + 1e-6; x += position_step) {
      for (double y = parameters_.map_origin[1]; y <= map_max_y + 1e-6; y += position_step) {
        for (double yaw = -kPi; yaw <= kPi + 1e-6; yaw += yaw_step) {
          GpuPose p;
          p.x = static_cast<float>(x);
          p.y = static_cast<float>(y);
          p.yaw = static_cast<float>(NormalizeAngle(yaw));
          gpu_poses.push_back(p);
        }
      }
    }
  }

  if (gpu_poses.empty()) {
    return {};
  }

  // Upload beams and score all poses on GPU
  std::vector<GpuBeam> gpu_beams(beams.size());
  for (size_t i = 0; i < beams.size(); ++i) {
    gpu_beams[i].angle = static_cast<float>(beams[i].angle);
    gpu_beams[i].range = static_cast<float>(beams[i].range);
  }
  gpu_->UploadBeams(gpu_beams);

  const int min_valid_hits = static_cast<int>(
    std::max<size_t>(4, beams.size() / 12));

  std::vector<float> scores;
  gpu_->ScorePoses(
    gpu_poses, scores,
    static_cast<float>(parameters_.map_origin[0]),
    static_cast<float>(parameters_.map_origin[1]),
    static_cast<float>(parameters_.map_origin[2]),
    static_cast<float>(parameters_.cell_size),
    static_cast<float>(base_link_to_lidar.x),
    static_cast<float>(base_link_to_lidar.y),
    static_cast<float>(base_link_to_lidar.yaw),
    static_cast<float>(parameters_.robot_radius),
    static_cast<float>(parameters_.max_beam_error),
    static_cast<float>(parameters_.out_of_range_threshold),
    min_valid_hits);

  // Collect finite-scored candidates and sort
  std::vector<CandidateScore> candidates;
  for (size_t i = 0; i < gpu_poses.size(); ++i) {
    if (std::isfinite(scores[i])) {
      CandidateScore c;
      c.pose.x = gpu_poses[i].x;
      c.pose.y = gpu_poses[i].y;
      c.pose.yaw = gpu_poses[i].yaw;
      c.score = scores[i];
      candidates.push_back(c);
    }
  }

  std::sort(
    candidates.begin(), candidates.end(),
    [](const CandidateScore & lhs, const CandidateScore & rhs) {
      return lhs.score < rhs.score;
    });
  if (candidates.size() > top_k) {
    candidates.resize(top_k);
  }
  return candidates;
}

}  // namespace occupancy_grid_localizer
}  // namespace isaac_ros
}  // namespace nvidia
