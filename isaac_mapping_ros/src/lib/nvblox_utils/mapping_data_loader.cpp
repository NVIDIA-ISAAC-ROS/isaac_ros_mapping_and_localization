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

#include "isaac_mapping_ros/nvblox_utils/mapping_data_loader.hpp"

#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <type_traits>

#include "nlohmann/json.hpp"
#include "nvblox/utils/timing.h"
#include "opencv2/opencv.hpp"

namespace nvblox
{
namespace datasets
{
namespace mapping_data
{
namespace
{

// Helper functions for safe JSON field access with default values
template<typename T>
T getJsonValue(const nlohmann::json & json, const std::string & key, const T & default_value = T{})
{
  if (json.contains(key) && !json[key].is_null()) {
    return json[key].get<T>();
  }
  return default_value;
}

template<typename T>
T getJsonValueFromPath(
  const nlohmann::json & json, const std::vector<std::string> & path,
  const T & default_value = T{})
{
  const nlohmann::json * current = &json;
  for (const auto & key : path) {
    if (!current->contains(key) || (*current)[key].is_null()) {
      return default_value;
    }
    current = &(*current)[key];
  }
  return current->get<T>();
}

std::string getPathForColorImage(
  const std::string & base_path,
  const std::vector<KeyframeMetadata> & keyframe_metadatas,
  const int frame_id)
{
  std::string path =
    base_path + "/" + keyframe_metadatas[frame_id].color_image_path;
  VLOG(1) << "Load color image from " << path;
  return path;
}

std::string getPathForDepthImage(
  const std::string & base_path,
  const std::vector<KeyframeMetadata> & keyframe_metadatas,
  const int frame_id)
{
  std::string path =
    base_path + "/" + keyframe_metadatas[frame_id].depth_image_path;
  VLOG(1) << "Load depth image from " << path;
  return path;
}

std::string ChangeFileExtension(const std::string & filename, const std::string & new_extension)
{
  // Find the last dot in the filename to identify the current extension
  size_t dot_position = filename.rfind('.');

  // If there's no dot, assume the filename has no extension
  if (dot_position == std::string::npos) {
    return filename + '.' + new_extension;
  }

  // Extract the base filename (without the current extension) and add the new extension
  return filename.substr(0, dot_position + 1) + new_extension;
}

template<typename T>
T stringToNumber(const std::string & str)
{
  if constexpr (std::is_same_v<T, int>) {
    return std::stoi(str);
  } else if constexpr (std::is_same_v<T, uint32_t>) {
    return std::stoul(str);
  } else if constexpr (std::is_same_v<T, uint64_t>) {
    return std::stoull(str);
  } else {
    throw std::invalid_argument("Unsupported type for conversion");
  }
}

// TODO(yuchen) since this is within isaac_mapping_ros repo
// we could directly use the c++ protobuf definition instead of reading json files
Transform transformFromRigidTransform3dJson(
  const nlohmann::json & json)
{
  Transform transform;
  float angle_deg = getJsonValueFromPath<float>(json, {"axis_angle", "angle_degrees"}, 0.0f);
  float qx = getJsonValueFromPath<float>(json, {"axis_angle", "x"}, 0.0f);
  float qy = getJsonValueFromPath<float>(json, {"axis_angle", "y"}, 0.0f);
  float qz = getJsonValueFromPath<float>(json, {"axis_angle", "z"}, 1.0f);  // Default to unit z-axis

  float x = getJsonValueFromPath<float>(json, {"translation", "x"}, 0.0f);
  float y = getJsonValueFromPath<float>(json, {"translation", "y"}, 0.0f);
  float z = getJsonValueFromPath<float>(json, {"translation", "z"}, 0.0f);

  transform.translation() = Eigen::Vector3f(x, y, z);

  // Ensure we have a valid axis for rotation
  Eigen::Vector3f axis(qx, qy, qz);
  if (axis.norm() < 1e-6) {
    axis = Eigen::Vector3f(0, 0, 1);  // Default to z-axis if no valid axis
  } else {
    axis.normalize();
  }

  Eigen::AngleAxisf axis_angle(angle_deg * M_PI / 180.0, axis);
  transform.linear() = Eigen::Matrix3f(axis_angle);
  return transform;
}

Camera cameraFromMonoCalibrationParametersJson(
  const nlohmann::json & json)
{
  // projection_matrix is a row-major stored 3x4 matrix:
  // fu, 0, cu, 0
  // 0, fv, cv, 0
  // 0,  0,  1, 0

  // Safe access to projection matrix data with default values
  auto getProjectionMatrixElement = [&](int index, double default_val) -> double {
      if (json.contains("projection_matrix") &&
        json["projection_matrix"].contains("data") &&
        json["projection_matrix"]["data"].is_array() &&
        json["projection_matrix"]["data"].size() > static_cast<size_t>(index))
      {
        return json["projection_matrix"]["data"][index].get<double>();
      }
      return default_val;
    };

  double fu = getProjectionMatrixElement(0, 1.0);  // Default focal length
  double fv = getProjectionMatrixElement(5, 1.0);  // Default focal length
  double cu = getProjectionMatrixElement(2, 0.0);  // Default principal point
  double cv = getProjectionMatrixElement(6, 0.0);  // Default principal point

  int width = getJsonValue<int>(json, "image_width", 0);
  int height = getJsonValue<int>(json, "image_height", 0);

  return Camera(fu, fv, cu, cv, width, height);
}

std::unique_ptr<ImageLoader<DepthImage>> createDepthImageLoader(
  const std::string & image_dir,
  const std::vector<KeyframeMetadata> & keyframe_metadatas,
  const bool multithreaded)
{
  return createImageLoader<DepthImage>(
    std::bind(
      getPathForDepthImage, image_dir, keyframe_metadatas,
      std::placeholders::_1),
    multithreaded);
}

// Custom color image loader that handles monochrome images by converting them to RGB
class MonochromeAwareColorImageLoader : public ImageLoader<ColorImage>
{
public:
  MonochromeAwareColorImageLoader(
    std::function<std::string(int)> path_getter_function,
    bool multithreaded)
  : ImageLoader<ColorImage>(path_getter_function),
    path_getter_function_(path_getter_function) {}

public:
  bool getNextImage(ColorImage * image_ptr) override
  {
    const std::string image_path = path_getter_function_(image_idx_);
    ++image_idx_;

    // Load image with OpenCV to check channels first
    cv::Mat cv_image = cv::imread(image_path, cv::IMREAD_UNCHANGED);
    if (cv_image.empty()) {
      LOG(ERROR) << "Failed to load image: " << image_path;
      return false;
    }

    // Convert to RGBA format as expected by nvblox
    cv::Mat rgba_image;
    if (cv_image.channels() == 1) {
      cv::cvtColor(cv_image, rgba_image, cv::COLOR_GRAY2RGBA);
    } else if (cv_image.channels() == 3) {
      // Convert BGR to RGBA (OpenCV default is BGR)
      cv::cvtColor(cv_image, rgba_image, cv::COLOR_BGR2RGBA);
    } else if (cv_image.channels() == 4) {
      // Convert BGRA to RGBA
      cv::cvtColor(cv_image, rgba_image, cv::COLOR_BGRA2RGBA);
    } else {
      LOG(ERROR) << "Unsupported number of channels: " << cv_image.channels()
                 << " in image: " << image_path;
      return false;
    }

    // Ensure we have a 4-channel RGBA image as expected by nvblox
    if (rgba_image.channels() != 4) {
      LOG(ERROR) << "Failed to convert to RGBA format for image: " << image_path;
      return false;
    }

    // Populate the ColorImage directly from OpenCV Mat
    const int height = rgba_image.rows;
    const int width = rgba_image.cols;

    // Convert OpenCV RGBA data to nvblox Color format
    image_ptr->copyFrom(height, width, reinterpret_cast<const nvblox::Color *>(rgba_image.data));

    return true;
  }

private:
  std::function<std::string(int)> path_getter_function_;
};

std::unique_ptr<ImageLoader<ColorImage>> createColorImageLoader(
  const std::string & image_dir,
  const std::vector<KeyframeMetadata> & keyframe_metadatas,
  const bool multithreaded)
{
  return std::make_unique<MonochromeAwareColorImageLoader>(
    std::bind(
      getPathForColorImage, image_dir, keyframe_metadatas,
      std::placeholders::_1),
    multithreaded);
}

bool loadKeyframeMetadataCollection(
  const std::string & color_image_dir,
  const std::string & depth_image_dir,
  const std::string & frames_meta_file,
  std::vector<KeyframeMetadata> * keyframe_metadatas,
  std::unordered_map<uint32_t, Camera> * cameras)
{
  std::ifstream ifs(frames_meta_file);
  if (!ifs.is_open()) {
    LOG(ERROR) << "Failed to open file: " << frames_meta_file;
    return false;
  }

  nlohmann::json json = nlohmann::json::parse(ifs);

  bool found_depth = false;
  bool change_depth_extension_to_png = false;
  try {
    if (!json.contains("keyframes_metadata") || json["keyframes_metadata"].is_null()) {
      LOG(WARNING) << "No keyframes_metadata found in JSON file";
      return true;  // Empty dataset is valid
    }

    for (const auto & metadata_j : json["keyframes_metadata"]) {
      KeyframeMetadata metadata;
      metadata.fromJson(metadata_j);

      if (!std::filesystem::exists(
          color_image_dir + "/" +
          metadata.color_image_path))
      {
        continue;
      }

      if (!found_depth) {
        if (std::filesystem::exists(depth_image_dir + "/" + metadata.depth_image_path)) {
          found_depth = true;
        } else if (std::filesystem::exists(
            depth_image_dir + "/" +
            ChangeFileExtension(metadata.depth_image_path, "png")))
        {
          found_depth = true;
          change_depth_extension_to_png = true;
          LOG(INFO) << "Found depth PNG files";
        }
      }

      if (change_depth_extension_to_png) {
        metadata.depth_image_path = ChangeFileExtension(metadata.depth_image_path, "png");
      }

      if (!std::filesystem::exists(
          depth_image_dir + "/" + metadata.depth_image_path))
      {
        continue;
      }

      keyframe_metadatas->push_back(metadata);
    }
    // sort by timestamps
    std::sort(
      keyframe_metadatas->begin(), keyframe_metadatas->end(),
      [](const KeyframeMetadata & metadata1,
      const KeyframeMetadata & metadata2) {
        return metadata1.timestamp_microseconds <
        metadata2.timestamp_microseconds;
      });
    LOG(INFO) << "Number of keyframes loaded: " << keyframe_metadatas->size();

    // Handle camera parameters safely
    if (json.contains("camera_params_id_to_camera_params") &&
      !json["camera_params_id_to_camera_params"].is_null())
    {
      for (const auto & [key, value] : json["camera_params_id_to_camera_params"].items()) {
        if (value.contains("calibration_parameters") &&
          !value["calibration_parameters"].is_null())
        {
          Camera camera = cameraFromMonoCalibrationParametersJson(value["calibration_parameters"]);
          try {
            cameras->emplace(stringToNumber<uint32_t>(key), camera);
          } catch (const std::exception &) {
            LOG(WARNING) << "Failed to parse camera_params_id: " << key;
          }
        } else {
          LOG(WARNING) << "Missing calibration_parameters for camera_params_id: " << key;
        }
      }
    } else {
      LOG(WARNING) << "No camera_params_id_to_camera_params found in JSON file";
    }
  } catch (const std::exception & e) {
    LOG(ERROR) << "Error: " << e.what();
    return false;
  }

  return true;
}
}  // namespace

std::unique_ptr<Fuser> createFuser(
  const std::string & color_image_dir,
  const std::string & depth_image_dir,
  const std::string & frames_meta_file, bool init_from_gflags,
  bool fit_to_z_plane,
  const std::string & output_dir)
{
  auto data_loader =
    DataLoader::create(
    color_image_dir, depth_image_dir, frames_meta_file,
    false /* if use multithread*/, fit_to_z_plane, output_dir);
  if (!data_loader) {
    return std::unique_ptr<Fuser>();
  }
  return std::make_unique<Fuser>(std::move(data_loader), init_from_gflags);
}

std::unique_ptr<DataLoader> DataLoader::create(
  const std::string & color_image_dir,
  const std::string & depth_image_dir,
  const std::string & frames_meta_file,
  bool multithreaded,
  bool fit_to_z_plane,
  const std::string & output_dir)
{
  LOG(INFO) << "Load color images from " << color_image_dir;
  LOG(INFO) << "Load depth images from " << depth_image_dir;
  LOG(INFO) << "Load frames_meta from " << frames_meta_file;

  // Construct a dataset loader but only return it if everything worked.
  std::vector<KeyframeMetadata> keyframe_metadatas;
  std::unordered_map<uint32_t, Camera> cameras;
  loadKeyframeMetadataCollection(
    color_image_dir, depth_image_dir, frames_meta_file, &keyframe_metadatas, &cameras);
  auto dataset_loader = std::make_unique<DataLoader>(
    color_image_dir, depth_image_dir, keyframe_metadatas, cameras, multithreaded, fit_to_z_plane,
    output_dir);
  if (dataset_loader->setup_success_) {
    return dataset_loader;
  } else {
    return std::unique_ptr<DataLoader>();
  }
}

void KeyframeMetadata::fromJson(const nlohmann::json & json)
{
  // In protobuf json serialization, int and uint64 are converted to strings.
  // Use safe field access with default values

  std::string timestamp_str = getJsonValue<std::string>(json, "timestamp_microseconds", "0");
  if (!timestamp_str.empty()) {
    try {
      timestamp_microseconds = stringToNumber<uint64_t>(timestamp_str);
    } catch (const std::exception &) {
      timestamp_microseconds = 0;
    }
  } else {
    timestamp_microseconds = 0;
  }

  color_image_path = getJsonValue<std::string>(json, "image_name", "");
  depth_image_path = color_image_path;

  std::string camera_id_str = getJsonValue<std::string>(json, "camera_params_id", "0");
  if (!camera_id_str.empty()) {
    try {
      camera_params_id = stringToNumber<uint32_t>(camera_id_str);
    } catch (const std::exception &) {
      camera_params_id = 0;
    }
  } else {
    camera_params_id = 0;
  }

  // Handle camera_to_world transform safely
  if (json.contains("camera_to_world") && !json["camera_to_world"].is_null()) {
    camera_to_world = transformFromRigidTransform3dJson(json["camera_to_world"]);
  } else {
    // Default to identity transform
    camera_to_world = Transform::Identity();
  }
}

DataLoader::DataLoader(
  const std::string & color_image_dir,
  const std::string & depth_image_dir,
  const std::vector<KeyframeMetadata> & keyframe_metadatas,
  const std::unordered_map<uint32_t, Camera> & cameras,
  bool multithreaded,
  bool fit_to_z_plane,
  const std::string & output_dir)
: RgbdDataLoaderInterface(),
  keyframe_metadatas_(keyframe_metadatas),
  cameras_(cameras),
  depth_image_loader_(createDepthImageLoader(depth_image_dir, keyframe_metadatas, multithreaded)),
  color_image_loader_(createColorImageLoader(color_image_dir, keyframe_metadatas, multithreaded)),
  fit_to_z_plane_(fit_to_z_plane),
  output_dir_(output_dir),
  T_world_to_z0_plane_(Transform::Identity()),
  has_z_plane_transform_(false)
{
  if (fit_to_z_plane_) {
    computeZPlaneTransform();
  }
}

DataLoadResult DataLoader::loadNext(
  DepthImage * depth_frame_ptr,
  Transform * T_L_C_ptr, Camera * camera_ptr,
  ColorImage * color_frame_ptr)
{
  CHECK(setup_success_);
  CHECK_NOTNULL(depth_frame_ptr);
  CHECK_NOTNULL(T_L_C_ptr);
  CHECK_NOTNULL(camera_ptr);

  // Because we might fail along the way, increment the frame number before we
  // start.
  ++frame_number_;

  if (frame_number_ > static_cast<int32_t>(keyframe_metadatas_.size())) {
    LOG(INFO) << "Reached the last frame";
    return DataLoadResult::kNoMoreData;
  }

  // Load the image into a Depth Frame.
  CHECK(depth_image_loader_);
  timing::Timer timer_file_depth("file_loading/depth_image");
  if (!depth_image_loader_->getNextImage(depth_frame_ptr)) {
    LOG(INFO) << "Couldn't find depth image";
    return DataLoadResult::kBadFrame;
  }
  timer_file_depth.Stop();

  // Load the color image into a ColorImage
  if (color_frame_ptr) {
    CHECK(color_image_loader_);
    timing::Timer timer_file_color("file_loading/color_image");
    if (!color_image_loader_->getNextImage(color_frame_ptr)) {
      LOG(INFO) << "Couldn't find color image";
      return DataLoadResult::kBadFrame;
    }
    timer_file_color.Stop();
  }

  int32_t current_frame_id = frame_number_ - 1;

  // Get the camera for this frame.
  timing::Timer timer_file_intrinsics("file_loading/camera");
  uint32_t camera_id = keyframe_metadatas_[current_frame_id].camera_params_id;
  auto camera_it = cameras_.find(camera_id);
  if (camera_it != cameras_.end()) {
    *camera_ptr = camera_it->second;
  } else {
    LOG(ERROR) << "Camera parameters not found for camera_params_id: " << camera_id
               << ". Using default camera parameters.";
    return DataLoadResult::kBadFrame;
  }
  timer_file_intrinsics.Stop();

  // Get the next pose
  timing::Timer timer_file_pose("file_loading/pose");
  *T_L_C_ptr = keyframe_metadatas_[current_frame_id].camera_to_world;

  // Apply z-plane alignment transform if enabled
  if (fit_to_z_plane_ && has_z_plane_transform_) {
    *T_L_C_ptr = T_world_to_z0_plane_ * (*T_L_C_ptr);
  }

  // Check that the loaded data doesn't contain NaNs or a faulty rotation
  // matrix. This does occur. If we find one, skip that frame and move to the
  // next.
  constexpr float kRotationMatrixDetEpsilon = 1e-4;
  if (!T_L_C_ptr->matrix().allFinite() ||
    std::abs(T_L_C_ptr->matrix().block<3, 3>(0, 0).determinant() - 1.0f) >
    kRotationMatrixDetEpsilon)
  {
    LOG(WARNING) << "Bad camera to world transform matrix";
    return DataLoadResult::kBadFrame;  // Bad data, but keep going.
  }

  VLOG(1) << "Current frame_id: " << current_frame_id
          << ", timestamp: " << keyframe_metadatas_[current_frame_id].timestamp_microseconds
          << ", camera_to_world:\n"
          << T_L_C_ptr->matrix() << "\ncamera_id: "
          << keyframe_metadatas_[current_frame_id].camera_params_id
          << ", camera:\n"
          << *camera_ptr;

  timer_file_pose.Stop();
  return DataLoadResult::kSuccess;
}

DataLoadResult DataLoader::loadNext(
  DepthImage * depth_frame_ptr,
  Transform * T_L_D_ptr,
  Camera * depth_camera_ptr,
  ColorImage * color_frame_ptr,
  Transform * T_L_C_ptr,
  Camera * color_camera_ptr)
{
  // NOTE: The other pointers are checked non-null below
  CHECK_NOTNULL(color_frame_ptr);
  CHECK_NOTNULL(T_L_C_ptr);
  CHECK_NOTNULL(color_camera_ptr);
  // For the replica dataset the depth and color cameras are the same, so just
  // copying over.
  auto result =
    loadNext(depth_frame_ptr, T_L_D_ptr, depth_camera_ptr, color_frame_ptr);
  *T_L_C_ptr = *T_L_D_ptr;
  *color_camera_ptr = *depth_camera_ptr;
  return result;
}

void DataLoader::computeZPlaneTransform()
{
  if (keyframe_metadatas_.empty()) {
    LOG(WARNING) << "No keyframe metadata available for z-plane alignment";
    return;
  }

  // Extract all camera positions
  std::vector<Eigen::Vector3f> positions;
  positions.reserve(keyframe_metadatas_.size());

  for (const auto & metadata : keyframe_metadatas_) {
    positions.push_back(metadata.camera_to_world.translation());
  }

  if (positions.size() < 3) {
    LOG(WARNING) << "Need at least 3 poses for z-plane alignment, got " << positions.size();
    return;
  }

  // Compute centroid
  Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
  for (const auto & pos : positions) {
    centroid += pos;
  }
  centroid /= static_cast<float>(positions.size());

  // Build matrix of centered positions
  Eigen::Matrix<float, 3, Eigen::Dynamic> centered_positions(3, positions.size());
  for (size_t i = 0; i < positions.size(); ++i) {
    centered_positions.col(i) = positions[i] - centroid;
  }

  // Compute SVD to find the best-fit plane
  Eigen::JacobiSVD<Eigen::Matrix<float, 3, Eigen::Dynamic>> svd(
    centered_positions, Eigen::ComputeFullU | Eigen::ComputeFullV);

  // The plane normal is the column of U corresponding to the smallest singular value
  Eigen::Vector3f plane_normal = svd.matrixU().col(2);

  // Ensure the normal points upward (positive z component in the original frame)
  if (plane_normal.z() < 0) {
    plane_normal = -plane_normal;
  }

  // Compute rotation to align plane normal with z-axis (0, 0, 1)
  Eigen::Vector3f target_normal(0.0f, 0.0f, 1.0f);

  // If the plane normal is already aligned with z-axis, use identity rotation
  if ((plane_normal - target_normal).norm() < 1e-6f) {
    T_world_to_z0_plane_.linear() = Eigen::Matrix3f::Identity();
  } else if ((plane_normal + target_normal).norm() < 1e-6f) {
    // If plane normal is opposite to z-axis, rotate 180 degrees around x-axis
    T_world_to_z0_plane_.linear() = Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitX()).matrix();
  } else {
    // Compute rotation using Rodrigues' formula
    Eigen::Vector3f rotation_axis = plane_normal.cross(target_normal).normalized();
    float cos_angle = plane_normal.dot(target_normal);
    float angle = std::acos(std::clamp(cos_angle, -1.0f, 1.0f));

    Eigen::AngleAxisf rotation(angle, rotation_axis);
    T_world_to_z0_plane_.linear() = rotation.matrix();
  }

  // Transform the centroid to the new coordinate system and set translation
  // so that the centroid lies on the z=0 plane
  Eigen::Vector3f transformed_centroid = T_world_to_z0_plane_.linear() * centroid;
  T_world_to_z0_plane_.translation() = Eigen::Vector3f(0.0f, 0.0f, -transformed_centroid.z());

  has_z_plane_transform_ = true;

  LOG(INFO) << "Computed z-plane alignment transform from " << positions.size() << " poses";
  LOG(INFO) << "Original plane normal: " << plane_normal.transpose();
  LOG(INFO) << "Centroid: " << centroid.transpose();
  LOG(INFO) << "Transform matrix:\n" << T_world_to_z0_plane_.matrix();

  // Save transform to JSON if output directory is provided
  if (!output_dir_.empty()) {
    saveTransformToJson();
  }
}

bool DataLoader::saveTransformToJson() const
{
  if (!has_z_plane_transform_ || output_dir_.empty()) {
    return false;
  }

  try {
    nlohmann::json transform_json;

    // Save rotation matrix
    const Eigen::Matrix3f & rotation = T_world_to_z0_plane_.linear();
    transform_json["rotation"] = {
      {rotation(0, 0), rotation(0, 1), rotation(0, 2)},
      {rotation(1, 0), rotation(1, 1), rotation(1, 2)},
      {rotation(2, 0), rotation(2, 1), rotation(2, 2)}
    };

    // Save translation
    const Eigen::Vector3f & translation = T_world_to_z0_plane_.translation();
    transform_json["translation"] = {translation.x(), translation.y(), translation.z()};

    // Save as 4x4 homogeneous matrix for convenience
    const Eigen::Matrix4f & matrix = T_world_to_z0_plane_.matrix();
    transform_json["homogeneous_matrix"] = {
      {matrix(0, 0), matrix(0, 1), matrix(0, 2), matrix(0, 3)},
      {matrix(1, 0), matrix(1, 1), matrix(1, 2), matrix(1, 3)},
      {matrix(2, 0), matrix(2, 1), matrix(2, 2), matrix(2, 3)},
      {matrix(3, 0), matrix(3, 1), matrix(3, 2), matrix(3, 3)}
    };

    std::string json_path = output_dir_ + "/T_world_to_z0.json";
    std::ofstream json_file(json_path);
    if (!json_file.is_open()) {
      LOG(ERROR) << "Failed to open file for writing: " << json_path;
      return false;
    }

    json_file << transform_json.dump(4);  // Pretty print with 4-space indent
    json_file.close();

    LOG(INFO) << "Saved z-plane transform to: " << json_path;
    return true;
  } catch (const std::exception & e) {
    LOG(ERROR) << "Failed to save transform to JSON: " << e.what();
    return false;
  }
}

}  // namespace mapping_data
}  // namespace datasets
}  // namespace nvblox
