# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.22.1)
project(isaac_mapping_ros LANGUAGES C CXX CUDA)

# Default to release build
if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

find_package(OpenCV REQUIRED)
find_package(urdf REQUIRED)
find_package(gflags REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(glog REQUIRED)

pkg_check_modules(FFMPEG REQUIRED libavcodec libavformat libavutil libswscale)

# Video Decoder
add_library(video_decoder SHARED src/video_decoder.cpp)
target_include_directories(video_decoder PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(video_decoder OpenCV rosbag2_cpp sensor_msgs)
target_link_libraries(video_decoder ${FFMPEG_LIBRARIES})

# Data Converter Utils
add_library(data_converter_utils SHARED src/data_converter_utils.cpp)
target_include_directories(data_converter_utils PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(data_converter_utils video_decoder yaml-cpp)
ament_target_dependencies(data_converter_utils nav_msgs urdf OpenCV rosbag2_cpp sensor_msgs tf2_ros cv_bridge isaac_ros_visual_mapping)

# Nvblox Utils
add_library(nvblox_utils SHARED src/lib/nvblox_utils/mapping_data_loader.cpp)
ament_target_dependencies(nvblox_utils OpenCV)
target_include_directories(nvblox_utils PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(nvblox_utils nvblox::nvblox_datasets)

# rosbag_to_mapping_data binary
ament_auto_add_executable(rosbag_to_mapping_data src/rosbag_to_mapping_data.cpp)
target_link_libraries(rosbag_to_mapping_data data_converter_utils gflags)
set_target_properties(rosbag_to_mapping_data PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE)

# rosbag_poses_to_tum_format binary
ament_auto_add_executable(rosbag_poses_to_tum_format src/rosbag_poses_to_tum_format.cpp)
ament_target_dependencies(rosbag_poses_to_tum_format rosbag2_cpp nav_msgs)
target_link_libraries(rosbag_poses_to_tum_format data_converter_utils gflags)
set_target_properties(rosbag_poses_to_tum_format PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE)

# mapping_pose_to_rosbag binary
ament_auto_add_executable(mapping_pose_to_rosbag src/mapping_pose_to_rosbag.cpp)
ament_target_dependencies(mapping_pose_to_rosbag isaac_ros_visual_mapping rosbag2_cpp nav_msgs tf2_msgs)
target_link_libraries(mapping_pose_to_rosbag gflags)
set_target_properties(mapping_pose_to_rosbag PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE)

# camera_frame_selection binary
ament_auto_add_executable(camera_frame_selection src/camera_frame_selection.cpp)
target_link_libraries(camera_frame_selection data_converter_utils gflags)
set_target_properties(camera_frame_selection PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE)

ament_auto_add_executable(decode_video src/decode_video.cpp)
target_link_libraries(decode_video video_decoder gflags)
ament_target_dependencies(decode_video isaac_ros_visual_mapping)
set_target_properties(decode_video PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE)

# run_nvblox binary
ament_auto_add_executable(run_nvblox src/run_nvblox.cpp)
target_link_libraries(run_nvblox nvblox_utils)
ament_target_dependencies(run_nvblox nvblox_ros)

# This is needed for the binary to find GXF related libs on aarch64
set_target_properties(run_nvblox PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE)

# select_frames_meta binary
ament_auto_add_executable(select_frames_meta src/select_frames_meta.cpp)
ament_target_dependencies(select_frames_meta isaac_ros_visual_mapping)
target_link_libraries(select_frames_meta gflags)
set_target_properties(select_frames_meta PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE)

# Tests
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

install(
  DIRECTORY include/
  DESTINATION include
)

# Install the library and headers
install(
  TARGETS data_converter_utils video_decoder nvblox_utils
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

install(PROGRAMS
  scripts/run_ess_trt_offline.py
  scripts/run_foundationstereo_trt_offline.py
  scripts/create_map_offline.py
  scripts/stereo_inference_base.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(ament_cmake nvblox nvblox_ros)

# Embed versioning information into installed files
ament_index_get_resource(ISAAC_ROS_COMMON_CMAKE_PATH isaac_ros_common_cmake_path isaac_ros_common)
include("${ISAAC_ROS_COMMON_CMAKE_PATH}/isaac_ros_common-version-info.cmake")
generate_version_info(${PROJECT_NAME})

ament_auto_package()
