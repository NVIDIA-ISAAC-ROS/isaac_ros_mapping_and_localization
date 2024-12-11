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
project(isaac_ros_visual_global_localization LANGUAGES C CXX)

set(CMAKE_WARN_DEPRECATED FALSE)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra)
endif()

find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

set(ISAAC_ROS_VISUAL_LOCALIZATION_NAME "isaac_ros_visual_global_localization")
set(ISAAC_ROS_APRILTAG_LOCALIZATION_NAME "isaac_ros_apriltag_localization")
set(ISAAC_ROS_GLOBAL_LOCALIZATION_MAPPER_NAME "isaac_ros_global_localization_mapper")
set(ISAAC_ROS_POINT_CLOUD_FILTER_NAME "isaac_ros_point_cloud_filter")

find_package(ament_cmake_auto REQUIRED)

ament_auto_find_build_dependencies()

# Dependencies
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(gflags REQUIRED)

# find those dependency again just to make sure if those are not missing slitenly
# in ament_auto_find_build_dependencies
find_package(isaac_ros_nitros REQUIRED)
find_package(isaac_ros_managed_nitros REQUIRED)
find_package(isaac_ros_nitros_image_type REQUIRED)
find_package(isaac_mapping_ros REQUIRED)
find_package(isaac_common REQUIRED)
find_package(isaac_ros_common REQUIRED)
find_package(isaac_ros_apriltag_interfaces REQUIRED)

include_directories(${isaac_common_INCLUDE_DIRS})

ament_auto_add_library(
  transform_manager SHARED
  src/transform_manager.cpp
)

target_link_libraries(
  transform_manager
  isaac_mapping_ros::common
)

ament_target_dependencies(transform_manager
  Eigen3
  rclcpp
  tf2_ros
  tf2_geometry_msgs
  nav_msgs
)

# visual Localization
ament_auto_add_library(
  visual_global_localization_node SHARED
  src/visual_global_localization_node.cpp
)

target_link_libraries(
  visual_global_localization_node
  transform_manager
  gflags
  isaac_mapping_ros::common
  isaac_mapping_ros::visual
)

# isaac_ros_visual_global_localization executable
ament_auto_add_executable(${ISAAC_ROS_VISUAL_LOCALIZATION_NAME}
  src/visual_global_localization_main.cpp
)

ament_target_dependencies(visual_global_localization_node
  Eigen3
  rclcpp
  sensor_msgs
  nav_msgs
  diagnostic_msgs
  tf2_geometry_msgs
  tf2_ros
  cv_bridge
  isaac_ros_nitros
  isaac_ros_managed_nitros
  isaac_ros_nitros_image_type
  std_srvs
)

# Apriltag Localization
ament_auto_add_library(
  apriltag_localization_node SHARED
  src/apriltag_localization_node.cpp
)

target_link_libraries(
  apriltag_localization_node
  transform_manager
  gflags
  isaac_mapping_ros::common
  isaac_mapping_ros::visual
)

# isaac_ros_apriltag_localization executable
ament_auto_add_executable(${ISAAC_ROS_APRILTAG_LOCALIZATION_NAME}
  src/apriltag_localization_main.cpp
)

ament_target_dependencies(apriltag_localization_node
  isaac_ros_apriltag_interfaces
  visualization_msgs
  rclcpp
  sensor_msgs
  tf2_ros
)

# global mapper node
ament_auto_add_library(
  global_localization_mapper_node SHARED
  src/global_localization_mapper_node.cpp
)

target_link_libraries(
  global_localization_mapper_node
  transform_manager
  gflags
  isaac_mapping_ros::common
  isaac_mapping_ros::visual
)

ament_auto_add_executable(${ISAAC_ROS_GLOBAL_LOCALIZATION_MAPPER_NAME}
  src/global_localization_mapper_main.cpp
)

ament_target_dependencies(global_localization_mapper_node
  rclcpp
  sensor_msgs
  nav_msgs
  cv_bridge
  isaac_ros_nitros
  isaac_ros_managed_nitros
  isaac_ros_nitros_image_type
)

# Point cloud filter
ament_auto_add_library(
  point_cloud_filter_node SHARED
  src/point_cloud_filter_node.cpp
)

target_link_libraries(
  point_cloud_filter_node
  gflags
)
ament_target_dependencies(point_cloud_filter_node
  rclcpp
  sensor_msgs
  tf2_ros
  tf2_sensor_msgs
)

ament_auto_add_executable(${ISAAC_ROS_POINT_CLOUD_FILTER_NAME}
  src/point_cloud_filter_main.cpp
)

include_directories(${CMAKE_CURRENT_BINARY_DIR}/isaac_mapping/generated)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

rclcpp_components_register_nodes(visual_global_localization_node "nvidia::isaac_ros::visual_global_localization::VisualGlobalLocalizationNode")
set(node_plugins "${node_plugins}nvidia::isaac_ros::visual_global_localization::VisualGlobalLocalizationNode;$<TARGET_FILE:visual_global_localization_node>\n")

rclcpp_components_register_nodes(apriltag_localization_node "nvidia::isaac_ros::visual_global_localization::AprilTagLocalizationNode")
set(node_plugins "${node_plugins}nvidia::isaac_ros::visual_global_localization::AprilTagLocalizationNode;$<TARGET_FILE:apriltag_localization_node>\n")

rclcpp_components_register_nodes(global_localization_mapper_node "nvidia::isaac_ros::visual_global_localization::GlobalLocalizationMapperNode")
set(node_plugins "${node_plugins}nvidia::isaac_ros::visual_global_localization::GlobalLocalizationMapperNode;$<TARGET_FILE:global_localization_mapper_node>\n")

target_link_libraries(${ISAAC_ROS_VISUAL_LOCALIZATION_NAME} visual_global_localization_node)
target_link_libraries(${ISAAC_ROS_APRILTAG_LOCALIZATION_NAME} apriltag_localization_node)
target_link_libraries(${ISAAC_ROS_GLOBAL_LOCALIZATION_MAPPER_NAME} global_localization_mapper_node)

rclcpp_components_register_nodes(point_cloud_filter_node "nvidia::isaac_ros::visual_global_localization::PointCloudFilterNode")
set(node_plugins "${node_plugins}nvidia::isaac_ros::visual_global_localization::PointCloudFilterNode;$<TARGET_FILE:point_cloud_filter_node>\n")

target_link_libraries(${ISAAC_ROS_VISUAL_LOCALIZATION_NAME} visual_global_localization_node)
target_link_libraries(${ISAAC_ROS_APRILTAG_LOCALIZATION_NAME} apriltag_localization_node)
target_link_libraries(${ISAAC_ROS_POINT_CLOUD_FILTER_NAME} point_cloud_filter_node)

# NOTE(alexmillane 14.02.2024): This is needed at the moment to find libgxf_isaac_optimizer.so
# See: https://nvidia.slack.com/archives/C023NB2F7SN/p1707529762925559.
set_target_properties(visual_global_localization_node PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE)

set_target_properties(apriltag_localization_node PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE)

set_target_properties(global_localization_mapper_node PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE)

install(TARGETS
  transform_manager ${ISAAC_ROS_VISUAL_LOCALIZATION_NAME} ${ISAAC_ROS_APRILTAG_LOCALIZATION_NAME} ${ISAAC_ROS_GLOBAL_LOCALIZATION_MAPPER_NAME} ${ISAAC_ROS_POINT_CLOUD_FILTER_NAME}
  LIBRARY DESTINATION lib
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/rviz
  DESTINATION share/${PROJECT_NAME})


# Embed versioning information into installed files
ament_index_get_resource(ISAAC_ROS_COMMON_CMAKE_PATH isaac_ros_common_cmake_path isaac_ros_common)
include("${ISAAC_ROS_COMMON_CMAKE_PATH}/isaac_ros_common-version-info.cmake")
generate_version_info(${PROJECT_NAME})

ament_auto_package(INSTALL_TO_SHARE launch)
