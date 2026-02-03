# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import isaac_ros_launch_utils.all_types as lut
import isaac_ros_launch_utils as lu
import yaml
import logging

logger = logging.getLogger('visual_global_localization')


def generate_nova_carter_remapping(camera_names: list[str], rectified: bool, node_name: str):
    # Generate the remapping by assuming the following camera names
    remappings = []
    camera_id = 0
    image_topic = 'image_rect' if rectified else 'image_raw'
    camera_info_topic = 'camera_info_rect' if rectified else 'camera_info'

    for stereo_camera in camera_names:
        for sub_topic in ["left", "right"]:
            remappings.append(
                (f'{node_name}/image_{camera_id}', f'/{stereo_camera}/{sub_topic}/{image_topic}'))
            remappings.append((f'{node_name}/camera_info_{camera_id}',
                               f'/{stereo_camera}/{sub_topic}/{camera_info_topic}'))
            camera_id += 1
    return remappings


def generate_remap_from_config_file(topic_config_file: str,
                                    node_name: str) -> list[tuple[str, str]]:
    with open(topic_config_file, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if 'stereo_cameras' not in config:
        logger.error("No stereo cameras found in config file: %s", topic_config_file)
        return []
    camera_id = 0
    remapping = []
    for stereo_camera in config['stereo_cameras']:
        for image_topic in ["left", "right"]:
            info_topic = image_topic + "_camera_info"
            if image_topic not in stereo_camera or info_topic not in stereo_camera:
                logger.error("Missing camera or camera_info topic for stereo camera: %s",
                             stereo_camera)
                return []
            remapping.append((f'{node_name}/image_{camera_id}', stereo_camera[image_topic]))
            remapping.append((f'{node_name}/camera_info_{camera_id}', stereo_camera[info_topic]))
            camera_id += 1
    return remapping


def create_point_cloud_filter(map_frame: str) -> lut.ComposableNode:
    point_cloud_filter_node = lut.ComposableNode(
        name='point_cloud_filter_node',
        package='isaac_ros_visual_global_localization',
        plugin='nvidia::isaac_ros::visual_global_localization::PointCloudFilterNode',
        parameters=[{
            'target_frame': map_frame
        }],
        remappings=[
            ('point_cloud', '/front_3d_lidar/lidar_points'),
            ('filtered_point_cloud', '/front_3d_lidar/lidar_points_filtered'),
            ('pose', '/visual_localization/pose'),
        ])
    return point_cloud_filter_node


def add_visual_global_localization(args: lu.ArgumentContainer) -> list[lut.Action]:
    node_name = 'visual_localization'

    if lu.is_valid(args.topic_config_file):
        remappings = generate_remap_from_config_file(args.topic_config_file, node_name)
    else:
        camera_names = args.vgl_enabled_stereo_cameras.split(',')
        remappings = generate_nova_carter_remapping(
            camera_names, args.vgl_rectified_images, node_name)

    remapping_string = ', '.join([f'{a} -> {b}' for a, b in remappings])

    actions = []
    actions.append(
        lu.log_info(["Visual global localization using remapping:'", remapping_string, "'"]))
    num_cameras = int(len(remappings) / 2)

    stereo_localizer_cam_ids = ','.join([str(i) for i in range(num_cameras)])
    visual_localization_node = lut.ComposableNode(
        name='visual_localization_node',
        package='isaac_ros_visual_global_localization',
        plugin='nvidia::isaac_ros::visual_global_localization::VisualGlobalLocalizationNode',
        parameters=[{
            'num_cameras': num_cameras,
            'stereo_localizer_cam_ids': stereo_localizer_cam_ids,
            'map_dir': str(args.vgl_map_dir),
            'config_dir': str(args.vgl_config_dir),
            'model_dir': str(args.vgl_model_dir),
            'debug_dir': str(args.vgl_debug_dir),
            'debug_map_raw_dir': str(args.vgl_debug_map_raw_dir),
            'map_frame': str(args.vgl_map_frame),
            'enable_rectify_images': not args.vgl_rectified_images,
            'publish_rectified_images': args.vgl_publish_rectified_images,
            'enable_continuous_localization': args.vgl_enable_continuous_localization,
            'use_initial_guess': args.vgl_use_initial_guess,
            'publish_map_to_base_tf': args.vgl_publish_map_to_base_tf,
            'image_sync_match_threshold_ms': args.vgl_image_sync_match_threshold_ms,
            'verbose_logging': args.vgl_verbose_logging,
            'init_glog': args.vgl_init_glog,
            'glog_v': args.vgl_glog_v,
            'localization_precision_level': args.vgl_localization_precision_level,
        }],
        remappings=remappings,
    )

    actions.append(lu.log_info(['Enabling visual localization']))
    actions.append(lu.load_composable_nodes(args.container_name, [visual_localization_node]))

    if args.vgl_enable_point_cloud_filter:
        actions.append(lu.log_info(['Enabling point cloud filter']))
        actions.append(
            lu.load_composable_nodes(args.container_name,
                                     [create_point_cloud_filter(args.vgl_map_frame)]))

    return actions


def generate_launch_description() -> lut.LaunchDescription:
    args = lu.ArgumentContainer()
    args.add_arg('container_name')
    args.add_arg('vgl_enabled_stereo_cameras')
    args.add_arg('vgl_map_dir')
    args.add_arg('vgl_config_dir', lu.get_path('isaac_ros_visual_mapping', 'configs/isaac'))
    args.add_arg('vgl_model_dir', lu.get_path('isaac_ros_visual_mapping', 'models'))
    args.add_arg('vgl_debug_dir', '')
    args.add_arg('vgl_debug_map_raw_dir', '')
    args.add_arg('vgl_map_frame', 'map')
    args.add_arg('vgl_publish_map_to_base_tf', False)
    args.add_arg('vgl_use_initial_guess', False)
    args.add_arg('vgl_enable_continuous_localization', False)
    args.add_arg('vgl_enable_point_cloud_filter', False)
    args.add_arg('vgl_image_sync_match_threshold_ms', 3.0)
    args.add_arg('vgl_verbose_logging', False)
    args.add_arg('vgl_init_glog', False)
    args.add_arg('vgl_glog_v', 0)
    args.add_arg('vgl_rectified_images', True)
    args.add_arg('vgl_publish_rectified_images', False)
    args.add_arg('vgl_localization_precision_level', 2)
    args.add_arg('topic_config_file', '')

    args.add_opaque_function(add_visual_global_localization)

    return lut.LaunchDescription(args.get_launch_actions())
