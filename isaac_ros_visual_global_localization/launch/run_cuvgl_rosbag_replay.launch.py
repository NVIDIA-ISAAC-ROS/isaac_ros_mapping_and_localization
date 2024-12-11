# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from ament_index_python.packages import get_package_share_directory
import os

# Remap image and camera info topics, we can modify this function to remap other topics if needed.
MAP_FRAME = "map"
OMAP_FRAME = "omap"


def get_camera_list(camera_names):
    if lu.is_equal(camera_names, "front"):
        return ['front_stereo_camera']
    elif lu.is_equal(camera_names, "front_left"):
        return ['front_stereo_camera', 'left_stereo_camera']
    elif lu.is_equal(camera_names, "front_left_right"):
        return ['front_stereo_camera', 'left_stereo_camera', 'right_stereo_camera']
    elif lu.is_equal(camera_names, 'front_back_left_right'):
        return [
            'front_stereo_camera', 'back_stereo_camera', 'left_stereo_camera',
            'right_stereo_camera'
        ]
    else:
        return []


def create_decoder(name: str, identifier: str):
    decoder_node = lut.ComposableNode(
        name='decoder_node',
        package='isaac_ros_h264_decoder',
        plugin='nvidia::isaac_ros::h264_decoder::DecoderNode',
        namespace=f'{name}/{identifier}',
        remappings=[('image_uncompressed', 'image_raw')],
    )
    return decoder_node


def create_decoder_nodes(camera_names):
    nodes = []
    for camera_name in camera_names:
        nodes.append(create_decoder(camera_name, 'left'))
        nodes.append(create_decoder(camera_name, 'right'))
    return nodes


def create_rectify_node(name: str, identifier: str):
    rectify_node = lut.ComposableNode(
        name='rectify_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::RectifyNode',
        namespace=f'{name}/{identifier}',
        parameters=[{
            'output_width': 1920,
            'output_height': 1200,
            'type_negotiation_duration_s': 1,
        }],
    )
    return rectify_node


def create_rectify_nodes(camera_names):
    nodes = []
    for camera_name in camera_names:
        nodes.append(create_rectify_node(camera_name, 'left'))
        nodes.append(create_rectify_node(camera_name, 'right'))
    return nodes


def create_format_converter_node(name: str, identifier: str, image_format: str,
                                 rectified_images: bool):
    remappings = [('image', f'image_{image_format}')]
    if rectified_images:
        remappings.append(('image_raw', 'image_rect'))
    reformat_node = lut.ComposableNode(
        name=f'reformat_node_{image_format}',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
        namespace=f'{name}/{identifier}',
        parameters=[{
            'image_width': 1920,
            'image_height': 1200,
            'type_negotiation_duration_s': 1,
            'encoding_desired': image_format,
        }],
        remappings=remappings)
    return reformat_node


def create_format_converter_nodes(camera_names, image_format: str, rectified_images: bool):
    nodes = []
    for camera_name in camera_names:
        nodes.append(
            create_format_converter_node(camera_name, 'left', image_format, rectified_images))
        nodes.append(
            create_format_converter_node(camera_name, 'right', image_format, rectified_images))
    return nodes


def create_omap_server(map_yaml_file):
    return lut.Node(package='nav2_map_server',
                    executable='map_server',
                    name='map_server',
                    output='screen',
                    parameters=[{
                        'yaml_filename': map_yaml_file,
                        'frame_id': 'omap'
                    }],
                    remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')])


def create_lifecycle_manager():
    return lut.TimerAction(
        period=5.0,
        actions=[
            lut.Node(
                package='nav2_lifecycle_manager',
                executable='lifecycle_manager',
                name='lifecycle_manager_map_server',
                parameters=[{
                    'autostart': True,
                    'node_names': ['map_server']
                }],
            )
        ],
    )


def create_omap_to_map(map_to_omap_x_offset, map_to_omap_y_offset):
    return lut.Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher',
        arguments=[
            str(map_to_omap_x_offset),
            str(map_to_omap_y_offset),
            '0.0',
            '0',
            '0',
            '0',
            MAP_FRAME,
            OMAP_FRAME,
        ],
    )


def create_rviz():
    rviz_config_file = 'rviz/omap.rviz'
    package_share_directory = get_package_share_directory("isaac_ros_visual_global_localization")
    rviz_config_path = os.path.join(package_share_directory, rviz_config_file)

    # Define the RViz node
    return lut.Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_path],
        output='screen',
    )


def create_visualization_actions(args):
    # nodes.append(create_odom_to_base_link())
    actions = []
    actions.append(create_omap_server(args.occupancy_map_yaml_file))
    # needed for map server to load automatically
    actions.append(create_lifecycle_manager())

    return actions


def add_visual_global_localization(args: lu.ArgumentContainer) -> list[lut.Action]:
    actions = []

    actions.append(
        lu.include(
            'isaac_ros_data_replayer',
            'launch/include/data_replayer_include.launch.py',
            launch_arguments={
                'rosbag': args.rosbag_path,
                'enabled_stereo_cameras': True,
                'enabled_fisheye_cameras': False,
                'enable_3d_lidar': args.enable_3d_lidar,
                'replay_rate': args.replay_rate
            }))

    camera_list = get_camera_list(args.camera_names)
    camera_names = ','.join(camera_list)

    rectified_images = False

    if args.enable_image_decoder:
        actions.append(
            lu.load_composable_nodes(args.container_name, create_decoder_nodes(camera_list)))

    if args.enable_image_rectify:
        rectified_images = True
        actions.append(
            lu.load_composable_nodes(args.container_name, create_rectify_nodes(camera_list)))

    actions.append(
        lu.include('isaac_ros_visual_global_localization',
                   'launch/include/visual_global_localization.launch.py',
                   launch_arguments={
                       'vgl_enabled_stereo_cameras': camera_names,
                       'vgl_rectified_images': rectified_images,
                       'container_name': args.container_name,
                   }))

    if args.use_cuvslam:
        actions.append(
            lu.include('isaac_ros_perceptor_bringup',
                       'launch/algorithms/vslam.launch.py',
                       launch_arguments={
                           'vslam_enabled_stereo_cameras': camera_names,
                           'container_name': args.container_name,
                       }))

    actions.append(create_omap_to_map(args.map_to_omap_x_offset, args.map_to_omap_y_offset))

    if args.occupancy_map_yaml_file != '':
        actions.extend(create_visualization_actions(args))

    if args.enable_foxglove_bridge:
        actions.append(
            lu.include(
                'isaac_ros_perceptor_bringup',
                'launch/tools/visualization.launch.py',
                launch_arguments={'use_foxglove_whitelist': args.use_foxglove_whitelist},
            ))
    else:
        actions.append(create_rviz())

    actions.append(lu.component_container(args.container_name))

    return actions


def generate_launch_description():
    """Replay a rosbag and decompress the images from the front stereo camera."""
    args = lu.ArgumentContainer()
    args.add_arg(
        'camera_names',
        default='front_back_left_right',
        choices=[
            'front',
            'front_left',
            'front_left_right',
            'front_back_left_right',
        ],
        cli=True,
    )
    args.add_arg('container_name', default='visual_localization_container', cli=True)
    args.add_arg(
        'localization_mode',
        default='visual_localization',
        choices=['visual_localization', 'apriltag_localization', 'global_localization_mapper'],
        cli=True,
    )
    args.add_arg(
        'rosbag_path',
        cli=True,
        description=(
            'Path of the rosbag to replay. If set, need to install isaac_ros_data_replayer package'
        ))
    args.add_arg(
        'replay_rate',
        cli=True,
        description='Rate of replaying the rosbag.',
        default='0.5',
    )
    args.add_arg(
        'enable_image_decoder',
        cli=True,
        description=(
            'If enable image decoder. If set, need to install isaac_ros_h264_decoder package'),
        default=True,
    )
    args.add_arg(
        'enable_image_rectify',
        cli=True,
        description=(
            'If enable image rectification. If set, need to install isaac_ros_image_proc package'),
        default=False)
    args.add_arg(
        'enable_format_converter',
        cli=True,
        description=(
            'If enable format converter. If set, need to install isaac_ros_image_proc package'),
        default=False)
    args.add_arg('enable_3d_lidar',
                 cli=True,
                 description='If enable 3d ldiar driver',
                 default=True)
    args.add_arg(
        'use_cuvslam',
        cli=True,
        description=(
            'Whether or not use cuvslam. If set, need to install isaac_ros_perceptor_bringup '
            'package'),
        default=False,
    )
    args.add_arg(
        'occupancy_map_yaml_file',
        cli=True,
        description='Yaml file for 2d omap. If set, need to install nav2_map_server package',
        default='',
    )
    args.add_arg(
        'map_to_omap_x_offset',
        cli=True,
        description='X offset from map to omap',
        default='0.0',
    )
    args.add_arg(
        'map_to_omap_y_offset',
        cli=True,
        description='Y offset for map to omap',
        default='0.0',
    )
    args.add_arg(
        'enable_foxglove_bridge',
        cli=True,
        description=(
            'If enable foxglove bridge. If set, need to install isaac_ros_perceptor_bringup '
            'package'),
        default=False)
    args.add_arg(
        'use_foxglove_whitelist',
        cli=True,
        description='If use foxglove whitelist',
        default=True,
    )
    args.add_arg(
        'vgl_publish_map_to_base_tf',
        cli=True,
        description='If publish vehicle pose to tf topic',
        default=True)
    args.add_arg(
        'vgl_enable_continuous_localization',
        cli=True,
        description=(
            'If enables trigger localization continuously after a localization has been computed '
            'previously'),
        default=True)
    args.add_arg(
        'vgl_enable_point_cloud_filter',
        cli=True,
        description='If enable point cloud filter',
        default=True)
    args.add_opaque_function(add_visual_global_localization)

    return lut.LaunchDescription(args.get_launch_actions())
