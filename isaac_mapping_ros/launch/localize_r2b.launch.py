# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os

import isaac_ros_launch_utils as lu
import isaac_ros_launch_utils.all_types as lut
from launch.conditions import IfCondition


def create_decoder(camera_name: str, identifier: str) -> lut.Action:
    decoder_node = lut.ComposableNode(
        name='decoder_node',
        package='isaac_ros_h264_decoder',
        plugin='nvidia::isaac_ros::h264_decoder::DecoderNode',
        namespace=f'{camera_name}/{identifier}',
        remappings=[
            ('image_uncompressed', 'image_raw'),
        ],
    )
    return lu.load_composable_nodes('nova_container', [decoder_node])


def add_nodes(args: lu.ArgumentContainer):
    actions = []
    cuvslam_map_dir = os.path.join(args.map_dir, 'cuvslam_map') if args.map_dir else ''
    cuvgl_map_dir = os.path.join(args.map_dir, 'cuvgl_map') if args.map_dir else ''
    ground_plane_published = False

    if args.enable_vgl:
        assert args.map_dir, 'map_dir is required when enable_vgl is true'

        vgl_launch_arguments = {
            'container_name': 'nova_container',
            'vgl_enabled_stereo_cameras': args.camera_names,
            'vgl_do_rectify_images': True,
            'vgl_map_frame': 'map',
            'vgl_map_dir': cuvgl_map_dir,
        }

        actions.append(
            lu.include(
                'isaac_ros_visual_global_localization',
                'launch/include/visual_global_localization.launch.py',
                launch_arguments=vgl_launch_arguments,
            ))

    if args.enable_vslam:
        launch_arguments = {
            'container_name': 'nova_container',
            'vslam_enabled_stereo_cameras': args.camera_names,
            'vslam_map_frame': 'map',
            'vslam_odom_frame': 'odom',
            'vslam_publish_map_to_odom_tf': True,
            'vslam_enable_slam': args.vslam_enable_slam,
            'vslam_use_rectified_images': False,
        }
        if args.map_dir:
            launch_arguments['vslam_load_map_folder_path'] = cuvslam_map_dir
            launch_arguments['vslam_enable_slam'] = True

        actions.append(
            lu.include(
                'isaac_mapping_ros',
                'launch/algorithms/vslam.launch.py',
                launch_arguments=launch_arguments,
            ))

    if args.map_dir:
        occupancy_map_yaml_file = os.path.join(args.map_dir, 'occupancy_map.yaml')
        assert os.path.exists(occupancy_map_yaml_file), (
            f'occupancy_map_yaml_file {occupancy_map_yaml_file} does not exist')

        actions.append(
            lu.include(
                'isaac_mapping_ros',
                'launch/tools/occupancy_map_server.launch.py',
                launch_arguments={
                    'occupancy_map_yaml_file': occupancy_map_yaml_file,
                    'omap_frame': 'omap',
                },
            ))

        ground_plane_file = os.path.join(args.map_dir, 'ground_plane.yaml')
        if os.path.exists(ground_plane_file):
            actions.append(lu.log_info(f'Publishing ground plane from {ground_plane_file}'))
            actions.append(
                lu.include(
                    'isaac_mapping_ros',
                    'launch/tools/publish_ground_plane.launch.py',
                    launch_arguments={
                        'ground_plane_file': ground_plane_file,
                        'parent_frame': 'map',
                        'child_frame': 'omap',
                    },
                ))
            ground_plane_published = True
        else:
            actions.append(lu.log_info('No ground plane file found'))
    else:
        actions.append(lu.log_info('No map_dir provided.'))

    if args.map_dir and not ground_plane_published:
        actions.append(lut.Node(
            name='map_to_omap_static_transform_publisher',
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=['0', '0', '0', '0', '0', '0', '1', 'map', 'omap'],
            output='screen',
        ))
        actions.append(lu.log_info('No ground plane file found, publishing identity transform'))

    if args.rosbag:
        for camera_name in args.camera_names.split(','):
            actions.append(create_decoder(camera_name, 'left'))
            actions.append(create_decoder(camera_name, 'right'))

    actions.append(lu.component_container('nova_container'))

    return actions


def generate_launch_description() -> lut.LaunchDescription:
    args = lu.ArgumentContainer()

    args.add_arg('rosbag', '', cli=True)
    args.add_arg(
        'camera_names',
        'front_stereo_camera,left_stereo_camera,right_stereo_camera,back_stereo_camera',
        cli=True)
    args.add_arg('replay_rate', '1.0', cli=True)
    args.add_arg('rosbag_start_delay_s', '3.0', cli=True)
    args.add_arg('replay_additional_args', '--disable-keyboard-controls', cli=True)

    args.add_arg('map_dir', '', cli=True)

    args.add_arg('enable_vgl', True, cli=True)
    args.add_arg('enable_vslam', True, cli=True)
    args.add_arg('vslam_enable_slam', False, cli=True)
    args.add_arg('enable_foxglove_bridge', True, cli=True)
    args.add_arg('use_foxglove_whitelist', True, cli=True)
    args.add_arg('type_negotiation_duration_s', lu.get_default_negotiation_time(), cli=True)

    args.add_opaque_function(add_nodes)

    actions = args.get_launch_actions()
    actions.append(
        lut.SetParameter('type_negotiation_duration_s', args.type_negotiation_duration_s))
    actions.append(
        lu.log_info([f'Using type negotiation duration: {args.type_negotiation_duration_s}']))
    actions.append(
        lu.include(
            'isaac_mapping_ros',
            'launch/tools/foxglove_bridge.launch.py',
            launch_arguments={
                'use_foxglove_whitelist': args.use_foxglove_whitelist,
                'rectified_images': False,
                'camera_names': args.camera_names,
            },
            condition=IfCondition(args.enable_foxglove_bridge),
        ))
    actions.append(
        lu.play_rosbag(args.rosbag,
                       rate=args.replay_rate,
                       delay=args.rosbag_start_delay_s,
                       additional_bag_play_args=args.replay_additional_args,
                       shutdown_on_exit=True,
                       condition=IfCondition(lu.is_valid(args.rosbag))))
    return lut.LaunchDescription(actions)
