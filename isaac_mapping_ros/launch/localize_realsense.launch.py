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

import isaac_ros_launch_utils as lu
import isaac_ros_launch_utils.all_types as lut
from launch.conditions import IfCondition, UnlessCondition
import os


def add_nodes(args: lu.ArgumentContainer):
    actions = []

    camera_optical_frames = 'camera_infra1_optical_frame,camera_infra2_optical_frame'
    base_frame = 'camera_link'
    if args.enable_vgl:
        assert args.map_dir, "map_dir is required when enable_vgl is true"
        actions.append(
            lu.include(
                'isaac_ros_visual_global_localization',
                'launch/include/visual_global_localization.launch.py',
                launch_arguments={
                    'container_name': 'nova_container',
                    'vgl_enabled_stereo_cameras': args.camera_name,
                    'vgl_do_rectify_images': False,
                    'vgl_map_frame': 'map',
                    'publish_rectified_images': False,
                    'vgl_camera_optical_frames': camera_optical_frames,
                    'vgl_map_dir': os.path.join(args.map_dir, 'cuvgl_map'),
                    'vgl_base_frame': base_frame,
                    'vgl_config_dir': lu.get_path('isaac_ros_visual_mapping',
                                                  'configs/single_stereo_localizer')
                },
            ))

    if args.enable_vslam:
        params = {
            'container_name': 'nova_container',
            'vslam_enabled_stereo_cameras': args.camera_name,
            'vslam_map_frame': 'map',
            'vslam_odom_frame': 'odom',
            'vslam_image_qos': 'SENSOR_DATA',
            'vslam_publish_map_to_odom_tf': True,
            'vslam_enable_visualization': args.vslam_enable_visualization,
            'vslam_enable_ground_constraint_in_odometry':
                args.vslam_enable_ground_constraint_in_odometry,
            'vslam_enable_ground_constraint_in_slam':
                args.vslam_enable_ground_constraint_in_slam,
            'vslam_camera_optical_frames': camera_optical_frames,
            'vslam_base_frame': base_frame,
            'vslam_use_rectified_images': True,
        }

        if args.map_dir != '':
            params['vslam_load_map_folder_path'] = os.path.join(args.map_dir, 'cuvslam_map')
            params['vslam_enable_slam'] = True

        actions.append(
            lu.include(
                'isaac_mapping_ros',
                'launch/algorithms/vslam.launch.py',
                launch_arguments=params,
            ))

    ground_plane_published = False

    if args.map_dir != '':
        occupancy_map_yaml_file = os.path.join(args.map_dir, 'occupancy_map.yaml')
        assert os.path.exists(occupancy_map_yaml_file), (
            f"occupancy_map_yaml_file {occupancy_map_yaml_file} does not exist")

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

    if not ground_plane_published:
        actions.append(lu.Node(
            name='map_to_omap_static_transform_publisher',
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=['0', '0', '0', '0', '0', '0', '1', 'map', 'omap'],
            output='screen'
        ))
        actions.append(lu.log_info('No ground plane file found, publishing identity transform'))

    actions.append(lu.component_container('nova_container'))

    return actions


def generate_launch_description() -> lut.LaunchDescription:
    args = lu.ArgumentContainer()

    args.add_arg('camera_name', 'realsense', cli=True)

    args.add_arg('rosbag', '', cli=True)
    args.add_arg('replay_rate', '1.0', cli=True)
    args.add_arg('replay_additional_args', '', cli=True)
    args.add_arg('rosbag_start_delay_s', '0.0', cli=True)

    args.add_arg('map_dir', '', cli=True)

    # vslam parameters
    args.add_arg('enable_vslam', True, cli=True)
    args.add_arg('vslam_enable_slam', True, cli=True)
    args.add_arg('vslam_enable_ground_constraint_in_odometry', False, cli=True)
    args.add_arg('vslam_enable_ground_constraint_in_slam', False, cli=True)
    args.add_arg('vslam_enable_visualization', False, cli=True)

    args.add_arg('enable_vgl', True, cli=True)

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
                'rectified_images': True,
                'camera_names': args.camera_name,
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

    actions.append(
        lu.include(
            'isaac_mapping_ros',
            'launch/sensors/realsense.launch.py',
            launch_arguments={
                'camera_name': args.camera_name,
            },
            condition=UnlessCondition(lu.is_valid(args.rosbag)),
        ))

    return lut.LaunchDescription(actions)
