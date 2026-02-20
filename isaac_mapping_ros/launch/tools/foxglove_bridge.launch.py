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

import isaac_ros_launch_utils.all_types as lut
import isaac_ros_launch_utils as lu

# Whitelisted topics are the only ones propagated to Foxglove. Set "use_foxglove_whitelist:=False"
# to make all topics available.
TOPIC_WHITELIST = [
    # CUVSLAM
    '/visual_slam/status',
    '/visual_slam/tracking/odometry',
    '/visual_slam/tracking/slam_path',
    '/visual_slam/tracking/vo_path',
    '/visual_slam/tracking/vo_pose',
    '/visual_slam/tracking/vo_pose_covariance',
    '/visual_slam/vis/landmarks_cloud',
    '/visual_slam/vis/observations_cloud',
    '/visual_slam/vis/gravity',
    '/visual_slam/vis/velocity',
    '/visual_slam/vis/slam_odometry',
    '/visual_slam/vis/loop_closure_cloud',
    '/visual_slam/vis/pose_graph_nodes',
    '/visual_slam/vis/pose_graph_edges',
    '/visual_slam/vis/pose_graph_edges2',
    '/visual_slam/vis/localizer',
    '/visual_slam/vis/localizer_map_cloud',
    '/visual_slam/vis/localizer_observations_cloud',
    '/visual_slam/vis/localizer_loop_closure_cloud',
    '/tf',
    '/tf_static',
    '/diagnostics',
    # Global localization
    '/visual_localization/pose',
    '/visual_localization/debug_image',
    '/visual_localization/camera_0/image_rect',
    '/visual_localization/camera_1/image_rect',
    '/visual_localization/trigger_localization',
    '/map',
]

RAW_TOPICS = [
    'left/image_raw',
    'right/image_raw',
    'left/camera_info',
    'right/camera_info',
]

RECTIFIED_TOPICS = [
    'left/image_rect',
    'left/camera_info_rect',
    'right/image_rect',
    'right/camera_info_rect',
]


def add_foxglove(args: lu.ArgumentContainer) -> list[lut.Action]:

    params = [{
        'send_buffer_limit': int(args.send_buffer_limit),
        'max_qos_depth': 30,
        'use_compression': False,
        'capabilities': ['clientPublish', 'connectionGraph', 'assets'],
    }]

    if args.camera_names:
        if lu.is_true(args.rectified_images):
            camera_name_list = args.camera_names.split(',')
            for camera_name in camera_name_list:
                for topic in RECTIFIED_TOPICS:
                    TOPIC_WHITELIST.append(f'/{camera_name}/{topic}')

        else:
            camera_name_list = args.camera_names.split(',')
            for camera_name in camera_name_list:
                for topic in RAW_TOPICS:
                    TOPIC_WHITELIST.append(f'/{camera_name}/{topic}')

    if lu.is_true(args.use_foxglove_whitelist):
        params[0].update({'topic_whitelist': TOPIC_WHITELIST})

    actions = []
    actions.append(
        lut.Node(
            package='foxglove_bridge',
            executable='foxglove_bridge',
            parameters=params,
            # Use error log level to reduce terminal cluttering from "send_buffer_limit reached"
            # warnings.
            arguments=['--ros-args', '--log-level', 'ERROR'],
        ))

    return actions


def generate_launch_description() -> lut.LaunchDescription:
    args = lu.ArgumentContainer()
    args.add_arg('use_foxglove_whitelist')
    args.add_arg('send_buffer_limit', 10000000)
    args.add_arg('camera_names', '', cli=True)
    args.add_arg('rectified_images', False, cli=True)

    args.add_opaque_function(add_foxglove)
    return lu.LaunchDescription(args.get_launch_actions())
