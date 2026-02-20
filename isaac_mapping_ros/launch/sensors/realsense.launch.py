# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


def launch_realsense(args: lu.ArgumentContainer) -> list[lut.Action]:

    actions = []

    # Prepare parameters
    parameters = []

    parameters.append({
        'infra1_enabled': True,
        'infra2_enabled': True,
        'infra1_qos': 'SENSOR_DATA',
        'infra2_qos': 'SENSOR_DATA',
        'infra1_info_qos': 'SENSOR_DATA',
        'infra2_info_qos': 'SENSOR_DATA',
        'publish_tf': True,
        'enable_depth': False,
        'enable_color': False,
        'enable_gyro': False,
        'enable_accel': False,
        'enable_motion': False,
        'enable_rgbd': False,
        'depth_module.emitter_enabled': 0,
    })

    remappings = [
        ('infra1/image_rect_raw', 'left/image_rect'),
        ('infra1/camera_info', 'left/camera_info_rect'),
        ('infra1/image_rect_raw/compressed', 'left/image_rect/compressed'),
        ('infra2/image_rect_raw', 'right/image_rect'),
        ('infra2/camera_info', 'right/camera_info_rect'),
        ('infra2/image_rect_raw/compressed', 'right/image_rect/compressed'),
    ]

    remapping_with_ns = []
    for remapping in remappings:
        remapping_with_ns.append(
            (
                f'{args.camera_name}/{remapping[0]}',
                f'{args.camera_name}/{remapping[1]}',
            )
        )

    realsense_node = lut.ComposableNode(
        package="realsense2_camera",
        plugin="realsense2_camera::RealSenseNodeFactory",
        name=args.camera_name,
        namespace='',
        parameters=parameters,
        remappings=remapping_with_ns,
    )

    actions.append(
        lu.log_info(
            f'Using RealSense camera: {args.camera_name} with default parameters'
        )
    )

    if args.run_standalone:
        actions.append(lu.component_container(args.container_name))
    actions.append(lu.load_composable_nodes(args.container_name, [realsense_node]))

    return actions


def generate_launch_description() -> lut.LaunchDescription:
    args = lu.ArgumentContainer()
    args.add_arg('container_name', 'nova_container')
    args.add_arg('run_standalone', False)
    args.add_arg('camera_name', 'realsense')
    args.add_opaque_function(launch_realsense)

    return lut.LaunchDescription(args.get_launch_actions())
