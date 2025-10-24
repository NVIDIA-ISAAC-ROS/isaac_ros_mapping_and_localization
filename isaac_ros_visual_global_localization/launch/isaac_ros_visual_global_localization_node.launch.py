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

from typing import List

from isaac_ros_launch_utils.all_types import Action, LaunchDescription
import isaac_ros_launch_utils as lu


def add_visual_global_localization(args: lu.ArgumentContainer) -> List[Action]:
    actions = []

    actions.append(
        lu.include(
            'isaac_ros_visual_global_localization',
            'launch/include/visual_global_localization.launch.py',
            launch_arguments={
                'vgl_enabled_stereo_cameras': args.camera_names,
                'container_name': args.container_name,
            }
        )
    )
    actions.append(lu.component_container(args.container_name))

    return actions


def generate_launch_description():
    """Replay a rosbag and decompress the images from the front stereo camera."""
    args = lu.ArgumentContainer()
    args.add_arg(
        'camera_names',
        cli=True,
        description='Comma-separated list of camera names to use for localization (e.g., '
                    '"front_stereo_camera,left_stereo_camera,right_stereo_camera")',
    )
    args.add_arg(
        'container_name',
        default='visual_localization_container',
        cli=True
    )
    args.add_opaque_function(add_visual_global_localization)

    return LaunchDescription(args.get_launch_actions())
