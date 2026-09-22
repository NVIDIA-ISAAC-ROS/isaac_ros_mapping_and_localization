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

import math
import os

import isaac_ros_launch_utils as lu
import isaac_ros_launch_utils.all_types as lut
import yaml


def normalize_plane_representation(normal, offset):
    """Normalize a plane representation and keep the z normal non-negative."""
    nx, ny, nz = normal[0], normal[1], normal[2]
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    if length < 1e-10:
        raise ValueError('Normal vector has zero length')

    nx, ny, nz = nx / length, ny / length, nz / length
    if nz < 0.0:
        nx, ny, nz = -nx, -ny, -nz
        offset = -offset

    return [nx, ny, nz], offset


def normal_to_quaternion(nx, ny, nz):
    """Convert a normal vector into a quaternion rotating +Z onto that normal."""
    if nz > 0.9999:
        return 0.0, 0.0, 0.0, 1.0

    qw = math.sqrt((1.0 + nz) / 2.0)
    if qw < 0.0001:
        return 1.0, 0.0, 0.0, 0.0

    scale = 1.0 / (2.0 * qw)
    qx = -ny * scale
    qy = nx * scale
    qz = 0.0
    return qx, qy, qz, qw


def load_ground_plane_from_yaml(yaml_file_path):
    """Load ground plane data from YAML and return normal and offset."""
    if not os.path.exists(yaml_file_path):
        raise FileNotFoundError(f'Transform YAML file not found: {yaml_file_path}')

    with open(yaml_file_path, 'r') as input_file:
        data = yaml.safe_load(input_file)

    if 'normal' not in data:
        raise ValueError("YAML file must contain 'normal' key")
    if 'offset' in data:
        offset = data['offset']
    elif 'height' in data:
        offset = data['height']
    else:
        raise ValueError("YAML file must contain either 'offset' or 'height' key")

    return data['normal'], offset


def create_static_transform_publisher(normal, offset, parent_frame, child_frame):
    """Create a static transform publisher for map-to-ground-plane alignment."""
    normal, offset = normalize_plane_representation(normal, offset)
    nz = normal[2]
    if abs(nz) < 1e-10:
        raise ValueError('Plane normal z-component is too small')

    qx, qy, qz, qw = normal_to_quaternion(normal[0], normal[1], normal[2])
    z_translation = -offset / nz

    return lut.Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher',
        arguments=[
            '--x', '0.0',
            '--y', '0.0',
            '--z', str(z_translation),
            '--qx', str(qx),
            '--qy', str(qy),
            '--qz', str(qz),
            '--qw', str(qw),
            '--frame-id', parent_frame,
            '--child-frame-id', child_frame,
        ],
        output='screen',
    )


def add_transform_publisher(args: lu.ArgumentContainer) -> list[lut.Action]:
    normal, offset = load_ground_plane_from_yaml(args.ground_plane_file)
    return [create_static_transform_publisher(
        normal,
        offset,
        args.parent_frame,
        args.child_frame,
    )]


def generate_launch_description():
    args = lu.ArgumentContainer()
    args.add_arg(
        'ground_plane_file',
        cli=True,
        description='Path to YAML file containing ground plane data',
    )
    args.add_arg('parent_frame', default='map', cli=True)
    args.add_arg('child_frame', default='ground_plane', cli=True)
    args.add_opaque_function(add_transform_publisher)
    return lut.LaunchDescription(args.get_launch_actions())
