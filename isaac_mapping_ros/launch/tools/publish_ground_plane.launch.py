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
import math
import yaml
import os


def normalize_plane_representation(normal, offset):
    """Normalize plane representation so that nz is always positive.
    If nz < 0, flip both the normal vector and the offset to maintain
    the same plane equation: n · p + d = 0.
    Returns normalized (normal, offset) tuple."""
    nx, ny, nz = normal[0], normal[1], normal[2]

    # Normalize the normal vector
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    if length < 1e-10:
        raise ValueError("Normal vector has zero length")
    nx, ny, nz = nx / length, ny / length, nz / length

    # If nz is negative, flip both normal and offset to keep nz positive
    if nz < 0.0:
        nx, ny, nz = -nx, -ny, -nz
        offset = -offset

    return [nx, ny, nz], offset


def normal_to_quaternion(nx, ny, nz):
    """Convert a normal vector to quaternion representing rotation from [0,0,1] to normal.
    Assumes nz is already positive (normalized)."""
    if nz > 0.9999:
        return 0.0, 0.0, 0.0, 1.0

    qw = math.sqrt((1.0 + nz) / 2.0)

    if qw < 0.0001:
        return 1.0, 0.0, 0.0, 0.0

    # Rotation axis from [0,0,1] to [nx, ny, nz] is the cross product:
    # [0,0,1] × [nx, ny, nz] = [-ny, nx, 0]
    # The negative sign on ny comes from this cross product
    scale = 1.0 / (2.0 * qw)
    qx = -ny * scale
    qy = nx * scale
    qz = 0.0

    return qx, qy, qz, qw


def load_ground_plane_from_yaml(yaml_file_path):
    """Load ground plane data from a YAML file and return normal and offset.
    Note: 'height' is a misnomer - it's actually the plane equation offset d."""
    if not os.path.exists(yaml_file_path):
        raise FileNotFoundError(f"Transform YAML file not found: {yaml_file_path}")

    with open(yaml_file_path, 'r') as f:
        data = yaml.safe_load(f)

    assert 'normal' in data, "YAML file must contain 'normal' key"
    assert 'offset' in data or 'height' in data, (
        "YAML file must contain either 'offset' or 'height' key")

    normal = data['normal']

    if 'offset' in data:
        offset = data['offset']
    else:
        offset = data['height']  # Legacy key name

    return normal, offset


def create_static_transform_publisher(normal, offset, parent_frame, child_frame):
    """Create a static transform publisher node from the provided normal and offset.
    The offset parameter is the plane equation offset d in: n · p + d = 0.
    The z-translation is computed as the z-coordinate of the ground plane at the
    origin of the parent frame (x=0, y=0 in parent frame): z = -d / n_z.

    The plane representation is normalized so that nz is always positive."""

    # Normalize plane representation: ensure nz > 0 by flipping if necessary
    normal, offset = normalize_plane_representation(normal, offset)
    nz = normal[2]

    if abs(nz) < 1e-10:
        raise ValueError("Plane normal z-component is too small (nearly vertical plane)")

    # Convert normalized normal to quaternion (nz is guaranteed to be positive)
    qx, qy, qz, qw = normal_to_quaternion(normal[0], normal[1], normal[2])

    # Compute z-translation: z-coordinate of ground plane at parent frame origin
    # The plane equation is: n · p + d = 0, where p is a point in the parent frame
    # At the origin of the parent frame (x=0, y=0): n_z * z + d = 0, so z = -d / n_z
    # This becomes the z-component of the translation in T_parent_child transform
    z_translation = -offset / nz

    return lut.Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher',
        arguments=[
            '0.0',
            '0.0',
            str(z_translation),
            str(qx),
            str(qy),
            str(qz),
            str(qw),
            parent_frame,
            child_frame,
        ],
        output='screen')


def add_transform_publisher(args: lu.ArgumentContainer) -> list[lut.Action]:
    """Add the static transform publisher to the launch."""
    actions = []

    normal, offset = load_ground_plane_from_yaml(args.ground_plane_file)

    transform_publisher = create_static_transform_publisher(normal, offset,
                                                            args.parent_frame, args.child_frame)

    actions.append(transform_publisher)
    return actions


def generate_launch_description():
    """Generate launch description with static transform publisher."""
    args = lu.ArgumentContainer()

    args.add_arg('ground_plane_file',
                 cli=True,
                 description='Path to YAML file containing ground plane data')

    args.add_arg('parent_frame',
                 default='map',
                 cli=True,
                 description='Parent frame for the transform')
    args.add_arg('child_frame',
                 default='ground_plane',
                 cli=True,
                 description='Child frame for the transform')

    args.add_opaque_function(add_transform_publisher)

    return lut.LaunchDescription(args.get_launch_actions())
