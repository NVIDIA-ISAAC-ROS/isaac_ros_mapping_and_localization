#!/usr/bin/env python3

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

import json
import logging
import math
import pathlib
import shutil
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

CUVSLAM_OPENCV_BACK_CONVERSION_TUM_SIGNS = [-1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0]


def get_default_nvblox_config() -> Dict[str, Any]:
    """Get default configuration for nvblox mapping."""
    return {
        'projective_integrator_max_integration_distance_m': 5.0,
        'use_2d_esdf_mode': True,
        'ground_points_candidates_min_z_m': -2.0,
        'slice_height_above_plane_m': 0.3,
        'slice_height_thickness_m': 0.6,
        'workspace_bounds_type': 1,  # WORKSPACE_BOUNDS_TYPE_HEIGHT_BOUNDS
        'workspace_bounds_min_height_m': -2.0,
        'workspace_bounds_max_height_m': 2.0,
        'mapping_type_dynamic': True,
        'fit_to_z0': False,
        'mapping_type_static_occupancy': False,
    }


def get_default_cuvslam_config() -> Dict[str, Any]:
    """Get default configuration for cuVSLAM."""
    return {
        # Core SLAM settings
        'cfg_enable_slam': True,
        'cfg_denoising': False,
        'cfg_max_frame_delta_s': 0.034,
        'cfg_horizontal': False,
        # Image masking
        'border_bottom': 0,
        'border_left': 0,
        'border_right': 0,
        'border_top': 0,
        # SLAM algorithm configuration
        'cfg_multicam_mode': 2,
        'cfg_odom_mode': 0,
        'cfg_planar': False,
        'cfg_slam_max_map_size': 300,
        'cfg_sync_slam': True,
        # Required for GetState() when SLAM is enabled (launcher: cfg_enable_export only)
        'cfg_enable_export': True,
        # export
        'debug_dump': False,
    }


def load_map_creation_config(config_path: Optional[pathlib.Path] = None,
                             cli_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load map creation configuration from YAML file with optional CLI overrides.

    Args:
        config_path: Path to YAML configuration file. If None, returns default config.
        cli_overrides: Dictionary of CLI overrides to apply on top of YAML config.
                      Format: {'nvblox': {'esdf_slice_height': 0.5}, 'cuvslam': {...}}

    Returns:
        Dictionary containing configuration with 'nvblox' and 'cuvslam' sections.
    """
    config = {
        'nvblox': get_default_nvblox_config(),
        'cuvslam': get_default_cuvslam_config(),
    }

    if config_path:
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    yaml_data = yaml.safe_load(f)
                    if yaml_data:
                        if 'nvblox' in yaml_data:
                            config['nvblox'].update(yaml_data['nvblox'])
                        if 'cuvslam' in yaml_data:
                            config['cuvslam'].update(yaml_data['cuvslam'])
                        print(f'Loaded map creation config from {config_path}')
                    else:
                        raise ValueError(f'Loaded empty config file {config_path}')
            except Exception as e:
                raise ValueError(f'Failed to load map creation config from {config_path}: {e}')
        else:
            raise FileNotFoundError(f'Config file {config_path} does not exist')

    # Apply CLI overrides if provided
    if cli_overrides:
        print('\nApplied configuration overrides:')
        for section, params in cli_overrides.items():
            if section in config and params:
                config[section].update(params)
                for param, value in params.items():
                    print(f'  {section}.{param} = {value}')

    return config


def build_command_from_config(
    base_command: List[str],
    config: Dict[str, Any],
    bool_as_flag: bool = False,
    exclude_fields: Optional[List[str]] = None,
) -> List[str]:
    """Build command arguments from a configuration dictionary.

    Args:
        base_command: Base command list to extend
        config: Dictionary containing configuration
        bool_as_flag: If True, boolean fields are added as --flag (when True).
                     If False, boolean fields are added as --flag=true/false.
        exclude_fields: List of field names to skip (for special handling)

    Returns:
        Extended command list with parameters from config
    """
    command = base_command.copy()

    if not config:
        return command

    exclude_fields = exclude_fields or []

    for key, value in config.items():
        if key in exclude_fields:
            continue

        if value is None or (isinstance(value, str) and value == ''):
            continue

        if isinstance(value, bool):
            if bool_as_flag:
                if value:
                    command.append(f'--{key}')
            else:
                command.append(f'--{key}={str(value).lower()}')
        else:
            command.append(f'--{key}={value}')

    return command


def parse_config_overrides(override_list: List[str]) -> Dict[str, Dict[str, Any]]:
    """Parse CLI config override strings into nested dict.

    Args:
        override_list: List of override strings in format 'section.param=value'
                      e.g., ['cuvslam.max_fps=30', 'nvblox.esdf_slice_height=0.5']

    Returns:
        Nested dictionary with overrides organized by section.
        e.g., {'cuvslam': {'max_fps': 30}, 'nvblox': {'esdf_slice_height': 0.5}}
    """
    overrides = {}

    for override_str in override_list:
        if '=' not in override_str:
            print(f'Warning: Skipping invalid override "{override_str}" (missing =)')
            continue

        key_path, value_str = override_str.split('=', 1)
        parts = key_path.split('.')

        if len(parts) != 2:
            print(f'Warning: Skipping invalid override "{override_str}" '
                  f'(expected format: section.param=value)')
            continue

        section, param = parts

        value = _parse_value(value_str)

        if section not in overrides:
            overrides[section] = {}
        overrides[section][param] = value

    return overrides


def _parse_value(value_str: str) -> Any:
    """Parse string value to appropriate type."""
    if value_str.lower() in ('true', 'false'):
        return value_str.lower() == 'true'

    try:
        return int(value_str)
    except ValueError:
        pass

    try:
        return float(value_str)
    except ValueError:
        pass

    return value_str


def build_cuvslam_command_api_launcher(base_command: List[str], cuvslam_config: Dict[str, Any],
                                       log_folder: pathlib.Path,
                                       output_poses_dir: pathlib.Path) -> List[str]:
    if not cuvslam_config:
        cuvslam_config = {}

    command = build_command_from_config(
        base_command=base_command,
        config=cuvslam_config,
        bool_as_flag=False,  # cuVSLAM uses --param=true/false format
        exclude_fields=[
            'debug_dump',
            'print_format',
            'ros_frame_conversion',
        ],  # Special handling below / not cuVSLAM CLI flags
    )

    if 'debug_dump' in cuvslam_config:
        debug_dump_enabled = cuvslam_config['debug_dump']
        if debug_dump_enabled:
            command.append(f'--debug_dump={log_folder}')

    repeat_count = cuvslam_config.get('repeat', 1)
    if repeat_count > 1:
        odom_file = 'odom_poses_repeated.tum'
        slam_file = 'slam_poses_repeated.tum'
        keyframe_file = 'keyframe_pose_repeated.tum'
    else:
        odom_file = 'odom_poses.tum'
        slam_file = 'slam_poses.tum'
        keyframe_file = 'keyframe_pose.tum'

    command.extend([
        '--ros_frame_conversion=true',
        '--print_format=tum',
        f'--print_odom_poses={output_poses_dir}/{odom_file}',
        f'--print_slam_poses={output_poses_dir}/{slam_file}',
        f'--print_map_keyframes={output_poses_dir}/{keyframe_file}',
    ])

    return command


def _convert_extrinsic_to_opencv_style(
    extrinsic: List[List[float]],
) -> List[List[float]]:
    if len(extrinsic) != 3 or any(len(row) != 4 for row in extrinsic):
        raise ValueError(f'Expected a 3x4 camera extrinsic in stereo.edex, got: {extrinsic}')

    # Apply C * [R | t] * C, where C = diag(1, -1, -1).
    # The translation column is affected only by the left multiplication.
    left_row_signs = [1.0, -1.0, -1.0]
    right_column_signs = [1.0, -1.0, -1.0, 1.0]

    return [
        [
            left_row_signs[row_index] * float(value) * right_column_signs[column_index]
            for column_index, value in enumerate(row)
        ]
        for row_index, row in enumerate(extrinsic)
    ]


def _write_opencv_rig_stereo_edex(input_stereo_edex: pathlib.Path,
                                  output_stereo_edex: pathlib.Path) -> int:
    with open(input_stereo_edex, 'r') as input_file:
        stereo_edex = json.load(input_file)

    if not isinstance(stereo_edex, list) or not stereo_edex:
        raise ValueError(f'Unexpected stereo.edex format: {input_stereo_edex}')

    camera_count = 0
    for rig in stereo_edex:
        if not isinstance(rig, dict):
            raise ValueError(f'Unexpected stereo.edex rig entry in {input_stereo_edex}: {rig}')
        for camera in rig.get('cameras', []):
            camera['transform'] = _convert_extrinsic_to_opencv_style(camera['transform'])
            camera_count += 1

    with open(output_stereo_edex, 'w') as output_file:
        json.dump(stereo_edex, output_file, indent=2)
        output_file.write('\n')

    return camera_count


def prepare_cuvslam_edex_dataset(
    edex_path: pathlib.Path,
) -> pathlib.Path:
    """Prepare the EDEX dataset convention expected by cuvslam_api_launcher."""
    stereo_edex = edex_path / 'stereo.edex'
    original_stereo_edex = edex_path / 'stereo.edex.original'

    if original_stereo_edex.exists():
        raise RuntimeError(
            f'Cannot apply OpenCV convention transform because backup file already exists: '
            f'{original_stereo_edex}. Regenerate the EDEX dataset before retrying.'
        )

    shutil.copy2(stereo_edex, original_stereo_edex)
    camera_count = _write_opencv_rig_stereo_edex(stereo_edex, stereo_edex)

    logger.info(
        'Prepared cuVSLAM EDEX dataset at %s: backed up stereo.edex to %s and enabled '
        'OpenCV convention transform; converted %d camera transforms.',
        edex_path,
        original_stereo_edex,
        camera_count,
    )
    return edex_path


def _format_tum_value(value: float) -> str:
    if abs(value) < 0.5e-12:
        value = 0.0
    return f'{value:.9f}'


def _convert_tum_line_from_opencv_style(
    line: str,
    line_number: int,
    tum_pose_file: pathlib.Path,
) -> str:
    stripped_line = line.strip()
    if not stripped_line or stripped_line.startswith('#'):
        return line

    fields = stripped_line.split()
    if len(fields) != 8:
        raise ValueError(
            f'{tum_pose_file}:{line_number}: expected 8 TUM fields, got {len(fields)}'
        )

    timestamp = fields[0]
    pose_values = [float(value) for value in fields[1:]]
    converted_pose_values = [
        sign * value
        for sign, value in zip(CUVSLAM_OPENCV_BACK_CONVERSION_TUM_SIGNS, pose_values)
    ]
    return ' '.join(
        [timestamp] +
        [_format_tum_value(value) for value in converted_pose_values]
    ) + '\n'


def _convert_tum_file_from_opencv_style(tum_pose_file: pathlib.Path) -> None:
    converted_lines = []
    with open(tum_pose_file, 'r') as input_file:
        for line_number, line in enumerate(input_file, start=1):
            converted_lines.append(_convert_tum_line_from_opencv_style(
                line,
                line_number,
                tum_pose_file,
            ))

    with open(tum_pose_file, 'w') as output_file:
        output_file.writelines(converted_lines)


def convert_cuvslam_pose_outputs_for_opencv_edex(output_poses_dir: pathlib.Path) -> None:
    """Convert cuVSLAM TUM pose output back when EDEX input was rewritten for OpenCV."""
    tum_pose_files = sorted(output_poses_dir.glob('*.tum'))
    for tum_pose_file in tum_pose_files:
        _convert_tum_file_from_opencv_style(tum_pose_file)

    logger.info(
        'Converted %d cuVSLAM pose output file(s) back from OpenCV EDEX convention '
        'using TUM pose signs %s.',
        len(tum_pose_files),
        CUVSLAM_OPENCV_BACK_CONVERSION_TUM_SIGNS,
    )


def _normalize_vector(vector: List[float]) -> List[float]:
    length = math.sqrt(sum(value * value for value in vector))
    if length < 1e-10:
        raise ValueError(f'Cannot normalize near-zero vector: {vector}')
    return [value / length for value in vector]


def normalize_ground_plane(
    normal: List[float],
    offset: float,
) -> Tuple[List[float], float]:
    normalized = _normalize_vector([float(value) for value in normal])
    if normalized[2] < 0.0:
        normalized = [-value for value in normalized]
        offset = -offset
    return normalized, float(offset)


def load_ground_plane(
    ground_plane_file: pathlib.Path,
) -> Tuple[List[float], float]:
    with open(ground_plane_file, 'r') as input_file:
        data = yaml.safe_load(input_file)

    if not isinstance(data, dict):
        raise ValueError(f'Unexpected ground plane YAML format: {ground_plane_file}')

    if 'normal' not in data:
        raise ValueError(f'Missing "normal" in ground plane file: {ground_plane_file}')

    if 'offset' in data:
        offset = data['offset']
    elif 'height' in data:
        offset = data['height']
    else:
        raise ValueError(f'Missing "offset" or "height" in ground plane file: {ground_plane_file}')

    return normalize_ground_plane(data['normal'], offset)


def _cross(lhs: List[float], rhs: List[float]) -> List[float]:
    return [
        lhs[1] * rhs[2] - lhs[2] * rhs[1],
        lhs[2] * rhs[0] - lhs[0] * rhs[2],
        lhs[0] * rhs[1] - lhs[1] * rhs[0],
    ]


def _dot(lhs: List[float], rhs: List[float]) -> float:
    return sum(left * right for left, right in zip(lhs, rhs))


def _rotation_matrix_from_axis_angle(
    axis: List[float],
    angle_rad: float,
) -> List[List[float]]:
    axis = _normalize_vector(axis)
    x, y, z = axis
    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)
    one_minus_cos = 1.0 - cos_angle

    return [
        [
            cos_angle + x * x * one_minus_cos,
            x * y * one_minus_cos - z * sin_angle,
            x * z * one_minus_cos + y * sin_angle,
        ],
        [
            y * x * one_minus_cos + z * sin_angle,
            cos_angle + y * y * one_minus_cos,
            y * z * one_minus_cos - x * sin_angle,
        ],
        [
            z * x * one_minus_cos - y * sin_angle,
            z * y * one_minus_cos + x * sin_angle,
            cos_angle + z * z * one_minus_cos,
        ],
    ]


def _rotation_matrix_from_z_to_normal(normal: List[float]) -> List[List[float]]:
    normal = _normalize_vector(normal)
    z_axis = [0.0, 0.0, 1.0]
    dot_product = max(-1.0, min(1.0, _dot(z_axis, normal)))

    if dot_product > 1.0 - 1e-10:
        return [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]

    if dot_product < -1.0 + 1e-10:
        return _rotation_matrix_from_axis_angle([1.0, 0.0, 0.0], math.pi)

    axis = _cross(z_axis, normal)
    angle = math.acos(dot_product)
    return _rotation_matrix_from_axis_angle(axis, angle)


def _transpose(matrix: List[List[float]]) -> List[List[float]]:
    return [
        [matrix[row][column] for row in range(len(matrix))]
        for column in range(len(matrix[0]))
    ]


def _matmul(lhs: List[List[float]], rhs: List[List[float]]) -> List[List[float]]:
    return [
        [
            sum(lhs[row][index] * rhs[index][column] for index in range(len(rhs)))
            for column in range(len(rhs[0]))
        ]
        for row in range(len(lhs))
    ]


def _matvec(matrix: List[List[float]], vector: List[float]) -> List[float]:
    return [
        sum(matrix[row][column] * vector[column] for column in range(len(vector)))
        for row in range(len(matrix))
    ]


def _rotation_matrix_from_axis_angle_dict(axis_angle: Dict[str, float]) -> List[List[float]]:
    angle_rad = math.radians(float(axis_angle.get('angle_degrees', 0.0)))
    if abs(angle_rad) < 1e-10:
        return [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]

    axis = [
        float(axis_angle.get('x', 0.0)),
        float(axis_angle.get('y', 0.0)),
        float(axis_angle.get('z', 1.0)),
    ]
    return _rotation_matrix_from_axis_angle(axis, angle_rad)


def _axis_angle_dict_from_rotation_matrix(rotation: List[List[float]]) -> Dict[str, float]:
    trace = rotation[0][0] + rotation[1][1] + rotation[2][2]
    cos_angle = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    angle_rad = math.acos(cos_angle)

    if abs(angle_rad) < 1e-10:
        return {
            'x': 0.0,
            'y': 0.0,
            'z': 1.0,
            'angle_degrees': 0.0,
        }

    sin_angle = math.sin(angle_rad)
    if abs(sin_angle) > 1e-6:
        axis = [
            (rotation[2][1] - rotation[1][2]) / (2.0 * sin_angle),
            (rotation[0][2] - rotation[2][0]) / (2.0 * sin_angle),
            (rotation[1][0] - rotation[0][1]) / (2.0 * sin_angle),
        ]
    else:
        xx = max(0.0, (rotation[0][0] + 1.0) / 2.0)
        yy = max(0.0, (rotation[1][1] + 1.0) / 2.0)
        zz = max(0.0, (rotation[2][2] + 1.0) / 2.0)
        axis = [math.sqrt(xx), math.sqrt(yy), math.sqrt(zz)]
        if rotation[2][1] - rotation[1][2] < 0.0:
            axis[0] = -axis[0]
        if rotation[0][2] - rotation[2][0] < 0.0:
            axis[1] = -axis[1]
        if rotation[1][0] - rotation[0][1] < 0.0:
            axis[2] = -axis[2]

    axis = _normalize_vector(axis)
    return {
        'x': axis[0],
        'y': axis[1],
        'z': axis[2],
        'angle_degrees': math.degrees(angle_rad),
    }


def _invert_rigid_transform(
    rotation: List[List[float]],
    translation: List[float],
) -> Tuple[List[List[float]], List[float]]:
    inverse_rotation = _transpose(rotation)
    inverse_translation = [-value for value in _matvec(inverse_rotation, translation)]
    return inverse_rotation, inverse_translation


def transform_frames_meta_to_ground_frame(
    input_frames_meta_file: pathlib.Path,
    ground_plane_file: pathlib.Path,
    output_frames_meta_file: pathlib.Path,
) -> pathlib.Path:
    normal, offset = load_ground_plane(ground_plane_file)
    map_from_omap_rotation = _rotation_matrix_from_z_to_normal(normal)
    map_from_omap_translation = [0.0, 0.0, -offset / normal[2]]
    omap_from_map_rotation, omap_from_map_translation = _invert_rigid_transform(
        map_from_omap_rotation,
        map_from_omap_translation,
    )

    with open(input_frames_meta_file, 'r') as input_file:
        frames_meta = json.load(input_file)

    for keyframe in frames_meta.get('keyframes_metadata', []):
        camera_to_world = keyframe.get('camera_to_world')
        if not camera_to_world:
            continue

        rotation = _rotation_matrix_from_axis_angle_dict(camera_to_world.get('axis_angle', {}))
        translation_dict = camera_to_world.get('translation', {})
        translation = [
            float(translation_dict.get('x', 0.0)),
            float(translation_dict.get('y', 0.0)),
            float(translation_dict.get('z', 0.0)),
        ]

        transformed_rotation = _matmul(omap_from_map_rotation, rotation)
        translated = [
            translation[index] - map_from_omap_translation[index]
            for index in range(3)
        ]
        transformed_translation = _matvec(omap_from_map_rotation, translated)

        keyframe['camera_to_world'] = {
            'translation': {
                'x': transformed_translation[0],
                'y': transformed_translation[1],
                'z': transformed_translation[2],
            },
            'axis_angle': _axis_angle_dict_from_rotation_matrix(transformed_rotation),
        }

    with open(output_frames_meta_file, 'w') as output_file:
        json.dump(frames_meta, output_file, indent=2)
        output_file.write('\n')

    logger.info(
        'Wrote ground-aligned frames metadata to %s using ground plane %s.',
        output_frames_meta_file,
        ground_plane_file,
    )
    return output_frames_meta_file


def build_nvblox_command(base_command: List[str], nvblox_config: Dict[str, Any],
                         output_dir: pathlib.Path) -> List[str]:

    command = build_command_from_config(
        base_command=base_command,
        config=nvblox_config,
        bool_as_flag=True,  # Nvblox uses boolean flags like --flag
    )

    if ('experimental_use_ground_plane_estimation' in nvblox_config
            and nvblox_config['experimental_use_ground_plane_estimation']):
        command.append(f'--ground_plane_output_path={output_dir}/ground_plane.yaml')

    return command
