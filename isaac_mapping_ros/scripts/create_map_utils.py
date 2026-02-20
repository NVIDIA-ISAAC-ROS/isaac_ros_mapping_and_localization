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

import pathlib
from typing import Dict, Any, List, Optional
import yaml


def get_default_nvblox_config() -> Dict[str, Any]:
    """Get default configuration for nvblox mapping."""
    return {
        'projective_integrator_max_integration_distance_m': 5.0,
        'use_2d_esdf_mode': True,
        'experimental_use_ground_plane_estimation': True,
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
        'ros_frame_conversion': True,
        'print_format': 'tum',
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
        # repeat
        'repeat': 1,
        # export
        'debug_dump': False,
    }


def load_map_creation_config(
    config_path: Optional[pathlib.Path] = None,
    cli_overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
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


def build_cuvslam_command_api_launcher(
        base_command: List[str],
        cuvslam_config: Dict[str, Any],
        log_folder: pathlib.Path,
        output_poses_dir: pathlib.Path
) -> List[str]:
    if not cuvslam_config:
        return base_command

    command = build_command_from_config(
        base_command=base_command,
        config=cuvslam_config,
        bool_as_flag=False,  # cuVSLAM uses --param=true/false format
        exclude_fields=['debug_dump'],  # Special handling below
    )

    if 'debug_dump' in cuvslam_config:
        debug_dump_enabled = cuvslam_config["debug_dump"]
        if debug_dump_enabled:
            command.append(f'--debug_dump={log_folder}')

    repeat_count = cuvslam_config.get("repeat", 1)
    if repeat_count > 1:
        odom_file = "odom_poses_repeated.tum"
        slam_file = "slam_poses_repeated.tum"
        keyframe_file = "keyframe_pose_repeated.tum"
    else:
        odom_file = "odom_poses.tum"
        slam_file = "slam_poses.tum"
        keyframe_file = "keyframe_pose.tum"

    command.extend([
        f'--print_odom_poses={output_poses_dir}/{odom_file}',
        f'--print_slam_poses={output_poses_dir}/{slam_file}',
        f'--print_map_keyframes={output_poses_dir}/{keyframe_file}',
    ])

    return command


def build_nvblox_command(
    base_command: List[str],
    nvblox_config: Dict[str, Any],
    output_dir: pathlib.Path
) -> List[str]:

    command = build_command_from_config(
        base_command=base_command,
        config=nvblox_config,
        bool_as_flag=True,  # Nvblox uses boolean flags like --flag
    )

    if (
        'experimental_use_ground_plane_estimation' in nvblox_config and
        nvblox_config['experimental_use_ground_plane_estimation']
    ):
        command.append(f'--ground_plane_output_path={output_dir}/ground_plane.yaml')

    return command
