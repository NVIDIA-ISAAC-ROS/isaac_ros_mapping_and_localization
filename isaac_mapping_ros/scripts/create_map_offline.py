#!/usr/bin/env python3

# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import math
import os
import pathlib
import subprocess
from datetime import datetime, timedelta

import ament_index_python.packages as packages

import rosbag2_py

from isaac_common_py import subprocess_utils
from isaac_common_py import filesystem_utils
from isaac_common_py import arg_utils
from create_map_utils import (
    load_map_creation_config,
    parse_config_overrides,
    build_cuvslam_command_api_launcher,
    build_nvblox_command,
)

ROS_WS = pathlib.Path(os.environ.get('ISAAC_ROS_WS'))
VISUAL_MAPPING_PACKAGE_NAME = 'isaac_ros_visual_mapping'
ISAAC_MAPPING_ROS_PACKAGE_NAME = 'isaac_mapping_ros'


def get_path(package: str, path: str) -> pathlib.Path:
    package_share = pathlib.Path(packages.get_package_share_directory(package))
    return package_share / path


def get_isaac_ros_visual_mapping_package_path() -> pathlib.Path:
    return pathlib.Path(packages.get_package_prefix(VISUAL_MAPPING_PACKAGE_NAME))


def get_visual_mapping_config_dir() -> pathlib.Path:
    return (get_isaac_ros_visual_mapping_package_path() / 'share' / VISUAL_MAPPING_PACKAGE_NAME /
            'configs' / 'isaac')


def get_visual_mapping_model_dir() -> pathlib.Path:
    return (get_isaac_ros_visual_mapping_package_path() / 'share' / VISUAL_MAPPING_PACKAGE_NAME /
            'models')


def get_visual_mapping_binary_dir() -> pathlib.Path:
    return (get_isaac_ros_visual_mapping_package_path() / 'lib' / VISUAL_MAPPING_PACKAGE_NAME)


def get_isaac_ros_visual_mapping_share_path() -> pathlib.Path:
    return (get_isaac_ros_visual_mapping_package_path() / 'share' / VISUAL_MAPPING_PACKAGE_NAME)


def get_isaac_mapping_ros_config_dir() -> pathlib.Path:
    return get_path(ISAAC_MAPPING_ROS_PACKAGE_NAME, 'configs')


def parse_args():
    parser = argparse.ArgumentParser(description='Script to build multiple maps from a rosbag.')
    parser.add_argument(
        '--sensor_data_bag',
        required=False,
        type=pathlib.Path,
        help='Path to the sensor data rosbag file. Required when running edex step or '
        'when --map_dir is not provided.',
    )
    parser.add_argument(
        '--base_output_folder',
        default=os.environ.get('ISAAC_MAPS_DIR', '/workspaces/isaac_ros-dev/ros_ws/data/maps'),
        type=pathlib.Path,
        help='Base output folder for the generated maps.',
    )
    parser.add_argument(
        '--prebuilt_bow_vocabulary_folder',
        type=pathlib.Path,
        default=None,
        help='Folder containing prebuilt BoW vocabulary files.',
    )
    parser.add_argument(
        '--print_mode',
        type=str,
        default='tail',
        choices=['none', 'tail', 'all'],
        help='Determines what is printed to stdout.',
    )
    parser.add_argument(
        '--steps_to_run',
        nargs='+',
        choices=[
            '',
            'edex',
            'compute_poses',
            'depth',
            'occupancy',
            'transform_map',
            'cuvgl',
        ],
        help='Specify which steps to run.',
    )
    parser.add_argument(
        '--map_dir',
        type=pathlib.Path,
        help='Directory containing existing map data.',
    )
    parser.add_argument(
        '--image_extension',
        type=str,
        default='.jpg',
        help='Determines the image format to save on disk.',
    )
    parser.add_argument(
        '--camera_topic_config',
        type=pathlib.Path,
        default=None,
        help='Path to camera topic configuration file for rosbag_to_mapping_data.',
    )
    parser.add_argument(
        '--map_creation_config',
        type=pathlib.Path,
        default=get_isaac_mapping_ros_config_dir() / 'map_creation_config.yaml',
        help='Path to YAML config file for map creation parameters.',
    )
    parser.add_argument(
        '--override',
        '-o',
        action='append',
        default=[],
        metavar='SECTION.PARAM=VALUE',
        help='Override config parameters from command line. Can be used multiple times. '
             'Format: SECTION.PARAM=VALUE. '
             'Examples: -o cuvslam.max_fps=30 -o nvblox.esdf_slice_height=0.5 '
             '-o cuvslam.verbosity=15',
    )
    parser.add_argument(
        '--base_link_name',
        type=str,
        default='base_link',
        help='The frame name of the base link',
    )
    parser.add_argument(
        '--use_raw_image',
        type=arg_utils.str_to_bool,
        nargs='?',
        const=True,
        default=True,
        help='If set, use raw (unrectified) images by passing '
             '--rectify_images=False to rosbag_to_mapping_data',
    )
    parser.add_argument(
        '--depth_model',
        type=str,
        default='foundationstereo',
        help='Stereo depth estimation model to use (default: foundationstereo)',
        choices=['ess', 'foundationstereo'],
    )
    parser.add_argument(
        '--use_cusfm',
        type=arg_utils.str_to_bool,
        nargs='?',
        const=True,
        default=False,
        help='Use CUSFM workflow: run cuvslam -> cusfm -> pose fitting -> optimized cuvslam',
    )
    parser.add_argument(
        '--skip_final_cuvslam',
        type=arg_utils.str_to_bool,
        nargs='?',
        const=True,
        default=True,
        help='Skip final cuvslam map creation in CUSFM workflow',
    )
    parser.add_argument(
        '--vgl_model_dir',
        type=pathlib.Path,
        default=None,
        help='Directory containing VGL models (default: uses $(ros2 pkg prefix --share '
             'isaac_ros_visual_mapping)/models/',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    steps_to_run = args.steps_to_run or [
        'edex', 'compute_poses', 'depth', 'occupancy', 'transform_map', 'cuvgl'
    ]

    # Validate sensor_data_bag requirements
    sensor_bag_required = 'edex' in steps_to_run or args.map_dir is None
    if sensor_bag_required and args.sensor_data_bag is None:
        raise ValueError(
            "sensor_data_bag is required when running 'edex' step or when --map_dir is not "
            "provided. Either provide --sensor_data_bag or specify --map_dir with steps that "
            "don't include 'edex'.")

    # Load map creation configuration
    # Parse CLI overrides
    cli_overrides = parse_config_overrides(args.override) if args.override else None

    # Load config with overrides (prints overrides internally)
    map_config = load_map_creation_config(
        args.map_creation_config,
        cli_overrides
    )

    if 'depth' in steps_to_run:
        check_running_depth_inference(args.depth_model)

    # Setup the output folder.
    start_time = datetime.now()
    timestamp = start_time.strftime('%Y-%m-%d_%H-%M-%S')
    if not args.map_dir:
        # Create new output folder using bag name
        bag_name = args.sensor_data_bag.name
        output_folder = filesystem_utils.create_workdir(
            args.base_output_folder,
            f'{timestamp}_{bag_name}',
            allow_sudo=True,
        )
    else:
        # Use existing map directory
        output_folder = args.map_dir
    print(f'Storing all maps and logs in {output_folder}.')

    log_folder = output_folder / 'logs'
    log_folder.mkdir(parents=True, exist_ok=True)

    # Create a metadata file.
    metadata_file = output_folder / 'metadata.yaml'
    metadata_file.write_text(f'output_folder: {output_folder}\n')

    # Get duration and estimate completion time only if we have a sensor bag
    if args.sensor_data_bag:
        duration = get_rosbag_duration(args.sensor_data_bag)
        # print a rough estimated processing time
        estimated_endtime = start_time + timedelta(seconds=duration * 20)
        print(f'Bag Duration: {int(duration)} seconds, ' +
              f'Map Creation Starts: {start_time.strftime("%H:%M:%S")}, ' +
              f'Estimated Completion: {estimated_endtime.strftime("%H:%M:%S")}')
    else:
        duration = 0  # Default for when no bag is provided
        print(f'Map Creation Starts: {start_time.strftime("%H:%M:%S")} '
              f'(no duration estimate without sensor bag)')

    edex_path = output_folder / 'edex'
    if 'edex' in steps_to_run:
        edex_timeout = math.ceil(duration * 5)
        edex_path.mkdir(parents=True, exist_ok=True)
        generate_edex_from_rosbag(args.sensor_data_bag, edex_path, args.image_extension,
                                  log_folder, args.print_mode, edex_timeout,
                                  args.camera_topic_config, args.base_link_name,
                                  args.use_raw_image)

    # Choose workflow based on --use_cusfm flag
    if 'compute_poses' in steps_to_run:
        if args.use_cusfm:
            # Run CUSFM workflow
            run_cusfm_workflow(
                edex_path,
                output_folder,
                log_folder,
                args.print_mode,
                duration,
                args.use_raw_image,
                map_config,
                args.skip_final_cuvslam,
            )
        else:
            # Run standard workflow
            run_standard_workflow(
                edex_path,
                output_folder,
                log_folder,
                args.print_mode,
                duration,
                args.use_raw_image,
                map_config
            )

        # Generate comparison reports (before depth inference)
        print('\n' + '=' * 50)
        print('GENERATING COMPARISON REPORTS')
        print('=' * 50)
        try:
            # Generate pose comparison for the current map (internal consistency check)
            generate_pose_plots(output_folder, log_folder, args.print_mode)
        except Exception as e:
            print(f'Warning: Failed to generate pose comparison: {e}')
        print('Comparison generation completed')

    # Always use rectified directory for consistency
    # When use_raw_image=True: raw images are rectified to this directory
    # When use_raw_image=False: already-rectified images are copied to this directory
    map_frames_rectified_dir = output_folder / 'map_frames' / 'rectified'
    final_meta_file = map_frames_rectified_dir / 'frames_meta.json'

    if not map_frames_rectified_dir.exists():
        raise RuntimeError(
            f"Cannot run depth step: Depth directory {map_frames_rectified_dir} does not exist. "
            f"Run 'compute_poses' step first.")

    num_map_frames = count_image_files(map_frames_rectified_dir)

    map_frames_depth_dir = output_folder / 'map_frames' / 'depth'
    if 'depth' in steps_to_run:
        depth_timeout = math.ceil(num_map_frames * 0.15 + 60) if num_map_frames > 0 else 60
        run_depth_inference(map_frames_rectified_dir, map_frames_depth_dir, final_meta_file,
                            log_folder, args.print_mode, depth_timeout, args.depth_model)

    if 'occupancy' in steps_to_run:
        nvblox_timeout = math.ceil(num_map_frames * 0.07) if num_map_frames > 0 else 60
        create_occupancy_map(output_folder, map_frames_rectified_dir, map_frames_depth_dir,
                             final_meta_file, log_folder, args.print_mode, nvblox_timeout,
                             map_config['nvblox'])

    if 'transform_map' in steps_to_run:
        if args.use_cusfm:
            transform_cusfm_map(output_folder / 'cusfm' / 'kpmap', output_folder, log_folder,
                                args.print_mode)

    if 'cuvgl' in steps_to_run:
        # also generate cuvgl map
        cuvgl_map_folder = output_folder / 'cuvgl_map'
        cuvgl_map_folder.mkdir(parents=True, exist_ok=True)
        create_cuvgl_map(cuvgl_map_folder, map_frames_rectified_dir, args.print_mode,
                         args.prebuilt_bow_vocabulary_folder)

    end_time = datetime.now()
    print(f'All maps can be found in {output_folder}, ' +
          f'Completed at: {end_time.strftime("%H:%M:%S")}, ' +
          f'Total Time: {int((end_time - start_time).total_seconds())} seconds')


def get_rosbag_duration(bag_path: pathlib.Path):
    # Create a reader
    reader = rosbag2_py.SequentialReader()

    # Open the bag file
    reader.open(rosbag2_py.StorageOptions(uri=str(bag_path)), rosbag2_py.ConverterOptions())

    # Get metadata
    metadata = reader.get_metadata()

    # Calculate duration
    duration_nanoseconds = metadata.duration.nanoseconds
    duration_seconds = duration_nanoseconds / 1e9

    return duration_seconds


def generate_edex_from_rosbag(sensor_data_bag: pathlib.Path,
                              output_folder: pathlib.Path,
                              image_extension: str,
                              log_folder: pathlib.Path,
                              print_mode: str,
                              timeout: int,
                              camera_topic_config: pathlib.Path = None,
                              base_link_name: str = 'base_link',
                              use_raw_image: bool = False):
    command = [
        'ros2',
        'run',
        'isaac_mapping_ros',
        'rosbag_to_mapping_data',
        f'--output_folder_path={output_folder}',
        f'--sensor_data_bag_file={sensor_data_bag}',
        '--min_inter_frame_distance=0.0',
        '--min_inter_frame_rotation_degrees=0.0',
        '--sample_sync_threshold_microseconds=100',
        '--generate_edex=True',
        f'--image_extension={image_extension}',
        f'--base_link_name={base_link_name}',
    ]

    if camera_topic_config:
        command.append(f'--camera_topic_config={camera_topic_config}')
    if use_raw_image:
        command.append('--rectify_images=False')

    subprocess_utils.run_command(
        mnemonic='Extract edex',
        command=command,
        log_file=log_folder / 'extract_edex.log',
        print_mode=print_mode,
        timeout=timeout,
    )


def run_cuvslam_api_launcher(edex_path: pathlib.Path,
                             output_map_dir: pathlib.Path,
                             output_poses_dir: pathlib.Path,
                             log_folder: pathlib.Path,
                             print_mode: str,
                             timeout: int,
                             cuvslam_config: dict = None,
                             use_raw_image: bool = True,
                             mnemonic: str = 'Run cuvslam_api_launcher'):
    additional_path = get_path('isaac_ros_visual_slam', '../cuvslam/lib/').resolve()
    ld_library_path = os.environ['LD_LIBRARY_PATH']
    os.environ['LD_LIBRARY_PATH'] = f'{ld_library_path}:{additional_path}'
    base_command = [
        'ros2',
        'run',
        'isaac_ros_visual_slam',
        'cuvslam_api_launcher',
        f'--dataset={edex_path}',
        f'--output_map={output_map_dir}',
    ]
    if cuvslam_config is None:
        cuvslam_config = {}
    else:
        cuvslam_config = cuvslam_config.copy()
    if use_raw_image:
        cuvslam_config['cfg_horizontal'] = False
    else:
        cuvslam_config['cfg_horizontal'] = True
    command = build_cuvslam_command_api_launcher(base_command, cuvslam_config, log_folder,
                                                 output_poses_dir)
    subprocess_utils.run_command(
        mnemonic=mnemonic,
        command=command,
        log_file=log_folder / 'run_cuvslam_api_launcher.log',
        print_mode=print_mode,
        timeout=timeout,
    )


def create_cuvslam_map(
    edex_path: pathlib.Path,
    output_cuvslam_map_dir: pathlib.Path,
    output_cuvslam_poses_dir: pathlib.Path,
    log_folder: pathlib.Path,
    print_mode: str,
    timeout: int,
    map_config: dict,
    use_raw_image: bool = True,
):
    if not map_config:
        raise ValueError("map_config is required")
    if not map_config.get('cuvslam'):
        raise ValueError("cuvslam config is required")
    run_cuvslam_api_launcher(edex_path=edex_path,
                             output_map_dir=output_cuvslam_map_dir,
                             output_poses_dir=output_cuvslam_poses_dir,
                             log_folder=log_folder,
                             print_mode=print_mode,
                             timeout=timeout,
                             cuvslam_config=map_config.get('cuvslam'),
                             use_raw_image=use_raw_image,
                             mnemonic='Create cuVSLAM map with cuvslam_api_launcher')


def select_map_frames(input_frames_meta_file: pathlib.Path, output_frames_meta_file: pathlib.Path,
                      log_folder: pathlib.Path, print_mode: str):
    subprocess_utils.run_command(
        mnemonic='Select map frames',
        command=[
            'ros2',
            'run',
            'isaac_mapping_ros',
            'select_frames_meta',
            f'--input_frames_meta_file={input_frames_meta_file}',
            f'--output_frames_meta_file={output_frames_meta_file}',
            '--min_inter_frame_distance=0.1',
            '--min_inter_frame_rotation_degrees=2',
        ],
        log_file=log_folder / 'select_map_frames.log',
        print_mode=print_mode,
    )


def optimize_vo_with_keyframe_pose(vo_pose_file: pathlib.Path, kf_pose_file: pathlib.Path,
                                   output_pose_file: pathlib.Path, log_folder: pathlib.Path,
                                   print_mode: str):
    config_file = (get_visual_mapping_config_dir() / 'vo_pose_optimize_ba_config.pb.txt')
    binary_path = get_visual_mapping_binary_dir() / 'optimize_vo_with_keyframe_pose_main'
    subprocess_utils.run_command(
        mnemonic='Optimize odometry pose with keyframe pose',
        command=[
            str(binary_path),
            f'--vo_tum_file={vo_pose_file}',
            f'--kf_tum_file={kf_pose_file}',
            f'--output_tum_file={output_pose_file}',
            f'--bundle_adjustment_config={config_file}',
        ],
        log_file=log_folder / 'optimize_vo_with_keyframe_pose.log',
        print_mode=print_mode,
    )


def update_frames_meta_pose(tum_pose_file: pathlib.Path, input_frames_meta_file: pathlib.Path,
                            output_frames_meta_file: pathlib.Path, log_folder: pathlib.Path,
                            print_mode: str):
    binary_path = get_visual_mapping_binary_dir() / 'update_keyframe_pose_main'
    subprocess_utils.run_command(
        mnemonic='Update frame metadata pose',
        command=[
            str(binary_path),
            f'--tum_pose_file={tum_pose_file}',
            f'--input_file={input_frames_meta_file}',
            f'--output_file={output_frames_meta_file}',
            '--max_pose_gap_seconds=0.2',
        ],
        log_file=log_folder / 'update_frames_meta_pose.log',
        print_mode=print_mode,
    )


def copy_raw_image_folder(input_raw_dir: pathlib.Path, input_frames_meta_file: pathlib.Path,
                          output_raw_dir: pathlib.Path, log_folder: pathlib.Path, print_mode: str):
    binary_path = get_visual_mapping_binary_dir() / 'copy_image_dir_main'
    command = [
        str(binary_path),
        f'--input_image_dir={input_raw_dir}',
        f'--output_image_dir={output_raw_dir}',
        f'--frames_meta_file={input_frames_meta_file}',
    ]
    subprocess_utils.run_command(
        mnemonic='Copy map frames',
        command=command,
        log_file=log_folder / 'copy_map_frames.log',
        print_mode=print_mode,
    )


def run_depth_inference(color_img_dir: pathlib.Path,
                        depth_img_dir: pathlib.Path,
                        frames_meta_file: pathlib.Path,
                        log_folder: pathlib.Path,
                        print_mode: str,
                        timeout: int,
                        depth_model: str = 'foundationstereo'):
    # Validate required directories and files exist
    if not color_img_dir.exists():
        raise RuntimeError(
            f"Cannot run depth inference: Image directory {color_img_dir} does not exist. "
            f"Run 'edex' and 'compute_poses' steps first.")

    num_images = count_image_files(color_img_dir)
    if num_images == 0:
        raise RuntimeError(f"Cannot run depth inference: No images found in {color_img_dir}.")

    if not frames_meta_file.exists():
        raise RuntimeError(
            f"Cannot run depth inference: Metadata file {frames_meta_file} does not exist. "
            f"Run 'compute_poses' step first.")

    if depth_model == 'foundationstereo':
        script_name = 'run_foundationstereo_trt_offline.py'
        mnemonic = 'Run Foundation Stereo inference'
        log_file = 'run_foundationstereo_inference.log'
    else:
        script_name = 'run_ess_trt_offline.py'
        mnemonic = 'Run ESS inference'
        log_file = 'run_ess_inference.log'

    command = [
        'ros2',
        'run',
        'isaac_mapping_ros',
        script_name,
        f'--image_dir={color_img_dir}',
        f'--output_dir={depth_img_dir}',
        f'--frames_meta_file={frames_meta_file}',
    ]
    subprocess_utils.run_command(
        mnemonic=mnemonic,
        command=command,
        log_file=log_folder / log_file,
        timeout=timeout,
        print_mode=print_mode,
    )


def create_occupancy_map(output_dir: pathlib.Path, color_img_dir: pathlib.Path,
                         depth_img_dir: pathlib.Path, frames_meta_file: pathlib.Path,
                         log_folder: pathlib.Path, print_mode: str, timeout: int,
                         nvblox_config: dict):
    # Validate required directories and files exist
    if not color_img_dir.exists():
        raise RuntimeError(
            f"Cannot create occupancy map: Image directory {color_img_dir} does not exist. "
            f"Run 'edex' and 'compute_poses' steps first.")

    if not depth_img_dir.exists():
        raise RuntimeError(
            f"Cannot create occupancy map: Depth directory {depth_img_dir} does not exist. "
            f"Run 'depth' step first.")

    if not frames_meta_file.exists():
        raise RuntimeError(
            f"Cannot create occupancy map: Metadata file {frames_meta_file} does not exist. "
            f"Run 'compute_poses' step first.")

    occupancy_map_path = f'{output_dir}/occupancy_map'
    mesh_output_path = f'{output_dir}/mesh.ply'
    base_command = [
        'ros2',
        'run',
        'nvblox_ros',
        'fuse_cusfm',
        f'--save_2d_occupancy_map_path={occupancy_map_path}',
        f'--color_image_dir={color_img_dir}',
        f'--frames_meta_file={frames_meta_file}',
        f'--depth_image_dir={depth_img_dir}',
        f'--mesh_output_path={mesh_output_path}',
    ]

    command = build_nvblox_command(base_command, nvblox_config)

    subprocess_utils.run_command(
        mnemonic='Run Nvblox',
        command=command,
        log_file=log_folder / 'fuse_cusfm.log',
        print_mode=print_mode,
        timeout=timeout,
    )

    # check occupancy map file is generated
    if not os.path.exists(occupancy_map_path + '.png'):
        raise RuntimeError(
            f'Something went wrong! Occupancy image not found at {occupancy_map_path}.png.')


def create_cuvgl_map(cuvgl_map_folder: pathlib.Path,
                     map_frames_image_dir: pathlib.Path,
                     print_mode: str,
                     prebuilt_bow_vocabulary_folder: pathlib.Path = None):
    # Validate required directory exists
    if not map_frames_image_dir.exists():
        raise RuntimeError(
            f"Cannot create cuvgl map: Image directory {map_frames_image_dir} does not exist. "
            f"Run 'edex' and 'compute_poses' steps first.")

    binary_folder = get_visual_mapping_binary_dir()
    config_folder = get_visual_mapping_config_dir()
    model_dir = get_visual_mapping_model_dir()
    command = [
        'ros2',
        'run',
        'isaac_ros_visual_mapping',
        'create_cuvgl_map.py',
        f'--map_folder={cuvgl_map_folder}',
        f'--raw_image_folder={map_frames_image_dir}',
        '--extract_feature',
        '--feature_type=aliked',
        f'--print_mode={print_mode}',
        f'--binary_folder_path={binary_folder}',
        f'--config_folder_path={config_folder}',
        f'--model_dir={model_dir}',
    ]
    if prebuilt_bow_vocabulary_folder:
        command.append(f'--prebuilt_bow_vocabulary_folder={prebuilt_bow_vocabulary_folder}')
    subprocess.run(command, check=True)


def count_image_files(image_folder: pathlib.Path):
    count = 0
    for root, dirs, files in os.walk(image_folder):
        # Count files that do not end with '.json' or '.txt'
        count += len([f for f in files if not (f.endswith('.json') or f.endswith('.txt'))])
    return count


def check_running_depth_inference(depth_model: str = 'ess'):
    result = subprocess.run(["ros2", "node", "list"], capture_output=True, text=True, check=True)
    active_nodes = result.stdout.splitlines()

    if depth_model == 'foundationstereo':
        stereo_processes = ['/foundationstereo_container', '/foundationstereo_decoder']
        model_name = 'Foundation Stereo'
    else:
        stereo_processes = ['/disparity', '/disparity/disparity_container']
        model_name = 'ESS'

    for name in stereo_processes:
        if name in active_nodes:
            raise RuntimeError(
                f'{model_name} node {name} is already running. Please wait for it to '
                f'finish, manually stop the node, or restart the Docker container '
                f'before running this command again.')


def transform_cusfm_map(cusfm_map_dir: pathlib.Path, map_folder: pathlib.Path,
                        log_folder: pathlib.Path, print_mode: str):
    transform_file = map_folder / 'T_world_to_z0.json'
    if not transform_file.exists():
        print(f"Warning: T_world_to_z0.json not found at {transform_file}")
        print("Skipping kpmap transformation - occupancy mapping may not have been run yet")
        print("The kpmap will remain in its original coordinate frame")
        return
    print(f"Found occupancy transform file: {transform_file}")
    print("Transforming kpmap to align with occupancy map coordinate frame...")
    transformed_kpmap_dir = map_folder / 'cusfm_map'
    subprocess_utils.run_command(
        mnemonic='Transform kpmap with occupancy coordinate alignment',
        command=[
            'ros2', 'run', 'isaac_ros_visual_mapping', 'transform_kp_map_main',
            f'--map_dir={str(cusfm_map_dir)}', f'--transform_file={str(transform_file)}',
            f'--output_dir={str(transformed_kpmap_dir)}'
        ],
        log_file=log_folder / 'transform_kpmap_occupancy.log',
        print_mode=print_mode,
        timeout=300,
    )
    print("Successfully transformed kpmap and aligned with occupancy coordinate frame")


def run_cusfm(rectified_images_dir: pathlib.Path, cusfm_base_dir: pathlib.Path,
              log_folder: pathlib.Path, print_mode: str):
    print(f'Running CUSFM on {rectified_images_dir}')
    cusfm_base_dir.mkdir(parents=True, exist_ok=True)
    model_dir = get_visual_mapping_model_dir()
    config_dir = get_visual_mapping_config_dir()
    binary_dir = get_visual_mapping_binary_dir()
    command = [
        'ros2',
        'run',
        'isaac_ros_visual_mapping',
        'cusfm_cli',
        f'--input_dir={rectified_images_dir}',
        f'--cusfm_base_dir={cusfm_base_dir}',
        f'--model_dir={model_dir}',
        f'--config_dir={config_dir}',
        f'--binary_dir={binary_dir}',
        '--skip_cuvslam',
        '--min_inter_frame_distance=0.0',
        '--min_inter_frame_rotation_degrees=0.0',
    ]
    subprocess_utils.run_command(
        mnemonic='Run CUSFM',
        command=command,
        log_file=log_folder / 'cusfm.txt',
        print_mode=print_mode,
        timeout=None
    )


def run_rectify_images_offline(raw_image_dir: pathlib.Path, rectified_image_dir: pathlib.Path,
                               log_folder: pathlib.Path, print_mode: str):
    print(f'Rectifying images from {raw_image_dir} to {rectified_image_dir}')
    rectified_image_dir.mkdir(parents=True, exist_ok=True)
    command = [
        'ros2',
        'run',
        'isaac_mapping_ros',
        'rectify_images_offline.py',
        f'--raw_image_dir={raw_image_dir}',
        f'--rectified_image_dir={rectified_image_dir}',
    ]
    if log_folder:
        command.extend([f'--log_folder={log_folder}'])
    subprocess_utils.run_command(mnemonic='Rectify images offline',
                                 command=command,
                                 log_file=log_folder / 'rectify_images.txt',
                                 print_mode=print_mode)


def run_pose_plane_fit(input_pose_file: pathlib.Path, fitted_pose_file: pathlib.Path,
                       log_folder: pathlib.Path, print_mode: str):
    print('Running pose plane fitting')
    if not input_pose_file.exists():
        raise FileNotFoundError(f'Input pose file not found: {input_pose_file}')
    command = [
        'ros2', 'run', 'isaac_mapping_ros', 'pose_plane_fit.py', f'--input_file={input_pose_file}',
        f'--output_file={fitted_pose_file}'
    ]
    subprocess_utils.run_command(mnemonic='Run pose plane fit',
                                 command=command,
                                 log_file=log_folder / 'pose_plane_fit.txt',
                                 print_mode=print_mode)


def run_cusfm_workflow(edex_path: pathlib.Path,
                       output_folder: pathlib.Path,
                       log_folder: pathlib.Path,
                       print_mode: str,
                       duration: float,
                       use_raw_image: bool,
                       map_config: dict,
                       skip_final_cuvslam: bool = False):
    print("=== Starting CUSFM Workflow ===")
    cusfm_base_dir = output_folder / 'cusfm'
    initial_cuvslam_map_dir = cusfm_base_dir / 'cuvslam_map'
    initial_cuvslam_poses_dir = cusfm_base_dir / 'cuvslam_poses'
    if use_raw_image:
        map_frames_raw_dir = output_folder / 'map_frames' / 'raw'
        map_frames_initial_dir = map_frames_raw_dir
    else:
        map_frames_rectified_dir = output_folder / 'map_frames' / 'rectified'
        map_frames_initial_dir = map_frames_rectified_dir
    map_frames_initial_dir.mkdir(parents=True, exist_ok=True)
    map_config['cuvslam']['repeat'] = 1
    print("Step 1: Running initial CUVSLAM")
    cuvslam_timeout = math.ceil(duration * 7)
    initial_cuvslam_map_dir.mkdir(parents=True, exist_ok=True)
    initial_cuvslam_poses_dir.mkdir(parents=True, exist_ok=True)
    create_cuvslam_map(edex_path, initial_cuvslam_map_dir, initial_cuvslam_poses_dir, log_folder,
                       print_mode, cuvslam_timeout, map_config, use_raw_image)
    print("Step 2: Running map frames step")
    cuvslam_odom_tum_file = initial_cuvslam_poses_dir / 'odom_poses.tum'
    cuvslam_kf_tum_file = initial_cuvslam_poses_dir / 'keyframe_pose.tum'
    cuvslam_kf_optimized_tum_file = initial_cuvslam_poses_dir / 'keyframe_pose_optimized.tum'
    all_frames_meta_file = edex_path / 'frames_meta.json'
    all_frames_meta_kf_poses_file = edex_path / 'frames_meta_kf_poses.json'
    initial_map_frames_meta_file = map_frames_initial_dir / 'frames_meta.json'
    optimize_vo_with_keyframe_pose(cuvslam_odom_tum_file, cuvslam_kf_tum_file,
                                   cuvslam_kf_optimized_tum_file, log_folder, print_mode)
    update_frames_meta_pose(cuvslam_kf_optimized_tum_file, all_frames_meta_file,
                            all_frames_meta_kf_poses_file, log_folder, print_mode)
    map_frames_initial_dir.mkdir(parents=True, exist_ok=True)
    select_map_frames(all_frames_meta_kf_poses_file, initial_map_frames_meta_file, log_folder,
                      print_mode)
    copy_raw_image_folder(
        edex_path,
        initial_map_frames_meta_file,
        map_frames_initial_dir,
        log_folder,
        print_mode,
    )
    if use_raw_image:
        print("Step 3: Generating rectified images")
        map_frames_rectified_dir = output_folder / 'map_frames' / 'rectified'
        run_rectify_images_offline(map_frames_initial_dir, map_frames_rectified_dir, log_folder,
                                   print_mode)
        map_frames_to_use = map_frames_rectified_dir
    else:
        map_frames_to_use = map_frames_initial_dir
    print("Step 4: Running CUSFM")
    run_cusfm(map_frames_to_use, cusfm_base_dir, log_folder, print_mode)
    cusfm_poses_dir = cusfm_base_dir / 'output_poses'
    merged_pose_file = cusfm_poses_dir / 'merged_pose_file.tum'
    pose_file_for_optimization = merged_pose_file
    map_frames_fitted_meta_file = map_frames_to_use / 'frames_meta.json'
    old_frames_meta_file = map_frames_to_use / 'frames_meta_old.json'
    map_frames_fitted_meta_file.rename(old_frames_meta_file)
    update_frames_meta_pose(
        merged_pose_file,
        old_frames_meta_file,
        map_frames_fitted_meta_file,
        log_folder,
        print_mode,
    )
    print("Step 7: Generating fully optimized poses")
    fully_optimized_pose_file = output_folder / 'poses' / 'fully_optimized_poses.tum'
    fully_optimized_pose_file.parent.mkdir(parents=True, exist_ok=True)
    optimize_vo_with_keyframe_pose(
        cuvslam_odom_tum_file,
        pose_file_for_optimization,
        fully_optimized_pose_file,
        log_folder,
        print_mode,
    )
    if skip_final_cuvslam:
        print("Step 8: Skipping final CUVSLAM map creation (--skip_final_cuvslam enabled)")
    else:
        print("Step 8: Final CUVSLAM with optimized poses not supported without pycuvslam")
        print("Note: cuvslam_api_launcher does not support external pose constraints")
    print("=== CUSFM Workflow Complete ===")


def run_standard_workflow(edex_path: pathlib.Path, output_folder: pathlib.Path,
                          log_folder: pathlib.Path, print_mode: str, duration: float,
                          use_raw_image: bool, map_config: dict):
    print("=== Starting Standard Workflow ===")
    output_cuvslam_map_dir = output_folder / 'cuvslam_map'
    output_cuvslam_poses_dir = output_folder / 'poses'
    cuvslam_timeout = math.ceil(duration * 7)
    output_cuvslam_map_dir.mkdir(parents=True, exist_ok=True)
    output_cuvslam_poses_dir.mkdir(parents=True, exist_ok=True)
    create_cuvslam_map(edex_path, output_cuvslam_map_dir, output_cuvslam_poses_dir, log_folder,
                       print_mode, cuvslam_timeout, map_config, use_raw_image)
    cuvslam_config = map_config.get('cuvslam', {})
    repeat_count = cuvslam_config.get('repeat', 1)
    print(f"Checking repeat count: {repeat_count}")
    if repeat_count > 1:
        print(f"Processing repeated poses for {repeat_count} runs...")
        process_repeated_poses(edex_path, output_cuvslam_poses_dir, repeat_count, log_folder,
                               print_mode)
    else:
        print(f"Repeat count is {repeat_count}, skipping repeated poses processing")
    cuvslam_odom_tum_file = output_cuvslam_poses_dir / 'odom_poses.tum'
    cuvslam_kf_tum_file = output_cuvslam_poses_dir / 'keyframe_pose.tum'
    cuvslam_kf_optimized_tum_file = output_cuvslam_poses_dir / 'keyframe_pose_optimized.tum'
    all_frames_meta_file = edex_path / 'frames_meta.json'
    all_frames_meta_kf_poses_file = edex_path / 'frames_meta_kf_poses.json'
    if use_raw_image:
        map_frames_raw_dir = output_folder / 'map_frames' / 'raw'
        map_frames_initial_dir = map_frames_raw_dir
        map_frames_initial_keyframe = map_frames_raw_dir / 'frames_meta.json'
    else:
        map_frames_rectified_dir = output_folder / 'map_frames' / 'rectified'
        map_frames_initial_dir = map_frames_rectified_dir
        map_frames_initial_keyframe = map_frames_rectified_dir / 'frames_meta.json'
    map_frames_initial_dir.mkdir(parents=True, exist_ok=True)
    optimize_vo_with_keyframe_pose(cuvslam_odom_tum_file, cuvslam_kf_tum_file,
                                   cuvslam_kf_optimized_tum_file, log_folder, print_mode)
    update_frames_meta_pose(cuvslam_kf_optimized_tum_file, all_frames_meta_file,
                            all_frames_meta_kf_poses_file, log_folder, print_mode)
    select_map_frames(
        all_frames_meta_kf_poses_file,
        map_frames_initial_keyframe,
        log_folder,
        print_mode,
    )
    copy_raw_image_folder(
        edex_path,
        map_frames_initial_keyframe,
        map_frames_initial_dir,
        log_folder,
        print_mode,
    )
    if use_raw_image:
        map_frames_rectified_dir = output_folder / 'map_frames' / 'rectified'
        run_rectify_images_offline(
            map_frames_initial_dir,
            map_frames_rectified_dir,
            log_folder,
            print_mode,
        )
    print("=== Standard Workflow Complete ===")


def process_repeated_poses(edex_path: pathlib.Path, output_poses_dir: pathlib.Path,
                           repeat_count: int, log_folder: pathlib.Path, print_mode: str):
    mapping_scripts_dir = pathlib.Path(os.environ.get('ISAAC_ROS_WS')) / 'src' / \
        'isaac_ros_mapping_and_localization' / 'isaac_mapping_ros' / 'scripts'
    split_script = mapping_scripts_dir / 'split_repeated_poses.py'
    if not split_script.exists():
        print(f"Warning: split_repeated_poses.py script not found at {split_script}")
        return
    subprocess_utils.run_command(
        mnemonic='Split repeated pose files',
        command=[
            'python3',
            str(split_script), '--poses_dir',
            str(output_poses_dir), '--edx_dir',
            str(edex_path), '--repeat_count',
            str(repeat_count)
        ],
        log_file=log_folder / 'split_repeated_poses.log',
        print_mode=print_mode,
        timeout=60,
    )


def generate_pose_plots(current_map_dir: pathlib.Path, log_folder: pathlib.Path, print_mode: str):
    mapping_scripts_dir = pathlib.Path(os.environ.get('ISAAC_ROS_WS')) / 'src' / \
        'isaac_ros_mapping_and_localization' / 'isaac_mapping_ros' / 'scripts'
    compare_poses_script = mapping_scripts_dir / 'compare_poses.py'
    if not compare_poses_script.exists():
        print(f"Warning: compare_poses.py script not found at {compare_poses_script}")
        return
    comparison_dir = current_map_dir / 'pose_comparison'
    comparison_dir.mkdir(parents=True, exist_ok=True)
    subprocess_utils.run_command(
        mnemonic='Generate pose comparison plots',
        command=[
            'python3',
            str(compare_poses_script), '--map_dir',
            str(current_map_dir), '--output',
            str(comparison_dir), '--stats'
        ],
        log_file=log_folder / 'pose_comparison.log',
        print_mode=print_mode,
        timeout=300,
    )
    print(f'Pose comparison plots and statistics saved to {comparison_dir}')


if __name__ == '__main__':
    main()
