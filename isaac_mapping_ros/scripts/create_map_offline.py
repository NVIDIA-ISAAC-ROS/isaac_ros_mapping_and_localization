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

ROS_WS = pathlib.Path(os.environ.get('ISAAC_ROS_WS'))
VISUAL_MAPPING_PACKAGE_NAME = 'isaac_ros_visual_mapping'


def get_path(package: str, path: str) -> pathlib.Path:
    package_share = pathlib.Path(packages.get_package_share_directory(package))
    return package_share / path


def get_isaac_ros_visual_mapping_package_path() -> pathlib.Path:
    return pathlib.Path(packages.get_package_prefix(VISUAL_MAPPING_PACKAGE_NAME))


def get_isaac_ros_visual_mapping_share_path() -> pathlib.Path:
    return (get_isaac_ros_visual_mapping_package_path() / 'share' / VISUAL_MAPPING_PACKAGE_NAME)


def parse_args():
    parser = argparse.ArgumentParser(description='Script to build multiple maps from a rosbag.')
    parser.add_argument(
        '--sensor_data_bag',
        required=True,
        type=pathlib.Path,
        help='Path to the sensor data rosbag file.',
    )
    parser.add_argument(
        '--base_output_folder',
        default='/mnt/nova_ssd/maps',
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
            'cuvslam',
            'map_frames',
            'depth',
            'occupancy',
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
        '--depth_model',
        type=str,
        default='foundationstereo',
        help='Stereo depth estimation model to use (default: foundationstereo)',
        choices=['ess', 'foundationstereo'],
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

    steps_to_run = args.steps_to_run if args.steps_to_run else [
        'edex', 'cuvslam', 'map_frames', 'depth', 'occupancy', 'cuvgl'
    ]

    if 'depth' in steps_to_run:
        check_running_depth_inference(args.depth_model)

    # Setup the output folder.
    start_time = datetime.now()
    timestamp = start_time.strftime('%Y-%m-%d_%H-%M-%S')
    bag_name = args.sensor_data_bag.name
    if not args.map_dir:
        output_folder = filesystem_utils.create_workdir(
            args.base_output_folder,
            f'{timestamp}_{bag_name}',
            allow_sudo=True,
        )
    else:
        output_folder = args.map_dir
    print(f'Storing all maps and logs in {output_folder}.')

    log_folder = output_folder / 'logs'
    log_folder.mkdir(parents=True, exist_ok=True)

    # Create a metadata file.
    metadata_file = output_folder / 'metadata.yaml'
    metadata_file.write_text(f'output_folder: {output_folder}\n')

    duration = get_rosbag_duration(args.sensor_data_bag)
    # print a rough estimated processing time
    estimated_endtime = start_time + timedelta(seconds=duration * 20)
    print(f'Bag Duration: {int(duration)} seconds, ' +
          f'Map Creation Starts: {start_time.strftime("%H:%M:%S")}, ' +
          f'Estimated Completion: {estimated_endtime.strftime("%H:%M:%S")}')

    edex_path = output_folder / 'edex'
    all_frames_meta_file = edex_path / 'frames_meta.json'
    if 'edex' in steps_to_run:
        edex_timeout = math.ceil(duration * 5)
        edex_path.mkdir(parents=True, exist_ok=True)
        generate_edex_from_rosbag(
            args.sensor_data_bag,
            edex_path,
            args.image_extension,
            log_folder,
            args.print_mode,
            edex_timeout,
            args.camera_topic_config,
        )

    output_cuvslam_map_dir = output_folder / 'cuvslam_map'
    output_cuvslam_poses_dir = output_folder / 'poses'
    if 'cuvslam' in steps_to_run:
        # Run cuvslam
        cuvslam_timeout = math.ceil(duration * 7)
        output_cuvslam_map_dir.mkdir(parents=True, exist_ok=True)
        output_cuvslam_poses_dir.mkdir(parents=True, exist_ok=True)
        create_cuvslam_map(edex_path, output_cuvslam_map_dir, output_cuvslam_poses_dir, log_folder,
                           args.print_mode, cuvslam_timeout)

    cuvslam_odom_tum_file = output_cuvslam_poses_dir / 'odom_poses.tum'
    cuvslam_kf_tum_file = output_cuvslam_poses_dir / 'keyframe_pose.tum'
    cuvslam_kf_optimized_tum_file = output_cuvslam_poses_dir / 'keyframe_pose_optimized.tum'
    all_frames_meta_kf_poses_file = edex_path / 'frames_meta_kf_poses.json'
    map_frames_image_dir = output_folder / 'map_frames' / 'raw'
    map_frames_meta_file = map_frames_image_dir / 'frames_meta.json'

    if 'map_frames' in steps_to_run:
        optimize_vo_with_keyframe_pose(cuvslam_odom_tum_file, cuvslam_kf_tum_file,
                                       cuvslam_kf_optimized_tum_file, log_folder, args.print_mode)
        update_frames_meta_pose(cuvslam_kf_optimized_tum_file, all_frames_meta_file,
                                all_frames_meta_kf_poses_file, log_folder, args.print_mode)
        map_frames_image_dir.mkdir(parents=True, exist_ok=True)
        select_map_frames(all_frames_meta_kf_poses_file, map_frames_meta_file, log_folder,
                          args.print_mode)
        copy_raw_image_folder(edex_path, map_frames_meta_file, map_frames_image_dir, log_folder,
                              args.print_mode)

    num_map_frames = count_image_files(map_frames_image_dir)
    map_frames_depth_dir = output_folder / 'map_frames' / 'depth'
    if 'depth' in steps_to_run:
        depth_timeout = math.ceil(num_map_frames * 0.15 + 60)
        run_depth_inference(map_frames_image_dir, map_frames_depth_dir, map_frames_meta_file,
                            log_folder, args.print_mode, depth_timeout, args.depth_model)

    if 'occupancy' in steps_to_run:
        nvblox_timeout = math.ceil(num_map_frames * 0.07)
        create_occupancy_map(output_folder, map_frames_image_dir, map_frames_depth_dir,
                             map_frames_meta_file, log_folder, args.print_mode, nvblox_timeout)

    cuvgl_map_folder = output_folder / 'cuvgl_map'
    if 'cuvgl' in steps_to_run:
        cuvgl_map_folder.mkdir(parents=True, exist_ok=True)
        create_cuvgl_map(cuvgl_map_folder, map_frames_image_dir, args.print_mode,
                         args.prebuilt_bow_vocabulary_folder, args.vgl_model_dir)

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
                              camera_topic_config: pathlib.Path):
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
    ]

    if camera_topic_config:
        command.append(f'--camera_topic_config={camera_topic_config}')

    subprocess_utils.run_command(
        mnemonic='Extract edex',
        command=command,
        log_file=log_folder / 'extract_edex.log',
        print_mode=print_mode,
        timeout=timeout,
    )


def create_cuvslam_map(edex_path: pathlib.Path, output_cuvslam_map_dir: pathlib.Path,
                       output_cuvslam_poses_dir: pathlib.Path, log_folder: pathlib.Path,
                       print_mode: str, timeout: int):
    # Create the cuVSLAM map.
    additional_path = get_path('isaac_ros_visual_slam', '../cuvslam/lib/').resolve()
    ld_library_path = os.environ['LD_LIBRARY_PATH']
    os.environ['LD_LIBRARY_PATH'] = f'{ld_library_path}:{additional_path}'
    subprocess_utils.run_command(
        mnemonic='Create cuVSLAM map',
        command=[
            'ros2',
            'run',
            'isaac_ros_visual_slam',
            'cuvslam_api_launcher',
            f'--dataset={edex_path}',
            f'--output_map={output_cuvslam_map_dir}',
            '--ros_frame_conversion=true',
            '--cfg_enable_slam=true',
            '--cfg_sync_slam',
            '--max_fps=15',
            '--print_format=tum',
            f'--print_odom_poses={output_cuvslam_poses_dir/"odom_poses.tum"}',
            f'--print_slam_poses={output_cuvslam_poses_dir/"slam_poses.tum"}',
            f'--print_map_keyframes={output_cuvslam_poses_dir}/keyframe_pose.tum',
        ],
        log_file=log_folder / 'create_cuvslam_map.log',
        print_mode=print_mode,
        timeout=timeout,
    )


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
            '--min_inter_frame_distance=0.2',
            '--min_inter_frame_rotation_degrees=5',
        ],
        log_file=log_folder / 'select_map_frames.log',
        print_mode=print_mode,
    )


def optimize_vo_with_keyframe_pose(vo_pose_file: pathlib.Path, kf_pose_file: pathlib.Path,
                                   output_pose_file: pathlib.Path, log_folder: pathlib.Path,
                                   print_mode: str):
    config_file = (get_isaac_ros_visual_mapping_share_path() /
                   'configs/isaac/vo_pose_optimize_ba_config.pb.txt')

    subprocess_utils.run_command(
        mnemonic='Optimize odometry pose with keyframe pose',
        command=[
            'ros2',
            'run',
            'isaac_ros_visual_mapping',
            'optimize_vo_with_keyframe_pose_main',
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
    subprocess_utils.run_command(
        mnemonic='Update frame metadata pose',
        command=[
            'ros2',
            'run',
            'isaac_ros_visual_mapping',
            'update_keyframe_pose_main',
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
    command = [
        'ros2',
        'run',
        'isaac_ros_visual_mapping',
        'copy_image_dir_main',
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
                         log_folder: pathlib.Path, print_mode: str, timeout: int):
    WORKSPACE_BOUNDS_TYPE_HEIGHT_BOUNDS = 1
    occupancy_map_path = f'{output_dir}/occupancy_map'
    command = [
        'ros2',
        'run',
        'isaac_mapping_ros',
        'run_nvblox',
        f'--save_2d_occupancy_map_path={occupancy_map_path}',
        f'--color_image_dir={color_img_dir}',
        f'--frames_meta_file={frames_meta_file}',
        f'--depth_image_dir={depth_img_dir}',
        '--mapping_type_dynamic',
        '--projective_integrator_max_integration_distance_m=2.5',
        '--esdf_slice_min_height=0.09',
        '--esdf_slice_max_height=0.65',
        '--esdf_slice_height=0.3',
        f'--workspace_bounds_type={WORKSPACE_BOUNDS_TYPE_HEIGHT_BOUNDS}',
        '--workspace_bounds_min_height_m=-0.3',
        '--workspace_bounds_max_height_m=2.0',
    ]

    subprocess_utils.run_command(
        mnemonic='Run Nvblox',
        command=command,
        log_file=log_folder / 'run_nvblox.log',
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
                     prebuilt_bow_vocabulary_folder: pathlib.Path = None,
                     vgl_model_dir: pathlib.Path = None):
    binary_folder = get_isaac_ros_visual_mapping_package_path() / 'bin/visual_mapping'
    config_folder = get_isaac_ros_visual_mapping_share_path() / 'configs/isaac'
    model_dir = (vgl_model_dir if vgl_model_dir else
                 get_isaac_ros_visual_mapping_share_path() / 'models')
    # Create global localization map.
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


def check_running_depth_inference(depth_model: str = 'foundationstereo'):
    result = subprocess.run(["ros2", "node", "list"], capture_output=True, text=True, check=True)
    active_nodes = result.stdout.splitlines()

    if depth_model == 'foundationstereo':
        # Check for Foundation Stereo nodes
        stereo_processes = ['/foundationstereo_container', '/foundationstereo_decoder']
        model_name = 'Foundation Stereo'
    else:
        # Check for ESS nodes (default)
        stereo_processes = ['/disparity', '/disparity/disparity_container']
        model_name = 'ESS'

    for name in stereo_processes:
        if name in active_nodes:
            raise RuntimeError(
                f'{model_name} node {name} is already running. Please wait for it to '
                f'finish, manually stop the node, or restart the Docker container '
                f'before running this command again.')


if __name__ == '__main__':
    main()
