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

# This script includes multiple tasks:
# - rosbag data conversion
# - generating cuvslam and cusfm pose
# - depth inference (stereo anything, todo: ESS)
# - running Nvblox
#
# It requires manual intervension to switch docker environment for depth inference.
# To run a specific step, you can comment out other steps at the bottom of the script.
#
# Example:
# python src/isaac_ros_mapping_and_localization/isaac_mapping_ros/scripts/create_occupancy_e2e.py
#   --sensor_bag=96_HLoop_carter3_4hi3dl_2024_05_30-14_10_43 --output_dir=data/cusfm/
#
import os
import argparse
import logging
import pathlib
import shutil

from isaac_common_py import subprocess_utils
import ament_index_python.packages


parser = argparse.ArgumentParser(description='Create 2d occupancy offline')
parser.add_argument(
    '--sensor_bag',
    required=False,
    type=pathlib.Path,
    help='Path to the sensor data bag'
)
parser.add_argument(
    '--output_dir',
    required=False,
    type=pathlib.Path,
    help='Path to the output dir'
)
parser.add_argument(
    '--print_mode',
    type=str,
    default='tail',
    choices=['none', 'tail', 'all'],
    help='Determines what is printed to stdout',
)
parser.add_argument(
    '--num_frames',
    type=int,
    default=-1,
    help='Number of frames to process. Negative means process all',
)
parser.add_argument(
    '--integration_distance',
    type=float,
    default=5.0,
    help='The maximum distance[m] to integrate the depth or color image values',
)
parser.add_argument(
    '--save_mesh',
    action=argparse.BooleanOptionalAction,
    default=False,
    help='Whether save the mesh cloud',
)
parser.add_argument(
    '--esdf_slice_height',
    type=float,
    default=0.3,
    help='The output slice height for the distance slice, ESDF pointcloud and occupancy map'
)
parser.add_argument(
    '--use_2d_esdf_mode',
    action=argparse.BooleanOptionalAction,
    default=False,
    help='Use the 2d ESDF mode (3D if false).',
)

args = parser.parse_args()

e2e_dir = args.output_dir
cusfm_dir = args.output_dir / 'cusfm/sift'
omap_dir = cusfm_dir / 'omap'

vslam_slam_frames_meta_file = f'{cusfm_dir}/raw/frames_meta.json'
vslam_odom_frames_meta_file = f'{cusfm_dir}/raw/frames_meta_vslam_odom.json'
vslam_kf_frames_meta_file = f'{cusfm_dir}/raw/frames_meta_vslam_kf.json'
cusfm_frames_meta_file = f'{cusfm_dir}/kpmap/keyframes/frames_meta.json'

vslam_odom_tum_file = f'{e2e_dir}/cuvslam/poses/odom_poses.tum'
vslam_slam_tum_file = f'{e2e_dir}/cuvslam/poses/slam_poses.tum'
vslam_kf_tum_file = f'{e2e_dir}/cuvslam/poses/keyframe_pose.tum'

home_dir = os.environ['HOME']
repo_base_dir = f'{home_dir}/workspaces'


def get_path(package: str, path: str) -> pathlib.Path:
    package_share = pathlib.Path(ament_index_python.packages.get_package_share_directory(package))
    return package_share / path


def run_cusfm():
    command = [
        'python',
        'src/isaac_ros_mapping_and_localization/isaac_mapping_ros/scripts/run_cusfm_e2e.py',
        f'--sensor_data_bag={args.sensor_bag}',
        '--image_extension=.png',
        f'--output_dir=${e2e_dir}',
        f'--print_mode=${args.print_mode}'
    ]
    subprocess_utils.run_command(
        mnemonic='run_cusfm',
        command=command,
    )


def copy_raw_image_folder():
    command = [
        './build/isaac_mapping_ros/isaac_mapping/tools/copy_image_dir_main',
        f'--input_image_dir={e2e_dir}/rosbag_mapping_data',
        f'--output_image_dir={cusfm_dir}/raw',
        f'--frames_meta_file={cusfm_dir}/keyframes/frames_meta.json',
    ]
    subprocess_utils.run_command(
        mnemonic='copy_raw_image_folder',
        command=command,
    )
    shutil.copy2(f'{cusfm_dir}/keyframes/frames_meta.json', f'{cusfm_dir}/raw/frames_meta.json')


def run_stereo_anything():
    script_path = get_path('isaac_ros_mapping', 'scripts/run_stereo_anything.sh').resolve()
    command = [
        'docker run --net=host --env=DISPLAY',
        f'-v {home_dir}/.Xauthority:/root/.Xauthority:rw',
        '-v /tmp/.X11-unix:/tmp/.X11-unix:rw',
        f'-v {repo_base_dir}:/workspaces',
        f'-v {cusfm_dir}:/workspace/data',
        'nvcr.io/nvidian/stereo_2024',
        'bash',
        script_path,
        '/workspaces/data'
    ]
    subprocess_utils.run_command(
        mnemonic='run_stereo_anything',
        command=command,
    )


def update_keyframe_pose(input_tum_file, input_frames_meta, output_frames_meta):
    command = [
        './build/isaac_mapping_ros/isaac_mapping/tools/update_keyframe_pose_main',
        f'--tum_pose_file={input_tum_file}',
        f'--input_frames_meta={input_frames_meta}',
        f'--output_frames_meta={output_frames_meta}'
    ]
    subprocess_utils.run_command(
        mnemonic='update_keyframe_pose',
        command=command,
    )


def run_nvblox(output_dir, img_dir, depth_dir, frames_meta_file):
    output_dir.mkdir(parents=True, exist_ok=True)

    WORKSPACE_BOUNDS_TYPE_HEIGHT_BOUNDS = 1

    command = [
        'ros2',
        'run',
        'isaac_mapping_ros',
        'run_nvblox',
        f'--save_2d_occupancy_map_path={output_dir}/occupancy_map',
        f'--color_image_dir={img_dir}',
        f'--frames_meta_file={frames_meta_file}',
        f'--depth_image_dir={depth_dir}',
        '--mapping_type_dynamic',
        f'--projective_integrator_max_integration_distance_m={args.integration_distance}',
        '--esdf_slice_min_height=0.09',
        '--esdf_slice_max_height=0.65',
        f'--esdf_slice_height={args.esdf_slice_height}',
        f'--workspace_bounds_type={WORKSPACE_BOUNDS_TYPE_HEIGHT_BOUNDS}',
        '--workspace_bounds_min_height_m=-0.3',
        '--workspace_bounds_max_height_m=2.0',
        '--use_2d_esdf_mode' if args.use_2d_esdf_mode else '',
        f'--num_frames={args.num_frames}' if args.num_frames > 0 else '',
        f'--mesh_output_path={output_dir}/mesh.ply' if args.save_mesh else ''
    ]

    subprocess_utils.run_command(
        mnemonic='Run Nvblox',
        command=command,
        log_file=output_dir / 'run_nvblox.log',
        print_mode=args.print_mode,
    )


def convert_tum_to_frames_meta():
    # update frames_meta using vslam_odom pose
    update_keyframe_pose(input_tum_file=vslam_odom_tum_file,
                         input_frames_meta=vslam_slam_frames_meta_file,
                         output_frames_meta=vslam_odom_frames_meta_file)

    # update frames_meta using vslam_kf pose
    update_keyframe_pose(input_tum_file=vslam_kf_tum_file,
                         input_frames_meta=vslam_slam_frames_meta_file,
                         output_frames_meta=vslam_kf_frames_meta_file)


def fuse_with_stereo_anything_depth():
    img_dir = cusfm_dir / 'raw'
    depth_dir = cusfm_dir / 'depth/original_size'
    run_nvblox(output_dir=omap_dir / 'vslam_odom_pose',
               img_dir=img_dir,
               depth_dir=depth_dir,
               frames_meta_file=vslam_odom_frames_meta_file)

    run_nvblox(output_dir=omap_dir / 'vslam_slam_pose',
               img_dir=img_dir,
               depth_dir=depth_dir,
               frames_meta_file=vslam_slam_frames_meta_file)

    run_nvblox(output_dir=omap_dir / 'vslam_kf_pose',
               img_dir=img_dir,
               depth_dir=depth_dir,
               frames_meta_file=vslam_kf_frames_meta_file)

    run_nvblox(output_dir=omap_dir / 'cusfm_pose',
               img_dir=img_dir,
               depth_dir=depth_dir,
               frames_meta_file=cusfm_frames_meta_file)


def fuse_with_ess_depth():
    img_dir = cusfm_dir / 'raw'
    depth_dir = cusfm_dir / 'ess_depth'
    run_nvblox(output_dir=omap_dir / 'ess_vslam_odom_pose',
               img_dir=img_dir,
               depth_dir=depth_dir,
               frames_meta_file=vslam_odom_frames_meta_file)

    run_nvblox(output_dir=omap_dir / 'ess_vslam_slam_pose',
               img_dir=img_dir,
               depth_dir=depth_dir,
               frames_meta_file=vslam_slam_frames_meta_file)

    run_nvblox(output_dir=omap_dir / 'ess_vslam_kf_pose',
               img_dir=img_dir,
               depth_dir=depth_dir,
               frames_meta_file=vslam_kf_frames_meta_file)

    run_nvblox(output_dir=omap_dir / 'ess_vslam_kf_pose',
               img_dir=img_dir,
               depth_dir=depth_dir,
               frames_meta_file=cusfm_frames_meta_file)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # run_cusfm()

    # copy_raw_image_folder()

    # run_stereo_anything()

    # TODO: add step for run_ess_depth()

    # fuse_with_stereo_anything_depth()

    fuse_with_ess_depth()
