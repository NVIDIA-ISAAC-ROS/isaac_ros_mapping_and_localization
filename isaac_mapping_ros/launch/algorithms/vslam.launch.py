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


def remap(i: int, name: str, identifier: str, rectified: bool) -> list[tuple[str, str]]:
    if rectified:
        return [
            (f'visual_slam/image_{i}', f'/{name}/{identifier}/image_rect'),
            (f'visual_slam/camera_info_{i}', f'/{name}/{identifier}/camera_info_rect'),
        ]
    else:
        return [
            (f'visual_slam/image_{i}', f'/{name}/{identifier}/image_raw'),
            (f'visual_slam/camera_info_{i}', f'/{name}/{identifier}/camera_info'),
        ]


def add_vslam(args: lu.ArgumentContainer) -> list[lut.Action]:
    camera_names = args.vslam_enabled_stereo_cameras.split(',')

    actions = []

    remappings = [
        ('visual_slam/imu', '/front_stereo_imu/imu'),
        ('visual_slam/initial_pose', '/visual_localization/pose'),
        ('visual_slam/trigger_hint', '/visual_localization/trigger_localization'),
    ]
    for i, camera_name in enumerate(camera_names):
        remappings.extend(remap(2 * i, camera_name, 'left', args.vslam_use_rectified_images))
        remappings.extend(remap(2 * i + 1, camera_name, 'right', args.vslam_use_rectified_images))

    camera_optical_frames = []
    if args.vslam_camera_optical_frames:
        camera_optical_frames = args.vslam_camera_optical_frames.split(',')

    # Isaac Sim publishes images at 20hz
    # Set image jitter threshold ms to keep the vslam node from spamming warnings
    image_jitter_threshold_ms = 51.0 if args.is_sim else 34.0
    visual_slam_node = lut.ComposableNode(
        name='visual_slam_node',
        package='isaac_ros_visual_slam',
        plugin='nvidia::isaac_ros::visual_slam::VisualSlamNode',
        parameters=[{
            # Images:
            'num_cameras': 2 * len(camera_names),
            'min_num_images': 2 * len(camera_names),
            'enable_image_denoising': False,
            'enable_localization_n_mapping':
                lut.ParameterValue(args.vslam_enable_slam, value_type=bool),
            'slam_throttling_time_ms': 10000,
            'enable_ground_constraint_in_odometry': lut.ParameterValue(
                args.vslam_enable_ground_constraint_in_odometry, value_type=bool),
            'enable_ground_constraint_in_slam': lut.ParameterValue(
                args.vslam_enable_ground_constraint_in_slam, value_type=bool),
            'enable_imu_fusion': lut.ParameterValue(args.vslam_enable_imu, value_type=bool),
            'gyro_noise_density': 0.000244,
            'gyro_random_walk': 0.000019393,
            'accel_noise_density': 0.001862,
            'accel_random_walk': 0.003,
            'calibration_frequency': 200.0,
            'image_qos': args.vslam_image_qos,
            'save_map_folder_path': args.vslam_save_map_folder_path,
            'load_map_folder_path': args.vslam_load_map_folder_path,
            'localize_on_startup':
                lut.ParameterValue(args.vslam_localize_on_startup, value_type=bool),
            'image_jitter_threshold_ms': image_jitter_threshold_ms,
            'enable_request_hint': args.vslam_enable_request_hint,
            # It's recommended to use image masking with unrectified images.
            'rectified_images': args.vslam_use_rectified_images,
            'img_mask_bottom': args.vslam_img_mask_bottom,
            'img_mask_left': args.vslam_img_mask_left,
            'img_mask_right': args.vslam_img_mask_right,
            # Frames:
            'map_frame': args.vslam_map_frame,
            'odom_frame': args.vslam_odom_frame,
            'base_frame': args.vslam_base_frame,
            'imu_frame': 'front_stereo_camera_imu',
            'publish_odom_to_base_tf': args.publish_odom_to_base_tf,
            'publish_map_to_odom_tf': args.vslam_publish_map_to_odom_tf,
            'invert_odom_to_base_tf': args.invert_odom_to_base_tf,
            # Visualization:
            'path_max_size': 1024,
            'enable_slam_visualization': lut.ParameterValue(
                args.vslam_enable_visualization, value_type=bool),
            'enable_observations_view': lut.ParameterValue(
                args.vslam_enable_visualization, value_type=bool),
            'enable_landmarks_view': lut.ParameterValue(
                args.vslam_enable_visualization, value_type=bool),
            'verbosity': 10,
            'camera_optical_frames': camera_optical_frames,
            }],
        remappings=remappings,
    )

    actions.append(lu.load_composable_nodes(args.container_name, [visual_slam_node]))
    actions.append(
        lu.log_info(["Enabling standard VSLAM for cameras '",
                    args.vslam_enabled_stereo_cameras, "'"]))

    return actions


def generate_launch_description() -> lut.LaunchDescription:
    args = lu.ArgumentContainer()
    args.add_arg('container_name', 'nova_container')
    args.add_arg('vslam_image_qos', 'SENSOR_DATA')
    args.add_arg('invert_odom_to_base_tf', False)
    args.add_arg('is_sim', False)
    args.add_arg('publish_odom_to_base_tf', True)
    args.add_arg('vslam_enable_imu', False)
    args.add_arg('vslam_publish_map_to_odom_tf', True)
    args.add_arg('vslam_enable_slam', False)
    args.add_arg('vslam_enabled_stereo_cameras')
    args.add_arg('vslam_load_map_folder_path', '')
    args.add_arg('vslam_save_map_folder_path', '')
    args.add_arg('vslam_localize_on_startup', False)
    args.add_arg('vslam_map_frame', 'map')
    args.add_arg('vslam_odom_frame', 'odom')
    args.add_arg('vslam_enable_request_hint', True)
    args.add_arg('vslam_use_rectified_images', False)
    args.add_arg('vslam_enable_ground_constraint_in_odometry', False)
    args.add_arg('vslam_enable_ground_constraint_in_slam', False)
    args.add_arg('vslam_img_mask_bottom', 0)
    args.add_arg('vslam_img_mask_left', 0)
    args.add_arg('vslam_img_mask_right', 0)
    args.add_arg('vslam_enable_visualization', False)
    args.add_arg('vslam_camera_optical_frames', '')
    args.add_arg('vslam_base_frame', 'base_link')

    args.add_opaque_function(add_vslam)

    return lut.LaunchDescription(args.get_launch_actions())
