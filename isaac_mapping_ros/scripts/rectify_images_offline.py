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
"""
Offline image rectification script for Isaac ROS mapping pipeline.

This script rectifies raw images using camera calibration information from
the KeyframesMetadataCollection proto structure. It follows the same pattern
as the native C++ implementation in lib/src/visual_mapping/src/common/image/image_rectifier.cc
"""

import argparse
import json
import pathlib
from typing import Optional
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


def apply_rectification_transform_to_sensor_transform(sensor_to_vehicle_transform,
                                                      rectification_matrix):
    """
    Apply the rectification matrix rotation to the sensor to vehicle transform.

    This follows the same logic as the C++ implementation in data_converter_utils.cpp line 302-311:
    - Map the rectification matrix R into a rotation matrix
    - Invert it to get rectified_camera_to_raw transform
    - Apply this transform to the existing sensor_to_vehicle_transform

    Args:
        sensor_to_vehicle_transform (dict): The sensor to vehicle transform from frames metadata
        rectification_matrix (np.ndarray): 3x3 rectification matrix

    Returns:
        dict: Updated sensor to vehicle transform
    """
    # Extract original transform components
    translation = sensor_to_vehicle_transform.get('translation', {'x': 0.0, 'y': 0.0, 'z': 0.0})
    translation_vector = np.array(
        [translation.get('x', 0.0),
         translation.get('y', 0.0),
         translation.get('z', 0.0)])

    # Get original rotation
    if 'axis_angle' in sensor_to_vehicle_transform:
        axis_angle = sensor_to_vehicle_transform.get('axis_angle', {'x': 0.0, 'y': 0.0, 'z': 1.0})
        axis = np.array(
            [axis_angle.get('x', 0.0),
             axis_angle.get('y', 0.0),
             axis_angle.get('z', 0.0)])
        angle_rad = np.deg2rad(axis_angle.get('angle_degrees', 0.0))
        original_rotation = R.from_rotvec(axis * angle_rad)
    else:
        original_rotation = R.from_matrix(np.eye(3))

    # Apply rectification adjustment following C++ logic:
    # rectified_camera_to_raw = SE3Transform(zero_translation, R.inverse())
    # adjusted_transform = sensor_to_vehicle_transform * rectified_camera_to_raw
    # Transpose = inverse for rotation matrix
    rectification_rotation_inv = R.from_matrix(rectification_matrix.T)

    # Compose the transformations: original_rotation * rectification_rotation_inv
    adjusted_rotation = original_rotation * rectification_rotation_inv

    # Convert back to axis-angle representation
    rotvec = adjusted_rotation.as_rotvec()
    angle_rad = np.linalg.norm(rotvec)
    if angle_rad > 1e-6:
        axis = rotvec / angle_rad
        angle_deg = np.rad2deg(angle_rad)
    else:
        axis = np.array([0.0, 0.0, 1.0])  # Default axis when angle is near zero
        angle_deg = 0.0

    # Create updated transform (translation remains the same)
    updated_transform = {
        'translation': {
            'x': translation_vector[0],
            'y': translation_vector[1],
            'z': translation_vector[2]
        },
        'axis_angle': {
            'x': axis[0],
            'y': axis[1],
            'z': axis[2],
            'angle_degrees': angle_deg
        }
    }

    return updated_transform


def rectify_images_offline(raw_image_dir: pathlib.Path,
                           rectified_image_dir: pathlib.Path,
                           frames_meta_file: Optional[pathlib.Path] = None,
                           log_folder: Optional[pathlib.Path] = None,
                           output_frames_meta_file: Optional[pathlib.Path] = None,
                           dry_run: bool = False):
    """Rectify raw images using camera calibration information.

    Args:
        raw_image_dir: Directory containing raw images organized by camera name
        rectified_image_dir: Output directory for rectified images
        frames_meta_file: Path to frames_meta.json file containing camera calibration
        log_folder: Optional log folder (for compatibility with main script)
        output_frames_meta_file: Optional custom output path for frames_meta.json
        dry_run: If True, only simulate operations without actually processing images
    """

    rectified_image_dir.mkdir(parents=True, exist_ok=True)

    if frames_meta_file is None:
        frames_meta_file = raw_image_dir / 'frames_meta.json'

    # Read frames metadata to get camera calibration
    with open(frames_meta_file, 'r') as f:
        frames_meta = json.load(f)

    if dry_run:
        print(f"[DRY RUN] Would rectify images from {raw_image_dir} to {rectified_image_dir}")
    else:
        print(f"Rectifying images from {raw_image_dir} to {rectified_image_dir}")

    # Get camera parameters lookup
    camera_params_lookup = frames_meta.get('camera_params_id_to_camera_params', {})
    if not camera_params_lookup:
        print("Error: No camera parameters found in frames metadata")
        return

    # Process each camera directory
    for camera_dir in raw_image_dir.iterdir():
        if not camera_dir.is_dir():
            continue

        camera_name = camera_dir.name
        rectified_camera_dir = rectified_image_dir / camera_name

        if dry_run:
            print(f"[DRY RUN] Would create camera directory: {rectified_camera_dir}")
            print(f"[DRY RUN] Would process camera: {camera_name}")
        else:
            rectified_camera_dir.mkdir(parents=True, exist_ok=True)
            print(f"Processing camera: {camera_name}")

        # Find camera parameters for this camera
        camera_params = None
        current_params_id = None
        current_params_data = None
        for params_id, params_data in camera_params_lookup.items():
            sensor_meta = params_data.get('sensor_meta_data', {})
            if sensor_meta.get('sensor_name') == camera_name:
                camera_params = params_data.get('calibration_parameters', {})
                current_params_id = params_id
                current_params_data = params_data
                break

        if not camera_params:
            print(f"Warning: No calibration data found for camera {camera_name}, skipping")
            continue

        # Extract all calibration parameters
        camera_matrix_data = camera_params.get('camera_matrix', {}).get('data', [])
        distortion_data = camera_params.get('distortion_coefficients', {}).get('data', [])
        rectification_data = camera_params.get('rectification_matrix', {}).get('data', [])
        projection_data = camera_params.get('projection_matrix', {}).get('data', [])

        if (not camera_matrix_data or not distortion_data or not rectification_data
                or not projection_data):
            raise ValueError(f"Missing calibration data for camera {camera_name}")

        # distortion can be size of 5 or 8
        if len(camera_matrix_data) != 9 or len(distortion_data) not in [
                5, 8
        ] or len(rectification_data) != 9 or len(projection_data) != 12:
            raise ValueError(f"Invalid calibration data for camera {camera_name}")

        # Convert to numpy arrays
        camera_matrix = np.array(camera_matrix_data).reshape(3, 3)
        dist_coeffs = np.array(distortion_data)
        rectification_matrix = np.array(rectification_data).reshape(3, 3)
        projection_matrix = np.array(projection_data).reshape(3, 4)

        # Get image size from calibration
        image_width = camera_params.get('image_width', 1280)
        image_height = camera_params.get('image_height', 720)

        # Get image files
        image_files = list(camera_dir.glob('*.jpg')) + list(camera_dir.glob('*.png'))
        if not image_files:
            print(f"Warning: No image files found for camera {camera_name}, skipping")
            continue

        # Verify image size with first image
        first_image = cv2.imread(str(image_files[0]))
        if first_image is not None:
            h, w = first_image.shape[:2]
            if w != image_width or h != image_height:
                print(f"Warning: Image size mismatch for {camera_name}. "
                      f"Expected {image_width}x{image_height}, got {w}x{h}")
                image_width, image_height = w, h

        # Create undistortion maps using all calibration parameters
        # This follows the same pattern as
        # lib/src/visual_mapping/src/common/image/image_rectifier.cc
        if not dry_run:
            map1, map2 = cv2.initUndistortRectifyMap(
                camera_matrix,
                dist_coeffs,
                rectification_matrix,
                projection_matrix[:3, :3],  # Use only the 3x3 part of projection matrix
                (image_width, image_height),
                cv2.CV_32F)
        else:
            print(
                f"[DRY RUN] Would create undistortion maps for {image_width}x{image_height} images"
            )

        # Process all images for this camera
        rectified_count = 0
        if dry_run:
            print(f"[DRY RUN] Would process {len(image_files)} images for camera {camera_name}")
            rectified_count = len(image_files)
        else:
            for image_file in image_files:
                # Read raw image
                img = cv2.imread(str(image_file))
                if img is None:
                    print(f"Warning: Could not read image {image_file}")
                    continue

                # Rectify image
                rectified_img = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)

                # Save rectified image
                output_path = rectified_camera_dir / image_file.name
                cv2.imwrite(str(output_path), rectified_img)
                rectified_count += 1

        print(f"Completed rectification for camera {camera_name}: {rectified_count} images")

        # Update sensor to vehicle transform based on rectification matrix if needed
        # This follows the same logic as data_converter_utils.cpp lines 302-311
        if np.allclose(rectification_matrix, np.eye(3)):
            print(
                f"Warning: Rectification matrix is identity for {camera_name}, "
                f"no adjustment needed"
            )

        if 'sensor_meta_data' not in current_params_data:
            raise ValueError(f"sensor_meta_data not found for {camera_name}")

        sensor_meta = current_params_data['sensor_meta_data']
        if 'sensor_to_vehicle_transform' not in sensor_meta:
            raise ValueError(f"Sensor to vehicle transform not found for {camera_name}")

        print(f"Adjusting sensor to vehicle transform for {camera_name} "
              f"based on rectification matrix")
        original_transform = sensor_meta['sensor_to_vehicle_transform']
        updated_transform = apply_rectification_transform_to_sensor_transform(
            original_transform, rectification_matrix)

        # Update the transform in the frames metadata
        frames_meta['camera_params_id_to_camera_params'][current_params_id]['sensor_meta_data'][
            'sensor_to_vehicle_transform'] = updated_transform
        # Set the camera projection model type to PINHOLE (rectified images are undistorted)
        frames_meta['camera_params_id_to_camera_params'][current_params_id][
            'camera_projection_model_type'] = 'PINHOLE'

    # Copy frames metadata to rectified directory
    if output_frames_meta_file is None:
        rectified_frames_meta = rectified_image_dir / 'frames_meta.json'
    else:
        rectified_frames_meta = output_frames_meta_file

    with open(rectified_frames_meta, 'w') as f:
        json.dump(frames_meta, f, indent=2)

    print(f"Image rectification completed. Rectified images saved to {rectified_image_dir}")
    print(f"Frames metadata copied to {rectified_frames_meta}")


def main():
    """Main function for standalone usage."""
    parser = argparse.ArgumentParser(description='Rectify raw images using camera calibration')
    parser.add_argument('--raw_image_dir',
                        required=True,
                        type=pathlib.Path,
                        help='Directory containing raw images organized by camera name')
    parser.add_argument('--rectified_image_dir',
                        required=True,
                        type=pathlib.Path,
                        help='Output directory for rectified images')
    parser.add_argument('--frames_meta_file',
                        required=False,
                        type=pathlib.Path,
                        help='Path to frames_meta.json file containing camera calibration')
    parser.add_argument('--log_folder',
                        type=pathlib.Path,
                        help='Optional log folder (for compatibility with main script)')
    parser.add_argument('--dry_run',
                        action='store_true',
                        help='Simulate operations without actually processing images')

    args = parser.parse_args()

    rectify_images_offline(args.raw_image_dir,
                           args.rectified_image_dir,
                           args.frames_meta_file,
                           args.log_folder,
                           dry_run=args.dry_run)


if __name__ == '__main__':
    main()
