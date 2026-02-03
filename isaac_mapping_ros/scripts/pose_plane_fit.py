#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

"""
Script to fit a plane to poses from a TUM pose file and transform them
such that all poses sit around a plane at the z-value of the first pose.

TUM format: timestamp tx ty tz qx qy qz qw
Where:
- timestamp: timestamp in seconds
- tx, ty, tz: translation in meters
- qx, qy, qz, qw: quaternion (x, y, z, w)
"""

import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys
import os


def read_tum_file(file_path):
    """
    Read poses from a TUM format file.

    Args:
        file_path (str): Path to the TUM format file

    Returns:
        dict: Dictionary containing timestamps, positions, and quaternions
    """
    try:
        # Skip comment lines starting with '#'
        data = np.loadtxt(file_path, comments='#')
    except Exception as e:
        raise ValueError(f"Failed to read TUM file {file_path}: {e}")

    if data.shape[1] != 8:
        raise ValueError(f"TUM file should have 8 columns, got {data.shape[1]}")

    poses = {
        'timestamps': data[:, 0],
        'positions': data[:, 1:4],  # tx, ty, tz
        'quaternions': data[:, 4:8]  # qx, qy, qz, qw
    }

    return poses


def fit_plane_to_points(points):
    """
    Fit a plane to 3D points using SVD.

    Args:
        points (np.ndarray): Nx3 array of 3D points

    Returns:
        tuple: (plane_normal, plane_point, plane_equation)
            - plane_normal: 3D unit normal vector of the fitted plane
            - plane_point: A point on the plane (centroid)
            - plane_equation: [a, b, c, d] where ax + by + cz + d = 0
    """
    # Center the points
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid

    # Perform SVD
    U, S, Vt = np.linalg.svd(centered_points)

    # The normal vector is the last column of V (last row of Vt)
    normal = Vt[-1, :]

    # Ensure normal points "up" (positive z component if possible)
    if normal[2] < 0:
        normal = -normal

    # Plane equation: ax + by + cz + d = 0
    # d = -normal · centroid
    d = -np.dot(normal, centroid)
    plane_equation = np.append(normal, d)

    return normal, centroid, plane_equation


def compute_transform_to_target_z_plane(plane_normal, plane_point, target_z):
    """
    Compute transformation matrix to align the fitted plane with target z plane.

    Args:
        plane_normal (np.ndarray): Normal vector of the fitted plane
        plane_point (np.ndarray): A point on the fitted plane
        target_z (float): Target z-value for the plane

    Returns:
        np.ndarray: 4x4 transformation matrix
    """
    # Target normal is z-axis
    target_normal = np.array([0, 0, 1])

    # Compute rotation to align plane normal with z-axis
    # Handle the case where vectors are already aligned or opposite
    dot_product = np.dot(plane_normal, target_normal)

    if np.abs(dot_product - 1.0) < 1e-6:
        # Already aligned
        rotation_matrix = np.eye(3)
    elif np.abs(dot_product + 1.0) < 1e-6:
        # Opposite direction, rotate 180 degrees around perpendicular axis
        # Choose x-axis if plane normal is not parallel to it
        if np.abs(plane_normal[0]) < 0.9:
            axis = np.array([1, 0, 0])
        else:
            axis = np.array([0, 1, 0])
        rotation_matrix = R.from_rotvec(np.pi * axis).as_matrix()
    else:
        # General case: compute rotation axis and angle
        rotation_axis = np.cross(plane_normal, target_normal)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        rotation_angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
        rotation_matrix = R.from_rotvec(
            rotation_angle * rotation_axis).as_matrix()

    # Compute translation to move plane point to target z
    # After rotation, find where the plane point ends up
    rotated_plane_point = rotation_matrix @ plane_point
    translation = np.array([0, 0, target_z - rotated_plane_point[2]])

    # Construct 4x4 transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = translation

    return transform


def apply_transform_to_poses(poses, transform):
    """
    Apply transformation to all poses.

    Args:
        poses (dict): Dictionary containing positions and quaternions
        transform (np.ndarray): 4x4 transformation matrix

    Returns:
        dict: Transformed poses
    """
    positions = poses['positions']
    quaternions = poses['quaternions']

    # Transform positions
    homogeneous_positions = np.hstack(
        [positions, np.ones((positions.shape[0], 1))])
    transformed_positions = (transform @ homogeneous_positions.T).T[:, :3]

    # Transform orientations
    rotation_matrix = transform[:3, :3]
    rotations = R.from_quat(quaternions)  # scipy uses (x,y,z,w) format

    # Apply rotation transformation
    transformed_rotations = R.from_matrix(rotation_matrix) * rotations
    transformed_quaternions = transformed_rotations.as_quat()

    transformed_poses = {
        'timestamps': poses['timestamps'],
        'positions': transformed_positions,
        'quaternions': transformed_quaternions
    }

    return transformed_poses


def write_tum_file(poses, file_path):
    """
    Write poses to a TUM format file.

    Args:
        poses (dict): Dictionary with timestamps, positions, and quaternions
        file_path (str): Output file path
    """
    with open(file_path, 'w') as f:
        f.write("# TUM format: timestamp tx ty tz qx qy qz qw\n")
        f.write("# Poses transformed to target z plane\n")

        for i in range(len(poses['timestamps'])):
            timestamp = poses['timestamps'][i]
            pos = poses['positions'][i]
            quat = poses['quaternions'][i]

            f.write(f"{timestamp:.9f} {pos[0]:.9f} {pos[1]:.9f} "
                    f"{pos[2]:.9f} {quat[0]:.9f} {quat[1]:.9f} "
                    f"{quat[2]:.9f} {quat[3]:.9f}\n")


def print_transform_info(transform, plane_normal, plane_point,
                         plane_equation, target_z):
    """
    Print information about the computed transformation.

    Args:
        transform (np.ndarray): 4x4 transformation matrix
        plane_normal (np.ndarray): Original plane normal
        plane_point (np.ndarray): Point on original plane
        plane_equation (np.ndarray): Plane equation coefficients [a,b,c,d]
        target_z (float): Target z-value for the plane
    """
    print("=" * 60)
    print("PLANE FITTING AND TRANSFORMATION RESULTS")
    print("=" * 60)

    print("\nOriginal fitted plane:")
    print(f"  Normal vector: [{plane_normal[0]:.6f}, "
          f"{plane_normal[1]:.6f}, {plane_normal[2]:.6f}]")
    print(f"  Point on plane: [{plane_point[0]:.6f}, "
          f"{plane_point[1]:.6f}, {plane_point[2]:.6f}]")
    print(f"  Plane equation: {plane_equation[0]:.6f}x + "
          f"{plane_equation[1]:.6f}y + {plane_equation[2]:.6f}z + "
          f"{plane_equation[3]:.6f} = 0")

    print(f"\nTarget z-plane: z = {target_z:.6f}")

    print("\nTransformation matrix (4x4):")
    for i in range(4):
        print(f"  [{transform[i, 0]:10.6f} {transform[i, 1]:10.6f} "
              f"{transform[i, 2]:10.6f} {transform[i, 3]:10.6f}]")

    # Extract rotation and translation components
    rotation_matrix = transform[:3, :3]
    translation = transform[:3, 3]
    rotation = R.from_matrix(rotation_matrix)
    euler_angles = rotation.as_euler('xyz', degrees=True)

    print("\nTransformation components:")
    print(f"  Translation: [{translation[0]:.6f}, {translation[1]:.6f}, "
          f"{translation[2]:.6f}]")
    print(f"  Rotation (Euler XYZ, degrees): [{euler_angles[0]:.3f}, "
          f"{euler_angles[1]:.3f}, {euler_angles[2]:.3f}]")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Fit a plane to TUM poses and transform them to first pose z-plane",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pose_plane_fit.py -i input.tum -o output.tum
  python pose_plane_fit.py --input_file trajectory.txt --output_file transformed.txt
        """
    )

    parser.add_argument('--input_file', '-i', required=True, help='Input TUM pose file')
    parser.add_argument('--output_file', '-o', required=True, help='Output TUM pose file')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed transformation information')

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.")
        sys.exit(1)

    try:
        # Read poses from TUM file
        print(f"Reading poses from: {args.input_file}")
        poses = read_tum_file(args.input_file)
        num_poses = len(poses['timestamps'])
        print(f"Loaded {num_poses} poses")

        if num_poses < 3:
            print("Error: Need at least 3 poses to fit a plane.")
            sys.exit(1)

        # Get target z-value from first pose
        target_z = poses['positions'][0, 2]
        print(f"Target z-plane: z = {target_z:.6f} (from first pose)")

        # Fit plane to pose positions
        print("Fitting plane to pose positions...")
        positions = poses['positions']
        plane_normal, plane_point, plane_equation = fit_plane_to_points(
            positions)

        # Compute transformation to align with target z plane
        print(f"Computing transformation to z = {target_z:.6f} plane...")
        transform = compute_transform_to_target_z_plane(plane_normal, plane_point, target_z)

        # Apply transformation to all poses
        print("Applying transformation to poses...")
        transformed_poses = apply_transform_to_poses(poses, transform)

        # Write transformed poses to output file
        print(f"Writing transformed poses to: {args.output_file}")
        write_tum_file(transformed_poses, args.output_file)

        # Print transformation information
        if args.verbose:
            print_transform_info(transform, plane_normal, plane_point,
                                 plane_equation, target_z)
        else:
            print("\nTransformation applied successfully!")
            print(f"Original plane normal: [{plane_normal[0]:.3f}, "
                  f"{plane_normal[1]:.3f}, {plane_normal[2]:.3f}]")
            print(f"Target z-plane: z = {target_z:.6f}")
            print("Use --verbose flag for detailed transformation information.")

        # Verify transformation by checking z-coordinates
        z_coords = transformed_poses['positions'][:, 2]
        z_mean = np.mean(z_coords)
        z_std = np.std(z_coords)
        print("\nTransformed pose z-coordinates:")
        print(f"  Mean: {z_mean:.6f}")
        print(f"  Std dev: {z_std:.6f}")
        print(f"  Range: [{np.min(z_coords):.6f}, {np.max(z_coords):.6f}]")

        print(f"\nDone! Transformed {num_poses} poses.")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
