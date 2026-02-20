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

import os
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np


class PoseType(Enum):
    VSLAM_SLAM_POSE = auto()
    VSLAM_ODOM_POSE = auto()
    VSLAM_KEYFRAME_POSE = auto()
    OPTIMIZED_KEYFRAME_POSE = auto()


vslam_run_pose_type_to_file_base = {
    PoseType.VSLAM_SLAM_POSE: 'slam_poses',
    PoseType.VSLAM_ODOM_POSE: 'odom_poses',
    PoseType.VSLAM_KEYFRAME_POSE: 'keyframe_pose',
}


def pose_dir(map_dir: str) -> str:
    return os.path.join(map_dir, 'poses')


def vslam_run_dir(map_dir: str) -> str:
    return os.path.join(pose_dir(map_dir), 'runs')


def load_tum_file(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a TUM file and extract timestamps and positions."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"TUM file not found: {file_path}")

    try:
        data = np.loadtxt(file_path)
        if data.shape[1] < 4:
            col_count = data.shape[1]
            msg = "TUM file must have at least 4 columns (timestamp, x, y, z), "
            msg += f"got {col_count}"
            raise ValueError(msg)

        timestamps = data[:, 0]
        positions = data[:, 1:4]
        return timestamps, positions
    except Exception as e:
        raise ValueError(f"Failed to load TUM file {file_path}: {e}")


def load_vslam_run_poses(map_dir: str) -> List[Dict[PoseType, Tuple[np.ndarray, np.ndarray]]]:
    """Load repeated run poses for all VSLAM sources."""
    poses_dir = vslam_run_dir(map_dir)
    pose_runs: List[Dict[PoseType, Tuple[np.ndarray, np.ndarray]]] = []

    if not os.path.isdir(poses_dir):
        return pose_runs

    run_idx = 0
    while True:
        run_data: Dict[PoseType, Tuple[np.ndarray, np.ndarray]] = {}
        found_any = False

        for pose_type, filename_base in vslam_run_pose_type_to_file_base.items():
            file_path = os.path.join(poses_dir, f"{filename_base}_{run_idx}.tum")
            if not os.path.exists(file_path):
                continue

            found_any = True
            try:
                timestamps, positions = load_tum_file(file_path)
                run_data[pose_type] = (timestamps, positions)
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")

        if run_idx == 0 and not found_any:
            fallback_data: Dict[PoseType, Tuple[np.ndarray, np.ndarray]] = {}
            for pose_type, filename_base in vslam_run_pose_type_to_file_base.items():
                file_path = os.path.join(poses_dir, f"{filename_base}.tum")
                if not os.path.exists(file_path):
                    continue
                try:
                    timestamps, positions = load_tum_file(file_path)
                    fallback_data[pose_type] = (timestamps, positions)
                except Exception as e:
                    print(f"Warning: Failed to load {file_path}: {e}")

            print(
                "Info: Missing *_0.tum run files; using non-indexed pose files and "
                "treating them as a single run."
            )
            return [fallback_data]

        if not found_any:
            break

        pose_runs.append(run_data)
        run_idx += 1

    return pose_runs


def load_optimized_keyframe(map_dir: str) -> Tuple[np.ndarray, np.ndarray] | None:
    """Load the optimized keyframe pose if it exists."""
    poses_dir = pose_dir(map_dir)
    file_path = os.path.join(poses_dir, 'keyframe_pose_optimized.tum')
    if not os.path.exists(file_path):
        print(f"Warning: Optimized keyframe pose not found: {file_path}")
        return None

    return load_tum_file(file_path)


def load_optimized_keyframe_min_max_z(map_dir: str) -> Optional[Tuple[float, float]]:
    """Load optimized keyframe poses and return min/max z coordinates."""
    optimized_poses = load_optimized_keyframe(map_dir)
    if optimized_poses is None:
        return None

    _, positions = optimized_poses
    if positions.size == 0:
        return None

    min_z = float(positions[:, 2].min())
    max_z = float(positions[:, 2].max())
    return min_z, max_z
