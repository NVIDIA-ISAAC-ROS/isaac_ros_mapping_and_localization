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

import argparse
import math
import sys
from typing import List, Tuple


Pose = Tuple[float, Tuple[float, float, float], Tuple[float, float, float, float]]


def load_tum_poses(path: str) -> List[Pose]:
    poses = []
    with open(path, 'r') as pose_file:
        for line_number, line in enumerate(pose_file, start=1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            fields = line.split()
            if len(fields) != 8:
                raise ValueError(f'{path}:{line_number}: expected 8 TUM fields, got {len(fields)}')
            timestamp, tx, ty, tz, qx, qy, qz, qw = (float(field) for field in fields)
            poses.append((timestamp, (tx, ty, tz), (qx, qy, qz, qw)))
    return poses


def translation_error_m(baseline_position: Tuple[float, float, float],
                        candidate_position: Tuple[float, float, float]) -> float:
    return math.sqrt(sum(
        (candidate_position[index] - baseline_position[index])**2
        for index in range(3)
    ))


def rotation_error_degrees(baseline_quaternion: Tuple[float, float, float, float],
                           candidate_quaternion: Tuple[float, float, float, float]) -> float:
    baseline_norm = math.sqrt(sum(value * value for value in baseline_quaternion))
    candidate_norm = math.sqrt(sum(value * value for value in candidate_quaternion))
    if baseline_norm == 0.0 or candidate_norm == 0.0:
        raise ValueError('Cannot compare zero-norm quaternion')

    dot = sum(
        (baseline_quaternion[index] / baseline_norm) *
        (candidate_quaternion[index] / candidate_norm)
        for index in range(4)
    )
    dot = min(1.0, max(-1.0, abs(dot)))
    return math.degrees(2.0 * math.acos(dot))


def compare_tum_poses(args: argparse.Namespace) -> int:
    baseline_poses = load_tum_poses(args.baseline)
    candidate_poses = load_tum_poses(args.candidate)
    if len(baseline_poses) != len(candidate_poses):
        print(
            f'Failure: pose count mismatch: baseline={len(baseline_poses)}, '
            f'candidate={len(candidate_poses)}'
        )
        return 1

    max_translation_error_m = 0.0
    max_rotation_error_degrees = 0.0
    max_timestamp_error_s = 0.0
    failing_pose_count = 0
    worst_pose_index = -1

    for pose_index, (baseline_pose, candidate_pose) in enumerate(
        zip(baseline_poses, candidate_poses)
    ):
        timestamp_error_s = abs(candidate_pose[0] - baseline_pose[0])
        trans_error_m = translation_error_m(baseline_pose[1], candidate_pose[1])
        rot_error_degrees = rotation_error_degrees(baseline_pose[2], candidate_pose[2])

        pose_failed = (
            timestamp_error_s > args.timestamp_threshold_s or
            trans_error_m > args.translation_threshold_m or
            rot_error_degrees > args.rotation_threshold_degrees
        )
        if pose_failed:
            failing_pose_count += 1
            if worst_pose_index == -1:
                worst_pose_index = pose_index

        if trans_error_m > max_translation_error_m:
            worst_pose_index = pose_index
        max_translation_error_m = max(max_translation_error_m, trans_error_m)
        max_rotation_error_degrees = max(max_rotation_error_degrees, rot_error_degrees)
        max_timestamp_error_s = max(max_timestamp_error_s, timestamp_error_s)

    print(f'Compared {len(baseline_poses)} TUM poses')
    print(f'Max timestamp error: {max_timestamp_error_s:.9f} s')
    print(f'Max translation error: {max_translation_error_m:.6f} m')
    print(f'Max rotation error: {max_rotation_error_degrees:.6f} deg')
    print(f'Failing poses: {failing_pose_count}')

    if failing_pose_count > args.max_failing_poses:
        print(f'Failure: worst pose index: {worst_pose_index}')
        return 1
    print('Success: TUM poses match baseline within thresholds')
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Compare two TUM pose files against translation and rotation thresholds.')
    parser.add_argument('--baseline', required=True, help='Baseline TUM pose file')
    parser.add_argument('--candidate', required=True, help='Candidate TUM pose file')
    parser.add_argument('--translation_threshold_m', type=float, default=0.25)
    parser.add_argument('--rotation_threshold_degrees', type=float, default=5.0)
    parser.add_argument('--timestamp_threshold_s', type=float, default=1e-6)
    parser.add_argument('--max_failing_poses', type=int, default=0)
    args = parser.parse_args()
    sys.exit(compare_tum_poses(args))


if __name__ == '__main__':
    main()
