#!/usr/bin/env python3
"""
SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION & AFFILIATES
SPDX-License-Identifier: Apache-2.0

Script to compare and plot poses from TUM format files and frames_meta files.
Plots all trajectories in the same reference axis with different colors.
"""

import argparse
import json
import os
import sys
from typing import Dict, Tuple, List

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

# Configure matplotlib to use non-interactive backend
matplotlib.use('Agg')


class PoseComparator:
    """Class to load and compare poses from different sources."""

    def __init__(self):
        # Define colors for trajectories (avoiding green and red for start/end points)
        self.colors = {
            'odom_poses': 'blue',
            'keyframe_pose_optimized': 'cyan',
            'slam_poses': 'hotpink',
            'frames_meta_raw_camera_0': 'orange',
            'frames_meta_raw_camera_1': 'darkorange',
            'frames_meta_rectified_camera_0': 'purple',
            'frames_meta_rectified_camera_1': 'darkviolet'
        }

        # Line styles for different pose types
        self.line_styles = {
            'odom_poses': '-',           # Solid line
            'keyframe_pose_optimized': '-',  # Solid line
            'slam_poses': '--',          # Dotted line
            'frames_meta_raw_camera_0': '-',
            'frames_meta_raw_camera_1': '-',
            'frames_meta_rectified_camera_0': '-',
            'frames_meta_rectified_camera_1': '-'
        }

    def load_tum_file(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
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
            positions = data[:, 1:4]  # Extract tx, ty, tz
            return timestamps, positions
        except Exception as e:
            raise ValueError(f"Failed to load TUM file {file_path}: {e}")

    def load_frames_meta(self, file_path: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Load frames_meta.json and extract camera positions separated by camera ID."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"frames_meta.json file not found: {file_path}")

        with open(file_path, 'r') as f:
            data = json.load(f)

        keyframes_metadata = data.get('keyframes_metadata', [])

        # Separate data by camera_params_id (typically 0=left, 1=right)
        camera_data = {}

        for keyframe in keyframes_metadata:
            timestamp_us = keyframe.get('timestamp_microseconds', 0)
            # Handle both string and numeric timestamps
            if isinstance(timestamp_us, str):
                timestamp_us = float(timestamp_us)
            timestamp = timestamp_us / 1e6  # Convert to seconds

            camera_id = keyframe.get('camera_params_id', '0')
            camera_to_world = keyframe.get('camera_to_world', {})

            if 'translation' in camera_to_world:
                translation = camera_to_world['translation']
                x = translation.get('x', 0)
                y = translation.get('y', 0)
                z = translation.get('z', 0)

                if camera_id not in camera_data:
                    camera_data[camera_id] = {'timestamps': [], 'positions': []}

                camera_data[camera_id]['timestamps'].append(timestamp)
                camera_data[camera_id]['positions'].append([x, y, z])

        if not camera_data:
            raise ValueError(f"No valid poses found in frames_meta file: {file_path}")

        # Convert to numpy arrays and return dict
        result = {}
        for camera_id, data in camera_data.items():
            result[f"camera_{camera_id}"] = (
                np.array(data['timestamps']),
                np.array(data['positions'])
            )

        return result

    def find_repeat_run_files(self, map_dir: str) -> Dict[str, List[str]]:
        """Find repeat run pose files (e.g., odom_poses_0.tum, odom_poses_1.tum)."""
        poses_dir = os.path.join(map_dir, 'poses')
        if not os.path.exists(poses_dir):
            return {}

        repeat_files = {}
        pose_types = ['odom_poses', 'slam_poses', 'keyframe_pose']

        for pose_type in pose_types:
            files = []
            run_idx = 0
            while True:
                file_path = os.path.join(poses_dir, f'{pose_type}_{run_idx}.tum')
                if os.path.exists(file_path):
                    files.append(file_path)
                    run_idx += 1
                else:
                    break

            if files:
                repeat_files[pose_type] = files
                print(f"Found {len(files)} repeat run files for {pose_type}")

        return repeat_files

    def find_pose_files(self, map_dir: str) -> Dict[str, str]:
        """Find all available pose files in the map directory."""
        pose_files = {}

        # TUM format files in poses/ directory
        poses_dir = os.path.join(map_dir, 'poses')
        if os.path.exists(poses_dir):
            tum_files = {
                'odom_poses': 'odom_poses.tum',
                'keyframe_pose_optimized': 'keyframe_pose_optimized.tum',
                'slam_poses': 'slam_poses.tum'
            }

            for key, filename in tum_files.items():
                file_path = os.path.join(poses_dir, filename)
                if os.path.exists(file_path):
                    pose_files[key] = file_path

        # frames_meta.json files
        raw_path = os.path.join(map_dir, 'map_frames', 'raw', 'frames_meta.json')
        rectified_path = os.path.join(map_dir, 'map_frames', 'rectified', 'frames_meta.json')
        frames_meta_files = {
            'frames_meta_raw': raw_path,
            'frames_meta_rectified': rectified_path
        }

        for key, file_path in frames_meta_files.items():
            if os.path.exists(file_path):
                pose_files[key] = file_path

        return pose_files

    def load_all_poses(self, map_dir: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Load all available pose files from the map directory."""
        pose_files = self.find_pose_files(map_dir)
        poses_data = {}

        for key, file_path in pose_files.items():
            try:
                if key.startswith('frames_meta'):
                    # Load frames_meta returns dict of camera trajectories
                    camera_trajectories = self.load_frames_meta(file_path)
                    for camera_key, (timestamps, positions) in camera_trajectories.items():
                        full_key = f"{key}_{camera_key}"
                        poses_data[full_key] = (timestamps, positions)
                        print(f"Loaded {full_key}: {len(positions)} poses from {file_path}")
                else:
                    timestamps, positions = self.load_tum_file(file_path)
                    poses_data[key] = (timestamps, positions)
                    print(f"Loaded {key}: {len(positions)} poses from {file_path}")

            except Exception as e:
                print(f"Warning: Failed to load {key} from {file_path}: {e}")

        return poses_data

    def load_repeat_run_data(self, map_dir: str) -> Dict[str, List[Tuple[np.ndarray, np.ndarray]]]:
        """Load repeat run pose files if they exist."""
        repeat_files = self.find_repeat_run_files(map_dir)
        if not repeat_files:
            return {}

        repeat_data = {}
        for pose_type, file_list in repeat_files.items():
            runs = []
            for file_path in file_list:
                try:
                    timestamps, positions = self.load_tum_file(file_path)
                    runs.append((timestamps, positions))
                except Exception as e:
                    print(f"Warning: Failed to load {file_path}: {e}")

            if runs:
                repeat_data[pose_type] = runs
                run_lengths = [len(r[1]) for r in runs]
                print(f"Loaded {pose_type}: {len(runs)} runs with {run_lengths} poses each")

        return repeat_data

    def plot_poses_2d(self, poses_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
                      output_path: str, map_name: str):
        """Plot all poses in 2D organized by source: TUM poses (row 1), frames_meta raw
        (row 2), frames_meta rectified (row 3), camera 0 comparison (row 4)."""
        if not poses_data:
            print("No pose data to plot")
            return

        # Organize data by source
        tum_data = {}
        frames_raw_data = {}
        frames_rectified_data = {}
        camera0_comparison_data = {}

        for key, (timestamps, positions) in poses_data.items():
            if key.startswith('frames_meta_raw'):
                frames_raw_data[key] = (timestamps, positions)
            elif key.startswith('frames_meta_rectified'):
                frames_rectified_data[key] = (timestamps, positions)
            else:
                tum_data[key] = (timestamps, positions)

        # Create camera 0 comparison data (raw vs rectified)
        if 'frames_meta_raw_camera_0' in poses_data:
            raw_camera_0 = poses_data['frames_meta_raw_camera_0']
            camera0_comparison_data['frames_meta_raw_camera_0'] = raw_camera_0
        if 'frames_meta_rectified_camera_0' in poses_data:
            rectified_camera_0 = poses_data['frames_meta_rectified_camera_0']
            camera0_comparison_data['frames_meta_rectified_camera_0'] = rectified_camera_0

        # Determine number of rows needed
        num_rows = 0
        if tum_data:
            num_rows += 1
        if frames_raw_data:
            num_rows += 1
        if frames_rectified_data:
            num_rows += 1
        if camera0_comparison_data:
            num_rows += 1

        if num_rows == 0:
            print("No data to plot")
            return

        # Create figure with GridSpec for precise control
        fig = plt.figure(figsize=(24, 4 * num_rows))  # Wider figure, less height per row

        # Create GridSpec: each row has 5 columns (legend, plot, space, plot, legend)
        gs = gridspec.GridSpec(
            num_rows, 5, figure=fig,
            width_ratios=[0.15, 1, 0.3, 1, 0.15],  # legend, plot, spacer, plot, legend
            height_ratios=[1] * num_rows,
            hspace=0.4, wspace=0.05  # Slightly more vertical space to prevent title clashing
        )

        # Create axes array to match the original structure
        axes = []
        for row in range(num_rows):
            # Create plot axes with spacer column in between (columns 1 and 3)
            ax_xy = fig.add_subplot(gs[row, 1])  # XY plot in column 1
            ax_xz = fig.add_subplot(gs[row, 3])  # XZ plot in column 3 (skip column 2 spacer)
            axes.append([ax_xy, ax_xz])

        axes = np.array(axes)

        row_idx = 0

        # Plot TUM data (first row)
        if tum_data:
            ax_xy, ax_xz = axes[row_idx]
            self._plot_trajectory_group(ax_xy, ax_xz, tum_data, f"TUM Poses - {map_name}")
            row_idx += 1

        # Plot frames_meta raw data (second row)
        if frames_raw_data:
            ax_xy, ax_xz = axes[row_idx]
            trajectory_title = f"Frames Meta Raw - {map_name}"
            self._plot_trajectory_group(ax_xy, ax_xz, frames_raw_data, trajectory_title)
            row_idx += 1

        # Plot frames_meta rectified data (third row)
        if frames_rectified_data:
            ax_xy, ax_xz = axes[row_idx]
            trajectory_title = f"Frames Meta Rectified - {map_name}"
            self._plot_trajectory_group(ax_xy, ax_xz, frames_rectified_data, trajectory_title)
            row_idx += 1

        # Plot camera 0 comparison data (fourth row)
        if camera0_comparison_data:
            ax_xy, ax_xz = axes[row_idx]
            trajectory_title = f"Camera 0 Comparison - {map_name}"
            self._plot_trajectory_group(ax_xy, ax_xz, camera0_comparison_data, trajectory_title)

        # Save plot - don't use tight_layout as we already adjusted spacing
        output_file = os.path.join(output_path, f"{map_name}_poses_comparison.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.1)
        print(f"Poses comparison plot saved to: {output_file}")

        plt.close(fig)

    def plot_repeat_runs_2d(self,
                            repeat_poses_data: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
                            output_path: str, map_name: str):
        """Plot poses separated by repeat runs in a 2-column layout (XY, XZ).
        Each row represents one run.
        """
        if not repeat_poses_data:
            print("No repeat pose data to plot")
            return

        # Determine maximum number of runs across all pose types
        max_runs = max(len(runs) for runs in repeat_poses_data.values())
        if max_runs <= 1:
            print("Only one run detected, skipping repeat run plot")
            return

        print(f"Creating repeat runs plot with {max_runs} runs")

        # Create figure with 2 columns (XY, XZ) and max_runs rows
        fig_height = max(8, 4 * max_runs)  # At least 8 inches, 4 inches per run
        fig, axes = plt.subplots(max_runs, 2, figsize=(16, fig_height))

        # Handle single row case
        if max_runs == 1:
            axes = axes.reshape(1, -1)

        # Plot each run
        for run_idx in range(max_runs):
            ax_xy = axes[run_idx, 0]
            ax_xz = axes[run_idx, 1]

            # Configure XY subplot
            ax_xy.set_title(f"Run {run_idx + 1} - XY View", fontsize=12, pad=10)
            ax_xy.set_xlabel("X (meters)")
            ax_xy.set_ylabel("Y (meters)")
            ax_xy.grid(True, alpha=0.3)
            ax_xy.minorticks_on()
            ax_xy.grid(True, which='minor', alpha=0.1)

            # Configure XZ subplot
            ax_xz.set_title(f"Run {run_idx + 1} - XZ View", fontsize=12, pad=10)
            ax_xz.set_xlabel("X (meters)")
            ax_xz.set_ylabel("Z (meters)")
            ax_xz.grid(True, alpha=0.3)
            ax_xz.minorticks_on()
            ax_xz.grid(True, which='minor', alpha=0.1)

            # Plot trajectories for this run
            for pose_type, runs in repeat_poses_data.items():
                if run_idx >= len(runs):
                    continue

                timestamps, positions = runs[run_idx]
                if len(positions) == 0:
                    continue

                color = self.colors.get(pose_type, 'black')
                line_style = self.line_styles.get(pose_type, '-')
                label = pose_type.replace('_', ' ').title()

                # XY plot
                ax_xy.plot(positions[:, 0], positions[:, 1],
                           color=color, linestyle=line_style, linewidth=2, label=label, alpha=0.8)
                ax_xy.scatter(positions[0, 0], positions[0, 1],
                              color='green', marker='o', s=50, zorder=5,
                              edgecolors='black', linewidth=1)
                ax_xy.scatter(positions[-1, 0], positions[-1, 1],
                              color='red', marker='s', s=50, zorder=5,
                              edgecolors='black', linewidth=1)

                # XZ plot
                ax_xz.plot(positions[:, 0], positions[:, 2],
                           color=color, linestyle=line_style, linewidth=2, label=label, alpha=0.8)
                ax_xz.scatter(positions[0, 0], positions[0, 2],
                              color='green', marker='o', s=50, zorder=5,
                              edgecolors='black', linewidth=1)
                ax_xz.scatter(positions[-1, 0], positions[-1, 2],
                              color='red', marker='s', s=50, zorder=5,
                              edgecolors='black', linewidth=1)

            # Add legends
            if run_idx == 0:  # Only add legend to first row
                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
                           markersize=8, markeredgecolor='black', markeredgewidth=1,
                           label='Start Point'),
                    Line2D([0], [0], marker='s', color='w', markerfacecolor='red',
                           markersize=8, markeredgecolor='black', markeredgewidth=1,
                           label='End Point')
                ]

                xy_handles, xy_labels = ax_xy.get_legend_handles_labels()
                xz_handles, xz_labels = ax_xz.get_legend_handles_labels()

                ax_xy.legend(handles=xy_handles + legend_elements,
                             loc='upper right', bbox_to_anchor=(1, 1))
                ax_xz.legend(handles=xz_handles + legend_elements,
                             loc='upper right', bbox_to_anchor=(1, 1))

        # Add overall title
        fig.suptitle(f"Repeat Runs Comparison - {map_name}", fontsize=16, y=0.98)

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save plot
        output_file = os.path.join(output_path, f"{map_name}_repeat_runs_comparison.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.1)
        print(f"Repeat runs comparison plot saved to: {output_file}")

        plt.close(fig)

    def _plot_trajectory_group(self, ax_xy, ax_xz,
                               data_group: Dict[str, Tuple[np.ndarray, np.ndarray]],
                               title_prefix: str):
        """Helper function to plot a group of trajectories on XY and XZ subplots."""
        # Configure XY subplot with shorter title
        ax_xy.set_title(f"XY - {title_prefix}", pad=20)  # Shorter title
        ax_xy.set_xlabel("X (meters)")
        ax_xy.set_ylabel("Y (meters)")
        ax_xy.grid(True, alpha=0.3)
        ax_xy.minorticks_on()  # Enable minor ticks
        ax_xy.grid(True, which='minor', alpha=0.1)  # Add minor grid
        # Remove aspect ratio constraint to allow GridSpec to control height

        # Configure XZ subplot with shorter title
        ax_xz.set_title(f"XZ - {title_prefix}", pad=20)  # Shorter title
        ax_xz.set_xlabel("X (meters)")
        ax_xz.set_ylabel("Z (meters)")
        ax_xz.grid(True, alpha=0.3)
        ax_xz.minorticks_on()  # Enable minor ticks
        ax_xz.grid(True, which='minor', alpha=0.1)  # Add minor grid
        # Remove aspect ratio constraint to allow GridSpec to control height

        # Plot each trajectory in the group
        for key, (timestamps, positions) in data_group.items():
            color = self.colors.get(key, 'black')
            line_style = self.line_styles.get(key, '-')
            label = key.replace('_', ' ').replace('frames meta', 'FM').title()

            # XY plot
            ax_xy.plot(positions[:, 0], positions[:, 1],
                       color=color, linestyle=line_style, linewidth=2, label=label, alpha=0.8)
            ax_xy.scatter(positions[0, 0], positions[0, 1],
                          color='green', marker='o', s=50, zorder=5,
                          edgecolors='black', linewidth=1)  # Start point - green
            ax_xy.scatter(positions[-1, 0], positions[-1, 1],
                          color='red', marker='s', s=50, zorder=5,
                          edgecolors='black', linewidth=1)  # End point - red

            # XZ plot
            ax_xz.plot(positions[:, 0], positions[:, 2],
                       color=color, linestyle=line_style, linewidth=2, label=label, alpha=0.8)
            ax_xz.scatter(positions[0, 0], positions[0, 2],
                          color='green', marker='o', s=50, zorder=5,
                          edgecolors='black', linewidth=1)  # Start point - green
            ax_xz.scatter(positions[-1, 0], positions[-1, 2],
                          color='red', marker='s', s=50, zorder=5,
                          edgecolors='black', linewidth=1)  # End point - red

        # Add start/end point legend elements
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
                   markersize=8, markeredgecolor='black', markeredgewidth=1, label='Start Point'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='red',
                   markersize=8, markeredgecolor='black', markeredgewidth=1, label='End Point')
        ]

        # Get existing legend handles and combine with new elements
        xy_handles, xy_labels = ax_xy.get_legend_handles_labels()
        xz_handles, xz_labels = ax_xz.get_legend_handles_labels()

        # Position legends in the dedicated columns (adjusted for 5-column layout)
        # XY plot legend on the left side (column 0)
        ax_xy.legend(handles=xy_handles + legend_elements,
                     loc='center right', bbox_to_anchor=(-0.1, 0.5))
        # XZ plot legend on the right side (column 4)
        ax_xz.legend(handles=xz_handles + legend_elements,
                     loc='center left', bbox_to_anchor=(1.1, 0.5))

    def generate_statistics(self, poses_data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> str:
        """Generate statistics about the loaded poses."""
        stats = []
        stats.append("POSE STATISTICS:")
        stats.append("=" * 50)

        for key, (timestamps, positions) in poses_data.items():
            stats.append(f"\n{key.replace('_', ' ').title()}:")
            stats.append(f"  Number of poses: {len(positions)}")
            stats.append(f"  Time span: {timestamps[-1] - timestamps[0]:.2f} seconds")

            # Compute trajectory length
            if len(positions) > 1:
                distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
                total_distance = np.sum(distances)
                stats.append(f"  Total distance: {total_distance:.2f} meters")

            # Position ranges
            x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
            y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
            z_min, z_max = positions[:, 2].min(), positions[:, 2].max()
            stats.append(f"  X range: [{x_min:.3f}, {x_max:.3f}] meters")
            stats.append(f"  Y range: [{y_min:.3f}, {y_max:.3f}] meters")
            stats.append(f"  Z range: [{z_min:.3f}, {z_max:.3f}] meters")

        return "\n".join(stats)

    def generate_repeat_statistics(
            self, repeat_poses_data: Dict[str, List[Tuple[np.ndarray, np.ndarray]]]
    ) -> str:
        """Generate statistics about repeat run poses."""
        stats = []
        stats.append("REPEAT RUNS STATISTICS:")
        stats.append("=" * 50)

        for pose_type, runs in repeat_poses_data.items():
            stats.append(f"\n{pose_type.replace('_', ' ').title()}:")
            stats.append(f"  Number of runs: {len(runs)}")

            for run_idx, (timestamps, positions) in enumerate(runs):
                stats.append(f"\n  Run {run_idx + 1}:")
                stats.append(f"    Number of poses: {len(positions)}")

                if len(timestamps) > 0:
                    stats.append(f"    Time span: {timestamps[-1] - timestamps[0]:.2f} seconds")
                    stats.append(f"    Start time: {timestamps[0]:.6f} seconds")
                    stats.append(f"    End time: {timestamps[-1]:.6f} seconds")

                # Compute trajectory length
                if len(positions) > 1:
                    distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
                    total_distance = np.sum(distances)
                    stats.append(f"    Total distance: {total_distance:.2f} meters")

                # Position ranges
                if len(positions) > 0:
                    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
                    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
                    z_min, z_max = positions[:, 2].min(), positions[:, 2].max()
                    stats.append(f"    X range: [{x_min:.3f}, {x_max:.3f}] meters")
                    stats.append(f"    Y range: [{y_min:.3f}, {y_max:.3f}] meters")
                    stats.append(f"    Z range: [{z_min:.3f}, {z_max:.3f}] meters")

        return "\n".join(stats)


def main():
    epilog_text = """
Examples:
  python3 compare_poses.py --map_dir /path/to/map --output /path/to/output
  python3 compare_poses.py --map_dir /path/to/map --output /path/to/output --stats

The script will automatically find and plot:
  - TUM files: odom_poses.tum, keyframe_pose_optimized.tum, slam_poses.tum
  - frames_meta.json files from map_frames/raw/ and map_frames/rectified/

Repeat Run Analysis:
  If repeat run files are found (e.g., odom_poses_0.tum, odom_poses_1.tum), the script
  automatically generates an additional plot showing per-run comparisons in XY/XZ views.
  These files are created by split_repeated_poses.py after running cuvslam with repeat > 1.

Colors and Styles:
  - odom_poses: blue (solid line)
  - keyframe_pose_optimized: cyan (solid line)
  - slam_poses: hot pink (dotted line)
  - frames_meta_raw: orange (solid line)
  - frames_meta_rectified: purple (solid line)
        """

    parser = argparse.ArgumentParser(
        description="Compare and plot poses from TUM files and frames_meta.json files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog_text
    )

    parser.add_argument(
        '--map_dir',
        required=True,
        help='Path to the map directory containing pose files'
    )

    parser.add_argument(
        '--output',
        help='Output directory for plots and statistics (default: map_dir/pose_comparison/)'
    )

    parser.add_argument(
        '--stats',
        action='store_true',
        help='Generate and save pose statistics'
    )

    args = parser.parse_args()

    # Validate map directory
    if not os.path.isdir(args.map_dir):
        print(f"Error: Map directory does not exist: {args.map_dir}", file=sys.stderr)
        sys.exit(1)

    # Set output directory - default to pose_comparison subdirectory in map_dir
    if args.output:
        output_dir = args.output
    else:
        output_dir = os.path.join(args.map_dir, 'pose_comparison')

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Extract map name from directory
    map_name = os.path.basename(args.map_dir.rstrip('/'))

    try:
        # Create pose comparator and load all poses
        comparator = PoseComparator()
        poses_data = comparator.load_all_poses(args.map_dir)

        if not poses_data:
            print("Error: No valid pose files found in the map directory", file=sys.stderr)
            sys.exit(1)

        # Generate and plot poses
        comparator.plot_poses_2d(poses_data, output_dir, map_name)

        # Check for repeat run files and generate comparison if found
        repeat_data = comparator.load_repeat_run_data(args.map_dir)
        if repeat_data:
            print("\nFound repeat run files, generating repeat run comparison...")
            comparator.plot_repeat_runs_2d(repeat_data, output_dir, map_name)

            # Generate repeat statistics if requested
            if args.stats:
                repeat_stats_content = comparator.generate_repeat_statistics(repeat_data)
                repeat_stats_file = os.path.join(
                    output_dir, f"{map_name}_repeat_runs_statistics.txt")

                with open(repeat_stats_file, 'w') as f:
                    f.write(repeat_stats_content)

                print(f"Repeat runs statistics saved to: {repeat_stats_file}")
                print("\n" + repeat_stats_content)

        # Generate statistics if requested
        if args.stats:
            stats_content = comparator.generate_statistics(poses_data)
            stats_file = os.path.join(output_dir, f"{map_name}_pose_statistics.txt")

            with open(stats_file, 'w') as f:
                f.write(stats_content)

            print(f"Pose statistics saved to: {stats_file}")
            print("\n" + stats_content)

    except Exception as e:
        print(f"Error during pose comparison: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
