#!/usr/bin/env python3
"""
SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION & AFFILIATES
SPDX-License-Identifier: Apache-2.0

Script to compare and plot poses from TUM format files.
Plots all trajectories in the same reference axis with different colors.
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from pose_loader import PoseType, load_optimized_keyframe, load_vslam_run_poses

matplotlib.use('Agg')

RUN_POSE_TYPES = [
    PoseType.VSLAM_ODOM_POSE,
    PoseType.VSLAM_SLAM_POSE,
    PoseType.VSLAM_KEYFRAME_POSE,
]

POSE_LABELS = {
    PoseType.VSLAM_ODOM_POSE: 'Odom Poses',
    PoseType.VSLAM_SLAM_POSE: 'SLAM Poses',
    PoseType.VSLAM_KEYFRAME_POSE: 'Keyframe Poses',
    PoseType.OPTIMIZED_KEYFRAME_POSE: 'Optimized Keyframe Pose',
}

POSE_COLORS = {
    PoseType.VSLAM_ODOM_POSE: 'blue',
    PoseType.VSLAM_SLAM_POSE: 'hotpink',
    PoseType.VSLAM_KEYFRAME_POSE: 'magenta',
    PoseType.OPTIMIZED_KEYFRAME_POSE: 'cyan',
}

POSE_LINE_STYLES = {
    PoseType.VSLAM_ODOM_POSE: '-',
    PoseType.VSLAM_SLAM_POSE: '--',
    PoseType.VSLAM_KEYFRAME_POSE: '-',
    PoseType.OPTIMIZED_KEYFRAME_POSE: '-',
}


class PosePlotter:
    """Plots pose data and exposes helper methods for statistics generation."""

    def __init__(self):
        self.colors = POSE_COLORS
        self.line_styles = POSE_LINE_STYLES
        self.labels = POSE_LABELS

    def plot_poses_2d(self, poses_data: Dict[PoseType, Tuple[np.ndarray, np.ndarray]],
                      output_path: str, map_name: str):
        """Plot all TUM poses in 2D (XY and XZ views)."""
        if not poses_data:
            print("No pose data to plot")
            return

        fig = plt.figure(figsize=(24, 4))

        gs = gridspec.GridSpec(
            1, 5, figure=fig,
            width_ratios=[0.15, 1, 0.3, 1, 0.15],
            hspace=0.4, wspace=0.05
        )

        ax_xy = fig.add_subplot(gs[0, 1])
        ax_xz = fig.add_subplot(gs[0, 3])

        self._plot_trajectories(
            ax_xy, ax_xz, poses_data,
            xy_title=f"XY - TUM Poses - {map_name}",
            xz_title=f"XZ - TUM Poses - {map_name}"
        )

        output_file = os.path.join(output_path, "optimized_keyframe_pose.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.1)
        print(f"Poses comparison plot saved to: {output_file}")

        plt.close(fig)

    def plot_repeat_runs_2d(self,
                            repeat_poses_data: List[Dict[PoseType, Tuple[np.ndarray, np.ndarray]]],
                            output_path: str, map_name: str):
        """Plot poses separated by repeat runs in a 2-column layout (XY, XZ).
        Each row represents one run.
        """
        if not repeat_poses_data:
            print("No repeat pose data to plot")
            return

        max_runs = len(repeat_poses_data)
        print(f"Creating repeat runs plot with {max_runs} runs")

        for run_idx in range(max_runs):
            run_data: Dict[PoseType, Tuple[np.ndarray, np.ndarray]] = {}
            for pose_type, (timestamps, positions) in repeat_poses_data[run_idx].items():
                if len(positions) > 0:
                    run_data[pose_type] = (timestamps, positions)

            if not run_data:
                continue

            fig, (ax_xy, ax_xz) = plt.subplots(1, 2, figsize=(16, 6))
            self._plot_trajectories(
                ax_xy, ax_xz, run_data,
                xy_title=f"Run {run_idx + 1} - XY View",
                xz_title=f"Run {run_idx + 1} - XZ View",
                add_legend=True
            )

            fig.suptitle(f"Repeat Run {run_idx + 1} - {map_name}", fontsize=16, y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.94])

            output_file = os.path.join(output_path, f"repeat_runs_{run_idx + 1}.png")
            plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.1)
            print(f"Repeat run plot saved to: {output_file}")
        plt.close(fig)

    def _plot_trajectories(self, ax_xy, ax_xz,
                           data_group: Dict[PoseType, Tuple[np.ndarray, np.ndarray]],
                           xy_title: str = None, xz_title: str = None,
                           add_legend: bool = True):
        """Plot trajectories on XY and XZ subplots.

        Args:
            ax_xy: Matplotlib axis for XY plot
            ax_xz: Matplotlib axis for XZ plot
            data_group: Dict mapping pose type to (timestamps, positions) tuple
            xy_title: Title for XY subplot (optional)
            xz_title: Title for XZ subplot (optional)
            add_legend: Whether to add legends to the plots
        """
        from matplotlib.lines import Line2D

        if xy_title:
            ax_xy.set_title(xy_title, pad=20)
        ax_xy.set_xlabel("X (meters)")
        ax_xy.set_ylabel("Y (meters)")
        ax_xy.grid(True, alpha=0.3)
        ax_xy.minorticks_on()
        ax_xy.grid(True, which='minor', alpha=0.1)

        if xz_title:
            ax_xz.set_title(xz_title, pad=20)
        ax_xz.set_xlabel("X (meters)")
        ax_xz.set_ylabel("Z (meters)")
        ax_xz.grid(True, alpha=0.3)
        ax_xz.minorticks_on()
        ax_xz.grid(True, which='minor', alpha=0.1)

        for key, (timestamps, positions) in data_group.items():
            color = self.colors.get(key, 'black')
            line_style = self.line_styles.get(key, '-')
            label = self.labels.get(key, key.name.replace('_', ' ').title())

            ax_xy.plot(positions[:, 0], positions[:, 1],
                       color=color, linestyle=line_style, linewidth=2, label=label, alpha=0.8)
            ax_xy.scatter(positions[0, 0], positions[0, 1],
                          color='green', marker='o', s=50, zorder=5,
                          edgecolors='black', linewidth=1)
            ax_xy.scatter(positions[-1, 0], positions[-1, 1],
                          color='red', marker='s', s=50, zorder=5,
                          edgecolors='black', linewidth=1)

            ax_xz.plot(positions[:, 0], positions[:, 2],
                       color=color, linestyle=line_style, linewidth=2, label=label, alpha=0.8)
            ax_xz.scatter(positions[0, 0], positions[0, 2],
                          color='green', marker='o', s=50, zorder=5,
                          edgecolors='black', linewidth=1)
            ax_xz.scatter(positions[-1, 0], positions[-1, 2],
                          color='red', marker='s', s=50, zorder=5,
                          edgecolors='black', linewidth=1)

        if add_legend:
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
                       markersize=8, markeredgecolor='black', markeredgewidth=1,
                       label='Start Point'),
                Line2D([0], [0], marker='s', color='w', markerfacecolor='red',
                       markersize=8, markeredgecolor='black', markeredgewidth=1,
                       label='End Point')
            ]

            xy_handles, _ = ax_xy.get_legend_handles_labels()
            xz_handles, _ = ax_xz.get_legend_handles_labels()

            ax_xy.legend(handles=xy_handles + legend_elements,
                         loc='center right', bbox_to_anchor=(-0.1, 0.5))
            ax_xz.legend(handles=xz_handles + legend_elements,
                         loc='center left', bbox_to_anchor=(1.1, 0.5))

    def _compute_trajectory_stats(self, timestamps: np.ndarray, positions: np.ndarray,
                                  indent: str = "  ") -> List[str]:
        """Compute statistics for a single trajectory.

        Args:
            timestamps: Array of timestamps
            positions: Array of positions (N x 3)
            indent: Indentation prefix for each line

        Returns:
            List of formatted statistic strings
        """
        stats = []
        stats.append(f"{indent}Number of poses: {len(positions)}")

        if len(timestamps) > 0:
            stats.append(f"{indent}Time span: {timestamps[-1] - timestamps[0]:.2f} seconds")

        if len(positions) > 1:
            distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
            total_distance = np.sum(distances)
            stats.append(f"{indent}Total distance: {total_distance:.2f} meters")

        if len(positions) > 0:
            x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
            y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
            z_min, z_max = positions[:, 2].min(), positions[:, 2].max()
            stats.append(f"{indent}X range: [{x_min:.3f}, {x_max:.3f}] meters")
            stats.append(f"{indent}Y range: [{y_min:.3f}, {y_max:.3f}] meters")
            stats.append(f"{indent}Z range: [{z_min:.3f}, {z_max:.3f}] meters")

        return stats

    def generate_statistics(
            self, poses_data: Dict[PoseType, Tuple[np.ndarray, np.ndarray]]
    ) -> str:
        """Generate statistics about the loaded poses."""
        stats = []

        for key, (timestamps, positions) in poses_data.items():
            label = self.labels.get(key, key.name.replace('_', ' ').title())
            stats.append(f"\n{label}:")
            stats.extend(self._compute_trajectory_stats(timestamps, positions))

        return "\n".join(stats)

    def generate_repeat_statistics(
            self, repeat_poses_data: List[Dict[PoseType, Tuple[np.ndarray, np.ndarray]]]
    ) -> str:
        """Generate statistics about repeat run poses."""
        stats = []
        for pose_type in RUN_POSE_TYPES:
            runs = [
                run_data[pose_type]
                for run_data in repeat_poses_data
                if pose_type in run_data
            ]
            if not runs:
                continue

            label = self.labels.get(pose_type, pose_type.name.replace('_', ' ').title())
            stats.append(f"\n{label}:")
            stats.append(f"  Number of runs: {len(runs)}")

            for run_idx, (timestamps, positions) in enumerate(runs):
                stats.append(f"\n  Run {run_idx + 1}:")
                stats.extend(self._compute_trajectory_stats(timestamps, positions, indent="    "))

        return "\n".join(stats)


def main():
    epilog_text = """
Examples:
  python3 compare_poses.py --map_dir /path/to/map --output /path/to/output

The script will find and plot TUM files:
  - poses/keyframe_pose_optimized.tum
  - poses/runs/odom_poses_*.tum
  - poses/runs/keyframe_pose_*.tum
  - poses/runs/slam_poses_*.tum
        """

    parser = argparse.ArgumentParser(
        description="Compare and plot poses from TUM format files",
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

    if not os.path.isdir(args.map_dir):
        print(f"Error: Map directory does not exist: {args.map_dir}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        output_dir = args.output
    else:
        output_dir = os.path.join(args.map_dir, 'pose_comparison')

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    map_name = os.path.basename(args.map_dir.rstrip('/'))

    try:
        plotter = PosePlotter()

        run_poses = load_vslam_run_poses(args.map_dir)
        optimized_poses = load_optimized_keyframe(args.map_dir)

        if optimized_poses is None:
            print("Error: Optimized keyframe pose not found", file=sys.stderr)
            sys.exit(1)

        print("Found optimized keyframe pose, plotting...")
        plotter.plot_poses_2d(
            {PoseType.OPTIMIZED_KEYFRAME_POSE: optimized_poses},
            output_dir,
            map_name
        )
        print("\nGenerating optimized keyframepose statistics...")
        pose_stats_content = plotter.generate_statistics(
            {PoseType.OPTIMIZED_KEYFRAME_POSE: optimized_poses}
        )
        stats_file = os.path.join(output_dir, "optimized_keyframe_pose_statistics.txt")
        with open(stats_file, 'w') as f:
            f.write(pose_stats_content)
        print(f"Pose statistics saved to: {stats_file}")

        if run_poses is None:
            print('Error: No repeat run files found', file=sys.stderr)
            sys.exit(1)

        print("\nFound repeat run files, generating repeat run comparison...")
        plotter.plot_repeat_runs_2d(run_poses, output_dir, map_name)
        print("\nGenerating repeat run poses statistics...")
        repeat_stats_content = plotter.generate_repeat_statistics(run_poses)
        repeat_stats_file = os.path.join(output_dir, "repeat_runs_statistics.txt")
        with open(repeat_stats_file, 'w') as f:
            f.write(repeat_stats_content)
        print(f"Repeat runs statistics saved to: {repeat_stats_file}")
        print(repeat_stats_content)

    except Exception as e:
        print(f"Error during pose comparison: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
