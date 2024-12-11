#!/bin/bash
set -ex

# This script runs relative localization evaluation.
# In order to evaluate how good the localization to map is, we need a tool to benchmark the relative error to a localization, a way to do this is:
# Inputs:
#    1. eval mapping pose file
#    2. gt mapping pose file
#    3. eval localization pose file
#    4. gt localization pose file

# approach:

# 1. first compute a relative mapping from eval pose to gt mapping pose, for each pose in input 1 find it's corresponding pose in input 2.
# 2. for each eval pose Ta in input3 and it’s corresponding gt localization pose Tb in input4
#      1. find the closest eval pose Tc in input 1
#      2. find the corresponding gt mapping pose Td
#      3. compute the relative transform delta_Ta between Ta and Tc
#      4. compute the relative transform delta_Tc between Tb and Td
#      5. compute the difference between the two delta transform.

# Usage:
# bash src/isaac_ros_mapping_and_localization/isaac_ros_camera_localization/script/run_evaluate_relative_localization.sh
# [eval_mapping_pose_file] [gt_mapping_pose_file] [eval_localization_pose_file] [gt_localization_pose_file] [output_folder]

EVAL_MAPPING_POSE_FILE=${1? "Please provide eval mapping pose file"}
GT_MAPPING_POSE_FILE=${2? "Please provide gt mapping pose file"}
EVAL_LOCALIZATION_POSE_FILE=${3? "Please provide eval localization pose file"}
GT_LOCALIZATION_POSE_FILE=${4? "Please provide gt localization pose file"}
OUTPUT_FOLDER=${5? "Please provide output folder"}

colcon build --symlink-install --packages-up-to isaac_mapping_ros
source ${ISAAC_ROS_WS}/install/setup.bash

echo "Evaluating relative localization ..."

${ISAAC_ROS_WS}/build/isaac_mapping_ros/isaac_mapping/tools/evaluate_relative_localization_main \
    --eval_mapping_pose_file $EVAL_MAPPING_POSE_FILE \
    --gt_mapping_pose_file $GT_MAPPING_POSE_FILE \
    --eval_localization_pose_file $EVAL_LOCALIZATION_POSE_FILE \
    --gt_localization_pose_file $GT_LOCALIZATION_POSE_FILE \
    --output_dir $OUTPUT_FOLDER

echo "Plotting pose error ..."

PLOT_TOOL=${ISAAC_ROS_WS}/build/isaac_mapping_ros/scripts/visual/metrics/plot_pose_error.py

python $PLOT_TOOL \
    $OUTPUT_FOLDER/pose_evaluation_result.json \
    --output_prefix $OUTPUT_FOLDER/pose_evaluation \
    --plot_line --plot_hist

echo "Plots are generated in $OUTPUT_FOLDER"