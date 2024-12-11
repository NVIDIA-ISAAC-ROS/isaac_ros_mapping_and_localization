#!/bin/bash
set -ex

# This script runs visual localization on a rosbag in replay. It also tries to get
# GT pose for the bag and runs evaluation.
# Usage:
# bash src/isaac_ros_mapping_and_localization/isaac_ros_visual_global_localization/script/run_visual_localization.sh
#   <sensor_data_bag_path> <glmap_dir> <omap_yaml_file> <x_offset> <y_offset>

# These are two example maps(omap + glmap) built with poses from Lidar pipeline:

####################
# hubble_lab_v0
####################
# Drive: https://drive.google.com/file/d/16l_M-9zzrEqVACQffrEgZM88M-O1AUe3/view
# from bag: hl4hl19_2024_04_05-16_05_35
# X_OFFSET: -2594.15966796875
# Y_OFFSET: 3445.883544921875

####################
# hubble_lab_0701
####################
# Drive: https://drive.google.com/drive/folders/12g_vkUbH7WVezeDy1hkd-pCNhGyFykm4
# from 6 bags:
# - small_hubble_lab_run_1_framedrop
# - small_hubble_lab_run_2_framedrop
# - 96_HLoop_carter3_4hi3dl_2024_05_30-14_10_43
# - 97_HLoop_carter3_4hi3dl_2024_05_30-14_21_18
# - 98_HLoop_carter14_release30_4hi3dl_2024_05_30-18_58_14
# - 99_HLoop_carter14_release30_4hi3dl_2024_05_30-19_03_04
# X_OFFSET: -2595.939208984375
# Y_OFFSET: 3445.929443359375

SENSOR_DATA_BAG=${1? "Please provide sensor data bag dir"}
GL_MAP_DIR=${2? "Please provide glmap dir"}

# Optional
OMAP_YAML_FILE=${3:-""}
X_OFFSET=${4:-0}
Y_OFFSET=${5:-0}
RAW_DATA_DIR=${6:-""}

BAG_NAME="$(basename $SENSOR_DATA_BAG)"
GL_DEBUG_DIR="$GL_MAP_DIR/results/$BAG_NAME"
mkdir -p $GL_DEBUG_DIR

# which launch file to use to start global loc
DO_MAPPER_IN_PERCEPTOR=false
LAUNCH_IN_PERCEPTOR=true
REPLAY_RATE="0.1"

function launch_global_localization_mapper_in_perceptor() {
    ros2 launch nova_carter_bringup perceptor.launch.py \
        stereo_camera_configuration:=front_configuration \
        disable_nvblox:=True \
        mode:=rosbag \
        disable_visual_global_localization:=False \
        rosbag:=$SENSOR_DATA_BAG \
        visual_global_localization_mode:=global_localization_mapper \
        visual_global_localization_map_dir:=$GL_MAP_DIR || true
}


function launch_global_loc_in_perceptor() {
    # Publish static_tf between vmap and omap
    ros2 run tf2_ros static_transform_publisher $X_OFFSET $Y_OFFSET 0 0 0 0 'map' 'omap' &
    static_tf_publisher_pid=$!

    VISUALIZATION_ARGS=""
    if [ ! -z $OMAP_YAML_FILE ]; then
        VISUALIZATION_ARGS="occupancy_map_yaml_file:=$OMAP_YAML_FILE "
        VISUALIZATION_ARGS+="omap_frame:=omap "
        VISUALIZATION_ARGS+="enable_3d_lidar:=True "
        VISUALIZATION_ARGS+="enable_point_cloud_filter:=True "
        VISUALIZATION_ARGS+="visual_global_localization_publish_map_to_base_tf:=True "
    fi

    ros2 launch nova_carter_bringup perceptor.launch.py \
        enable_cuvslam:=False \
        disable_nvblox:=True \
        disable_visual_global_localization:=False \
        visual_global_localization_map_dir:=$GL_MAP_DIR \
        stereo_camera_configuration:=front_configuration \
        visual_global_localization_debug_dir:=$GL_DEBUG_DIR \
        rosbag:=$SENSOR_DATA_BAG \
        mode:=rosbag \
        enable_continuous_localization:=True \
        replay_shutdown_on_exit:=True \
        replay_rate:=$REPLAY_RATE \
        ${OMAP_YAML_FILE:+"$VISUALIZATION_ARGS"} || true

    echo 'Stopping static tf publisher...'
    kill ${static_tf_publisher_pid} && wait ${static_tf_publisher_pid} || true
}

function launch_global_loc() {
    VISUALIZATION_ARGS=""
    if [ ! -z $OMAP_YAML_FILE ]; then
        VISUALIZATION_ARGS="omap_yaml_file:=$OMAP_YAML_FILE "
        VISUALIZATION_ARGS+="vmap_to_omap_x_offset:=$X_OFFSET "
        VISUALIZATION_ARGS+="vmap_to_omap_y_offset:=$Y_OFFSET "
        VISUALIZATION_ARGS+="enable_point_cloud_filter:=True "
    fi

    ros2 launch isaac_ros_visual_global_localization isaac_ros_visual_global_localization_e2e.launch.py \
        rosbag_path:=$SENSOR_DATA_BAG \
        map_dir:=$GL_MAP_DIR \
        debug_dir:=$GL_DEBUG_DIR \
        localization_mode:=visual_localization \
        enable_continuous_localization:=True \
        replay_shutdown_on_exit:=True \
        replay_rate:=$REPLAY_RATE \
        ${OMAP_YAML_FILE:+"$VISUALIZATION_ARGS"} \
        ${RAW_DATA_DIR:+"debug_map_raw_dir:=$RAW_DATA_DIR"} || true
}

if [ "$DO_MAPPER_IN_PERCEPTOR" == true ]; then
    launch_global_localization_mapper_in_perceptor
else
    echo "DO_MAPPER_IN_PERCEPTOR is false, will not run global localization mapper in perceptor"
fi

if [ "$LAUNCH_IN_PERCEPTOR" == true ]; then
    launch_global_loc_in_perceptor
else
    launch_global_loc
fi

GT_POSE_FILE=$SENSOR_DATA_BAG/timestamp_poses_tum.txt
if [ ! -f $GT_POSE_FILE ]; then
    wget https://pdx.s8k.io/v1/AUTH_team-isaac/inca/data_platform/benchmark/ground_truth_dataset/$BAG_NAME/timestamp_poses_tum.txt -O $GT_POSE_FILE
fi

# If GT file exists or is downloaded successfully
if [ -f $GT_POSE_FILE ]; then
    METRIC_DIR=$GL_DEBUG_DIR/eval
    mkdir -p $METRIC_DIR

    ${ISAAC_ROS_WS}/build/isaac_mapping_ros/isaac_mapping/tools/compute_pose_error_main \
        -gt_pose_file $GT_POSE_FILE \
        -gt_pose_type tum \
        -eval_pose_file $GL_DEBUG_DIR/loc_timestamp_poses_tum.txt \
        -eval_pose_type tum \
        -fit_gt_to_eval_tf_method "min_diff" \
        -output_dir $METRIC_DIR

    PLOT_TOOL=${ISAAC_ROS_WS}/build/isaac_mapping_ros/scripts/visual/metrics/plot_pose_error.py

    python $PLOT_TOOL \
        $METRIC_DIR/absolute_error.json \
        --output_prefix $METRIC_DIR/absolute_error \
        --plot_line --plot_hist --plot_location
    echo "Plots are generated in $METRIC_DIR"
else
    echo "GT file was not found"
fi
