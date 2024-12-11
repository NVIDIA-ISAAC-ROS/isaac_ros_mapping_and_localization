#!/bin/bash
set -ex

# This script runs visual localization on a rosbag in replay. It also tries to get
# GT pose for the bag and runs evaluation.
# Usage:
# bash src/isaac_ros_mapping_and_localization/isaac_ros_visual_global_localization/script/run_visual_localization.sh
#   --sensor_data_bag=<sensor_data_bag_path> --glmap_dir=<glmap_dir> --omap_yaml_file=<omap_yaml_file>

# Default values for optional parameters
OMAP_YAML_FILE=""
X_OFFSET="0.0"
Y_OFFSET="0.0"
RAW_DATA_DIR=""
LAUNCH_IN_PERCEPTOR=true
REPLAY_RATE="0.1"
IMAGE_SYNC_MATCH_THRESHOLD_MS="3.0"
STEREO_CAMERA_CONFIGURATION="front_back_left_right_vgl_configuration"
CAMERA_NAMES="front_back_left_right"
VERBOSE="False"
INIT_GLOG="False"
GLOG_V=0
DEBUG_MAP_RAW_DIR=""

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
    --sensor_data_bag=*)
        SENSOR_DATA_BAG="${1#*=}"
        shift
        ;;
    --glmap_dir=*)
        GL_MAP_DIR="${1#*=}"
        shift
        ;;
    --omap_yaml_file=*)
        OMAP_YAML_FILE="${1#*=}"
        shift
        ;;
    --x_offset=*)
        X_OFFSET="${1#*=}"
        shift
        ;;
    --y_offset=*)
        Y_OFFSET="${1#*=}"
        shift
        ;;
    --raw_data_dir=*)
        RAW_DATA_DIR="${1#*=}"
        shift
        ;;
    --launch_in_perceptor=*)
        LAUNCH_IN_PERCEPTOR="${1#*=}"
        shift
        ;;
    --replay_rate=*)
        REPLAY_RATE="${1#*=}"
        shift
        ;;
    --image_sync_match_threshold_ms=*)
        IMAGE_SYNC_MATCH_THRESHOLD_MS="${1#*=}"
        shift
        ;;
    --stereo_camera_configuration=*)
        STEREO_CAMERA_CONFIGURATION="${1#*=}"
        shift
        ;;
    --camera_names=*)
        CAMERA_NAMES="${1#*=}"
        shift
        ;;
    --verbose=*)
        VERBOSE="${1#*=}"
        shift
        ;;
    --init_glog=*)
        INIT_GLOG="${1#*=}"
        shift
        ;;
    --glog_v=*)
        GLOG_V="${1#*=}"
        shift
        ;;
    --debug_map_raw_dir=*)
        DEBUG_MAP_RAW_DIR="${1#*=}"
        shift
        ;;
    *)
        echo "Unknown parameter passed: $1"
        exit 1
        ;;
    esac
done

# Check for mandatory parameters
if [ -z "$SENSOR_DATA_BAG" ] || [ -z "$GL_MAP_DIR" ]; then
    echo "Error: Missing required parameters."
    echo "Usage: $0 --sensor_data_bag=xxx --glmap_dir=xxx [optional parameters]"
    exit 1
fi

BAG_NAME="$(basename $SENSOR_DATA_BAG)"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
GL_DEBUG_DIR=${GL_MAP_DIR}/results/${TIMESTAMP}_${BAG_NAME}
LATEST_RESULT=${GL_MAP_DIR}/results/latest
mkdir -p $GL_DEBUG_DIR
rm -rf $LATEST_RESULT
ln -s $GL_DEBUG_DIR $LATEST_RESULT

function launch_global_loc_in_perceptor() {
    # Publish static_tf between vmap and omap
    ros2 run tf2_ros static_transform_publisher $X_OFFSET $Y_OFFSET 0 0 0 0 'map' 'omap' &
    static_tf_publisher_pid=$!

    VISUALIZATION_ARGS=""
    if [ ! -z "$OMAP_YAML_FILE" ]; then
        VISUALIZATION_ARGS="occupancy_map_yaml_file:=$OMAP_YAML_FILE "
        VISUALIZATION_ARGS+="omap_frame:=omap "
        VISUALIZATION_ARGS+="enable_3d_lidar:=True "
        VISUALIZATION_ARGS+="vgl_enable_point_cloud_filter:=True "
        VISUALIZATION_ARGS+="vgl_publish_map_to_base_tf:=True "
    fi

    RAW_DATA_DIR_ARGS=""
    if [ ! -z "${DEBUG_MAP_RAW_DIR}" ]; then
        RAW_DATA_DIR_ARGS+="vgl_debug_map_raw_dir:=$DEBUG_MAP_RAW_DIR "
    fi

    ros2 launch nova_carter_bringup perceptor.launch.py \
        stereo_camera_configuration:=$STEREO_CAMERA_CONFIGURATION \
        enable_cuvslam:=False \
        disable_nvblox:=True \
        disable_vgl:=False \
        vgl_map_dir:=$GL_MAP_DIR \
        vgl_debug_dir:=$GL_DEBUG_DIR \
        vgl_mode:=visual_localization \
        vgl_enable_continuous_localization:=True \
        vgl_image_sync_match_threshold_ms:=$IMAGE_SYNC_MATCH_THRESHOLD_MS \
        rosbag:=$SENSOR_DATA_BAG \
        mode:=rosbag \
        replay_rate:=$REPLAY_RATE \
        replay_shutdown_on_exit:=True \
        use_foxglove_whitelist:=False \
        vgl_verbose_logging:=$VERBOSE \
        vgl_init_glog:=$INIT_GLOG \
        vgl_glog_v:=$GLOG_V \
        ${VISUALIZATION_ARGS} ${RAW_DATA_DIR_ARGS} || true

    echo 'Stopping static tf publisher...'
    kill ${static_tf_publisher_pid} && wait ${static_tf_publisher_pid} || true
}

function launch_global_loc() {
    VISUALIZATION_ARGS=""
    if [ ! -z "$OMAP_YAML_FILE" ]; then
        VISUALIZATION_ARGS="occupancy_map_yaml_file:=$OMAP_YAML_FILE "
        VISUALIZATION_ARGS+="map_to_omap_x_offset:=$X_OFFSET "
        VISUALIZATION_ARGS+="map_to_omap_y_offset:=$Y_OFFSET "
        VISUALIZATION_ARGS+="enable_3d_lidar:=True "
        VISUALIZATION_ARGS+="vgl_enable_point_cloud_filter:=True "
        VISUALIZATION_ARGS+="vgl_publish_map_to_base_tf:=True "
    fi

    RAW_DATA_DIR_ARGS=""
    if [ ! -z "${DEBUG_MAP_RAW_DIR}" ]; then
        RAW_DATA_DIR_ARGS+="vgl_debug_map_raw_dir:=$DEBUG_MAP_RAW_DIR "
    fi

    ros2 launch isaac_ros_visual_global_localization run_cuvgl_rosbag_replay.launch.py \
        camera_names:=$CAMERA_NAMES \
        use_cuvslam:=False \
        enable_image_decoder:=True \
        enable_image_rectify:=True \
        enable_format_converter:=False \
        vgl_publish_rectified_images:=False \
        vgl_map_dir:=$GL_MAP_DIR \
        vgl_debug_dir:=$GL_DEBUG_DIR \
        vgl_mode:=visual_localization \
        vgl_enable_continuous_localization:=True \
        vgl_image_sync_match_threshold_ms:=$IMAGE_SYNC_MATCH_THRESHOLD_MS \
        rosbag_path:=$SENSOR_DATA_BAG \
        replay_rate:=$REPLAY_RATE \
        replay_shutdown_on_exit:=True \
        vgl_verbose_logging:=$VERBOSE \
        vgl_init_glog:=$INIT_GLOG \
        vgl_glog_v:=$GLOG_V \
        ${VISUALIZATION_ARGS} ${RAW_DATA_DIR_ARGS} || true
}

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
        -output_dir $METRIC_DIR \
        --algorithm_name VGL

    PLOT_TOOL=${ISAAC_ROS_WS}/build/isaac_mapping_ros/scripts/visual/metrics/plot_pose_error.py

    python $PLOT_TOOL \
        $METRIC_DIR/absolute_error.json \
        --output_prefix $METRIC_DIR/absolute_error \
        --plot_line --plot_hist --plot_location

    ${ISAAC_ROS_WS}/build/isaac_mapping_ros/isaac_mapping/tools/extract_image_retrieval_result_info \
        -loc_results_file $GL_DEBUG_DIR/loc_results.json \
        -gt_pose_tum_file $GT_POSE_FILE \
        -map_keyframe_metadata_file $GL_MAP_DIR/keyframes/frames_meta.pb.txt \
        -algorithm_name "bow" \
        -output_file $METRIC_DIR/image_retrieval_result_info.json

    python ${ISAAC_ROS_WS}/build/isaac_mapping_ros/scripts/visual/metrics/plot_image_retrieval_result_info.py \
        $METRIC_DIR/image_retrieval_result_info.json \
        --output_prefix $METRIC_DIR/image_retrieval_result \
        --plot_hist --plot_scatter --plot_kf_distance_to_loc_error

    echo "Plots are generated in $METRIC_DIR"
else
    echo "GT file was not found"
fi
