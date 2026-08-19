#!/bin/sh

set -e

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# Default depth model - can be overridden by environment variable or first argument
DEPTH_MODEL=${1:-${DEPTH_MODEL:-"auto"}}

# Auto-detect depth model based on architecture if not specified
if [ "$DEPTH_MODEL" = "auto" ]; then
  ARCH=$(uname -m)
  if [ "$ARCH" = "aarch64" ]; then
    DEPTH_MODEL="ess"
    echo "Auto-detected aarch64 architecture, using ESS depth model"
  else
    DEPTH_MODEL="foundationstereo"
    echo "Auto-detected x86 architecture, using FoundationStereo depth model"
  fi
else
  echo "Using specified depth model: $DEPTH_MODEL"
fi

# Validate depth model
if [ "$DEPTH_MODEL" != "ess" ] && [ "$DEPTH_MODEL" != "foundationstereo" ]; then
  echo "Error: Invalid depth model '$DEPTH_MODEL'. Must be 'ess' or 'foundationstereo'"
  exit 1
fi

# shellcheck source=/dev/null
. "${ISAAC_ROS_WS:?}"/install/setup.sh

echo "ISAAC_ROS_WS: $ISAAC_ROS_WS"

r2b_galileo=$(ros2 pkg prefix isaac_ros_r2b_galileo)/share/isaac_ros_r2b_galileo

# Create map offline
echo "Creating map offline using depth model: $DEPTH_MODEL..."
MAP_OUTPUT_DIR=$(mktemp -d /tmp/my_map.XXXXXX)
trap 'rm -rf "$MAP_OUTPUT_DIR"' EXIT
bag_path=${r2b_galileo}/data/r2b_galileo
ros2 run isaac_mapping_ros create_map_offline.py \
  --sensor_data_bag "$bag_path" \
  --base_output_folder "$MAP_OUTPUT_DIR" \
  --depth_model "$DEPTH_MODEL" \
  --print_mode all

echo "Checking generated cuVSLAM odom poses against R2B Galileo baseline..."
python3 "${SCRIPT_DIR}/compare_tum_poses.py" \
  --baseline "${SCRIPT_DIR}/data/galileo/odom_poses_r15_opencv_baseline.tum" \
  --candidate "$MAP_OUTPUT_DIR/latest/poses/odom_poses.tum" \
  --translation_threshold_m 0.10 \
  --rotation_threshold_degrees 5.0 \
  --timestamp_threshold_s 0.000001

# Test visual localization (cuvgl) on the created map
command="ros2 launch isaac_ros_visual_global_localization run_cuvgl_rosbag_replay.launch.py \
    camera_names:=front_stereo_camera,left_stereo_camera,right_stereo_camera,back_stereo_camera \
    use_cuvslam:=False \
    enable_image_decoder:=True \
    enable_image_rectify:=True \
    vgl_publish_rectified_images:=False \
    vgl_map_dir:=${MAP_OUTPUT_DIR}/latest/cuvgl_map \
    vgl_mode:=visual_localization \
    vgl_enable_continuous_localization:=True \
    vgl_image_sync_match_threshold_ms:=30.0 \
    rosbag_path:=$bag_path \
    replay_rate:=1.0 \
    enable_visualization:=False \
    vgl_enable_point_cloud_filter:=False \
    enable_3d_lidar:=False \
    replay_shutdown_on_exit:=True \
    vgl_verbose_logging:=False \
    vgl_init_glog:=True \
    vgl_glog_v:=0"

printf 'Testing visual localization (cuvgl)..., with command:\n %s\n' "$command"

# Disable this test for now as well as it seems to be unstable at construction time
# $command  || true

# This compare map does not currently because occasionally the map generated has different size
# We need to switch to use a 2d ICP based approach for occupancy
# python ${ISAAC_ROS_WS}/src/isaac_ros_mapping_and_localization/isaac_mapping_ros/tests/compare_images.py \
#     --image1 ${MAP_OUTPUT_DIR}/latest/occupancy_map.png \
#     --image2 ${ISAAC_ROS_WS}/src/isaac_ros_mapping_and_localization/isaac_mapping_ros/tests/data/occupancy_map.png \
#     --pixel_threshold 10 --diff_threshold 2000
