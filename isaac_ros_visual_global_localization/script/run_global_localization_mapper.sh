set -ex

WORK_PATH="${ISAAC_ROS_WS:?}"

# Mapping tracks
ROS_BAG_PATH=/mnt/nova_ssd/datasets/Robotics/hl4hl19_2024_04_05-16_05_35/hl4hl19_2024_04_05-16_05_35_0.mcap
WORKSPACE_DIR=/mnt/nova_ssd/datasets/Robotics/7-4-global-localization-mapper-test
CONFIG_DIR=$WORK_PATH/src/isaac_ros_mapping_and_localization/isaac_mapping_ros/isaac_mapping/configs
MAP_DIR=$WORKSPACE_DIR/map_dir

LOGS_FILE="$WORKSPACE_DIR/global_localization_mapper_logs.txt"
DEBUG_DIR=$WORKSPACE_DIR/debug_dir

rm -rf $LOGS_FILE
colcon build --symlink-install --packages-up-to isaac_ros_visual_global_localization
source ./install/setup.bash

ros2 launch isaac_ros_visual_global_localization isaac_ros_visual_global_localization_e2e.launch.py \
    rosbag_path:=$ROS_BAG_PATH \
    map_dir:=$MAP_DIR \
    debug_dir:=$DEBUG_DIR \
    config_dir:=$CONFIG_DIR \
    use_cuvslam:=True \
    localization_mode:=global_localization_mapper \
    2>&1 | tee -a "$LOGS_FILE"
