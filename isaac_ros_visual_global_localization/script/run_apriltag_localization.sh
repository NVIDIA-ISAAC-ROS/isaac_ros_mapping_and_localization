set -ex

# Mapping tracks
ROS_BAG_PATH=/home/admin/datasets/Robotics/hl4hl19_2024_04_05-16_05_35/rosbag/hl4hl19_2024_04_05-16_05_35_0.mcap
WORKSPACE_DIR=/home/admin/datasets/Robotics/6-6-test-data
GT_POSE_FILE=/home/admin/datasets/Robotics/6-5-test-data/alignment_poses_19782/19782/alignment_id-19782_track_id-233938_pose_file_tum.txt
X_OFFSET="-2594.15966796875"
Y_OFFSET="3445.883544921875"

MAP_DIR=$WORKSPACE_DIR/april_tag_map
OMAP_YAML_FILE=$WORKSPACE_DIR/omap/map_metadata.yaml
LOGS_FILE="$MAP_DIR/apriltag_localization_log.txt"

LOCALIZATION_DIR=$WORKSPACE_DIR/localization_233939
METRIC_DIR=$LOCALIZATION_DIR/eval

colcon build --symlink-install --packages-up-to isaac_ros_visual_global_localization
source ./install/setup.bash

ros2 launch isaac_ros_visual_global_localization isaac_ros_visual_global_localization_e2e.launch.py \
    rosbag_path:=$ROS_BAG_PATH \
    map_dir:=$MAP_DIR \
    debug_dir:=$LOCALIZATION_DIR \
    omap_yaml_file:=$OMAP_YAML_FILE \
    localization_mode:=apriltag_localization \
    vmap_to_omap_x_offset:=$X_OFFSET \
    vmap_to_omap_y_offset:=$Y_OFFSET \
    enable_continuous_localization:=True \
    2>&1 | tee -a "$LOGS_FILE"

# ./install/isaac_ros_visual_global_localization/bin/isaac_mapping/compute_pose_error_main \
#     -gt_pose_file $GT_POSE_FILE \
#     -gt_pose_type tum \
#     -eval_pose_file $LOCALIZATION_DIR/loc_timestamp_poses_tum.txt \
#     -eval_pose_type tum \
#     -output_dir $METRIC_DIR

# PLOT_TOOL=./src/isaac_ros_mapping_and_localization/isaac_mapping_ros/isaac_mapping/scripts/visual/metrics/plot_pose_error.py

# python $PLOT_TOOL \
#     $METRIC_DIR/absolute_error.json \
#     --output_prefix $METRIC_DIR/absolute_error \
#     --plot_line --plot_hist --plot_location
