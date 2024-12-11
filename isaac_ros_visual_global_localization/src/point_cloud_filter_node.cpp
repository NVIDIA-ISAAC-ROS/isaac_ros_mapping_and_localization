#include "isaac_ros_visual_global_localization/point_cloud_filter_node.hpp"
#include "tf2_sensor_msgs/tf2_sensor_msgs.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace visual_global_localization
{

// Convert ROS Header timestamp to timestamp seconds
double ROSTimestampToSeconds(
  const builtin_interfaces::msg::Time & stamp)
{
  return stamp.sec + static_cast<double>(stamp.nanosec) / 1e9;
}

PointCloudFilterNode::PointCloudFilterNode(const rclcpp::NodeOptions & options)
: Node("point_cloud_filter", options)
{
  getParameters();
  subscribeToTopics();
  advertiseTopics();
}

void PointCloudFilterNode::getParameters()
{
  target_frame_ = this->declare_parameter<std::string>("target_frame", "omap");
  cloud_queue_size_ = this->declare_parameter<int>("cloud_queue_size", 100);
  time_thresh_sec_ = this->declare_parameter<float>("time_thresh_sec", 0.05);
}

void PointCloudFilterNode::subscribeToTopics()
{
  cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
    "point_cloud", 10, std::bind(
      &PointCloudFilterNode::pointCloudCallback, this,
      std::placeholders::_1));
  pose_sub_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
    "pose", 10,
    std::bind(&PointCloudFilterNode::poseCallback, this, std::placeholders::_1));

  // Initialize tf buffer
  tf_buffer_ =
    std::make_unique<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ =
    std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

}

void PointCloudFilterNode::advertiseTopics()
{
  filtered_cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
    "filtered_point_cloud", 10);
}

void PointCloudFilterNode::pointCloudCallback(
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
{
  cloud_queue_.push_back(msg);
  while (cloud_queue_.size() > cloud_queue_size_) {
    cloud_queue_.pop_front();
  }
}

void PointCloudFilterNode::poseCallback(
  const geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr msg)
{
  double pose_time = ROSTimestampToSeconds(msg->header.stamp);
  RCLCPP_INFO(get_logger(), "Receive pose_msg at %.2f", pose_time);

  while (!cloud_queue_.empty()) {
    const auto & cloud = cloud_queue_.front();
    double cloud_time = ROSTimestampToSeconds(cloud->header.stamp);
    double time_diff = cloud_time - pose_time;
    if (time_diff < -time_thresh_sec_) {
      cloud_queue_.pop_front();
      RCLCPP_DEBUG(
        get_logger(),
        "Cloud is outdated. time-diff: %.2f, cloud time: %.2f, pose time: %.2f",
        time_diff,
        cloud_time, pose_time);
    } else if (abs(time_diff) < time_thresh_sec_) {
      sensor_msgs::msg::PointCloud2 out_cloud;
      if (!transformCloud(cloud, msg->header.stamp, out_cloud)) {
        break;
      }
      filtered_cloud_pub_->publish(out_cloud);
      cloud_queue_.pop_front();
      RCLCPP_INFO(
        get_logger(),
        "Time diff within threshold: %.2f, publish cloud at: %.2f, pose time: %.2f", time_diff, cloud_time,
        pose_time);
    } else {
      RCLCPP_DEBUG(
        get_logger(),
        "Pose is outdated, time-diff %.2f, cloud time: %.2f, pose time: %.2f", time_diff, cloud_time,
        pose_time);
      break;
    }
  }
}

bool PointCloudFilterNode::transformCloud(
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr & in_cloud,
  const builtin_interfaces::msg::Time & pose_stamp,
  sensor_msgs::msg::PointCloud2 & out_cloud) const
{
  geometry_msgs::msg::TransformStamped tf_msg;
  try {
    tf_msg = tf_buffer_->lookupTransform(
      target_frame_, in_cloud->header.frame_id,
      rclcpp::Time(pose_stamp.sec, pose_stamp.nanosec));
  } catch (const tf2::TransformException & ex) {
    RCLCPP_WARN(
      this->get_logger(), "Could not transform between %s and %s at %.2f",
      target_frame_.c_str(), in_cloud->header.frame_id.c_str(), ROSTimestampToSeconds(pose_stamp));
    return false;
  }
  tf2::doTransform(*(in_cloud.get()), out_cloud, tf_msg);
  out_cloud.header.frame_id = target_frame_;
  return true;
}


} // namespace visual_global_localization
} // namespace isaac_ros
} // namespace nvidia

// Register as a component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(
  nvidia::isaac_ros::visual_global_localization::PointCloudFilterNode)
