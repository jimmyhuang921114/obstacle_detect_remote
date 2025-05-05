//ros2 msg
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/header.hpp>

// other lib
#include <random>
#include <climits>
#include <algorithm>
#include <math.h>
#include <mutex>

//voxel grid
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filiters/voxel_grid.h>
#include <pcl-1.12/filiters/approximate_voxel_grid.h>
#include <chrono>


using namespace std::chrono_literals;
using std::placeholders::_1;

class ObstacleCloudPointDetect : public rclcpp::Noode{
public:
    ObstacleCloudPointDetect(): Node("obstacle_cloud_point_detect"){
        decla
    }
}

private:
    void point_cloud_callback(const sensor_msgs::msg::Image::SharedPtr msg){
        std::lock_guard<std::mutex> lock(image_mutex_);
            if (color_image_.empty()) return;

            try {
                cv::Mat depth = cv_bridge::toCvCopy(msg)->image;
                cv::Mat debug_img = color_image_.clone();
            pcl::VoxelGrid<pcl::PointXYZI> sor;
            sor.setInputCloud(cloud);
            sor.setLeafSize(0.01f,0.01f,0.01f);
            sor.filiter(*cloud_filitered);

            }
        }