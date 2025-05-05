#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <cv_bridge/cv_bridge.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>  // 統計濾波
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/centroid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <Eigen/Dense>

using namespace std::chrono_literals;

class DepthProcessor : public rclcpp::Node {
public:
  DepthProcessor() : Node("depth_processor") {
    // 參數系統 (可通過 ros2 param 動態調整)
    declare_parameter("fx", 554.25);
    declare_parameter("fy", 554.25);
    declare_parameter("cx", 320.5);
    declare_parameter("cy", 240.5);
    declare_parameter("min_depth", 0.5);
    declare_parameter("max_depth", 5.0);
    declare_parameter("ground_threshold", 0.15);
    declare_parameter("cluster_tolerance", 0.3);
    declare_parameter("min_cluster_size", 50);

    // ROS 接口配置
    depth_sub_ = create_subscription<sensor_msgs::msg::Image>(
      "/camera/depth", 10,
      [this](const sensor_msgs::msg::Image::SharedPtr msg) { depth_callback(msg); });
    
    ground_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>("/ground_params", 10);
    obstacle_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>("/obstacles", 10);
    cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("/filtered_cloud", 10);

    // PCL 算法初始化
    init_pcl_components();
  }

private:
  void init_pcl_components() {
    // RANSAC 地面分割配置
    seg_.setOptimizeCoefficients(true);
    seg_.setModelType(pcl::SACMODEL_PLANE);
    seg_.setMethodType(pcl::SAC_RANSAC);
    seg_.setMaxIterations(1000);  // 提高迭代次數適應複雜地形
    seg_.setDistanceThreshold(get_parameter("ground_threshold").as_double());

    // 統計濾波配置 (去除浮動噪聲)
    sor_.setMeanK(50);
    sor_.setStddevMulThresh(1.0);

    // 聚類提取配置
    ec_.setClusterTolerance(get_parameter("cluster_tolerance").as_double());
    ec_.setMinClusterSize(get_parameter("min_cluster_size").as_int());
    ec_.setMaxClusterSize(25000);
  }

  void depth_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
    try {
      // ==== 深度圖轉換 ====
      cv::Mat depth_img = cv_bridge::toCvCopy(msg, "16UC1")->image;
      
      // ==== 點雲生成 ====
      auto cloud = generate_pointcloud(depth_img);
      if (cloud->empty()) {
        RCLCPP_WARN(get_logger(), "點雲生成失敗: 無有效深度數據");
        return;
      }

      // ==== 數據預處理 ====
      pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
      preprocess_pointcloud(cloud, filtered_cloud);

      // ==== 地面分割 ====
      auto [ground_coeffs, obstacle_cloud] = detect_ground(filtered_cloud);
      if (!ground_coeffs) {
        RCLCPP_ERROR(get_logger(), "地面檢測失敗，跳過本幀處理");
        return;
      }

      // ==== 障礙物處理 ====
      auto clusters = detect_obstacles(obstacle_cloud);
      publish_results(ground_coeffs, clusters, obstacle_cloud, msg->header);

    } catch (const cv_bridge::Exception& e) {
      RCLCPP_ERROR(get_logger(), "CV橋接異常: %s", e.what());
    } catch (const std::exception& e) {
      RCLCPP_ERROR(get_logger(), "處理異常: %s", e.what());
    }
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr generate_pointcloud(const cv::Mat& depth_img) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    const float fx = get_parameter("fx").as_double();
    const float fy = get_parameter("fy").as_double();
    const float cx = get_parameter("cx").as_double();
    const float cy = get_parameter("cy").as_double();
    const float min_depth = get_parameter("min_depth").as_double();
    const float max_depth = get_parameter("max_depth").as_double();

    cloud->width = depth_img.cols;
    cloud->height = depth_img.rows;
    cloud->is_dense = false;
    cloud->points.resize(cloud->width * cloud->height);

    for (int v = 0; v < depth_img.rows; ++v) {
      for (int u = 0; u < depth_img.cols; ++u) {
        const uint16_t depth_value = depth_img.at<uint16_t>(v, u);
        auto& point = cloud->points[v * depth_img.cols + u];

        if (depth_value == 0) {
          point.x = point.y = point.z = std::numeric_limits<float>::quiet_NaN();
          continue;
        }

        const float z = depth_value / 1000.0f;
        if (z < min_depth || z > max_depth) {
          point.x = point.y = point.z = std::numeric_limits<float>::quiet_NaN();
          continue;
        }

        point.x = (u - cx) * z / fx;
        point.y = (v - cy) * z / fy;
        point.z = z;
      }
    }
    return cloud;
  }

  void preprocess_pointcloud(pcl::PointCloud<pcl::PointXYZ>::Ptr input,
                            pcl::PointCloud<pcl::PointXYZ>::Ptr output) {
    // 移除 NaN 點
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*input, *output, indices);

    // 統計離群點濾波
    sor_.setInputCloud(output);
    sor_.filter(*output);

    if (output->empty()) {
      throw std::runtime_error("預處理後點雲為空");
    }
  }

  std::pair<pcl::ModelCoefficients::Ptr, pcl::PointCloud<pcl::PointXYZ>::Ptr>
  detect_ground(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

    seg_.segment(*inliers, *coefficients);

    if (inliers->indices.empty()) {
        RCLCPP_WARN(get_logger(), "平面偵測失敗：沒有 inliers");
        return {nullptr, nullptr};
    }


    // 提取障礙物
    pcl::PointCloud<pcl::PointXYZ>::Ptr obstacle_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);
    extract.setNegative(true);
    extract.filter(*obstacle_cloud);

    return {coefficients, obstacle_cloud};
  }

  std::vector<pcl::PointIndices> detect_obstacles(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
    if (cloud->empty()) return {};

    // 構建搜索樹
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud);

    // 執行聚類
    std::vector<pcl::PointIndices> cluster_indices;
    ec_.setSearchMethod(tree);
    ec_.setInputCloud(cloud);
    ec_.extract(cluster_indices);

    return cluster_indices;
  }

  void publish_results(pcl::ModelCoefficients::Ptr ground_coeffs,
                      const std::vector<pcl::PointIndices>& clusters,
                      pcl::PointCloud<pcl::PointXYZ>::Ptr obstacle_cloud,
                      const std_msgs::msg::Header& header) {
    // 發布地面參數 (平面方程 ax+by+cz+d=0)
    std_msgs::msg::Float32MultiArray ground_msg;
    ground_msg.data.assign(ground_coeffs->values.begin(), ground_coeffs->values.end());
    ground_pub_->publish(ground_msg);

    // 發布障礙物質心
    std_msgs::msg::Float32MultiArray obstacles_msg;
    for (const auto& cluster : clusters) {
      Eigen::Vector4f centroid;
      pcl::compute3DCentroid(*obstacle_cloud, cluster.indices, centroid);
      obstacles_msg.data.insert(obstacles_msg.data.end(), {centroid[0], centroid[1], centroid[2]});
    }
    obstacle_pub_->publish(obstacles_msg);

    // 發布處理後的點雲
    sensor_msgs::msg::PointCloud2 output;
    pcl::toROSMsg(*obstacle_cloud, output);
    output.header = header;
    cloud_pub_->publish(output);
  }

  // ROS 成員
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr ground_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr obstacle_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;

  // PCL 組件
  pcl::SACSegmentation<pcl::PointXYZ> seg_;
  pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor_;
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DepthProcessor>());
  rclcpp::shutdown();
  return 0;
}