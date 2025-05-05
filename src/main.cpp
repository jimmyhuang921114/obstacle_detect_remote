#include "new_obs_filiter/plane_estimator.hpp"
#include "new_obs_filiter/obstacle_detector.hpp"
#include "new_obs_filiter/visualization.hpp"
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <mutex>
#include <chrono>

using namespace std::chrono_literals;

namespace new_obs_filiter {

class ObstacleDetectorNode : public rclcpp::Node {
public:
    ObstacleDetectorNode() 
    : Node("obstacle_detector"),
      color_img_(cv::Mat(480, 640, CV_8UC3, cv::Scalar(0, 0, 0))) 
    {
        // 宣告 ROS 參數
        declare_ros_parameters();
        
        // 初始化模組
        plane_estimator_ = std::make_unique<PlaneEstimator>(init_plane_params());
        obstacle_detector_ = std::make_unique<ObstacleDetector>(*plane_estimator_, init_obstacle_params());
        visualizer_ = std::make_unique<Visualizer>(init_viz_params());

        // 建立訂閱者
        auto qos = rclcpp::SensorDataQoS();
        depth_sub_ = create_subscription<sensor_msgs::msg::Image>(
            "/camera/depth", qos,
            [this](sensor_msgs::msg::Image::ConstSharedPtr msg) {
                depth_callback(msg);
            });
            
        color_sub_ = create_subscription<sensor_msgs::msg::Image>(
            "/camera/color", qos,
            [this](sensor_msgs::msg::Image::ConstSharedPtr msg) {
                color_callback(msg);
            });

        // 建立 debug 圖像 publisher
        debug_image_pub_ = create_publisher<sensor_msgs::msg::Image>("/debug_image", 10);

        // 初始化 OpenCV 窗口（可保留）
        try {
            cv::namedWindow("Obstacle Detection", cv::WINDOW_NORMAL);
            cv::resizeWindow("Obstacle Detection", 1280, 720);
        } catch (const cv::Exception& e) {
            RCLCPP_ERROR(get_logger(), "OpenCV GUI init failed: %s", e.what());
        }
    }

private:
    // ROS components
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr color_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_image_pub_;

    // modules
    std::unique_ptr<PlaneEstimator> plane_estimator_;
    std::unique_ptr<ObstacleDetector> obstacle_detector_;
    std::unique_ptr<Visualizer> visualizer_;
    
    // image data
    cv::Mat color_img_;
    std::mutex img_mutex_;

    // 宣告 ROS2 參數
    void declare_ros_parameters() {
        rcl_interfaces::msg::ParameterDescriptor plane_desc;
        plane_desc.set__description("Plane fitting threshold in meters");
        declare_parameter<double>("plane_threshold", 0.05, plane_desc);

        rcl_interfaces::msg::FloatingPointRange range;
        range.from_value = 0.1;
        range.to_value = 5.0;
        rcl_interfaces::msg::ParameterDescriptor range_desc;
        range_desc.set__floating_point_range({range});
        declare_parameter<double>("max_detection_distance", 5.0, range_desc);
    }

    PlaneEstimator::Parameters init_plane_params() {
        PlaneEstimator::Parameters p;
        p.ransac_iterations = 300;
        p.plane_threshold = 0.05;
        p.normal_constraint = 0.7;
        return p;
    }

    ObstacleDetector::Parameters init_obstacle_params() {
        ObstacleDetector::Parameters p;
        p.max_detection_distance = get_parameter("max_detection_distance").as_double();
        p.obstacle_height_threshold = 0.15f;
        p.ground_margin = 0.05f;
        p.fx = 610.0;
        p.fy = 610.0;
        p.cx = 320.0;
        p.cy = 240.0;
        return p;
    }

    Visualizer::Parameters init_viz_params() {
        Visualizer::Parameters p;
        p.info_font_scale = 0.8;
        p.ground_alpha = 0.4;
        p.info_text_color = cv::Scalar(200, 200, 0);
        p.fps_text_color = cv::Scalar(255, 255, 0);
        return p;
    }

    void color_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
        std::lock_guard<std::mutex> lock(img_mutex_);
        try {
            color_img_ = cv_bridge::toCvCopy(msg, "bgr8")->image;
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(get_logger(), "CV Bridge error: %s", e.what());
        }
    }

    void depth_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
        auto start = std::chrono::high_resolution_clock::now();
        
        std::lock_guard<std::mutex> lock(img_mutex_);
        if (color_img_.empty()) return;

        try {
            cv::Mat depth = cv_bridge::toCvCopy(msg)->image;
            cv::Mat debug_img = color_img_.clone();

            const float depth_scale = 0.001f;
            int obstacle_count = obstacle_detector_->detect(depth, debug_img, depth_scale);
            
            double fps = 0.0;
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            if (duration.count() > 0) {
                fps = 1e9 / duration.count();
            }

            if (obstacle_detector_ && plane_estimator_) {
                const int ground_row = obstacle_detector_->get_ground_row();
                const cv::Vec4f plane = plane_estimator_->get_average_plane();

                visualizer_->visualize(
                    debug_img,
                    ground_row,
                    obstacle_count,
                    plane,
                    fps,
                    depth.rows,
                    debug_img.rows
                );

                // 發佈 debug image topic
                auto debug_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", debug_img).toImageMsg();
                debug_image_pub_->publish(*debug_msg);
            }
        } catch (const std::exception& e) {
            RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 1000, 
                "Processing error: %s", e.what());
        }
    }
};

} // namespace new_obs_filiter

// main函式
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<new_obs_filiter::ObstacleDetectorNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
