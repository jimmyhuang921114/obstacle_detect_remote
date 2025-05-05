#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <vector>

using std::placeholders::_1;

class XZVisualizer : public rclcpp::Node {
public:
    XZVisualizer() : Node("xz_visualizer") {
        // 参数配置
        declare_parameter<double>("max_z", 5.0);       // 最大显示距离
        declare_parameter<double>("x_scale", 100.0);   // X轴缩放系数 (像素/米)
        declare_parameter<double>("z_scale", 100.0);   // Z轴缩放系数 (像素/米)
        
        // 相机内参（根据实际标定修改）
        declare_parameter<double>("fx", 638.17);
        declare_parameter<double>("cx", 638.64);

        // 订阅深度图像
        depth_sub_ = create_subscription<sensor_msgs::msg::Image>(
            "/camera/depth", 10,
            std::bind(&XZVisualizer::depth_callback, this, _1));

        // 可视化图像发布
        viz_pub_ = create_publisher<sensor_msgs::msg::Image>("xz_viz", 10);

        // 初始化可视化画布
        viz_image_ = cv::Mat(600, 800, CV_8UC3, cv::Scalar(40, 40, 40));
        
        RCLCPP_INFO(get_logger(), "XZ平面可视化节点已启动");
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr viz_pub_;
    cv::Mat viz_image_;
    
    struct XZPoint {
        float x;
        float z;
    };

    void depth_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        try {
            cv::Mat depth_image = cv_bridge::toCvCopy(msg, msg->encoding)->image;
            std::vector<XZPoint> points = process_depth(depth_image);
            update_visualization(points);
            publish_viz_image(msg->header);
        }
        catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "处理深度图像错误: %s", e.what());
        }
    }

    std::vector<XZPoint> process_depth(const cv::Mat& depth_img) {
        std::vector<XZPoint> points;
        const int step = 2;  // 采样步长
        const float max_z = get_parameter("max_z").as_double();
        const float fx = get_parameter("fx").as_double();
        const float cx = get_parameter("cx").as_double();

        for (int v = 0; v < depth_img.rows; v += step) {
            for (int u = 0; u < depth_img.cols; u += step) {
                float z = depth_img.at<uint16_t>(v, u) / 1000.0f;
                
                if (z > 0.3 && z < max_z) {
                    float x = (u - cx) * z / fx;
                    points.push_back({x, z});
                }
            }
        }
        return points;
    }

    void update_visualization(const std::vector<XZPoint>& points) {
        // 清空画布
        viz_image_.setTo(cv::Scalar(40, 40, 40));
        
        // 绘制坐标系
        draw_coordinate_system();

        // 转换参数
        const float x_scale = get_parameter("x_scale").as_double();
        const float z_scale = get_parameter("z_scale").as_double();
        const int center_x = viz_image_.cols / 2;
        const int base_y = viz_image_.rows - 20;

        // 绘制所有点
        for (const auto& p : points) {
            int img_x = center_x + static_cast<int>(p.x * x_scale);
            int img_y = base_y - static_cast<int>(p.z * z_scale);
            
            if (img_x >= 0 && img_x < viz_image_.cols && 
                img_y >= 0 && img_y < viz_image_.rows) {
                cv::circle(viz_image_, cv::Point(img_x, img_y), 
                         2, cv::Scalar(0, 200, 255), -1);
            }
        }
    }

    void draw_coordinate_system() {
        const int center_x = viz_image_.cols / 2;
        const int base_y = viz_image_.rows - 20;
        const float max_z = get_parameter("max_z").as_double();
        const float z_scale = get_parameter("z_scale").as_double();

        // 绘制X轴（左右方向）
        cv::line(viz_image_, 
                cv::Point(0, base_y), 
                cv::Point(viz_image_.cols, base_y), 
                cv::Scalar(200, 200, 200), 1);

        // 绘制Z轴（深度方向）
        cv::line(viz_image_, 
                cv::Point(center_x, base_y), 
                cv::Point(center_x, 20), 
                cv::Scalar(200, 200, 200), 1);

        // 添加刻度
        for (float z = 1.0; z <= max_z; z += 1.0) {
            int y_pos = base_y - static_cast<int>(z * z_scale);
            cv::line(viz_image_, 
                    cv::Point(center_x - 5, y_pos),
                    cv::Point(center_x + 5, y_pos),
                    cv::Scalar(150, 150, 150), 1);
            cv::putText(viz_image_, std::to_string((int)z), 
                       cv::Point(center_x + 10, y_pos + 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.4,
                       cv::Scalar(200, 200, 200), 1);
        }
    }

    void publish_viz_image(const std_msgs::msg::Header& header) {
        auto img_msg = cv_bridge::CvImage(
            header, "bgr8", viz_image_).toImageMsg();
        viz_pub_->publish(*img_msg);
        
        // 本地显示
        cv::imshow("XZ Plane Visualization", viz_image_);
        cv::waitKey(1);
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<XZVisualizer>());
    rclcpp::shutdown();
    return 0;
}