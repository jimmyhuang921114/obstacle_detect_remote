#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <mutex>
#include <deque>
#include <chrono>

using namespace std::chrono_literals;
using std::placeholders::_1;

class ObstacleDetector : public rclcpp::Node {
public:
    ObstacleDetector() : Node("obstacle_detector") {
        // 参数声明
        declare_parameter<double>("plane_threshold", 0.05);
        declare_parameter<int>("ransac_iterations", 500);
        declare_parameter<double>("min_ground_height", 0.3);
        declare_parameter<double>("ground_tolerance", 0.07);
        declare_parameter<double>("max_obstacle_height", 1.2);
        declare_parameter<double>("dynamic_factor", 0.2);
        declare_parameter<double>("normal_constraint", 0.8);
        declare_parameter<double>("min_distance", 0.5);
        declare_parameter<double>("max_distance", 5.0);
        declare_parameter<double>("roi_width", 0.8);

        // 相机内参
        fx_ = 610.0f;
        fy_ = 610.0f;
        cx_ = 320.0f;
        cy_ = 240.0f;

        // 订阅器
        depth_sub_ = create_subscription<sensor_msgs::msg::Image>(
            "/farm/depth", 10, std::bind(&ObstacleDetector::depth_callback, this, _1));
        color_sub_ = create_subscription<sensor_msgs::msg::Image>(
            "/farm/color", 10, std::bind(&ObstacleDetector::color_callback, this, _1));

        // 初始化显示窗口
        cv::namedWindow("Obstacle Detection", cv::WINDOW_NORMAL);
        cv::resizeWindow("Obstacle Detection", 1280, 720);

        RCLCPP_INFO(get_logger(), "Obstacle Detector Initialized");
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_, color_sub_;
    cv::Mat color_image_, U_, V_;
    std::mutex data_mutex_;
    float fx_, fy_, cx_, cy_;
    cv::Vec4f current_plane_{0,0,1,0};
    std::deque<cv::Vec4f> plane_history_;
    cv::Size last_size_{0,0};  // 新增：记录最后处理的图像尺寸

    void check_and_update_uv(const cv::Size& size) {  // 新增：动态尺寸检查
        if (size != last_size_) {
            precompute_uv_mats(size);
            last_size_ = size;
            RCLCPP_INFO(get_logger(), "Updated UV matrices for size: %dx%d", size.width, size.height);
        }
    }

    void precompute_uv_mats(const cv::Size& size) {
        U_.create(size, CV_32F);
        V_.create(size, CV_32F);
        for (int v = 0; v < size.height; ++v) {
            for (int u = 0; u < size.width; ++u) {
                U_.at<float>(v, u) = (u - cx_) / fx_;
                V_.at<float>(v, u) = (v - cy_) / fy_;
            }
        }
    }

    void depth_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        auto start = std::chrono::high_resolution_clock::now();
        std::lock_guard<std::mutex> lock(data_mutex_);

        try {
            cv::Mat depth = cv_bridge::toCvCopy(msg)->image;
            
            // 关键修改：动态检查并更新UV矩阵
            check_and_update_uv(depth.size());

            if (color_image_.empty()) return;
            cv::Mat debug_img = color_image_.clone();

            // 地面检测
            int ground_row = detect_ground(depth);
            auto ground_points = sample_ground_points(depth, ground_row);

            // 平面估计
            if (ground_points.size() > 100) {
                estimate_plane(ground_points);
                update_plane_model();
            }

            // 障碍物检测
            int obstacle_count = detect_obstacles(depth, debug_img);

            // 可视化
            visualize_results(debug_img, ground_row, obstacle_count, start);

            cv::imshow("Obstacle Detection", debug_img);
            cv::waitKey(1);

        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "Processing error: %s", e.what());
        }
    }

    void color_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        try {
            color_image_ = cv_bridge::toCvCopy(msg, "bgr8")->image;
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(get_logger(), "CV Bridge error: %s", e.what());
        }
    }

    int detect_ground(const cv::Mat& depth) {
        const int step = 5;
        const int min_points = 50;
        const double roi_width = get_parameter("roi_width").as_double();
        const int width_start = depth.cols * (1 - roi_width) / 2;
        const int width_end = depth.cols - width_start;

        for (int v = depth.rows-1; v >= depth.rows/2; v -= step) {
            int valid_count = 0;
            for (int u = width_start; u < width_end; u += step) {
                if (depth.at<uint16_t>(v,u) > get_parameter("min_ground_height").as_double() * 1000)
                    valid_count++;
            }
            if (valid_count > min_points) return std::max(0, v-100);
        }
        return depth.rows - 100;
    }

    std::vector<cv::Point3f> sample_ground_points(const cv::Mat& depth, int ground_row) {
        const int roi_height = 300;
        const int end_row = std::min(ground_row + roi_height, depth.rows-1);
        const double roi_width = get_parameter("roi_width").as_double();

        cv::Mat roi_mask = cv::Mat::zeros(depth.size(), CV_8U);
        roi_mask(cv::Range(ground_row, end_row),
                 cv::Range(depth.cols*(1-roi_width)/2, depth.cols*(1+roi_width)/2)) = 1;

        cv::Mat z_mat;
        depth.convertTo(z_mat, CV_32F, 1e-3);
        cv::Mat valid_mask = (z_mat > get_parameter("min_ground_height").as_double()) &
                             (z_mat < 8.0) &
                             roi_mask;

        std::vector<cv::Point3f> points;
        for (int v = 0; v < z_mat.rows; ++v) {
            for (int u = 0; u < z_mat.cols; ++u) {
                if (valid_mask.at<uchar>(v,u)) {
                    points.emplace_back(
                        U_.at<float>(v,u) * z_mat.at<float>(v,u),
                        V_.at<float>(v,u) * z_mat.at<float>(v,u),
                        z_mat.at<float>(v,u)
                    );
                }
            }
        }
        return points;
    }

    void estimate_plane(const std::vector<cv::Point3f>& points) {
        const int max_iter = get_parameter("ransac_iterations").as_int();
        const float dist_thresh = get_parameter("plane_threshold").as_double();
        const float normal_thresh = get_parameter("normal_constraint").as_double();

        cv::Vec4f best_plane;
        size_t best_inliers = 0;
        std::random_device rd;
        std::mt19937 gen(rd());

        for (int iter = 0; iter < max_iter; ++iter) {
            std::vector<cv::Point3f> samples(3);
            std::sample(points.begin(), points.end(), samples.begin(), 3, gen);

            cv::Vec4f plane;
            fitPlaneToPoints3D(samples, plane);

            if (plane[2] < normal_thresh) continue;

            size_t inliers = 0;
            for (const auto& pt : points) {
                float dist = std::abs(plane[0]*pt.x + plane[1]*pt.y + plane[2]*pt.z + plane[3]);
                if (dist < dist_thresh) ++inliers;
            }

            if (inliers > best_inliers) {
                best_inliers = inliers;
                best_plane = plane;
            }
        }

        if (best_inliers > points.size() * 0.3) {
            current_plane_ = best_plane;
        }
    }

    void fitPlaneToPoints3D(const std::vector<cv::Point3f>& points, cv::Vec4f& plane) {
        cv::Point3f centroid(0,0,0);
        for (const auto& p : points) centroid += p;
        centroid *= (1.0f / points.size());

        float xx=0, xy=0, xz=0, yy=0, yz=0, zz=0;
        for (const auto& p : points) {
            cv::Point3f r = p - centroid;
            xx += r.x * r.x;
            xy += r.x * r.y;
            xz += r.x * r.z;
            yy += r.y * r.y;
            yz += r.y * r.z;
            zz += r.z * r.z;
        }

        cv::Matx33f cov(xx, xy, xz, xy, yy, yz, xz, yz, zz);
        cv::Vec3f normal;
        cv::eigen(cov, normal);

        normal = normal / cv::norm(normal);
        plane = cv::Vec4f(normal[0], normal[1], normal[2], -normal.dot(centroid));
    }

    int detect_obstacles(const cv::Mat& depth, cv::Mat& debug_img) {
        cv::Mat Z;
        depth.convertTo(Z, CV_32F, 1e-3f);  // 确保转换为浮点型

        // 确保矩阵尺寸统一
        CV_Assert(U_.size() == Z.size() && V_.size() == Z.size());

        cv::Mat X = U_.mul(Z);
        cv::Mat Y = V_.mul(Z);

        // 平面距离计算
        cv::Vec4f plane = get_average_plane();
        const float norm_factor = 1.0f / cv::norm(plane.val);
        cv::Mat distance = (plane[0] * X + plane[1] * Y + plane[2] * Z + plane[3]) * norm_factor;

        // 动态阈值
        cv::Mat threshold = get_parameter("plane_threshold").as_double() * 
                          (1.0 + get_parameter("dynamic_factor").as_double() * Z);

        // 生成障碍物掩膜
        cv::Mat obstacle_mask;
        cv::abs(distance).convertTo(obstacle_mask, CV_32F);
        cv::inRange(obstacle_mask, threshold, 100.0, obstacle_mask);

        // 后处理
        cv::morphologyEx(obstacle_mask, obstacle_mask, cv::MORPH_OPEN, 
                        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5,5)));

        // 轮廓检测
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(obstacle_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // 绘制结果
        int valid_count = 0;
        for (const auto& contour : contours) {
            if (cv::contourArea(contour) < 150) continue;
            valid_count++;
            cv::rectangle(debug_img, cv::boundingRect(contour), cv::Scalar(0,255,255), 2);
        }

        return valid_count;
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ObstacleDetector>());
    rclcpp::shutdown();
    return 0;
}