#include "new_obs_filiter/obstacle_detector.hpp" // import the obstacle detector header

namespace new_obs_filiter {//define the namespace


ObstacleDetector::ObstacleDetector(const PlaneEstimator& plane_estimator,
                                  const Parameters& params)//set the parameters and the plane estimator
    : plane_estimator_(plane_estimator), 
      params_(params),
      ground_row_(0) {}  // initialize the ground row to 0

int ObstacleDetector::detect(const cv::Mat& depth_image,
                           cv::Mat& debug_img,
                           float depth_scale) 
{
    
    cv::Mat height_map = calculate_height_map(depth_image, depth_scale);
    
    // 步骤2：生成障碍物掩膜
    cv::Mat obstacle_mask;
    const float threshold = params_.obstacle_height_threshold * 1000;  // 转换为毫米
    cv::threshold(height_map, obstacle_mask, threshold, 255, cv::THRESH_BINARY);
    
    // 步骤3：查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(obstacle_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    // 步骤4：计算地面行号（示例逻辑）
    const int image_center_y = depth_image.rows / 2;
    ground_row_ = image_center_y + static_cast<int>(params_.ground_margin * 1000 / depth_scale);
    
    // 步骤5：在调试图像绘制结果
    cv::cvtColor(obstacle_mask, debug_img, cv::COLOR_GRAY2BGR);
    cv::line(debug_img, 
            cv::Point(0, ground_row_), 
            cv::Point(debug_img.cols, ground_row_),
            cv::Scalar(0, 255, 0), 2);
    
    return static_cast<int>(contours.size());
}

cv::Mat ObstacleDetector::calculate_height_map(const cv::Mat& depth_image,
                                             float depth_scale) const 
{
    cv::Mat height_map(depth_image.size(), CV_32FC1, cv::Scalar(0));
    const cv::Vec4f plane = plane_estimator_.get_average_plane();

    // 从参数获取相机内参
    const double fx = params_.fx;
    const double fy = params_.fy;
    const double cx = params_.cx;
    const double cy = params_.cy;

    for(int y = 0; y < depth_image.rows; ++y) {
        for(int x = 0; x < depth_image.cols; ++x) {
            const float depth = depth_image.at<float>(y, x) * depth_scale;
            if(depth <= 0.1f) continue;  // 忽略无效深度

            // 3D坐标计算
            const float X = (x - cx) * depth / fx;
            const float Y = (y - cy) * depth / fy;
            const float Z = depth;

            // 平面方程计算高度
            const float height = plane[0] * X + plane[1] * Y + plane[2] * Z + plane[3];
            height_map.at<float>(y, x) = std::abs(height);
        }
    }
    return height_map;
}

} // namespace new_obs_filiter