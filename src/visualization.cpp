#include "new_obs_filiter/visualization.hpp"

namespace new_obs_filiter {

Visualizer::Parameters::Parameters() :
    info_font_scale(0.6f),
    info_font_thickness(2),
    ground_color(0,100,0),
    info_text_color(200,200,0),
    fps_text_color(255,255,0),
    ground_alpha(0.3f) {}

Visualizer::Visualizer(const Parameters& params) : params_(params) {}

void Visualizer::visualize(cv::Mat& frame,
                         int ground_row,
                         int obstacle_count,
                         const cv::Vec4f& plane,
                         double fps,
                         int depth_height,
                         int debug_height) {
    float height_ratio = static_cast<float>(ground_row)/depth_height;
    draw_ground_layer(frame, ground_row, height_ratio);
    draw_info_text(frame, plane, obstacle_count, fps);
}

void Visualizer::draw_ground_layer(cv::Mat& img, int ground_pixel, float height_ratio) const {
    cv::Mat overlay = img.clone();
    cv::rectangle(overlay, 
                 cv::Point(0, ground_pixel), 
                 cv::Point(img.cols, img.rows), 
                 params_.ground_color, 
                 cv::FILLED);
    cv::addWeighted(overlay, params_.ground_alpha, img, 1 - params_.ground_alpha, 0, img);
}

void Visualizer::draw_info_text(cv::Mat& img,
                              const cv::Vec4f& plane,
                              int obstacle_count,
                              double fps) const {
    std::string plane_info = cv::format("Plane: [%.2f, %.2f, %.2f, %.2f]", plane[0], plane[1], plane[2], plane[3]);
    cv::putText(img, plane_info, 
               cv::Point(10,30), 
               cv::FONT_HERSHEY_SIMPLEX, 
               params_.info_font_scale, 
               params_.info_text_color, 
               params_.info_font_thickness);

    cv::putText(img, cv::format("FPS: %.1f", fps), 
               cv::Point(10,60), 
               cv::FONT_HERSHEY_SIMPLEX, 
               params_.info_font_scale, 
               params_.fps_text_color, 
               params_.info_font_thickness);

    cv::putText(img, cv::format("Obstacles: %d", obstacle_count), 
               cv::Point(10,90), 
               cv::FONT_HERSHEY_SIMPLEX, 
               params_.info_font_scale, 
               params_.info_text_color, 
               params_.info_font_thickness);
}

}






// // 在障碍物检测类中新增可视化方法
// void ObstacleDetector::visualize_plane_extent(cv::Mat& debug_img, const cv::Vec4f& plane) {
//     // 生成网格点（假设相机俯仰角30度）
//     const int step = 20; // 网格间距（像素）
//     for(int y = 0; y < debug_img.rows; y += step) {
//         for(int x = 0; x < debug_img.cols; x += step) {
//             // 将像素坐标反向投影到3D平面
//             float depth = estimate_depth_from_plane(x, y, plane);
//             if(depth > 0) {
//                 cv::circle(debug_img, cv::Point(x,y), 2, cv::Scalar(0,200,0), -1);
//             }
//         }
//     }
// }

// // 根据平面方程估计某像素的深度
// float ObstacleDetector::estimate_depth_from_plane(int x, int y, const cv::Vec4f& plane) {
//     // 平面方程：ax + by + cz + d = 0
//     // 相机坐标系下：z = -(ax + by + d)/c
//     const float a = plane[0], b = plane[1], c = plane[2], d = plane[3];
//     if(std::abs(c) < 1e-6) return -1; // 避免除以零
    
//     // 像素到相机坐标的转换（假设已校正）
//     float X = (x - params_.cx) / params_.fx;
//     float Y = (y - params_.cy) / params_.fy;
    
//     return -(a*X + b*Y + d) / c;
// }