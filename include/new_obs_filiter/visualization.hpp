#ifndef VISUALIZATION_HPP
#define VISUALIZATION_HPP

#include <opencv2/opencv.hpp>
#include <string>

namespace new_obs_filiter {

class Visualizer {
public:
    struct Parameters {
        float info_font_scale;
        int info_font_thickness;
        cv::Scalar ground_color;
        cv::Scalar info_text_color;
        cv::Scalar fps_text_color;
        float ground_alpha;

        Parameters();
    };

    explicit Visualizer(const Parameters& params = Parameters());
    
    void visualize(cv::Mat& frame,
                 int ground_row,
                 int obstacle_count,
                 const cv::Vec4f& plane,
                 double fps,
                 int depth_height,
                 int debug_height);

private:
    Parameters params_;

    void draw_ground_layer(cv::Mat& img, int ground_pixel, float height_ratio) const;
    void draw_info_text(cv::Mat& img,
                      const cv::Vec4f& plane,
                      int obstacle_count,
                      double fps) const;
};

} // namespace

#endif