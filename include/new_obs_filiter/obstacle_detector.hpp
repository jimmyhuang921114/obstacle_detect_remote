#ifndef OBSTACLE_DETECTOR_HPP
#define OBSTACLE_DETECTOR_HPP

#include <opencv2/opencv.hpp>
#include "new_obs_filiter/plane_estimator.hpp"

namespace new_obs_filiter {

class ObstacleDetector {
public:
    struct Parameters {
        double max_detection_distance = 5.0;
        float obstacle_height_threshold = 0.15f;
        float ground_margin = 0.05f;
        double fx = 610.0;
        double fy = 610.0;
        double cx = 320.0;
        double cy = 240.0;
    };

    ObstacleDetector(const PlaneEstimator& plane_estimator, const Parameters& params);
    
    int detect(const cv::Mat& depth_image, cv::Mat& debug_img, float depth_scale);
    int get_ground_row() const { return ground_row_; }

private:
    const PlaneEstimator& plane_estimator_;
    Parameters params_;
    int ground_row_ = 0;

    cv::Mat calculate_height_map(const cv::Mat& depth_image, float depth_scale) const;
};

} // namespace new_obs_filiter

#endif // OBSTACLE_DETECTOR_HPP