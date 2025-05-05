#ifndef PLANE_ESTIMATOR_HPP // define the header 
#define PLANE_ESTIMATOR_HPP // define the header 

#include <vector> // vector for points
#include <deque> // deque for plane history
#include <opencv2/core.hpp> // for vector and matrix

//setting the namespace 
namespace new_obs_filiter {

//define the class for the pararmeters to be used in the plane estimator 
class PlaneEstimator {
public:
    struct Parameters {
        int ransac_iterations; //number of iterations for ransac
        double plane_threshold; //threshold for the distance from the plane
        double normal_constraint; //constraint for the normal vector

        Parameters();
    };

    explicit PlaneEstimator(const Parameters& params = Parameters());
    
    bool estimate(const std::vector<cv::Point3f>& points);
    cv::Vec4f get_average_plane() const;

private:
    Parameters params_;
    std::deque<cv::Vec4f> plane_history_;
    cv::Vec4f current_plane_;

    void fit_plane(const std::vector<cv::Point3f>& points, cv::Vec4f& plane);
};

} // namespace

#endif