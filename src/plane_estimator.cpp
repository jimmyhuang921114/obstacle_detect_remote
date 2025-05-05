#include "new_obs_filiter/plane_estimator.hpp" // import the plane estimator header
#include <opencv2/calib3d.hpp>// import the pca function



namespace new_obs_filiter { //define the namespace

//get the default parameters  for plane estimator hpp
PlaneEstimator::Parameters::Parameters() ://initialize the parameters with default values
    ransac_iterations(500),//set the number of iterations
    plane_threshold(0.02),//set the threshold for the distance from the plane
    normal_constraint(0.8) {}//set the constraint for the normal vector

//constructor for the plane estimator
PlaneEstimator::PlaneEstimator(const Parameters& params) 
    : params_(params), current_plane_(0,0,1,0) {}

bool PlaneEstimator::estimate(const std::vector<cv::Point3f>& points) {//check if the point cloud has enough points to estimate a plane
    if(points.size() < 3) return false;//if the number of points is less than 3, return false
    
    cv::Vec4f plane;
    fit_plane(points, plane);//get the plane from the point cloud 
    
    if(std::abs(plane[2]) < params_.normal_constraint) return false;//if the normal vector is too small, return false
    
    plane_history_.push_back(plane);//add the plane to the history 
    if(plane_history_.size() > 10) plane_history_.pop_front();//if the history is too long,remove the oldest data
    
    current_plane_ = plane;//update the current plane 
    return true;
}
//get the average plane from the history

cv::Vec4f PlaneEstimator::get_average_plane() const {
    if(plane_history_.empty()) return current_plane_;//if the history is empty, return the current plane
    
    cv::Vec4f sum(0,0,0,0);//initialize the sum vector
    for(const auto& p : plane_history_) sum += p;//add all the planes in the history to the sum vector
    return sum * (1.0f / plane_history_.size());//return the average plane
}

void PlaneEstimator::fit_plane(const std::vector<cv::Point3f>& points, cv::Vec4f& plane) {//fit a plane to the point cloud using pca
    cv::Mat points_mat = cv::Mat(points).reshape(1);
    cv::PCA pca(points_mat, cv::Mat(), cv::PCA::DATA_AS_ROW);
    cv::Vec3f normal(pca.eigenvectors.row(2));
    float d = -normal.dot(pca.mean.reshape(1,3));
    plane = cv::Vec4f(normal[0], normal[1], normal[2], d);
}

}