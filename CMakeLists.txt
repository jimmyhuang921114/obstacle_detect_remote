cmake_minimum_required(VERSION 3.16)
project(new_obs_filiter)

# 設定編譯選項
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 查找依賴包
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)

# 包含目錄
include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

# 加入 library（如果有其他模組）
add_library(plane_estimator 
  src/plane_estimator.cpp
)

add_library(visualization
  src/visualization.cpp
)

# 加入 plane_filiter2 的可執行檔（重點）
add_executable(plane_filiter2_node
  src/plane_filiter2.cpp
)

# 鏈接
target_link_libraries(plane_estimator
  ${OpenCV_LIBS}
)

target_link_libraries(visualization
  ${OpenCV_LIBS}
)

target_link_libraries(plane_filiter2_node
  plane_estimator
  visualization
  ${OpenCV_LIBS}
)

# 加上 ROS2 的依賴
ament_target_dependencies(plane_estimator
  rclcpp
)

ament_target_dependencies(visualization
  rclcpp
)

ament_target_dependencies(plane_filiter2_node
  rclcpp
  sensor_msgs
  cv_bridge
)

# 安裝
install(TARGETS
  plane_filiter2_node
  plane_estimator
  visualization
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)

# 匯出
ament_export_include_directories(include)
ament_export_libraries(
  plane_estimator
  visualization
)

ament_package()
