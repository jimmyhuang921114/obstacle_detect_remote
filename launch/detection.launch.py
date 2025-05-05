from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='new_obs_filiter',  #  package 名稱
            executable='obstacle_detector_node',
            name='obstacle_detector',
            output='screen',
            parameters=[{
                'plane_threshold': 0.05,
                'ransac_iterations': 300,
                'min_ground_height': 0.3,
                'max_obstacle_height': 1.2,
                'min_detection_distance': 0.5,
                'max_detection_distance': 5.0
            }]
        )
    ])
