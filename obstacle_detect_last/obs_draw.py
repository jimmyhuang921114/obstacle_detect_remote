#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

class Visualizer(Node):
    def __init__(self):
        super().__init__('visualizer')
        self.bridge = CvBridge()

        # Matplotlib setup
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.xlim = (-4.0, 4.0)
        self.zlim = (0.0, 4.0)
        self.setup_plot()

        # Image storage
        self.latest_color = None
        self.latest_depth_vis = None

        # FPS calculation
        self.last_display_time = None
        self.avg_fps = 0.0
        self.alpha = 0.9  # Smoothing factor

        # ROS2 Subscribers
        self.depth_sub = self.create_subscription(Image, '/processed/depth', self.depth_callback, 10)
        self.obstacle_sub = self.create_subscription(Float32MultiArray, '/obstacles', self.obstacle_callback, 10)
        self.color_sub = self.create_subscription(Image, '/camera/color', self.color_callback, 10)

    def setup_plot(self):
        """Initialize radar plot"""
        self.ax.set_title("Real-time Obstacle Radar")
        self.ax.set_xlabel("Lateral Distance (m)")
        self.ax.set_ylabel("Depth (m)")
        self.ax.set_xlim(*self.xlim)
        self.ax.set_ylim(*self.zlim)
        self.ax.grid(True, linestyle='--', alpha=0.6)
        self.scatter = self.ax.scatter([], [], s=50, c='red', alpha=0.8)
        plt.ion()
        plt.show()

    def depth_callback(self, msg):
        try:
            self.latest_depth_vis = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.show_combined_view()
        except Exception as e:
            self.get_logger().error(f"Depth Error: {str(e)}")

    def color_callback(self, msg):
        try:
            self.latest_color = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.show_combined_view()
        except Exception as e:
            self.get_logger().error(f"Color Error: {str(e)}")

    def obstacle_callback(self, msg):
        points = np.array(msg.data).reshape(-1, 2)
        if points.size > 0:
            self.scatter.set_offsets(points)
            self.ax.draw_artist(self.scatter)
            self.fig.canvas.blit(self.fig.bbox)
            self.fig.canvas.flush_events()

    def show_combined_view(self):
        if self.latest_color is None or self.latest_depth_vis is None:
            return

        # FPS calculation
        current_time = time.time()
        if self.last_display_time is not None:
            delta = current_time - self.last_display_time
            current_fps = 1.0 / delta
            self.avg_fps = self.alpha * self.avg_fps + (1 - self.alpha) * current_fps
        self.last_display_time = current_time

        # Resize images
        target_size = (854, 480)
        color_resized = cv2.resize(self.latest_color, target_size)
        depth_resized = cv2.resize(self.latest_depth_vis, target_size)

        # Add FPS text
        fps_text = f"FPS: {self.avg_fps:.1f}"
        cv2.putText(color_resized, fps_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(depth_resized, fps_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        combined = np.hstack((color_resized, depth_resized))
        cv2.namedWindow("Color + Depth View", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Color + Depth View", 1708, 480)
        cv2.imshow("Color + Depth View", combined)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    visualizer = Visualizer()
    try:
        rclpy.spin(visualizer)
    finally:
        cv2.destroyAllWindows()
        plt.close()
        visualizer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()