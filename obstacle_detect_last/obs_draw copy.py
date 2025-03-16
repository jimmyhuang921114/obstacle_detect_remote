#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np
import matplotlib.pyplot as plt

class Visualizer(Node):
    def __init__(self):
        super().__init__('visualizer')
        self.bridge = CvBridge()
        
        # ✅ Matplotlib setup for real-time obstacle radar
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.xlim = (0.0, 4.0)  # Image pixel width range
        self.zlim = (0.0, 4.0)  # Depth range (meters)
        self.setup_plot()

        # ✅ Store latest images
        self.latest_color = None
        self.latest_depth_vis = None
        
        # ✅ ROS2 Subscribers
        self.depth_sub = self.create_subscription(Image, '/processed/depth', self.depth_callback, 10)
        self.obstacle_sub = self.create_subscription(Float32MultiArray, '/obstacles', self.obstacle_callback, 10)
        self.color_sub = self.create_subscription(Image, '/camera/color', self.color_callback, 10)

    def setup_plot(self):
        """Initialize Matplotlib plot."""
        self.ax.set_title("Real-time Obstacle Radar")
        self.ax.set_xlabel("Pixel X (RealSense)")
        self.ax.set_ylabel("Depth Z (m)")
        self.ax.set_xlim(*self.xlim)
        self.ax.set_ylim(*self.zlim)
        self.ax.grid(True, linestyle='--', alpha=0.6)

        # Scatter plot for obstacles
        self.scatter = self.ax.scatter([], [], s=50, c='red', alpha=0.8)

        plt.ion()  # Interactive mode ON
        plt.show()

    def depth_callback(self, msg):
        """Convert depth image for visualization."""
        try:
            depth_img = self.bridge.imgmsg_to_cv2(msg, '16UC1')

            # ✅ Normalize and apply color map
            vis_img = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX)
            vis_img = np.uint8(vis_img)
            vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_JET)

            # ✅ Store processed depth image
            self.latest_depth_vis = vis_img

            # ✅ Try displaying combined view
            self.show_combined_view()
        except Exception as e:
            self.get_logger().error(f"Depth Image Processing Error: {str(e)}")

    def color_callback(self, msg):
        """Receive color image and store it."""
        try:
            color_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.latest_color = color_img  # Store latest color frame

            self.show_combined_view()  # Try displaying combined view
        except Exception as e:
            self.get_logger().error(f"Color Image Processing Error: {str(e)}")

    def obstacle_callback(self, msg):
        """Update obstacle scatter plot in Matplotlib."""
        points = np.array(msg.data).reshape(-1, 2)

        if points.shape[0] == 0:
            return  # Avoid errors if no obstacles

        # ✅ Clip points within valid RealSense range
        points[:, 0] = np.clip(points[:, 0], 0, 640)  # X (Image pixels)
        points[:, 1] = np.clip(points[:, 1], 0, 4)    # Z (Depth meters)

        # ✅ Update scatter plot
        self.scatter.set_offsets(points)

        # ✅ Redraw Matplotlib figure
        self.ax.draw_artist(self.scatter)
        self.fig.canvas.blit(self.fig.bbox)
        self.fig.canvas.flush_events()

    def show_combined_view(self):
        """Display combined Color + Depth Image."""
        if self.latest_color is None or self.latest_depth_vis is None:
            return  # Wait until both images are received

        target_size = (1280, 720)

        # ✅ Resize images to a fixed size
        color_resized = cv2.resize(self.latest_color, target_size, interpolation=cv2.INTER_AREA)
        depth_resized = cv2.resize(self.latest_depth_vis, target_size, interpolation=cv2.INTER_AREA)

        # ✅ Horizontally stack images
        combined = np.hstack((color_resized, depth_resized))

        # ✅ Display with OpenCV
        cv2.imshow("Color + Depth View (640x360)", combined)
        cv2.waitKey(1)  # Required to refresh window

def main(args=None):
    """ROS2 Node Execution"""
    rclpy.init(args=args)
    node = Visualizer()

    try:
        rclpy.spin(node)
    finally:
        cv2.destroyAllWindows()  # Ensure OpenCV windows are closed
        plt.close()  # Close Matplotlib
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
