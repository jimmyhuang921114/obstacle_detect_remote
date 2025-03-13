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
        
        # Matplotlib 相關 (顯示障礙物雷達圖)
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.xlim = (-2.0, 2.0)
        self.zlim = (0.0, 4.0)

        # 儲存最新的彩色 & 深度影像(可視化後)
        self.latest_color = None
        self.latest_depth_vis = None
        
        # 初始化訂閱
        self.depth_sub = self.create_subscription(Image, '/processed/depth', self.depth_callback, 10)
        self.obstacle_sub = self.create_subscription(Float32MultiArray, '/obstacles', self.obstacle_callback, 10)
        self.color_sub = self.create_subscription(Image, '/camera/color', self.color_callback, 10)
        
        # 設定障礙物雷達圖
        self.setup_plot()

    def setup_plot(self):
        """初始化雷達圖顯示"""
        self.ax.set_title("Real-time Obstacle Radar")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Z (m)")
        self.ax.grid(True, linestyle='--', alpha=0.6)
        self.ax.set_xlim(*self.xlim)
        self.ax.set_ylim(*self.zlim)
        self.scatter = self.ax.scatter([], [], s=50, c='red', alpha=0.8)
        plt.ion()
        plt.show()

    def depth_callback(self, msg):
        """接收深度影像，轉成彩色深度圖後存起來"""
        try:
            # ROS Image → OpenCV: depth (16UC1)
            depth_img = self.bridge.imgmsg_to_cv2(msg, '16UC1')
            # 將深度圖做可視化 (ex: COLORMAP_JET)
            vis_img = cv2.convertScaleAbs(depth_img, alpha=0.03)
            vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_JET)
            
            # 儲存到類別變數
            self.latest_depth_vis = vis_img

            # 嘗試合併 & 顯示
            self.show_combined_view()
        except Exception as e:
            self.get_logger().error(f"可視化错误: {str(e)}")
            
    def color_callback(self, msg):
        """接收彩色影像，存起來"""
        try:
            color_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            # 儲存到類別變數
            self.latest_color = color_img

            # 嘗試合併 & 顯示
            self.show_combined_view()
        except Exception as e:
            self.get_logger().error(f"可視化错误: {str(e)}")

    def obstacle_callback(self, msg):
        """障礙物點雲更新 (使用 Matplotlib 做即時雷達顯示)"""
        points = np.array(msg.data).reshape(-1, 2)
        self.scatter.set_offsets(points)

        # 部分後端需要 redraw + flush 才能即時顯示
        self.ax.draw_artist(self.scatter)
        self.fig.canvas.blit(self.fig.bbox)
        self.fig.canvas.flush_events()

    def show_combined_view(self):
        """將最新的彩色影像與深度影像都強制縮放到 640×360，然後合併顯示"""
        if self.latest_color is None or self.latest_depth_vis is None:
            return

        # 指定最終顯示的大小
        target_width = 640
        target_height = 360

        # 先縮放彩色影像
        color_resized = cv2.resize(
            self.latest_color, 
            (target_width, target_height), 
            interpolation=cv2.INTER_AREA
        )

        # 縮放深度影像
        depth_resized = cv2.resize(
            self.latest_depth_vis, 
            (target_width, target_height), 
            interpolation=cv2.INTER_AREA
        )

        # 左右合併
        combined = np.hstack((color_resized, depth_resized))

        # 顯示在同一個視窗
        cv2.imshow("Color + Depth @640x360", combined)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = Visualizer()
    try:
        rclpy.spin(node)
    finally:
        cv2.destroyAllWindows()
        plt.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
