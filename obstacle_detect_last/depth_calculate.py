#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import numpy as np
import cv2

class DepthProcessor(Node):
    def __init__(self):
        super().__init__('depth_processor')
        
        # 声明参数（包含相机内参）
        self.declare_parameters(
            namespace='',
            parameters=[
                ('roi_x', 640),  # ROI 中心 X
                ('roi_y', 360),  # ROI 中心 Y
                ('roi_width', 400),  # ROI 宽度
                ('roi_height', 800),  # ROI 高度
                ('num_slices', 100),  # 水平切片数
                ('min_depth', 0.1),  # 最小深度
                ('max_depth', 2.5),  # 最大深度
                ('fx', 607.033), ('fy', 606.141), 
                ('cx', 640), ('cy', 360)  # **修正 cx/cy 使中心对齐**
            ]
        )

        # ROS 订阅 & 发布
        self.bridge = CvBridge()
        self.origin_depth_sub = self.create_subscription(Image, '/camera/depth', self.depth_callback, 10)
        self.obstacle_pub = self.create_publisher(Float32MultiArray, '/obstacles', 10)
        self.depth_pub = self.create_publisher(Image, '/processed/depth', 10)        # 初始化参数
        self.param_update_needed = True
        self.update_params()
        self.add_on_set_parameters_callback(self.param_callback)

    def param_callback(self, params):
        self.param_update_needed = True
        return rclpy.node.SetParametersResult(successful=True)

    def update_params(self):
        base_params = self.get_parameters([
            'roi_x', 'roi_y', 'roi_width', 'roi_height',
            'num_slices', 'min_depth', 'max_depth'
        ])
        camera_params = self.get_parameters(['fx', 'fy', 'cx', 'cy'])
        
        # 参数检查 & 更新
        self.roi_x = max(0, min(base_params[0].value, 1280-1))
        self.roi_y = max(0, min(base_params[1].value, 720-1))
        self.roi_width = max(10, min(base_params[2].value, 1280-self.roi_x))
        self.roi_height = max(10, min(base_params[3].value, 720-self.roi_y))
        self.num_slices = max(1, base_params[4].value)
        self.min_depth = max(0.1, base_params[5].value)
        self.max_depth = max(self.min_depth+0.1, base_params[6].value)

        self.camera_params = {
            'fx': camera_params[0].value,
            'fy': camera_params[1].value,
            'cx': camera_params[2].value,
            'cy': camera_params[3].value
        }

        self.slice_width = self.roi_width // self.num_slices
        self.param_update_needed = False
        self.get_logger().info(f"相机参数更新：{self.camera_params}")

    def depth_callback(self, msg):
        """ 处理深度图像 """
        try:
            depth_img = self.bridge.imgmsg_to_cv2(msg, '16UC1')
            # 处理深度图
            marked_img, points = self.process_depth(depth_img)
            # 发布障碍物数据
            self.publish_obstacles(points)
            # 发布处理后的图像
            self.depth_pub.publish(self.bridge.cv2_to_imgmsg(marked_img, '16UC1'))
            
        except Exception as e:
            self.get_logger().error(f"处理错误: {str(e)}")

    def process_depth(self, img):
        """ 处理 ROI 并转换为 3D 点 """
        marked_img = img.copy()
        all_points = []

        # 计算 ROI 范围，确保居中
        x_start = max(0, self.roi_x - self.roi_width // 2)
        y_start = max(0, self.roi_y - self.roi_height // 2)
        x_end = x_start + self.roi_width
        y_end = y_start + self.roi_height

        # 绘制 ROI 边界
        cv2.rectangle(marked_img, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

        # 处理每个切片区域
        for i in range(self.num_slices):
            slice_x_start = x_start + i * self.slice_width
            slice_x_end = slice_x_start + self.slice_width

            # 绘制分割线
            cv2.line(marked_img, (slice_x_end, y_start), (slice_x_end, y_end), (255, 255, 255), 2)

            # 提取 ROI
            roi = img[y_start:y_end, slice_x_start:slice_x_end]
            points = self.convert_to_3d(roi, slice_x_start)

            # 计算中心点
            if len(points) > 0:
                center_point = np.mean(points, axis=0)
            else:
                center_point = [slice_x_start + self.slice_width/2 - self.camera_params['cx'], 0]
            all_points.append(center_point)

        return marked_img, np.array(all_points)


    # def convert_to_3d(self, roi, x_offset):
    #     """ 将 ROI 转换为 3D 点 """
    #     rows, cols = roi.shape
    #     u = np.arange(cols) + x_offset
    #     v = np.arange(rows) + self.roi_y
    #     uu, vv = np.meshgrid(u, v)


    #     z = roi.astype(np.float32) / 1000.0
    #     valid = (z > self.min_depth) & (z < self.max_depth)

    #     # 直接使用相机内参，转换到相机坐标系 (X, Z)
    #     # X = (u - cx) * Z / fx
    #     # Y 轴你暂时不需要，所以只保留 X, Z
    #     X = (uu[valid] - self.camera_params['cx']) * z[valid] / self.camera_params['fx']
    #     Z = z[valid]

    #     # 返回完整的实际坐标，不再做裁剪/归一化处理
    #     return np.column_stack((X, Z))

    def convert_to_3d(self, roi, x_offset):
        """将 ROI 区域直接转换为相机坐标系中的实际 XZ，并计算平均深度，带有「有效面積」限制"""

        # 1) 取得 ROI 影像尺寸
        rows, cols = roi.shape

        # 2) 計算每個像素在影像中的 (u, v) 座標
        u = np.arange(cols) + x_offset
        v = np.arange(rows) + self.roi_y
        uu, vv = np.meshgrid(u, v)

        # 3) 將 mm (16UC1) 轉為米
        z = roi.astype(np.float32) / 1000.0

        # 4) 取得有效像素遮罩
        valid = (z > self.min_depth) & (z < self.max_depth)

        # 5) 判斷有效像素數量，若不足就略過
        min_valid_count = 50  # 這裡可依需求自行設定
        valid_count = np.count_nonzero(valid)

        if valid_count < min_valid_count:
            self.get_logger().warn(
                f"該 ROI 區域內有效像素數量 ({valid_count}) < 閾值 ({min_valid_count})，跳過處理。")
            # 依需求，你可以直接 return 空陣列，或回傳 [0,0]，或做別的處理
            return np.array([])

        # ---------- 計算平均深度 -----------
        avg_depth = np.mean(z[valid])
        self.get_logger().info(f"ROI 平均深度: {avg_depth:.3f} m, 有效像素: {valid_count}")

        # 6) 轉為相機座標系 (X, Z)
        X = (uu[valid] - self.camera_params['cx']) * z[valid] / self.camera_params['fx']
        Z = z[valid]

        return np.column_stack((X, Z))

    # def convert_to_3d(self, roi, x_offset):
    #     """ 取得 ROI 區域內最近的點（X, Z） """
    #     rows, cols = roi.shape
    #     u = np.arange(cols) + x_offset
    #     v = np.arange(rows) + self.roi_y
    #     uu, vv = np.meshgrid(u, v)

    #     # 將深度值從 mm 轉換為 m
    #     z = roi.astype(np.float32) / 1000.0

    #     # 過濾有效的深度點
    #     valid = (z > self.min_depth) & (z < self.max_depth)

    #     if np.any(valid):  # 確保有有效點
    #         # 取得最小 Z（最近的點）
    #         min_index = np.argmin(z[valid])  # 找到最小 Z 的索引
    #         min_z = np.min(z[valid])  # 取得最小的 Z 值
    #         min_x = (uu[valid][min_index] - self.camera_params['cx']) * min_z / self.camera_params['fx']

    #         self.get_logger().info(f"ROI 最近障礙物: Z = {min_z:.3f} m")
    #     else:
    #         mix_x = x_offset + self.slice_width / 2 - self.camera_params['cx']
    #         min_z = 0.0
    #         self.get_logger().warn("該 ROI 區域內無有效深度點")
  

    #     return np.array([[min_x, min_z]])  # 只回傳最近的點


    def publish_obstacles(self, points):
        """ 发布障碍物坐标 """
        msg = Float32MultiArray()
        msg.data = points.flatten().tolist()
        self.obstacle_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = DepthProcessor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()