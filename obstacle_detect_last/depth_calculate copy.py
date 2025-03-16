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
                ('roi_width', 500),  # ROI 宽度
                ('roi_height', 800),  # ROI 高度
                ('num_slices', 100),  # 水平切片数
                ('min_depth', 0.02),# 最小深度
                ('max_depth', 4),  # 最大深度
                ('fx', 384.5462341308594), ('fy', 384.5462341308594), 
                ('cx', 322.67999267578125), ('cy', 236.3990478515625)  # **修正 cx/cy 使中心对齐**
            ]
        )

        # ROS 订阅 & 发布
        self.bridge = CvBridge()
        # self.origin_color_sub = self.create_subscription(Image,'/camera/color',self.color_callback,10)
        self.origin_depth_sub = self.create_subscription(Image, '/camera/depth', self.depth_callback, 10)
        self.obstacle_pub = self.create_publisher(Float32MultiArray, '/obstacles', 10)
        self.depth_pub = self.create_publisher(Image, '/processed/depth', 10)        # 初始化参数
        # self.color_pub = self.create_publisher(Image,'/processed/color',10)
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
        self.roi_height = max(10, min(base_params[3].value, 720-self.roi_y))  # ✅ 确保 roi_height 受限
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
        self.get_logger().info(f"相机参数更新：{self.camera_params}, ROI 高度: {self.roi_height}")


    def depth_callback(self, msg):
        """ 订阅深度图像并处理 """
        try:
            depth_img = self.bridge.imgmsg_to_cv2(msg, '16UC1')  # 16 位深度图
            marked_img, points = self.process_depth(depth_img)  # 处理深度图
            
            self.publish_obstacles(points)  # 发布障碍物数据

            # ✅ **转换 `16UC1` 到 `bgr8`**
            vis_img = cv2.normalize(marked_img, None, 0, 255, cv2.NORM_MINMAX)  # 归一化
            vis_img = np.uint8(vis_img)  # 转换为 8bit 格式
            vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_JET)  # 伪彩色处理
            assert vis_img.shape[-1] == 3, "转换后应为 `bgr8` 格式"

            # ✅ **发布 `bgr8` 格式数据**
            self.depth_pub.publish(self.bridge.cv2_to_imgmsg(vis_img, encoding='bgr8'))  
            self.get_logger().info("✅ 发布 `/processed/depth` 话题，格式 `bgr8`")
        except Exception as e:
            self.get_logger().error(f"处理深度图像时出错: {str(e)}")


    # def mark_color_image(self, img):
    #     """ Draw ROI and slices on the color image """
    #     marked_img = img.copy()
    #     x_start = max(0, self.roi_x - self.roi_width // 2)
    #     y_start = max(0, self.roi_y - self.roi_height // 2)
    #     x_end = x_start + self.roi_width
    #     y_end = y_start + self.roi_height

    #     # Draw ROI rectangle
    #     cv2.rectangle(marked_img, (x_start, y_start), (x_end, y_end), (0, 0, 0), 2)

    #     # Draw slice lines
    #     for i in range(self.num_slices):
    #         slice_x_end = x_start + (i + 1) * self.slice_width
    #         cv2.line(marked_img, (slice_x_end, y_start), (slice_x_end, y_end), (0, 0, 0), 2)

    #     return marked_img
    # def color_callback(self,msg):
    #     try:
    #         color_img = self.bridge.imgmsg_to_cv2(msg, '8UC1')

    #         marked_img,_ = self.process_depth(color_img)
            
    #         # 发布处理后的图像
    #         self.color_pub.publish(self.bridge.cv2_to_imgmsg(marked_img, '8UC1'))
            
    #     except Exception as e:
    #         self.get_logger().error(f"处理错误: {str(e)}")

        
    def process_depth(self, img):
        """ 处理深度图并转换为 3D 点，并绘制可视化框 """
        depth_vis = cv2.convertScaleAbs(img, alpha=0.03)
        marked_img = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)  # 转换成 BGR 以便绘制彩色框

        all_points = []

        # 计算 ROI 范围
        x_start = max(0, self.roi_x - self.roi_width // 2)
        x_end = x_start + self.roi_width
        y_start = max(0, self.roi_y - self.roi_height // 2)
        y_end = y_start + self.roi_height

        # **✅ 在深度图上绘制 ROI 矩形**
        cv2.rectangle(marked_img, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)  # 绿色框

        # 处理每个 X 方向的切片
        for i in range(self.num_slices):
            slice_x_start = x_start + i * self.slice_width
            slice_x_end = slice_x_start + self.slice_width

            # **✅ 在图像上绘制切片分割线**
            cv2.line(marked_img, (slice_x_end, y_start), (slice_x_end, y_end), (0, 255, 255), 1)  # 黄色分割线

            # 提取 ROI（X 轴分割，同时限制 Y 轴）
            roi = img[y_start:y_end, slice_x_start:slice_x_end]
            points = self.convert_to_3d(roi, slice_x_start, y_start)

            if points.shape[0] > 0:
                all_points.append(points)

        if all_points:
            all_points = np.vstack(all_points)
        else:
            all_points = np.empty((0, 2))

        return marked_img, all_points


    def convert_to_3d(self, roi, x_offset, y_offset):
        """ 
        取得 ROI 内所有有效点的平均值（X, Z），并进行滤波处理 
        1. **跳过面积过小的区域**
        2. **去除 10% 最小值和 90% 最大值**
        3. **计算中值，减少噪声影响**
        """
        
        rows, cols = roi.shape
        u = np.arange(cols) + x_offset  # **计算像素坐标 X**
        v = np.arange(rows) + y_offset  # **计算像素坐标 Y**
        uu, vv = np.meshgrid(u, v)

        z = roi.astype(np.float32) / 1000.0  # **深度值转换成米**
        valid = (z > self.min_depth) & (z < self.max_depth)  # **筛选有效深度数据**

        valid_depths = z[valid]  # **提取有效深度数据**
        if valid_depths.size == 0:
            self.get_logger().warn("⚠️ ROI 内无有效深度数据，跳过处理。")
            return np.empty((0, 2))

        # ✅ **去除 10% 最小值和 90% 最大值（降低噪声）**
        lower_thresh = np.percentile(valid_depths, 5)  # 10% 分位数
        upper_thresh = np.percentile(valid_depths, 95)  # 90% 分位数
        valid = valid & (z > lower_thresh) & (z < upper_thresh)

        # ✅ **确保仍有足够数据**
        if np.count_nonzero(valid) < 300:
            self.get_logger().warn(f"⚠️ 经过滤波后有效点过少 ({np.count_nonzero(valid)} < 300)，跳过处理。")
            return np.empty((0, 2))

        # ✅ **计算实际 X, Z 坐标**
        X_real = (uu[valid] - self.camera_params['cx']) * z[valid] / self.camera_params['fx']
        Z_real = z[valid]

        # ✅ **计算中值，减少极端值干扰**
        median_x = np.median(X_real)  # **真实 X 坐标（米）**
        median_z = np.median(Z_real)  # **真实 Z 坐标（米）**

        self.get_logger().info(f"✅ 经过滤波后障碍物位置: X = {median_x:.3f}, Z = {median_z:.3f} m")

        return np.array([[median_x, median_z]])  # **返回 (1, 2) 形状数组**




    # def convert_to_3d(self, roi, x_offset):
    #     """ 取得 ROI 區域內最近的點（X, Z），优化滤波去噪 """
        
    #     rows, cols = roi.shape
    #     u = np.arange(cols) + x_offset
    #     v = np.arange(rows) + self.roi_y
    #     uu, vv = np.meshgrid(u, v)

    #     # 将深度值从 mm 转换为 m
    #     z = roi.astype(np.float32) / 1000.0

    #     # 过滤有效深度点
    #     valid = (z > self.min_depth) & (z < self.max_depth)

    #     # **✅ 设定最小有效面积**
    #     min_valid_area = 100  # 可以根据噪声情况调整
    #     if np.count_nonzero(valid) < min_valid_area:
    #         self.get_logger().warn(f"ROI 有效区域过小 ({np.count_nonzero(valid)} 像素)，跳过处理。")
    #         return np.array([])

    #     # **✅ 计算 Z 方向均值，去除离群点**
    #     depth_values = z[valid]
    #     mean_depth = np.mean(depth_values)  # 计算均值
    #     std_depth = np.std(depth_values)  # 计算标准差
        
    #     # **✅ 移除最小 5% 的深度值（避免极端点干扰）**
    #     depth_threshold = np.percentile(depth_values, 10)
    #     valid = valid & (z > depth_threshold)  # 只保留大于该阈值的点
        
    #     # **✅ 再次检查有效点**
    #     if np.count_nonzero(valid) == 0:
    #         self.get_logger().warn("过滤最近 5% 后，ROI 内无有效深度点")
    #         return np.array([])

    #     # **✅ 计算中值，避免离群点影响**
    #     min_z = np.median(z[valid])  
    #     min_x = (np.median(uu[valid]) - self.camera_params['cx']) * min_z / self.camera_params['fx']   # 右移 325mm (0.325m)
    #     self.get_logger().info(f"ROI 最近障碍物: X = {min_x:.3f}, Z = {min_z:.3f} m")

    #     return np.array([[min_x, min_z]])  # 只返回最近的点


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