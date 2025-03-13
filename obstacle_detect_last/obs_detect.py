#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
import torch
from ultralytics import YOLO

# topic name
#sub
TOPIC_COLOR_IMG = '/camera/color'
#pub
TOPIC_OBSTACLE_MASKED = '/obstacle_masked_image'  
TOPIC_OBSTACLE_PIXELS = '/obstacle_pixels'        

# yolo detect permete
MODEL_PATH = "yolov8m-seg.pt"
TARGET_SIZE = (1280 ,720 )  # (width, height)
CONF_THRESH = 0.5
ALPHA = 0.4  # use for show mask 

class ObstacleDetectPublisher(Node):
    def __init__(self):
        super().__init__('obstacle_detect_publisher')
        
        self.bridge = CvBridge()
        self.color_img = None
        
        # check cuda 
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self._init_model()
        
        # sub node
        self.color_sub = self.create_subscription(Image, TOPIC_COLOR_IMG, self.color_cb, 10)
        # pub node
        self.masked_pub = self.create_publisher(Image, TOPIC_OBSTACLE_MASKED, 10) 
        self.pixel_pub = self.create_publisher(Float32MultiArray, TOPIC_OBSTACLE_PIXELS, 10)
        
        
        self.frame_count = 0
        self.last_log_time = self.get_clock().now()
        self.get_logger().info(f"ObstacleDetectPublisher 节点启动成功, 使用设备: {self.device}")

    def _init_model(self):
        try:
            model = YOLO(MODEL_PATH)
            model.to(self.device)
            model.fuse()  # 模型 fuse() 优化
            self.get_logger().info("YOLOv8 模型加载成功")
            return model
        except Exception as e:
            self.get_logger().error(f"模型加载失败: {str(e)}")
            raise RuntimeError("模型初始化失败")

    def color_cb(self, msg):
        try:
            # 转换图像格式
            self.color_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self._process_frame()
        except CvBridgeError as e:
            self.get_logger().error(f"图像转换失败: {str(e)}")

    def _process_frame(self):
        # 图像预处理 (縮放至TARGET_SIZE, 加快推理速度)
        color_resized = cv2.resize(self.color_img, TARGET_SIZE)
        
        try:
            # YOLO推理 (seg)
            results = self.model(color_resized, conf=CONF_THRESH, verbose=False)
            
            # 生成带遮罩影像和像素座標
            masked_image, pixel_coords = self._apply_transparent_mask(color_resized, results)
            
            # 发布結果
            self._publish_results(masked_image, pixel_coords)
            
            # 簡單 FPS 日志 (每30帧輸出一次)
            self.frame_count += 1
            if self.frame_count % 30 == 0:
                now = self.get_clock().now()
                elapsed_ns = (now - self.last_log_time).nanoseconds
                fps = 30 / (elapsed_ns * 1e-9)
                self.get_logger().info(f"推理帧率: {fps:.1f} FPS")
                self.last_log_time = now
                
        except Exception as e:
            self.get_logger().error(f"处理帧数据失败: {str(e)}")

    def _apply_transparent_mask(self, image, results):
        """对图像叠加半透明掩码，并获取障碍物像素坐标"""
        overlay = image.copy()
        pixel_coords = []  # 存储障碍物像素点
        
        if results is None or not results or results[0].masks is None:
            # 没有检测到物体，直接返回原图
            return image, pixel_coords

        # 遍历每个檢測 mask
        for mask_data in results[0].masks.data:
            mask_np = mask_data.cpu().numpy()
            # 尺寸对齐
            mask_resized = cv2.resize(mask_np, (image.shape[1], image.shape[0]))
            mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255  # 转二值圖

            # 创建红色遮罩 (BGR)
            color_mask = np.zeros_like(image, dtype=np.uint8)
            color_mask[:, :, 2] = 255  # 紅色遮罩

            # 覆蓋
            overlay[mask_binary > 0] = color_mask[mask_binary > 0]

            # 半透明混合
            masked_image = cv2.addWeighted(overlay, ALPHA, image, 1 - ALPHA, 0)

            # 获取障碍物像素坐标
            y_coords, x_coords = np.where(mask_binary > 0)
            # flatten: [x1, y1, x2, y2, ...]
            pixel_coords.extend(np.column_stack((x_coords, y_coords)).flatten().tolist())

        return masked_image, pixel_coords

    def _publish_results(self, masked_image, pixels):
        """发布带遮罩图像 和 像素坐标"""
        try:
            # 发布带遮罩影像
            masked_msg = self.bridge.cv2_to_imgmsg(masked_image, "bgr8")
            self.masked_pub.publish(masked_msg)

            # 发布像素坐标 (float32)
            pixel_msg = Float32MultiArray()
            pixel_msg.data = [float(coord) for coord in pixels]  # 确保数据类型为 float
            self.pixel_pub.publish(pixel_msg)

        except CvBridgeError as e:
            self.get_logger().error(f"结果发布失败: {str(e)}")

    def __del__(self):
        """资源清理"""
        try:
            if hasattr(self, 'model'):
                del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            self.get_logger().warn(f"资源清理失败: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    try:
        node = ObstacleDetectPublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"节点异常: {str(e)}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
