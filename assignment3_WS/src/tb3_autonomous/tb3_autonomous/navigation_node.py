#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np

class PIDNavigationNode(Node):
    def __init__(self):
        super().__init__('pid_navigation_node')
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',  # 请确保该话题与摄像头插件一致
            self.image_callback,
            10
        )
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.bridge = CvBridge()

        # PD 控制参数（可根据实际情况调试）
        self.Kp = 0.005
        self.Kd = 0.001
        self.prev_error = 0.0
        self.dt = 0.1  # 假定的时间间隔（你也可以基于实际时间戳计算）

        # 运动参数：提高移动速度
        self.base_speed = 0.8  # 提高至 0.8 m/s
        self.max_speed_reduction = 0.8  # 根据障碍比例最大减速量

    def image_callback(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"图像转换失败: {e}")
            return

        height, width, _ = image.shape

        # 提取 ROI：选择图像下半部分中间区域（例如下50%的中间40%）
        roi_top = int(height * 0.6)
        roi_left = int(width * 0.3)
        roi_right = int(width * 0.7)
        roi = image[roi_top:height, roi_left:roi_right]

        roi_height, roi_width, _ = roi.shape
        roi_center_x = roi_width / 2.0

        # 转换为 HSV 空间并进行阈值分割
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # 此处采用亮度阈值，假设障碍物为较暗区域；你也可以根据具体情况改为颜色检测
        lower_thresh = np.array([0, 0, 0])
        upper_thresh = np.array([180, 255, 100])
        mask = cv2.inRange(hsv, lower_thresh, upper_thresh)

        # 计算障碍物区域占比
        obstacle_area = cv2.countNonZero(mask)
        total_area = roi_height * roi_width
        obstacle_ratio = obstacle_area / total_area

        # 计算障碍物区域的重心（moments）
        M = cv2.moments(mask)
        if M["m00"] > 0:
            obstacle_cx = M["m10"] / M["m00"]
        else:
            obstacle_cx = roi_center_x  # 若无障碍，则重心为 ROI 中心

        # 误差为 ROI 中心与障碍物重心的水平距离（正值表示障碍物偏左，机器人应右转）
        error = roi_center_x - obstacle_cx

        # PD 控制：计算角速度
        derivative = (error - self.prev_error) / self.dt
        angular_z = self.Kp * error + self.Kd * derivative
        self.prev_error = error

        # 计算线速度：障碍物占比越高，前进速度越低
        linear_x = self.base_speed * (1 - min(obstacle_ratio, 1.0) * self.max_speed_reduction)
        # 当障碍物占比较高时（例如超过50%），直接停止前进
        if obstacle_ratio > 0.5:
            linear_x = 0.0

        # 构造 Twist 指令并发布
        twist = Twist()
        twist.linear.x = linear_x
        twist.angular.z = angular_z
        self.cmd_vel_pub.publish(twist)

        # 调试输出
        self.get_logger().info(
            f"误差: {error:.2f}, 导数: {derivative:.2f}, 障碍占比: {obstacle_ratio:.2f}, 线速度: {linear_x:.2f}, 角速度: {angular_z:.2f}"
        )

def main(args=None):
    rclpy.init(args=args)
    node = PIDNavigationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
