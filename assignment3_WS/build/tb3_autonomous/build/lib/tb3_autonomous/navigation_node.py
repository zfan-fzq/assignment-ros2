#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import math

class TurtleBot3VisionNavigator(Node):
    def __init__(self):
        super().__init__('tb3_vision_navigator')
        # 订阅摄像头数据话题
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            qos_profile_sensor_data
        )
        # 发布速度指令
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.bridge = CvBridge()
        self.forward_speed = 0.2   # 前进速度
        self.turn_speed = 0.5      # 转弯速度

    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CvBridge 转换图像失败: {e}")
            return

        # 灰度化和模糊处理
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # 霍夫变换检测直线
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=40, maxLineGap=10)

        obstacle_detected = False      
        boundary_front_detected = False  
        boundary_left_detected = False   
        boundary_right_detected = False  

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                if angle < 0:
                    angle += 180.0
                mid_x = (x1 + x2) / 2.0
                mid_y = (y1 + y2) / 2.0

                # 判断水平直线
                if angle < 15.0 or angle > 165.0:
                    if mid_y > cv_image.shape[0] * 0.6:
                        boundary_front_detected = True
                # 判断垂直直线
                elif 75.0 < angle < 105.0:
                    if mid_x < cv_image.shape[1] * 0.3:
                        boundary_left_detected = True
                    elif mid_x > cv_image.shape[1] * 0.7:
                        boundary_right_detected = True
                    else:
                        obstacle_detected = True
                else:
                    if cv_image.shape[1] * 0.3 < mid_x < cv_image.shape[1] * 0.7:
                        obstacle_detected = True

        twist = Twist()
        if boundary_left_detected or boundary_right_detected or boundary_front_detected:
            twist.linear.x = 0.0
            if boundary_left_detected:
                twist.angular.z = -abs(self.turn_speed)
            elif boundary_right_detected:
                twist.angular.z = abs(self.turn_speed)
            else:
                twist.angular.z = abs(self.turn_speed)
        elif obstacle_detected:
            twist.linear.x = 0.0
            twist.angular.z = abs(self.turn_speed)
        else:
            twist.linear.x = self.forward_speed
            twist.angular.z = 0.0

        self.cmd_vel_pub.publish(twist)
        if obstacle_detected or boundary_front_detected or boundary_left_detected or boundary_right_detected:
            self.get_logger().info(
                f"Obstacle: {obstacle_detected}, Boundary front: {boundary_front_detected}, "
                f"left: {boundary_left_detected}, right: {boundary_right_detected}"
            )

def main(args=None):
    rclpy.init(args=args)
    node = TurtleBot3VisionNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
