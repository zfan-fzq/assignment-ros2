#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import time

class TurtlebotNavigator(Node):
    def __init__(self):
        super().__init__('turtlebot_navigator')
        # 订阅相机话题
        self.image_sub = self.create_subscription(
            Image, 
            '/camera/image_raw',    # 根据实际情况修改话题名称
            self.image_callback, 
            10)
        # 发布速度指令
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.bridge = CvBridge()
        
        # PID 参数（可根据需要调节）
        self.Kp = 0.005
        self.Ki = 0.0
        self.Kd = 0.001
        self.prev_error = 0.0
        self.integral_error = 0.0
        
        # 速度设置
        self.normal_speed = 0.2   # 正常线速度（m/s）
        self.slow_speed = 0.05    # 减速时的速度（m/s）
        self.max_turn_speed = 1.0 # 最大角速度（rad/s）
        
        # 定义各区域的比例（根据图像尺寸调整）
        self.boundary_slow_frac = 0.7   # 图像底部 30% 为边界预警区
        self.boundary_stop_frac = 0.9   # 图像底部 10% 为紧急停止区
        
        # 紧急转向状态参数
        self.emergency_turn = False
        self.turn_start_time = None
        self.emergency_turn_duration = 2.0  # 紧急转向持续时间（秒），1 rad/s * 2 s ≈ 114°转角
        self.emergency_turn_direction = 0.0  # 紧急转向方向（正：左转，负：右转）
        
        # 障碍物检测阈值（轮廓面积，增大此值表示只有障碍物更近时才触发）
        self.obstacle_area_threshold = 2500

    def classify_lines(self, lines, image_shape):
        """
        根据检测到的线段简单分类：
        - 中场线：出现在图像上半部分且接近水平
        - 禁区线：介于中场线和边界之间
        - 边界线：出现在图像下部
        返回一个字典，包含三个类别对应的线段列表。
        """
        classifications = {"midfield": [], "penalty": [], "boundary": []}
        height, width = image_shape[0], image_shape[1]
        for line in lines:
            x1, y1, x2, y2 = line[0]
            avg_y = (y1 + y2) / 2.0
            # 计算线段角度（单位：度）
            angle = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))
            abs_angle = abs(angle)
            if avg_y < height * 0.4 and abs_angle < 10:
                classifications["midfield"].append(line)
            elif avg_y > height * 0.7:
                classifications["boundary"].append(line)
            else:
                classifications["penalty"].append(line)
        return classifications

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"图像转换失败: {e}")
            return
        
        height, width, _ = frame.shape
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # -------------------------------------
        # 1. 白线检测（用于检测边界、禁区、中场线）
        # -------------------------------------
        lower_white = np.array([0, 0, 200], dtype=np.uint8)
        upper_white = np.array([179, 50, 255], dtype=np.uint8)
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        
        # 使用 HoughLinesP 检测线段
        lines = cv2.HoughLinesP(white_mask, 1, np.pi / 180, threshold=40, minLineLength=30, maxLineGap=5)
        line_classes = {"midfield": [], "penalty": [], "boundary": []}
        if lines is not None:
            line_classes = self.classify_lines(lines, frame.shape)
            self.get_logger().info(
                f"中场线: {len(line_classes['midfield'])}, 禁区线: {len(line_classes['penalty'])}, 边界线: {len(line_classes['boundary'])}"
            )
        
        # 判断边界情况：检测图像底部白色区域情况
        slow_region_y = int(self.boundary_slow_frac * height)
        stop_region_y = int(self.boundary_stop_frac * height)
        slow_for_boundary = cv2.countNonZero(white_mask[slow_region_y:, :]) > 0
        near_boundary = cv2.countNonZero(white_mask[stop_region_y:, :]) > 0

        # -------------------------------------
        # 2. 障碍物检测（以 construction cone 为例，假设为橙色）
        # -------------------------------------
        lower_orange = np.array([5, 100, 100], dtype=np.uint8)
        upper_orange = np.array([15, 255, 255], dtype=np.uint8)
        orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
        orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel)
        
        # 仅关注图像下半部分区域（假设障碍物主要出现在机器人前方）
        roi_obstacle = orange_mask[int(height * 0.5):, :]
        # 利用轮廓提取判断障碍物是否非常近
        contours, _ = cv2.findContours(roi_obstacle, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        obstacle_close = False
        largest_contour = None
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            if area > self.obstacle_area_threshold:
                obstacle_close = True

        # -------------------------------------
        # 3. 紧急转向处理（优先处理紧急状态）
        # -------------------------------------
        if self.emergency_turn:
            elapsed = time.time() - self.turn_start_time
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = self.emergency_turn_direction  # 固定角速度转动
            self.cmd_pub.publish(twist)
            self.get_logger().info(f"紧急转向中，已转向 {elapsed:.1f}s")
            if elapsed >= self.emergency_turn_duration:
                self.emergency_turn = False
                self.turn_start_time = None
                self.get_logger().info("紧急转向完成")
            return  # 紧急转向期间不处理其他逻辑

        # -------------------------------------
        # 4. 检测障碍物紧急情况：只有当障碍物非常近时才触发
        # -------------------------------------
        if obstacle_close:
            # 利用障碍物轮廓质心判断转向方向
            M = cv2.moments(largest_contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
            else:
                cx = width / 2
            # 如果障碍物质心在图像左侧，则向右转；反之则向左转
            if cx < width / 2:
                self.emergency_turn_direction = -1.0  # 右转
            else:
                self.emergency_turn_direction = 1.0   # 左转
            self.emergency_turn = True
            self.turn_start_time = time.time()
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = self.emergency_turn_direction
            self.cmd_pub.publish(twist)
            self.get_logger().info("检测到近距离障碍物，启动紧急转向")
            return

        # -------------------------------------
        # 5. 检测边界紧邻情况，启动紧急转向
        # -------------------------------------
        if near_boundary:
            # 根据左右区域白色分布决定转向方向
            left_white = cv2.countNonZero(white_mask[stop_region_y:, 0:width//2])
            right_white = cv2.countNonZero(white_mask[stop_region_y:, width//2:])
            if right_white > left_white:
                self.emergency_turn_direction = 1.0   # 向左转
            else:
                self.emergency_turn_direction = -1.0  # 向右转
            self.emergency_turn = True
            self.turn_start_time = time.time()
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = self.emergency_turn_direction
            self.cmd_pub.publish(twist)
            self.get_logger().info("紧急情况：接近边界，启动紧急转向")
            return
        
        # -------------------------------------
        # 6. 正常导航（无紧急情况时）
        # -------------------------------------
        twist = Twist()
        twist.linear.x = self.normal_speed
        if slow_for_boundary:
            twist.linear.x = self.slow_speed
            self.get_logger().info("检测到白线：减速")
        
        # 利用禁区线或中场线辅助调整航向
        if len(line_classes["penalty"]) > 0:
            xs = []
            for line in line_classes["penalty"]:
                x1, _, x2, _ = line[0]
                xs.append((x1 + x2) / 2)
            if xs:
                penalty_center = np.mean(xs)
                error = penalty_center - (width / 2)
                p_term = error
                self.integral_error += error * 0.1
                d_term = (error - self.prev_error) / 0.1
                ang_z = self.Kp * p_term + self.Ki * self.integral_error + self.Kd * d_term
                ang_z = np.clip(ang_z, -self.max_turn_speed, self.max_turn_speed)
                twist.angular.z = -ang_z  # 负号保证转向与误差方向相反
                self.prev_error = error
                self.get_logger().info(f"禁区线检测：error={error:.2f}, ang_z={twist.angular.z:.2f}")
        elif len(line_classes["midfield"]) > 0:
            xs = []
            for line in line_classes["midfield"]:
                x1, _, x2, _ = line[0]
                xs.append((x1 + x2) / 2)
            if xs:
                midfield_center = np.mean(xs)
                error = midfield_center - (width / 2)
                p_term = error
                self.integral_error += error * 0.1
                d_term = (error - self.prev_error) / 0.1
                ang_z = self.Kp * p_term + self.Ki * self.integral_error + self.Kd * d_term
                ang_z = np.clip(ang_z, -self.max_turn_speed, self.max_turn_speed)
                twist.angular.z = -ang_z
                self.prev_error = error
                self.get_logger().info(f"中场线检测：error={error:.2f}, ang_z={twist.angular.z:.2f}")
        else:
            self.integral_error *= 0.9
            self.prev_error *= 0.9
        
        self.cmd_pub.publish(twist)
        # 可选：调试时显示图像（启用时取消注释）
        # cv2.imshow("White Mask", white_mask)
        # cv2.imshow("Orange Mask", orange_mask)
        # cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    navigator = TurtlebotNavigator()
    try:
        rclpy.spin(navigator)
    except KeyboardInterrupt:
        navigator.get_logger().info("导航节点被用户中断")
    finally:
        navigator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
