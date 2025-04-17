#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import sys
import select
import termios
import tty
import threading
import time
import threading
import time
import math

import tf2_ros
from tf_transformations import euler_from_quaternion

class ArucoNavigator(Node):
    def __init__(self):
        super().__init__('aruco_navigator')
        self.get_logger().info('Aruco navigator started.')

        # Publishers & subscribers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.bridge = CvBridge()

        # TF2 for map (world) → base_link
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ArUco setup
        self.aruco_dict    = cv2.aruco.getPredefinedDictionary(
                                cv2.aruco.DICT_4X4_50)
        self.marker_length = 0.16  # 16 cm

        # Dummy intrinsics (replace with your calibration)
        self.camera_matrix = np.array([
            [525.0,   0.0, 320.5],
            [  0.0, 525.0, 240.5],
            [  0.0,   0.0,   1.0]
        ], dtype=np.float32)
        self.dist_coeffs = np.zeros((5, 1))

        self.robot_x = None
        self.robot_y = None

        self.marker_positions = {}

        # once we have 4 corners, compute this center
        self.target_center = None     # (x, z)

        # timer to run control at 10 Hz

        # Keyboard control (optional)
        if sys.stdin.isatty():
            self.keyboard_input = sys.stdin
        else:
            try:
                self.keyboard_input = open('/dev/tty')
            except Exception:
                self.keyboard_input = None

        if self.keyboard_input:
            thr = threading.Thread(target=self.keyboard_loop, daemon=True)
            thr.start()

    def keyboard_loop(self):
        settings = termios.tcgetattr(self.keyboard_input)
        def getKey():
            tty.setraw(self.keyboard_input.fileno())
            r, _, _ = select.select([self.keyboard_input], [], [], 0.1)
            key = self.keyboard_input.read(1) if r else ''
            termios.tcsetattr(self.keyboard_input, termios.TCSADRAIN, settings)
            return key

        self.get_logger().info("Keyboard: W/A/S/D to move, Q to quit")
        while rclpy.ok():
            k = getKey()
            t = Twist()
            if k=='w':      t.linear.x = 0.5
            elif k=='s':    t.linear.x = -0.5
            elif k=='a':    t.angular.z = 0.5
            elif k=='d':    t.angular.z = -0.5
            elif k == 'g':
                threading.Thread(target=self.gotoposition, daemon=True).start()
            elif k=='q':
                self.get_logger().info("Shutting down.")
                rclpy.shutdown()
                break
            self.cmd_pub.publish(t)
            time.sleep(0.1)

    def image_callback(self, msg):
        try:
            # 1) CV conversion & detect
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict)
            if ids is None:
                return

            # 2) Pose in camera frame
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_length,
                self.camera_matrix, self.dist_coeffs)

            # 3) Robot pose in world/map
            tf_map_base = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            t = tf_map_base.transform.translation
            q = tf_map_base.transform.rotation
            _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

            # 4) For each marker, compute world‐relative vector
            for i, mid in enumerate(ids.flatten()):
                x_cam, _, z_cam = tvecs[i][0]  # drop camera height

                # rotate into world frame → this is the RELATIVE vector
                dx_world =  np.cos(yaw)*x_cam - np.sin(yaw)*z_cam
                dz_world =  np.sin(yaw)*x_cam + np.cos(yaw)*z_cam
                dist_world = np.hypot(dx_world, dz_world)

                # self.get_logger().info(
                #     f"Marker {mid} relative in world frame: "
                #     f"dx={dx_world:.2f} m, dz={dz_world:.2f} m "
                #     f"(d={dist_world:.2f} m)\n"
                # )

                if mid == 1:
                    self.robot_x = -dx_world
                    self.robot_z = -dz_world
                    self.marker_positions[mid] = (0, 0)
                    self.get_logger().info(
                        f"Robot position (world frame): x={self.robot_x:.2f} m, z={self.robot_z:.2f} m\n"
                    )

                if mid == 2 and self.robot_x is not None:
                    # compute marker 2 world pose
                    m2_x = self.robot_x + dx_world
                    m2_z = self.robot_z + dz_world
                    self.marker_positions[mid] = (m2_x, m2_z)

                if mid == 3 and self.robot_x is not None:
                    # compute marker 2 world pose
                    m2_x = self.robot_x + dx_world
                    m2_z = self.robot_z + dz_world
                    self.marker_positions[mid] = (m2_x, m2_z)


                if mid == 4 and self.robot_x is not None:
                    # compute marker 2 world pose
                    m2_x = self.robot_x + dx_world
                    m2_z = self.robot_z + dz_world
                    self.marker_positions[mid] = (m2_x, m2_z)

                # draw axes for sanity
                # cv2.aruco.drawAxis(
                #     img, self.camera_matrix, self.dist_coeffs,
                #     rvecs[i], tvecs[i], self.marker_length*0.5)
            
            for mid, (mx, mz) in self.marker_positions.items():
                self.get_logger().info(f"Marker {mid}: x={mx:.2f} m, z={mz:.2f} m")


            if len(self.marker_positions) >= 4 and self.target_center is None:
                xs = [p[0] for p in self.marker_positions.values()]
                zs = [p[1] for p in self.marker_positions.values()]
                self.target_center = (sum(xs)/4.0, sum(zs)/4.0)
                self.get_logger().info(
                    f"Quad center: x={self.target_center[0]:.2f} m, z={self.target_center[1]:.2f} m"
                )

            # debug view
            # cv2.imshow('Aruco', img)
            # cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Image processing error: {e}")

    def gotoposition(self):
        if self.target_center is None:
            self.get_logger().warn("Center not yet computed.")
            return
        self.get_logger().info("Navigating to center...")
        rate = self.create_rate(10)
        while rclpy.ok():
            try:
                tf_map_base = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
                t = tf_map_base.transform.translation
                q = tf_map_base.transform.rotation
                _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
                robot_x, robot_z = t.x, t.z
            except Exception:
                continue

            cx, cz = self.target_center
            dx = cx - robot_x
            dz = cz - robot_z
            dist = math.hypot(dx, dz)
            if dist < 0.05:
                self.get_logger().info("Arrived at center.")
                self.cmd_pub.publish(Twist())
                break

            theta_target = math.atan2(dx, dz)
            angle_error = (theta_target - yaw + math.pi) % (2*math.pi) - math.pi

            cmd = Twist()
            if abs(angle_error) > 0.1:
                cmd.angular.z = 1.0 * angle_error
            else:
                cmd.linear.x = 0.3 * dist

            cmd.linear.x  = max(min(cmd.linear.x, 0.5), -0.5)
            cmd.angular.z = max(min(cmd.angular.z, 1.0), -1.0)
            self.cmd_pub.publish(cmd)
            rate.sleep()

def main(args=None):
    rclpy.init(args=args)
    node = ArucoNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__=='__main__':
    main()
