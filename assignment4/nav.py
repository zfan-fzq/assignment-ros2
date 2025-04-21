#!/usr/bin/env python3

import select

import termios
import threading
import time
import math
from collections import deque, defaultdict

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

from cv_bridge import CvBridge
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


class ArucoNavigator(Node):
    def __init__(self):
        super().__init__("aruco_navigator")
        self.get_logger().info("Aruco navigator started.")

        # QoS for image subscriber
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Publishers & subscribers
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.image_sub = self.create_subscription(
            Image, "/camera/color/image_raw", self.image_callback, qos_profile=qos
        )
        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self.odom_callback, 50
        )
        self.bridge = CvBridge()

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,  # BEST_EFFORT: attempt to deliver samples, but may lose them if the network is not robust.
            durability=DurabilityPolicy.VOLATILE,  # VOLATILE: no attempt is made to persist samples.
            history=HistoryPolicy.KEEP_LAST,  # KEEP_LAST: only store up to N samples, configurable via the queue depth option.
            depth=10,  # a queue size of 10 to buffer messages if they arrive faster than they can be processed
        )

        self.scan_subscription = self.create_subscription(
            LaserScan,
            "scan",
            self.scan_callback,
            qos_profile=qos_profile,  # Replace with your lidar topic
        )

        # ArUco detection params
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.marker_length = 0.16  # meters
        self.camera_matrix = np.array(
            [
                [456.82000732, 0.0, 326.66424561],
                [0.0, 456.82000732, 243.38911438],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)

        # initialize

        # Odometry state
        self.odom_x = self.odom_y = self.odom_yaw = 0.0
        self.odom0_x = 0.0
        self.odom0_y = 0.0
        self.odom0_yaw = 0.0

        # internal state
        self._search_phase = "spin"  # or "pause"
        self._phase_start = time.time()

        self.marker_distance = {}
        self.marker_camera = {}
        self.marker_update = {1: False, 2: False, 3: False, 4: False}

        # Map-frame state (after initialization)
        self.robot_x = self.robot_y = self.robot_yaw = 0.0
        self.offset_x = self.offset_y = self.offset_yaw = 0.0
        self.initialized = False

        # Marker estimates & smoothing buffers
        self.marker_positions = {}  # mid -> (x,y)
        self._marker_buffers = defaultdict(lambda: deque(maxlen=5))

        # Center point & navigation state
        self.target_center = None
        self.state = None

        self.obstacle_too_close = False

        self.T_odom_to_map = np.eye(3)  # 3x3 identity matrix

        self.base_id = None
        self.id_map = {}

        # Control loop timer
        self.create_timer(0.1, self.control_loop)

        self.alpha = 0.2  # EMA weight for updates

        # how long to spin (s) and pause (s)
        self.search_spin_time = 0.2  # spin for 0.5 s
        self.search_pause_time = 0.5  # then pause 1 s

        # control gains & thresholds
        self.kp_ang = 1.0  # radian error → angular speed
        self.max_ang_vel = 0.5  # rad/s
        self.forward_speed = 0.3  # m/s
        self.angle_tol = 0.15  # tolerance
        self.dist_tol = 0.9  # stop when closer than 1 m

        # Keyboard setup
        self._setup_keyboard()

    def reset(self):
        # Odometry state
        self.odom_x = self.odom_y = self.odom_yaw = 0.0
        self.odom0_x = 0.0
        self.odom0_y = 0.0
        self.odom0_yaw = 0.0

        # internal state
        self._search_phase = "spin"  # or "pause"
        self._phase_start = time.time()

        self.marker_distance = {}
        self.marker_camera = {}
        self.marker_update = {1: False, 2: False, 3: False, 4: False}

        # Map-frame state (after initialization)
        self.robot_x = self.robot_y = self.robot_yaw = 0.0
        self.offset_x = self.offset_y = self.offset_yaw = 0.0
        self.initialized = False

        # Marker estimates & smoothing buffers
        self.marker_positions = {}  # mid -> (x,y)
        self._marker_buffers = defaultdict(lambda: deque(maxlen=5))

        # Center point & navigation state
        self.target_center = None

        self.obstacle_too_close = False

        self.T_odom_to_map = np.eye(3)  # 3x3 identity matrix

        self.base_id = None
        self.id_map = {}

    def scan_callback(self, scan: LaserScan):
        # we only care about +/-10° around straight ahead (i.e. index center)
        # LaserScan.angle_min, angle_increment → compute index range
        mid = len(scan.ranges) // 2
        window = int(math.radians(10) / scan.angle_increment)
        front_ranges = scan.ranges[mid - window : mid + window + 1]
        # filter out invalid readings
        front = [r for r in front_ranges if not math.isinf(r)]

        if front and min(front) < 0.5:
            self.obstacle_too_close = True
        else:
            self.obstacle_too_close = False

    def _setup_keyboard(self):
        try:
            self._tty = open("/dev/tty")
        except OSError:
            self.get_logger().warn("Could not open /dev/tty for keyboard input")
            return

        fd = self._tty.fileno()
        self._orig_termios = termios.tcgetattr(fd)
        new_t = termios.tcgetattr(fd)
        new_t[3] &= ~(termios.ECHO | termios.ICANON)
        new_t[3] |= termios.ISIG
        new_t[6][termios.VMIN] = 0
        new_t[6][termios.VTIME] = 0
        termios.tcsetattr(fd, termios.TCSADRAIN, new_t)

        def keyboard_loop():
            self.get_logger().info(
                "Keyboard: W/A/S/D to move, G to go to center, Q to quit"
            )
            try:
                while rclpy.ok():
                    r, _, _ = select.select([self._tty], [], [], 0.1)
                    k = self._tty.read(1) if r else ""
                    if k == "\x03" or k == "q":
                        self.get_logger().info("Shutting down.")
                        rclpy.shutdown()
                        break

                    cmd = Twist()
                    if k == "w":
                        cmd.linear.x = 0.5
                    elif k == "s":
                        cmd.linear.x = -0.5
                    elif k == "a":
                        cmd.angular.z = 1.5
                    elif k == "d":
                        cmd.angular.z = -1.5
                    elif k == "i":
                        self.state = "explore"
                    elif k == "g":
                        threading.Thread(target=self.gotoposition).start()
                    elif k == "2":
                        self.state = "approach2"
                    elif k == "3":
                        self.state = "approach3"
                    elif k == "4":
                        self.state = "approach4"
                    elif k == "n":
                        self.reset()
                        self.state = "turning_go"

                    if k in ("w", "a", "s", "d"):
                        self.state = "manual"
                    elif k == "i":
                        self.state = "explore"
                    elif k == "g":
                        self.state = "goto"
                    # if cmd.linear.x > 0 and self.obstacle_too_close:
                    #     cmd.linear.x = 0.0
                    #     self.get_logger().warn("Stopping: obstacle < 0.5m ahead")
                    if self.state == "manual":
                        self.cmd_pub.publish(cmd)
                    time.sleep(0.1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, self._orig_termios)
                try:
                    self._tty.close()
                except:
                    pass

        threading.Thread(target=keyboard_loop).start()

    def odom_callback(self, msg: Odometry):
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        r = R.from_quat([q.x, q.y, q.z, q.w])
        e_xyz = r.as_euler("xyz", degrees=False)
        self.odom_x, self.odom_y, self.odom_yaw = px, py, e_xyz[2]

        # logging all odometry values
        # self.get_logger().info(
        #     f"Odometry: px={px:.2f}, py={py:.2f}, odom_yaw={math.degrees(e_xyz[2]):.2f}"
        # )

        if self.initialized:
            # 2) apply your homogeneous T_odom_to_map transform
            T = self.T_odom_to_map  # 3×3 matrix: [[c, -s, tx], [s, c, ty], [0,0,1]]
            # multiply [odom_x, odom_y, 1]
            x_map = T[0, 0] * self.odom_x + T[0, 1] * self.odom_y + T[0, 2]
            y_map = T[1, 0] * self.odom_x + T[1, 1] * self.odom_y + T[1, 2]
            self.robot_x, self.robot_y = x_map, y_map

            # 3) yaw just adds the offset_yaw (and wrap to [-π,π])
            raw = self.odom_yaw + self.offset_yaw
            self.robot_yaw = (raw + math.pi) % (2 * math.pi) - math.pi

            # self.get_logger().info(
            #     f"Robot: x={self.robot_x:.2f}, y={self.robot_y:.2f}, degree={math.degrees(self.robot_yaw):.2f}"
            # )

    def image_callback(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict)
            if ids is None:
                # self.get_logger().info("No markers detected.")
                return

            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_length, self.camera_matrix, self.dist_coeffs
            )

            for i, mid in enumerate(ids.flatten()):
                x_cam, _, z_cam = tvecs[i][0]
                forward, left = z_cam, -x_cam

                if self.base_id is None and mid in (1, 3):
                    # first-ever detected marker becomes the “1”
                    self.base_id = mid
                    # build a map so that ids rotate: base_id→1, base_id+1→2, etc.
                    self.id_map = {
                        orig: ((orig - self.base_id) % 4) + 1 for orig in (1, 2, 3, 4)
                    }
                    self.get_logger().info(
                        f"Using marker {self.base_id} as base → id_map={self.id_map}"
                    )

                mapped_mid = self.id_map.get(mid, mid)

                # Initial alignment on marker 1
                if mapped_mid == 1:

                    # compute camera yaw relative to marker
                    R_cm, _ = cv2.Rodrigues(rvecs[i][0])  # marker → camera
                    R_mc = R_cm.T  # camera → marker

                    # 1) get the ZYX angles (camera→marker)
                    e_xyz = R.from_matrix(R_mc).as_euler("xyz", degrees=False)

                    # 2) extract the pitch (second element)
                    yaw_cam = e_xyz[1]

                    marker_world_yaw = -math.pi / 2
                    measured_yaw = marker_world_yaw + yaw_cam

                    dx = (
                        math.cos(self.robot_yaw) * forward
                        - math.sin(self.robot_yaw) * left
                    )
                    dy = (
                        math.sin(self.robot_yaw) * forward
                        + math.cos(self.robot_yaw) * left
                    )

                    # self.get_logger().info(f"Marker {mid}: dx={dx:.2f}, dy={dy:.2f}")

                    meas_x, meas_y = -dx, -dy

                    # logging forword, left, yaw and dx and dy
                    # self.get_logger().info(
                    #     f"Marker {mid}: forward={forward:.2f}, left={left:.2f}"
                    # )

                    if not self.initialized:

                        # first detection: set offsets directly
                        self.robot_x, self.robot_y, self.robot_yaw = (
                            meas_x,
                            meas_y,
                            measured_yaw,
                        )

                        self.odom0_x = self.odom_x
                        self.odom0_y = self.odom_y
                        self.odom0_yaw = self.odom_yaw

                        # your odom pose at that same instant
                        ox, oy, oyaw = self.odom_x, self.odom_y, self.odom_yaw

                        # 1) compute yaw offset
                        delta_yaw = self.robot_yaw - oyaw
                        c, s = math.cos(delta_yaw), math.sin(delta_yaw)

                        # 2) compute translation offset so that:
                        #    [x_map]   [ c -s  tx ] [x_odom]
                        #    [y_map] = [ s  c  ty ] [y_odom]
                        #    [  1  ]   [ 0  0   1 ] [  1   ]
                        tx = self.robot_x - (c * ox - s * oy)
                        ty = self.robot_y - (s * ox + c * oy)

                        # 3) stash your full transform matrix
                        T = np.array(
                            [
                                [c, -s, tx],
                                [s, c, ty],
                                [0, 0, 1],
                            ]
                        )
                        self.T_odom_to_map = T

                        self.offset_x = tx
                        self.offset_y = ty
                        self.offset_yaw = delta_yaw

                        self.initialized = True
                        self.get_logger().info(
                            f"Init from marker1 Robot: x={self.robot_x:.2f}, y={self.robot_y:.2f}, yaw={math.degrees(measured_yaw):.2f}"
                        )
                    else:
                        self.robot_x, self.robot_y, self.robot_yaw = (
                            meas_x,
                            meas_y,
                            measured_yaw,
                        )

                        self.odom0_x = self.odom_x
                        self.odom0_y = self.odom_y
                        self.odom0_yaw = self.odom_yaw

                        # your odom pose at that same instant
                        ox, oy, oyaw = self.odom_x, self.odom_y, self.odom_yaw

                        # 1) compute yaw offset
                        delta_yaw = self.robot_yaw - oyaw
                        c, s = math.cos(delta_yaw), math.sin(delta_yaw)

                        # 2) compute translation offset so that:
                        #    [x_map]   [ c -s  tx ] [x_odom]
                        #    [y_map] = [ s  c  ty ] [y_odom]
                        #    [  1  ]   [ 0  0   1 ] [  1   ]
                        tx = self.robot_x - (c * ox - s * oy)
                        ty = self.robot_y - (s * ox + c * oy)

                        # 3) stash your full transform matrix
                        T = np.array(
                            [
                                [c, -s, tx],
                                [s, c, ty],
                                [0, 0, 1],
                            ]
                        )
                        self.T_odom_to_map = T

                        self.offset_x = tx
                        self.offset_y = ty
                        self.offset_yaw = delta_yaw

                        # self.get_logger().info(f"Refined from marker1")

                    # always store marker1 at origin
                    self.marker_positions[mapped_mid] = (0.0, 0.0)
                    self.marker_distance[mapped_mid] = forward
                    self.marker_camera[mapped_mid] = (x_cam, 0.0, z_cam)
                    self.marker_update[mapped_mid] = True

                    if self.state == "explore":
                        self.state = "approach1"
                        self.get_logger().info(
                            "→ marker1 spotted, switching to APPROACH1"
                        )

                # For markers 2–4, after initialization
                if self.initialized and mapped_mid in (2, 3, 4):
                    self.marker_distance[mapped_mid] = forward
                    self.marker_camera[mapped_mid] = (x_cam, 0.0, z_cam)
                    self.marker_update[mapped_mid] = True
                    # self.get_logger().info(
                    #     f"update marker camera {mapped_mid}: {x_cam:.2f}, {z_cam:.2f}"
                    # )

                    # rotate into world frame by current robot_yaw
                    dx = (
                        math.cos(self.robot_yaw) * forward
                        - math.sin(self.robot_yaw) * left
                    )
                    dy = (
                        math.sin(self.robot_yaw) * forward
                        + math.cos(self.robot_yaw) * left
                    )

                    # self.get_logger().info(
                    #     f"Marker {mid}: forward={forward:.2f}, left={left:.2f}"
                    # )
                    # logging dx and dy
                    # self.get_logger().info(f"Marker {mid}: dx={dx:.2f}, dy={dy:.2f}")

                    # absolute marker position in world
                    meas_mx = self.robot_x + dx
                    meas_my = self.robot_y + dy

                    # smooth via EMA if we’ve seen this marker before
                    if mapped_mid in self.marker_positions:
                        old_x, old_y = self.marker_positions[mapped_mid]
                        mx = (1 - self.alpha) * old_x + self.alpha * meas_mx
                        my = (1 - self.alpha) * old_y + self.alpha * meas_my
                    else:
                        mx, my = meas_mx, meas_my

                    self.marker_positions[mapped_mid] = (mx, my)
                    self.get_logger().info(
                        f"Marker {mapped_mid}: x={mx:.2f}, y={my:.2f}"
                    )

            # logging all marker positions
            self.get_logger().info("Marker positions:")
            for mid, (mx, my) in self.marker_positions.items():
                self.get_logger().info(f"Marker {mid}: x={mx:.2f}, y={my:.2f}")

            # compute center when all 4 markers seen
            if (
                self.initialized
                and len(self.marker_positions) >= 4
                and self.target_center is None
            ):
                xs = [p[0] for p in self.marker_positions.values()]
                ys = [p[1] for p in self.marker_positions.values()]
                self.target_center = (sum(xs) / 4.0, sum(ys) / 4.0)

                if self.state == "turning_go":
                    self.state = "goto"
                    self.get_logger().info("→ all markers seen, switching to GOTO")
                    threading.Thread(target=self.gotoposition).start()

                self.get_logger().info(
                    f"Quad center: x={self.target_center[0]:.2f}, y={self.target_center[1]:.2f}"
                )

        except Exception as e:
            self.get_logger().error(f"Image processing error: {e}")

    def wrap_angle(self, a: float) -> float:
        return (a + math.pi) % (2 * math.pi) - math.pi

    def control_loop(self):
        auto_states = (
            "explore",
            "approach1",
            "approach2",
            "approach3",
            "approach4",
            "turning_go",
        )
        if self.state not in auto_states:
            return

        twist = Twist()

        if self.state == "turning_go":
            now = time.time()
            elapsed = now - self._phase_start

            if self._search_phase == "spin":
                # fast spin
                twist.angular.z = 2.0

                # after spin time, switch to pause
                if elapsed >= self.search_spin_time:
                    self._search_phase = "pause"
                    self._phase_start = now

            else:  # pause phase
                twist.angular.z = 0.0

                # after pause time, switch back to spin
                if elapsed >= self.search_pause_time:
                    self._search_phase = "spin"
                    self._phase_start = now
        # 1) spin until marker1
        elif self.state == "explore":
            twist.angular.z = -0.15

        # 2) approach marker1 by ArUco‐bearing
        elif self.state == "approach1":
            self.get_logger().info(f"marker1 update: {self.marker_update[1]}")
            angle_error = math.atan2(self.marker_camera[1][0], self.marker_camera[1][2])
            # rotate to center
            if abs(angle_error) > self.angle_tol:
                vel = max(
                    -self.max_ang_vel, min(self.max_ang_vel, self.kp_ang * angle_error)
                )
                twist.angular.z = -vel * 0.4
                self.get_logger().info(
                    f"approach1: angle_error={math.degrees(angle_error):.2f}, robot_yaw={math.degrees(self.robot_yaw):.2f}"
                )
            # drive straight if centered
            elif self.marker_distance[1] > 0.7:
                twist.linear.x = self.forward_speed
            else:
                # done with marker1 → move to approach2
                twist = Twist()
                self.state = "approach2"
                # record where we start this leg
                self.start_x, self.start_y = self.robot_x, self.robot_y
                self.get_logger().info("→ reached marker1, switching to APPROACH2")

        # 3) approach2: turn to +90° then go 1 m forward
        elif self.state == "approach2":

            desired_yaw = math.radians(90)
            yaw_err = self.wrap_angle(desired_yaw - self.robot_yaw)
            if abs(yaw_err) > self.angle_tol and 2 not in self.marker_distance:
                twist.angular.z = self.kp_ang * yaw_err
                self.get_logger().info("approach2: turning")
            else:
                if 2 not in self.marker_distance:
                    self.get_logger().warn("Marker 2 not found!")
                    twist.linear.x = self.forward_speed
                else:
                    angle_error = math.atan2(
                        self.marker_camera[2][0], self.marker_camera[2][2]
                    )
                    # rotate to center
                    if abs(angle_error) > self.angle_tol:
                        vel = max(
                            -self.max_ang_vel,
                            min(self.max_ang_vel, self.kp_ang * angle_error),
                        )
                        twist.angular.z = -vel * 0.4

                        self.get_logger().info(
                            f"approach2: angle_error={math.degrees(angle_error):.2f}, robot_yaw={math.degrees(self.robot_yaw):.2f}"
                        )
                    elif self.marker_distance[2] > self.dist_tol:
                        self.get_logger().info(
                            f"approach2: {self.marker_distance[2]:.2f}"
                        )
                        twist.linear.x = self.forward_speed
                    else:
                        # done → approach3
                        twist = Twist()
                        self.state = "approach3"
                        self.start_x, self.start_y = self.robot_x, self.robot_y
                        self.get_logger().info("→ done leg2, switching to APPROACH3")

        # 4) approach3: turn to 0° then go 1 m
        elif self.state == "approach3":
            desired_yaw = 0.0
            yaw_err = self.wrap_angle(desired_yaw - self.robot_yaw)
            if abs(yaw_err) > self.angle_tol and 3 not in self.marker_distance:
                twist.angular.z = self.kp_ang * yaw_err
                self.get_logger().info("approach3: turning")
            else:
                # check self.marker_distance have key 3
                if 3 not in self.marker_distance:
                    self.get_logger().warn("Marker 3 not found!")
                    twist.linear.x = self.forward_speed
                else:
                    angle_error = math.atan2(
                        self.marker_camera[3][0], self.marker_camera[3][2]
                    )
                    # rotate to center
                    if self.marker_update[3] and abs(angle_error) > self.angle_tol:
                        vel = max(
                            -self.max_ang_vel,
                            min(self.max_ang_vel, self.kp_ang * angle_error),
                        )
                        twist.angular.z = -vel * 0.4
                        self.get_logger().info(
                            f"approach3: angle_error={math.degrees(angle_error):.2f}, robot_yaw={math.degrees(self.robot_yaw):.2f}"
                        )
                    elif self.marker_distance[3] > self.dist_tol:
                        self.get_logger().info(
                            f"approach3: {self.marker_distance[3]:.2f}"
                        )
                        twist.linear.x = self.forward_speed
                    else:
                        twist = Twist()
                        self.state = "approach4"
                        self.start_x, self.start_y = self.robot_x, self.robot_y
                        self.get_logger().info("→ done leg3, switching to APPROACH4")

        # 5) approach4: turn to –90° then go 1 m
        elif self.state == "approach4":
            desired_yaw = math.radians(-90)
            yaw_err = self.wrap_angle(desired_yaw - self.robot_yaw)
            if abs(yaw_err) > self.angle_tol and 4 not in self.marker_distance:
                twist.angular.z = self.kp_ang * yaw_err
                self.get_logger().info("approach4: turning")
            else:
                if 4 not in self.marker_distance:
                    self.get_logger().warn("Marker 4 not found!")
                    twist.linear.x = self.forward_speed
                else:
                    angle_error = math.atan2(
                        self.marker_camera[4][0], self.marker_camera[4][2]
                    )
                    # rotate to center
                    if abs(angle_error) > self.angle_tol:
                        vel = max(
                            -self.max_ang_vel,
                            min(self.max_ang_vel, self.kp_ang * angle_error),
                        )
                        twist.angular.z = -vel * 0.4
                    elif self.marker_distance[4] > self.dist_tol:
                        self.get_logger().info(
                            f"approach4: {self.marker_distance[4]:.2f}"
                        )
                        twist.linear.x = self.forward_speed
                    else:
                        twist = Twist()
                        self.state = "goto"
                        threading.Thread(target=self.gotoposition).start()
                        self.get_logger().info("→ done leg4, all markers visited!")

        # 6) any other state → safe stop
        else:
            twist = Twist()

        for mid in self.marker_update:
            self.marker_update[mid] = False

        self.cmd_pub.publish(twist)

    def gotoposition(self):
        if not self.initialized or self.target_center is None:
            self.get_logger().warn("Not ready: waiting for initial pose & all markers.")
            return

        self.get_logger().info("Navigating to center…")
        rate = self.create_rate(10)
        while rclpy.ok():
            dx = self.target_center[0] - self.robot_x
            dy = self.target_center[1] - self.robot_y
            dist = math.hypot(dx, dy)
            if dist < 0.2:
                self.get_logger().info("Arrived at center.")
                break

            theta = math.atan2(dy, dx)
            err = (theta - self.robot_yaw + math.pi) % (2 * math.pi) - math.pi
            cmd = Twist()
            if abs(err) > 0.4:
                # turn faster the larger the error, but keep a minimum base turn
                turn_speed = 0.1 + 1.5 * abs(err)
                # choose direction based on sign of err
                cmd.angular.z = turn_speed if err > 0 else -turn_speed
            else:
                # only drive forward when roughly facing the target
                cmd.linear.x = 0.1 + 0.5 * dist

            cmd.linear.x = max(min(cmd.linear.x, 0.5), -0.5)
            cmd.angular.z = max(min(cmd.angular.z, 1.0), -1.0)
            if self.state == "goto":
                self.cmd_pub.publish(cmd)
            self.get_logger().info(
                f"center: x={self.target_center[0]:.2f}, y={self.target_center[1]:.2f}"
            )
            self.get_logger().info(
                f"Robot: x={self.robot_x:.2f}, y={self.robot_y:.2f}, degree={math.degrees(self.robot_yaw):.2f}"
            )
            self.get_logger().info(f"To center: dist={dist:.2f}, angle_err={err:.2f}")
            rate.sleep()


def main(args=None):
    rclpy.init(args=args)
    node = ArucoNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # nothing else needed — keyboard thread restores terminal
        node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
