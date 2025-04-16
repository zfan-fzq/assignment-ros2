#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import tf2_ros
import tf_transformations


class ArucoNavigator(Node):
    def __init__(self):
        super().__init__('aruco_navigator')

        # Publishers & Subscribers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )

        # TF2 listener for map → base_link
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # OpenCV bridge
        self.bridge = CvBridge()

        # ArUco detection
        self.aruco_dict    = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50)
        self.marker_length = 0.16  # meters

        # Dummy camera intrinsics (replace with your calibration)
        self.camera_matrix = np.array([
            [525.0,   0.0, 320.5],
            [  0.0, 525.0, 240.5],
            [  0.0,   0.0,   1.0]
        ], dtype=np.float32)
        self.dist_coeffs = np.zeros((5,1))

        # Transform from camera_link → base_link
        cam_xyz = [0.064, -0.065, 0.094]
        cam_rpy = [0.0, 0.0, 0.0]
        self.T_cam_base = tf_transformations.concatenate_matrices(
            tf_transformations.translation_matrix(cam_xyz),
            tf_transformations.euler_matrix(*cam_rpy)
        )

        # State
        self.state        = 'explore'    # explore → goto → done
        self.marker_poses_map = {}       # id → 4×4 map→marker transform
        self.origin       = None         # (x0,y0) of marker 1 in map frame
        self.yaw_marker1  = None         # θ of marker 1 in map frame
        self.goal_rel     = None         # (x_goal, y_goal) in marker‑1 frame

        # For debugging: ground truth positions (don't use for math!)
        self.GT = {
            1: (0.0, 0.0),
            2: (0.0, 3.6),
            3: (4.0, 3.6),
            4: (4.0, 0.0),
        }

        # Control loop at 10 Hz
        self.create_timer(0.1, self.control_loop)

        self.get_logger().info('Aruco SLAM Navigator ready.')

    def image_callback(self, msg):
        # 1) Convert ROS image to OpenCV
        cv_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        gray   = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        # 2) Detect ArUco markers
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.aruco_dict)
        if ids is None:
            return

        # 3) Estimate each marker’s pose in the camera frame
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, self.marker_length,
            self.camera_matrix, self.dist_coeffs
        )

        # 4) Get current robot pose in the SLAM map frame
        try:
            tf_map_base: TransformStamped = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time())
            t = tf_map_base.transform.translation
            q = tf_map_base.transform.rotation
            T_map_base = tf_transformations.concatenate_matrices(
                tf_transformations.translation_matrix([t.x, t.y, t.z]),
                tf_transformations.quaternion_matrix([q.x, q.y, q.z, q.w])
            )
        except Exception as e:
            self.get_logger().warn(f'No map→base_link yet: {e}')
            return

        # 5) For each detected marker, compute its map‐frame transform
        for i, mid in enumerate(ids.flatten()):
            tvec = tvecs[i][0]
            rvec = rvecs[i][0]
            R_marker, _ = cv2.Rodrigues(rvec)

            T_cam_marker = tf_transformations.identity_matrix()
            T_cam_marker[:3, :3] = R_marker
            T_cam_marker[:3, 3]  = tvec

            T_map_marker = T_map_base @ self.T_cam_base @ T_cam_marker
            self.marker_poses_map[mid] = T_map_marker

            # log once
            if mid not in getattr(self, '_logged_ids', set()):
                pos = tf_transformations.translation_from_matrix(T_map_marker)
                self.get_logger().info(
                    f"Found marker {mid} at map coords "
                    f"({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
                )
                self._logged_ids = getattr(self, '_logged_ids', set()) | {mid}

        # 6) Once we see all 4, fit rectangle & set goal in marker‑1 frame
        if self.state == 'explore' and len(self.marker_poses_map) >= 4 and 1 in self.marker_poses_map:
            # --- compute marker‑1 origin & yaw in map frame ---
            T1 = self.marker_poses_map[1]
            t1 = tf_transformations.translation_from_matrix(T1)
            self.origin      = (t1[0], t1[1])
            quat1 = tf_transformations.quaternion_from_matrix(T1)
            _, _, self.yaw_marker1 = tf_transformations.euler_from_quaternion(quat1)
            self.get_logger().info(
                f"Marker 1 origin = ({self.origin[0]:.2f},{self.origin[1]:.2f}), "
                f"yaw = {self.yaw_marker1:.2f} rad"
            )

            # --- compute all corners in marker‑1 frame (accounts for rotation!) ---
            T1_inv = np.linalg.inv(T1)
            rel = {}
            for mid, Tm in self.marker_poses_map.items():
                T_rel = T1_inv @ Tm
                p_rel = tf_transformations.translation_from_matrix(T_rel)
                rel[mid] = p_rel
                self.get_logger().info(
                    f"Marker {mid} in M1 frame: ({p_rel[0]:.2f}, {p_rel[1]:.2f})"
                )

            # snap to perfect rectangle
            y2, y3 = rel[2][1], rel[3][1]
            x3, x4 = rel[3][0], rel[4][0]
            H = (y2 + y3) / 2.0
            W = (x3 + x4) / 2.0

            ideal = {
                1: (0.0, 0.0),
                2: (0.0, H),
                3: (W,   H),
                4: (W,   0.0),
            }
            for mid, (ix, iy) in ideal.items():
                z = rel[mid][2]
                rel[mid] = (ix, iy, z)

            # centroid in marker‑1 frame
            cx, cy = W/2.0, H/2.0
            self.goal_rel = (cx, cy)
            self.get_logger().info(
                f"Fitted rectangle W={W:.2f}, H={H:.2f}; "
                f"centroid = ({cx:.2f}, {cy:.2f})"
            )

            # DEBUG: compare to ground truth
            for mid, (ex, ey, _) in rel.items():
                gx, gy = self.GT[mid]
                err = np.hypot(ex-gx, ey-gy)
                self.get_logger().info(
                    f"  Marker {mid}: GT=({gx:.2f},{gy:.2f}), "
                    f"est=({ex:.2f},{ey:.2f}), err={err:.2f} m"
                )

            # store for goto
            self.state = 'goto'
            self.markers_in_marker1_frame = rel

    def control_loop(self):
        # 1) still exploring → spin in place
        if self.state == 'explore':
            twist = Twist()
            twist.angular.z = 0.2
            self.cmd_pub.publish(twist)
            return

        # 2) goto: drive to self.goal_rel in marker‑1 frame
        if self.state == 'goto' and self.goal_rel is not None:
            # get robot pose in map
            try:
                tf_map_base = self.tf_buffer.lookup_transform(
                    'map', 'base_link', rclpy.time.Time())
                px = tf_map_base.transform.translation.x
                py = tf_map_base.transform.translation.y
                q  = tf_map_base.transform.rotation
                _, _, yaw_map = tf_transformations.euler_from_quaternion(
                    [q.x, q.y, q.z, q.w])
            except Exception:
                return

            # 1) translate into marker1 origin
            dx = px - self.origin[0]
            dy = py - self.origin[1]

            # 2) rotate by –yaw_marker1 into marker‑1 axes
            c, s = np.cos(-self.yaw_marker1), np.sin(-self.yaw_marker1)
            x_r = c*dx - s*dy
            y_r = s*dx + c*dy

            # errors in marker‑1 frame
            gx, gy = self.goal_rel
            ex, ey = gx - x_r, gy - y_r
            dist = np.hypot(ex, ey)
            angle_to_goal = np.arctan2(ey, ex)

            # **FIXED**: robot’s heading in marker‑1 frame
            yaw_r = yaw_map - self.yaw_marker1
            angle_error = (angle_to_goal - yaw_r + np.pi) % (2*np.pi) - np.pi

            self.get_logger().info(
                f"[ROBOT] pos=({x_r:.2f},{y_r:.2f}), yaw_r={yaw_r:.2f}\n"
                f"[GOAL ] pos=({gx:.2f},{gy:.2f}), dist={dist:.2f}, err={angle_error:.2f}"
            )

            # simple P‐controller
            kp_lin, kp_ang = 0.5, 1.0
            lin = min(0.3, kp_lin * dist)
            ang = np.clip(kp_ang * angle_error, -0.8, 0.8)

            twist = Twist()
            if abs(angle_error) > 0.15:
                twist.linear.x  = 0.0
                twist.angular.z = ang * 0.5
            else:
                twist.linear.x  = lin
                twist.angular.z = 0.0

            if dist < 0.1:
                self.get_logger().info('Reached centroid, stopping.')
                self.cmd_pub.publish(Twist())
                self.state = 'done'
            else:
                self.cmd_pub.publish(twist)

        # 3) done → idle
        elif self.state == 'done':
            return


def main(args=None):
    rclpy.init(args=args)
    node = ArucoNavigator()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
