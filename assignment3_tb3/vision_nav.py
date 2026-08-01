import math
from dataclasses import dataclass

import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import TwistStamped
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


# Developed with assistance from OpenAI Codex; cite this usage in the report.


@dataclass
class Detection:
    area_ratio: float
    center_offset: float
    bottom_ratio: float


@dataclass
class LineDetections:
    boundary: Detection
    internal: Detection
    edge_green: Detection
    edge_gray: Detection


class VisionNavigator(Node):
    def __init__(self):
        super().__init__("vision_nav")

        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("linear_speed", 0.30)
        self.declare_parameter("turn_speed", 0.95)
        self.declare_parameter("slow_speed", 0.15)
        self.declare_parameter("image_timeout", 1.0)

        self.bridge = CvBridge()
        self.latest_image = None
        self.latest_stamp = None
        self.last_turn = 1.0
        self.status_count = 0
        self.nav_state = "normal"
        self.state_ticks = 0
        self.recovery_turn = 1.0
        self.last_field_score = None
        self.best_recovery_score = 0.0
        self.bad_field_ticks = 0
        self.good_field_ticks = 0
        self.safe_field_ticks = 0
        self.last_outside_offset = 0.0
        self.outside_memory_ticks = 0

        image_topic = self.get_parameter("image_topic").value
        cmd_topic = self.get_parameter("cmd_vel_topic").value

        self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            qos_profile_sensor_data,
        )
        self.cmd_pub = self.create_publisher(TwistStamped, cmd_topic, 10)
        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info(
            f"Camera-only navigator started: image={image_topic}, cmd={cmd_topic}"
        )

    def image_callback(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.latest_stamp = self.get_clock().now()
        except Exception as exc:
            self.get_logger().warning(f"Could not convert camera image: {exc}")

    def control_loop(self):
        if self.latest_image is None or self.image_is_stale():
            self.publish_cmd(0.0, 0.0)
            return

        image = self.latest_image
        field = self.detect_field(image)
        outside = self.detect_outside(image)
        cone = self.detect_cones(image)
        lines = self.detect_lines(image)
        self.update_outside_memory(outside)

        field_score = self.field_score(field)
        self.update_field_trend(field_score)
        self.state_ticks += 1

        if self.nav_state == "search_field":
            linear, angular, mode = self.run_search_state(
                field, outside, lines, field_score
            )
        elif self.nav_state == "recover_field":
            linear, angular, mode = self.run_recover_state(
                field, outside, lines, field_score
            )
        else:
            if self.should_search_field(field):
                self.enter_state("search_field", field, outside, lines, field_score)
                linear, angular, mode = self.run_search_state(
                    field, outside, lines, field_score
                )
            elif self.should_recover_field(field, outside, lines):
                self.enter_state("recover_field", field, outside, lines, field_score)
                linear, angular, mode = self.run_recover_state(
                    field, outside, lines, field_score
                )
            else:
                linear, angular, mode = self.run_normal_state(field, outside, cone, lines)

        if abs(angular) > 0.05:
            self.last_turn = math.copysign(1.0, angular)

        self.publish_cmd(linear, angular)
        self.log_status(mode, field, outside, cone, lines, linear, angular)

    def field_score(self, field):
        return 0.65 * field.bottom_ratio + 0.35 * field.area_ratio

    def update_field_trend(self, field_score):
        if self.last_field_score is None:
            self.last_field_score = field_score
            return

        if field_score < self.last_field_score - 0.015:
            self.bad_field_ticks += 1
            self.good_field_ticks = 0
        elif field_score > self.last_field_score + 0.010:
            self.good_field_ticks += 1
            self.bad_field_ticks = 0
        else:
            self.bad_field_ticks = max(self.bad_field_ticks - 1, 0)
            self.good_field_ticks = max(self.good_field_ticks - 1, 0)

        self.last_field_score = field_score

    def update_outside_memory(self, outside):
        if outside.area_ratio > 0.06 and abs(outside.center_offset) > 0.04:
            self.last_outside_offset = outside.center_offset
            self.outside_memory_ticks = 20
        elif self.outside_memory_ticks > 0:
            self.outside_memory_ticks -= 1

    def enter_state(self, state, field, outside, lines, field_score):
        if state == self.nav_state:
            return

        self.nav_state = state
        self.state_ticks = 0
        self.best_recovery_score = field_score
        self.bad_field_ticks = 0
        self.good_field_ticks = 0
        self.safe_field_ticks = 0

        if state == "recover_field":
            self.recovery_turn = self.choose_recovery_turn(field, outside, lines)
        elif state == "search_field":
            self.recovery_turn = self.last_turn

        self.get_logger().info(f"state={state}")

    def should_search_field(self, field):
        return field.area_ratio < 0.045 and field.bottom_ratio < 0.12

    def should_recover_field(self, field, outside, lines):
        if lines.edge_green.area_ratio > 0.0015 or lines.boundary.area_ratio > 0.0015:
            return True
        if outside.bottom_ratio > 0.18:
            return True
        if outside.bottom_ratio > 0.08 and field.bottom_ratio < 0.78:
            return True
        if outside.area_ratio > 0.22:
            return True
        if outside.area_ratio > 0.18 and field.area_ratio < 0.82:
            return True
        if field.bottom_ratio < 0.32 or abs(field.center_offset) > 0.32:
            return True
        return self.bad_field_ticks >= 4 and field.bottom_ratio < 0.58

    def run_normal_state(self, field, outside, cone, lines):
        linear = float(self.get_parameter("linear_speed").value)
        angular = 0.06 * self.last_turn
        mode = "normal"

        if outside.bottom_ratio > 0.08 or (
            outside.area_ratio > 0.18 and field.area_ratio < 0.82
        ):
            linear = min(float(self.get_parameter("slow_speed").value), 0.10)
            if outside.area_ratio > 0.08 and abs(outside.center_offset) > 0.08:
                angular = self.turn_away_from(outside.center_offset, gain=0.90)
            else:
                angular = self.recovery_angular(field, outside, lines, gain=0.90)
            mode = "avoid_gray"
        elif cone.bottom_ratio > 0.025 or cone.area_ratio > 0.035:
            linear = float(self.get_parameter("slow_speed").value)
            angular = self.turn_away_from(cone.center_offset, gain=1.15)
            mode = "avoid_cone"
        elif cone.area_ratio > 0.012:
            linear *= 0.75
            angular = self.turn_away_from(cone.center_offset, gain=0.65)
            mode = "steer_from_cone"
        elif abs(field.center_offset) > 0.18:
            linear *= 0.85
            angular = self.turn_toward(field.center_offset, gain=0.40)
            mode = "center_field"

        return linear, angular, mode

    def run_recover_state(self, field, outside, lines, field_score):
        field_is_safe = (
            self.state_ticks >= 18
            and field.area_ratio > 0.35
            and field.bottom_ratio > 0.72
            and abs(field.center_offset) < 0.24
            and outside.bottom_ratio < 0.08
            and outside.area_ratio < 0.14
        )
        if field_is_safe:
            self.safe_field_ticks += 1
        else:
            self.safe_field_ticks = 0

        if self.safe_field_ticks >= 25:
            self.enter_state("normal", field, outside, lines, field_score)
            return self.run_normal_state(
                field, outside, Detection(0.0, 0.0, 0.0), lines
            )
        if self.safe_field_ticks > 0:
            return self.run_recover_advance_state(field, outside)

        if self.can_recover_escape(field, outside, lines):
            return self.run_recover_escape_state(field, outside, lines)

        if field_score > self.best_recovery_score + 0.010:
            self.best_recovery_score = field_score

        if self.state_ticks > 8 and self.should_search_field(field):
            self.enter_state("search_field", field, outside, lines, field_score)
            return self.run_search_state(field, outside, lines, field_score)

        if self.state_ticks > 10 and self.bad_field_ticks >= 4:
            self.recovery_turn *= -1.0
            self.bad_field_ticks = 0

        linear = 0.0
        gain = 1.30 if outside.bottom_ratio > 0.20 else 1.15
        angular = self.recovery_angular(field, outside, lines, gain=gain)
        mode = "recover_field"
        return linear, angular, mode

    def run_recover_advance_state(self, field, outside):
        linear = min(float(self.get_parameter("slow_speed").value), 0.10)
        if outside.area_ratio > 0.08 and abs(outside.center_offset) > 0.05:
            angular = self.turn_away_from(outside.center_offset, gain=0.45)
        elif self.outside_memory_ticks > 0 and abs(self.last_outside_offset) > 0.06:
            angular = self.turn_away_from(self.last_outside_offset, gain=0.35)
        elif abs(field.center_offset) > 0.10:
            angular = self.turn_toward(field.center_offset, gain=0.45)
        else:
            angular = 0.05 * self.last_turn
        return linear, angular, "recover_advance"

    def can_recover_escape(self, field, outside, lines):
        side_gray = outside.area_ratio > 0.12
        side_edge = lines.edge_green.area_ratio > 0.001 or lines.boundary.area_ratio > 0.001
        return (
            field.bottom_ratio > 0.74
            and field.area_ratio > 0.42
            and outside.bottom_ratio < 0.05
            and (side_gray or side_edge)
        )

    def run_recover_escape_state(self, field, outside, lines):
        linear = min(float(self.get_parameter("slow_speed").value), 0.08)
        if outside.area_ratio > 0.08 and abs(outside.center_offset) > 0.05:
            angular = self.turn_away_from(outside.center_offset, gain=0.55)
        elif lines.edge_green.area_ratio > 0.001:
            angular = self.turn_toward(lines.edge_green.center_offset, gain=0.55)
        elif abs(field.center_offset) > 0.10:
            angular = self.turn_toward(field.center_offset, gain=0.45)
        else:
            angular = self.recovery_angular(field, outside, lines, gain=0.90)
        return linear, angular, "recover_escape"

    def run_search_state(self, field, outside, lines, field_score):
        if field.area_ratio > 0.075 or field.bottom_ratio > 0.16:
            self.enter_state("recover_field", field, outside, lines, field_score)
            return self.run_recover_state(field, outside, lines, field_score)

        if self.state_ticks > 18 and self.bad_field_ticks >= 3:
            self.recovery_turn *= -1.0
            self.bad_field_ticks = 0

        turn_speed = float(self.get_parameter("turn_speed").value)
        return 0.0, self.recovery_turn * turn_speed * 0.80, "search_field"

    def choose_recovery_turn(self, field, outside, lines):
        target_offset = self.recovery_target_offset(field, outside, lines)
        if target_offset is None or abs(target_offset) < 0.08:
            return self.last_turn
        return math.copysign(1.0, self.turn_toward(target_offset, gain=1.0))

    def recovery_target_offset(self, field, outside, lines):
        if lines.edge_green.area_ratio > 0.001:
            return lines.edge_green.center_offset
        if outside.bottom_ratio > 0.08 and abs(outside.center_offset) > 0.06:
            return -outside.center_offset
        if field.area_ratio > 0.025:
            return field.center_offset
        if outside.area_ratio > 0.10 and abs(outside.center_offset) > 0.06:
            return -outside.center_offset
        if lines.edge_gray.area_ratio > 0.001:
            return -lines.edge_gray.center_offset
        return None

    def recovery_angular(self, field, outside, lines, gain):
        target_offset = self.recovery_target_offset(field, outside, lines)
        if target_offset is not None:
            angular = self.turn_toward(target_offset, gain)
            if abs(angular) >= 0.25:
                self.recovery_turn = math.copysign(1.0, angular)
                return angular

        turn_speed = float(self.get_parameter("turn_speed").value)
        return self.recovery_turn * turn_speed * 0.90

    def image_is_stale(self):
        timeout = float(self.get_parameter("image_timeout").value)
        age = (self.get_clock().now() - self.latest_stamp).nanoseconds / 1e9
        return age > timeout

    def detect_cones(self, image):
        height, width = image.shape[:2]
        roi = image[height // 3 :, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        orange_low = np.array([3, 80, 80], dtype=np.uint8)
        orange_high = np.array([28, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, orange_low, orange_high)
        mask = self.clean_mask(mask)

        return self.measure_mask(mask, width)

    def detect_field(self, image):
        height, width = image.shape[:2]
        roi = image[int(height * 0.45) :, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        green_low = np.array([35, 45, 35], dtype=np.uint8)
        green_high = np.array([90, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, green_low, green_high)
        mask = self.clean_mask(mask)

        return self.measure_mask(mask, width)

    def detect_outside(self, image):
        height, width = image.shape[:2]
        roi = image[int(height * 0.45) :, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        gray_low = np.array([0, 0, 70], dtype=np.uint8)
        gray_high = np.array([180, 45, 215], dtype=np.uint8)
        mask = cv2.inRange(hsv, gray_low, gray_high)
        mask = self.clean_mask(mask)

        return self.measure_mask(mask, width)

    def detect_lines(self, image):
        height, width = image.shape[:2]
        roi = image[int(height * 0.50) :, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        white_low = np.array([0, 0, 220], dtype=np.uint8)
        white_high = np.array([180, 55, 255], dtype=np.uint8)
        white_mask = cv2.inRange(hsv, white_low, white_high)
        white_mask = self.clean_mask(white_mask)

        green_low = np.array([35, 45, 35], dtype=np.uint8)
        green_high = np.array([90, 255, 255], dtype=np.uint8)
        green_mask = cv2.inRange(hsv, green_low, green_high)

        gray_low = np.array([0, 0, 70], dtype=np.uint8)
        gray_high = np.array([180, 45, 215], dtype=np.uint8)
        gray_mask = cv2.inRange(hsv, gray_low, gray_high)

        boundary_mask = np.zeros_like(white_mask)
        internal_mask = np.zeros_like(white_mask)
        edge_green_mask = np.zeros_like(white_mask)
        edge_gray_mask = np.zeros_like(white_mask)
        contours, _ = cv2.findContours(
            white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        ring_kernel = np.ones((25, 25), np.uint8)
        core_kernel = np.ones((5, 5), np.uint8)
        for contour in contours:
            if cv2.contourArea(contour) < 20.0:
                continue

            component = np.zeros_like(white_mask)
            cv2.drawContours(component, [contour], -1, 255, thickness=cv2.FILLED)

            dilated = cv2.dilate(component, ring_kernel, iterations=1)
            core = cv2.dilate(component, core_kernel, iterations=1)
            ring = cv2.bitwise_and(dilated, cv2.bitwise_not(core))
            ring_pixels = max(cv2.countNonZero(ring), 1)

            green_near = cv2.countNonZero(cv2.bitwise_and(green_mask, ring))
            gray_near = cv2.countNonZero(cv2.bitwise_and(gray_mask, ring))
            green_ratio = green_near / ring_pixels
            gray_ratio = gray_near / ring_pixels

            # Boundary lines split green field from gray outside floor.
            # Midfield and penalty-box lines are white too, but their neighborhood is green.
            if green_ratio > 0.08 and gray_ratio > 0.08:
                boundary_mask = cv2.bitwise_or(boundary_mask, component)
                edge_green_mask = cv2.bitwise_or(
                    edge_green_mask, cv2.bitwise_and(green_mask, ring)
                )
                edge_gray_mask = cv2.bitwise_or(
                    edge_gray_mask, cv2.bitwise_and(gray_mask, ring)
                )
            else:
                internal_mask = cv2.bitwise_or(internal_mask, component)

        return LineDetections(
            boundary=self.measure_mask(boundary_mask, width),
            internal=self.measure_mask(internal_mask, width),
            edge_green=self.measure_mask(edge_green_mask, width),
            edge_gray=self.measure_mask(edge_gray_mask, width),
        )

    def clean_mask(self, mask):
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    def measure_mask(self, mask, width):
        mask_pixels = int(cv2.countNonZero(mask))
        total_pixels = mask.shape[0] * mask.shape[1]
        area_ratio = mask_pixels / max(total_pixels, 1)

        bottom_band = mask[int(mask.shape[0] * 0.62) :, :]
        bottom_ratio = cv2.countNonZero(bottom_band) / max(bottom_band.size, 1)

        moments = cv2.moments(mask)
        if moments["m00"] > 0:
            center_x = moments["m10"] / moments["m00"]
            center_offset = (center_x - width / 2.0) / (width / 2.0)
        else:
            center_offset = 0.0

        return Detection(area_ratio, center_offset, bottom_ratio)

    def turn_away_from(self, offset, gain):
        turn_speed = float(self.get_parameter("turn_speed").value)
        if abs(offset) < 0.12:
            direction = self.last_turn
        else:
            direction = math.copysign(1.0, offset)
        return max(min(direction * turn_speed * gain, 1.2), -1.2)

    def turn_toward(self, offset, gain):
        turn_speed = float(self.get_parameter("turn_speed").value)
        if abs(offset) < 0.08:
            return 0.0
        direction = -math.copysign(1.0, offset)
        return max(min(direction * turn_speed * gain, 1.2), -1.2)

    def turn_to_edge_green(self, lines, field, gain):
        target_offset = lines.edge_green.center_offset
        if lines.edge_green.area_ratio < 0.01:
            target_offset = field.center_offset

        if abs(target_offset) < 0.08:
            if abs(lines.edge_gray.center_offset) > 0.08:
                target_offset = -lines.edge_gray.center_offset
            else:
                target_offset = field.center_offset

        angular = self.turn_toward(target_offset, gain)
        if abs(angular) < 0.15:
            angular = math.copysign(
                float(self.get_parameter("turn_speed").value) * 0.75,
                lines.boundary.center_offset or self.last_turn,
            )
        return angular

    def publish_cmd(self, linear_x, angular_z):
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = "base_link"
        cmd.twist.linear.x = float(linear_x)
        cmd.twist.angular.z = float(angular_z)
        self.cmd_pub.publish(cmd)

    def log_status(self, mode, field, outside, cone, lines, linear, angular):
        self.status_count += 1
        if self.status_count % 20 != 0:
            return
        boundary = lines.boundary
        internal = lines.internal
        self.get_logger().info(
            "state=%s mode=%s linear=%.2f angular=%.2f field=%.3f/%.3f gray=%.3f/%.3f/%+.2f cone=%.3f/%.3f edge=%.3f/%.3f edge_green=%.3f/%+.2f inner=%.3f/%.3f"
            % (
                self.nav_state,
                mode,
                linear,
                angular,
                field.area_ratio,
                field.bottom_ratio,
                outside.area_ratio,
                outside.bottom_ratio,
                outside.center_offset,
                cone.area_ratio,
                cone.bottom_ratio,
                boundary.area_ratio,
                boundary.bottom_ratio,
                lines.edge_green.area_ratio,
                lines.edge_green.center_offset,
                internal.area_ratio,
                internal.bottom_ratio,
            )
        )


def main(args=None):
    rclpy.init(args=args)
    node = VisionNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if rclpy.ok():
                node.publish_cmd(0.0, 0.0)
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
