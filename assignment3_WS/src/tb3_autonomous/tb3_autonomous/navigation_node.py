#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError

class VisionNavigation(Node):
    def __init__(self):
        # Initialize the ROS node
        super().__init__('vision_navigation')
        
        # Set up the OpenCV bridge
        self.bridge = CvBridge()
        
        # Subscribe to the camera topic
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',  # Adjust according to your TurtleBot3 setup
            self.image_callback,
            10)
        
        # Publisher for velocity commands
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Control parameters
        self.linear_speed = 0.3  # Default forward speed
        self.angular_speed = 0.3  # Default turning speed
        
        # Processing parameters
        # Orange cone detection
        self.obstacle_color_lower = np.array([5, 5, 5])  # HSV range for orange obstacles
        self.obstacle_color_upper = np.array([25, 255, 255])

        self.obstacle_distance = 0

        self.obstacle_turning_direction = 1
        
        # White line detection (for boundaries)
        self.line_threshold = 100  # Threshold for white line detection
        
        # Control variables
        self.obstacle_detected = False
        self.boundary_detected = False
        self.obstacle_direction = 0  # Direction to obstacle (negative = left, positive = right)
        self.boundary_direction = 0  # Direction to boundary (negative = left, positive = right)
        
        # Parameters for boundary detection
        self.boundary_threshold = 500  # Minimum white pixels to consider as boundary
        self.boundary_ratio_threshold = 0.1  # Boundary pixels ratio threshold

        self.turning_back = 0
        self.keep_turning = 0
        self.last_direction = None
        
        self.get_logger().info("Vision navigation node initialized with white line boundary detection")

    def image_callback(self, data):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")
            return
            
        # Process the image to detect obstacles and boundaries
        self.process_image(cv_image)
        
        # Make navigation decisions based on processed data
        self.navigate()

    def process_image(self, image):
        # Get image dimensions
        height, width, _ = image.shape
        
        # Convert to HSV for obstacle detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Convert to grayscale for line detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to detect white lines
        _, line_mask = cv2.threshold(gray, self.line_threshold, 255, cv2.THRESH_BINARY)

        bottom_filed = hsv[int(height*0.55):height, :]
        b_mask = cv2.inRange(bottom_filed, np.array([0, 0, 30]), np.array([180, 40, 70]))

        # Morphological operations to remove noise
        kernel = np.ones((5, 5), np.uint8)

        # Remove small noise (Opening: Erosion → Dilation)
        b_mask = cv2.morphologyEx(b_mask, cv2.MORPH_OPEN, kernel)

        # Fill small holes (Closing: Dilation → Erosion)
        b_mask = cv2.morphologyEx(b_mask, cv2.MORPH_CLOSE, kernel)

        # Optional: Apply Gaussian Blur for smooth edges
        b_mask = cv2.GaussianBlur(b_mask, (5, 5), 0)

        # Create regions of interest for boundary detection
        # We'll check bottom, left and right sides of the image
        bottom_roi = line_mask[int(height*0.7):height, :]
        left_roi = line_mask[int(height*0.7):height, :int(width*0.4)]
        right_roi = line_mask[int(height*0.7):height, int(width*0.6):]
        
        # Count white pixels in each region
        bottom_white = cv2.countNonZero(bottom_roi)
        left_white = cv2.countNonZero(left_roi)
        right_white = cv2.countNonZero(right_roi)
        
        # Calculate white pixel ratios
        bottom_ratio = bottom_white / (bottom_roi.shape[0] * bottom_roi.shape[1])
        left_ratio = left_white / (left_roi.shape[0] * left_roi.shape[1])
        right_ratio = right_white / (right_roi.shape[0] * right_roi.shape[1])
        
        outside = cv2.countNonZero(b_mask)

        # self.get_logger().info(f"outside detected {outside}")
        if (outside > 0):
            # Determine if boundary is detected and in which direction
            self.boundary_detected = (bottom_ratio > self.boundary_ratio_threshold or 
                                    left_ratio > self.boundary_ratio_threshold or 
                                    right_ratio > self.boundary_ratio_threshold)
        else:
            if self.keep_turning == 0:
                self.boundary_detected = False
            
        # Determine direction to turn if boundary detected
        if self.boundary_detected:
            if bottom_ratio > self.boundary_ratio_threshold:
                # Boundary at bottom, check left vs right
                if left_ratio > right_ratio:
                    self.boundary_direction = 1  # Turn right
                else:
                    self.boundary_direction = -1  # Turn left
            elif left_ratio > right_ratio:
                self.boundary_direction = 1  # Boundary on left, turn right
            else:
                self.boundary_direction = -1  # Boundary on right, turn left

        # Detect obstacles (orange cones)
        obstacle_mask = cv2.inRange(hsv, self.obstacle_color_lower, self.obstacle_color_upper)

        # Detect black color (cone base)
        black_lower = np.array([0, 0, 0])  # Adjust as needed
        black_upper = np.array([30, 25, 25])  # Vary upper V for better detection
        black_mask = cv2.inRange(hsv, black_lower, black_upper)

                # Detect black color (cone base)
        black_lower = np.array([100, 150, 20])  # Adjust as needed
        black_upper = np.array([140, 255, 80])  # Vary upper V for better detection
        blue_mask = cv2.inRange(hsv, black_lower, black_upper)


        # Combine both masks
        combined_mask = cv2.bitwise_or(obstacle_mask, black_mask)
        # combined_mask = cv2.bitwise_or(combined_mask, blue_mask)

        # Focus on middle region
        obstacle_middle = combined_mask[:, int(width * 0.2):int(width * 0.8)]
        obstacle_pixels = cv2.countNonZero(obstacle_middle)
                
        # Check if obstacles are detected
        self.obstacle_detected = obstacle_pixels > 4000
        
        if self.obstacle_detected:

            # Find contours of obstacles
            contours, _ = cv2.findContours(obstacle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self.obstacle_distance = 0  # Initialize with the maximum possible y-value

            if contours:
                closest_contour = None
                max_y = 0
                for contour in contours:
                    for point in contour:
                        x, y = point[0]
                        if y > max_y:
                            max_y = y
                            closest_contour = contour
                
                self.obstacle_distance = max_y
                # Find the largest contour (main obstacle)
                largest_contour = max(contours, key=cv2.contourArea)
                self.get_logger().info(f"Obstacle detected! {obstacle_pixels} distance: {self.obstacle_distance}")
            
                # Find center of the obstacle
                M = cv2.moments(closest_contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    # Calculate obstacle direction relative to center
                    self.obstacle_direction = cx - (width // 2)
        else:
            self.obstacle_distance = 0

        # Visualization for debugging (uncomment if needed)
        # cv2.imshow("Camera View", image)
        # cv2.imshow("Line Mask", b_mask)
        # cv2.imshow("Bottom ROI", bottom_roi)
        # cv2.imshow("Obstacle ROI", obstacle_middle)
        # cv2.imshow("Obstacle Mask", combined_mask)
        # cv2.waitKey(1)

    def navigate(self):
        # Create Twist message for robot control
        twist_cmd = Twist()
        
        # Default behavior: move forward
        twist_cmd.linear.x = self.linear_speed
        twist_cmd.angular.z = 0.0

        # Priority 1: Avoid obstacles
        if self.obstacle_detected and self.obstacle_distance > 330:

            twist_cmd.linear.x = 0.0  # Slow down

            self.keep_turning = 0
            
            # Turn away from obstacle
            if self.obstacle_direction < 0:
                self.obstacle_turning_direction = -1
                twist_cmd.angular.z = -self.angular_speed  # Turn right
                self.get_logger().info("Turning right to avoid obstacle")
            elif self.obstacle_direction > 0:
                self.obstacle_turning_direction = 1
                twist_cmd.angular.z = self.angular_speed  # Turn left
                self.get_logger().info("Turning left to avoid obstacle")
    
        # Priority 2: Avoid white line boundaries
        else:
            if self.obstacle_turning_direction != 0:
                self.boundary_direction = self.obstacle_turning_direction
            self.obstacle_turning_direction = 0
            if self.keep_turning > 0:
                twist_cmd.linear.x = 0.0
                twist_cmd.angular.z = -self.angular_speed * self.last_direction
                self.keep_turning -= 1
                self.get_logger().info(f"Keep Turning {self.keep_turning}")
            elif self.boundary_detected:
                self.get_logger().info("White boundary line detected!")
                twist_cmd.linear.x = 0.0  # Slow down significantly
                
                # Turn away from boundary
                twist_cmd.angular.z = -self.angular_speed * self.boundary_direction

                self.last_direction = self.boundary_direction
                self.keep_turning = 150

                if self.boundary_direction > 0:
                    self.get_logger().info("Turning right to avoid boundary")
                else:
                    self.get_logger().info("Turning left to avoid boundary")
        
        # Publish velocity command
        self.cmd_vel_pub.publish(twist_cmd)

    def draw_debug_visualization(self, image, line_mask, obstacle_mask):
        """Create a debug visualization image (for development purposes)"""
        height, width, _ = image.shape
        
        # Draw boundary detection regions
        debug_img = image.copy()
        
        # Mark bottom ROI
        cv2.rectangle(debug_img, 
                    (0, int(height*0.7)), 
                    (width, height), 
                    (0, 255, 0), 2)
        
        # Mark left ROI
        cv2.rectangle(debug_img, 
                    (0, 0), 
                    (int(width*0.3), height), 
                    (255, 0, 0), 2)
        
        # Mark right ROI
        cv2.rectangle(debug_img, 
                    (int(width*0.7), 0), 
                    (width, height), 
                    (0, 0, 255), 2)
        
        # Add text with detection status
        text = "Obstacle: {}".format("Yes" if self.obstacle_detected else "No")
        cv2.putText(debug_img, text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        text = "Boundary: {}".format("Yes" if self.boundary_detected else "No")
        cv2.putText(debug_img, text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return debug_img

def main(args=None):
    rclpy.init(args=args)
    node = VisionNavigation()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()