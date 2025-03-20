#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from time import time

class IntegratedRobot(Node):
    def __init__(self):
        super().__init__('integrated_robot')

        # Initialize ROS2 node
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, '/camera/color/image_raw', self.image_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.twist = Twist()
        
        # Line following settings
        self.min_angular_vel = -0.3
        self.max_angular_vel = 0.3
        self.last_line_detection_time = time()
        self.max_time_without_line = 1.5  # Seconds before stopping if no line detected
        
        # Object detection settings
        self.obstacle_distance_threshold = 0.2  # 20 cm for obstacles
        self.closest_object_distance = float('inf')

    def scan_callback(self, msg):
        """ Detect the closest object using LiDAR """
        self.closest_object_distance = min(msg.ranges) if msg.ranges else float('inf')

    def image_callback(self, msg):
        """ Process camera feed for line following and object detection """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"CV Bridge Error: {e}")
            return

        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # === OBJECT DETECTION ===
        color_ranges = {
            "RED": [(np.array([0, 120, 70]), np.array([10, 255, 255])),
                    (np.array([170, 120, 70]), np.array([180, 255, 255]))],
            "BLUE": [(np.array([100, 150, 70]), np.array([140, 255, 255]))],
            "GREEN": [(np.array([40, 50, 50]), np.array([90, 255, 255]))]
        }
        detected_colors = []
        
        for color, ranges in color_ranges.items():
            mask = np.zeros_like(hsv[:, :, 0])
            for lower, upper in ranges:
                mask += cv2.inRange(hsv, lower, upper)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:
                    detected_colors.append(color)

        # If an unknown object is within 20cm, classify it as an obstacle
        if detected_colors:
            self.get_logger().info(f"Detected cones: {', '.join(detected_colors)}")
        elif self.closest_object_distance < self.obstacle_distance_threshold:
            self.get_logger().info("🚨 Obstacle detected within 20 cm! Stopping robot.")
            self.stop_robot()
            return

        # === LINE FOLLOWING ===
        height, width, _ = cv_image.shape
        crop_height = height // 2  # Focus on lower half
        cropped_image = cv_image[crop_height:, :]

        # Threshold for white line detection
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        mask = cv2.inRange(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV), lower_white, upper_white)

        M = cv2.moments(mask)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            error = cx - cropped_image.shape[1] / 2

            # Move forward following the line
            self.twist.linear.x = 0.1
            self.twist.angular.z = -float(error) / 100
            self.twist.angular.z = max(self.min_angular_vel, min(self.max_angular_vel, self.twist.angular.z))

            self.last_line_detection_time = time()
        else:
            # If no line detected for too long, stop the robot
            if time() - self.last_line_detection_time > self.max_time_without_line:
                self.stop_robot()
                return

        # Publish movement
        self.cmd_vel_pub.publish(self.twist)

    def stop_robot(self):
        """ Stop the robot """
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.0
        self.cmd_vel_pub.publish(self.twist)
        self.get_logger().info("Robot stopped.")

def main(args=None):
    rclpy.init(args=args)
    node = IntegratedRobot()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()