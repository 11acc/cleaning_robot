import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge

class ObjectDetector(Node):
    def __init__(self):
        super().__init__('object_detector')
        self.bridge = CvBridge()

        self.image_sub = self.create_subscription(Image, '/camera/color/image_raw', self.image_callback, 10)

        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        self.obstacle_distance_threshold = 0.2  # 20 cm
        self.closest_object_distance = float('inf')

    def scan_callback(self, msg):
        """ Get the closest object distance from LiDAR. """
        self.closest_object_distance = min(msg.ranges) if msg.ranges else float('inf')

    def image_callback(self, msg):
        """ Detects colors and classifies obstacles based on distance. """
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        color_ranges = {
            "RED": [(np.array([0, 120, 70]), np.array([10, 255, 255])),  # Lower red range
                    (np.array([170, 120, 70]), np.array([180, 255, 255]))],  # Upper red range
            "BLUE": [(np.array([100, 150, 70]), np.array([140, 255, 255]))],  # Blue range
            "GREEN": [(np.array([40, 50, 50]), np.array([90, 255, 255]))]  # Green range
        }

        detected_colors = []

 
        for color, ranges in color_ranges.items():
            mask = np.zeros_like(hsv[:, :, 0])
            for lower, upper in ranges:
                mask += cv2.inRange(hsv, lower, upper)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Filter out small objects
                    detected_colors.append(color)

        # Determine if an object is an obstacle
        if detected_colors:
            self.get_logger().info(f"Detected objects: {', '.join(detected_colors)}")
        elif self.closest_object_distance < self.obstacle_distance_threshold:
            self.get_logger().info("Obstacle detected within 20 cm!")
        else:
            self.get_logger().info("No objects detected.")

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()