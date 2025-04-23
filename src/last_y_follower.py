#!/usr/bin/env python3

import rospy
import cv2
import cv_bridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import numpy as np

class Follower:
    def __init__(self):
        self.node_name = "rosbot_line_follower"
        rospy.init_node(self.node_name)
        self.bridge = cv_bridge.CvBridge()

        self.image_sub = rospy.Subscriber(
           '/camera/color/image_raw', 
          Image, self.image_callback)
        self.cmd_vel_pub = rospy.Publisher(
           '/cmd_vel',
           Twist,
           queue_size=1)
        self.speed = Twist()
        self.last_seen_angular_speed = 0
        self.rate = rospy.Rate(10)
        
        # Parameters for Hough transform
        self.rho = 1                # Distance resolution in pixels
        self.theta = np.pi/180      # Angular resolution in radians
        self.threshold = 15         # Minimum number of votes
        self.min_line_length = 30   # Minimum line length
        self.max_line_gap = 20      # Maximum allowed gap between line segments
        
        # For visualization
        self.debug = True
        self.debug_publisher = rospy.Publisher('/line_debug', Image, queue_size=1)

    def image_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # Store original image for debugging
        debug_image = image.copy()
        
        # Get image dimensions
        height, width, _ = image.shape
        
        # Crop the top 50% of the image
        crop_height = int(height * 0.5)
        cropped_image = image[crop_height:, :]
        
        # Convert to HSV for better line detection
        hsv_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
        
        # Define black color range in HSV
        lower_black = np.array([0, 0, 0], dtype=np.uint8)
        upper_black = np.array([180, 255, 50], dtype=np.uint8)
        
        # Create mask for black pixels
        mask = cv2.inRange(hsv_image, lower_black, upper_black)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Apply edge detection
        edges = cv2.Canny(mask, 50, 150, apertureSize=3)
        
        # Apply Hough Line Transform
        lines = cv2.HoughLinesP(edges, self.rho, self.theta, self.threshold, 
                               minLineLength=self.min_line_length, 
                               maxLineGap=self.max_line_gap)
        
        # Initialize variables to track line position
        cx = width // 2  # Default to center
        found_line = False
        
        # Process detected lines
        if lines is not None and len(lines) > 0:
            # Variables to calculate average line position
            x_sum = 0
            count = 0
            
            # Draw lines on debug image
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Draw on debug image (adjust y-coordinates for cropping)
                if self.debug:
                    cv2.line(debug_image, (x1, y1 + crop_height), (x2, y2 + crop_height), (0, 255, 0), 2)
                
                # Calculate midpoint of line
                x_mid = (x1 + x2) // 2
                x_sum += x_mid
                count += 1
            
            if count > 0:
                # Calculate average x position of detected lines
                cx = x_sum // count
                found_line = True
                
                # Draw center point on debug image
                if self.debug:
                    cv2.circle(debug_image, (cx, height - 20), 5, (0, 0, 255), -1)
                    cv2.putText(debug_image, f"Line Position: {cx}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Calculate error from center of image
        error = cx - (width // 2)
        
        # Determine control action based on line detection
        if found_line:
            # Calculate turn ratio based on error
            max_error = width // 2  # Maximum possible error
            turn_ratio = -float(error) / max_error
            
            # Set robot movement
            max_turn_speed = 0.5  # Maximum angular speed
            turn_speed = max_turn_speed * turn_ratio
            
            self.speed.angular.z = turn_speed
            self.last_seen_angular_speed = turn_speed
            self.speed.linear.x = 0.15  # Forward speed
            
            if self.debug:
                cv2.putText(debug_image, f"Turn: {turn_speed:.2f}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        else:
            # Line lost, turn in the direction it was last seen
            search_speed = 0.3 * np.sign(self.last_seen_angular_speed) 
            if self.last_seen_angular_speed == 0:  # If no history, turn right
                search_speed = 0.3
                
            self.speed.angular.z = search_speed
            self.speed.linear.x = 0  # Stop forward movement while searching
            
            if self.debug:
                cv2.putText(debug_image, "Searching for line", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Publish movement command
        self.cmd_vel_pub.publish(self.speed)
        
        # Publish debug image
        if self.debug:
            # Draw line marking the crop area
            cv2.line(debug_image, (0, crop_height), (width, crop_height), (255, 0, 0), 2)
            
            try:
                self.debug_publisher.publish(self.bridge.cv2_to_imgmsg(debug_image, "bgr8"))
            except cv_bridge.CvBridgeError as e:
                rospy.logerr(f"Could not publish debug image: {e}")
                
        self.rate.sleep()

if __name__ == '__main__':
    try:
        follower = Follower()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
