#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import datetime
import os

class LineFollower:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('line_follower', anonymous=True)
        
        self.bridge = CvBridge()# Bridge to convert ROS images to OpenCV format
        self.twist = Twist() # Twist message to store movement commands

        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)  # Publisher to send movement commands to the robot
        rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback) # Subscribe to image data from the robot's camera

        # HSV range for detecting the black line
        self.lower_black = np.array([0, 0, 0])
        self.upper_black = np.array([180, 255, 60])

        rospy.loginfo("Line follower node started, waiting for images...")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8") # Convert ROS image to OpenCV format
        except Exception as e:
            rospy.logerr("CV Bridge error: %s", e)
            return

        # Convert image to HSV color space for better color filtering
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        height, width, _ = cv_image.shape

        # Crop for bottom 20%
        crop_img = hsv[int(height * 0.65):height, :]

        # Apply Gaussian blur to reduce image noise
        blurred = cv2.GaussianBlur(crop_img, (5, 5), 0)

        # Create a binary mask for black color
        mask = cv2.inRange(blurred, self.lower_black, self.upper_black)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Check if any contours were found
        if contours:
            # Use the largest contour, assuming it’s the line
            largest_contour = max(contours, key=cv2.contourArea)
            # Ignore small contours (noise)
            if cv2.contourArea(largest_contour) > 1000:
                # Calculate the center of mass of the contour
                M = cv2.moments(largest_contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    # Error is the horizontal distance from image center
                    error = cx - (width // 2)

                    # Visualize the centroid
                    cv2.circle(mask, (cx, int(mask.shape[0] / 2)), 5, (255, 0, 0), -1)

                    # Set forward speed and adjust angular speed based on error
                    self.twist.linear.x = 0.05
                    # Use smaller gain (increase the divisor) to make turning less sensitive
                    self.twist.angular.z = -float(error) / 350.0
                else:
                    # No valid mass found; rotate to search for line
                    self.twist.linear.x = 0.0
                    self.twist.angular.z = 0.15
            else:
                # Contour too small; rotate in place
                self.twist.linear.x = 0.0
                self.twist.angular.z = 0.15
        else:
            # No contours found; rotate to find line
            self.twist.linear.x = 0.0
            self.twist.angular.z = 0.15

        # Publish the movement command
        self.cmd_pub.publish(self.twist)

        # Show debug windows
        cv2.imshow("Cropped Image", crop_img)
        cv2.imshow("Mask", mask)
        cv2.waitKey(1)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = LineFollower()
        node.run()
    except rospy.ROSInterruptException:
        pass
