#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from geometry_msgs.msg import Twist
from time import time

class LineFollower:
    def __init__(self):
        rospy.init_node('line_follower')
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.twist = Twist()

        # Angular velocity limits
        self.min_angular_vel = -0.3
        self.max_angular_vel = 0.3

        # For 90-degree turns
        self.turning_vel_90 = 0.5

        # Flags and state
        self.is_turning = False
        self.is_following_line = False
        self.last_line_detection_time = time()

        # Timeout for detecting loss of line
        self.max_time_without_line_detection = 1.5

        # Margin for detecting sharp turns (as a % of width)
        self.turn_margin = 0.35  # i.e. 35% from left or right edge

    def move_forward(self, error):
        self.twist.linear.x = 0.1  # Forward speed
        self.twist.angular.z = -float(error) / 100  # Proportional control
        self.twist.angular.z = max(self.min_angular_vel, min(self.max_angular_vel, self.twist.angular.z))

        self.is_turning = False
        self.is_following_line = True
        print("Following line. Error:", error)

    def stop_robot(self):
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.0
        self.cmd_vel_pub.publish(self.twist)
        self.is_turning = True
        self.is_following_line = False
        print("Robot stopped (no line detected)")

    def turn(self, direction):
        self.twist.linear.x = 0.0
        self.twist.angular.z = self.turning_vel_90 if direction == 'left' else -self.turning_vel_90
        self.is_turning = True
        print("Turning", direction)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)
            return

        height, width, _ = cv_image.shape
        crop_height = height // 2
        cropped_image = cv_image[crop_height:, :]

        # Convert to grayscale
        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

        # Threshold for black (adjust 50 if needed)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        line_contour = None
        max_area = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h if h > 0 else 0
                if 2 < aspect_ratio < 10:  # Adjust as needed
                    if area > max_area:
                        max_area = area
                        line_contour = cnt

        if line_contour is not None:
            M = cv2.moments(line_contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])

                # Turn detection thresholds
                left_margin = width * self.turn_margin
                right_margin = width * (1 - self.turn_margin)

                if cx < left_margin:
                    self.turn('left')
                elif cx > right_margin:
                    self.turn('right')
                else:
                    error = cx - width / 2
                    self.move_forward(error)

                self.last_line_detection_time = time()
        else:
            if self.is_following_line and time() - self.last_line_detection_time > self.max_time_without_line_detection:
                self.stop_robot()

        self.cmd_vel_pub.publish(self.twist)
        cv2.waitKey(3)

if __name__ == '__main__':
    try:
        follower = LineFollower()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
