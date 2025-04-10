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

        # Define minimum and maximum angular velocities
        self.min_angular_vel = -0.3
        self.max_angular_vel = 0.3

        # Define the turning angle for each increment
        self.turn_angle = np.pi / 2  # 90 degrees in radians

        # Define the turning velocity for 90-degree turns
        self.turning_vel_90 = 0.5

        # Flag to indicate whether the robot is currently turning
        self.is_turning = False

        # Flag to indicate whether the robot is currently following the line
        self.is_following_line = False

        # Timestamp to track when the line was last detected
        self.last_line_detection_time = time()

        # Maximum time without line detection before considering an obstacle (in seconds)
        self.max_time_without_line_detection = 1.5

    def move_forward(self, error):
        # Proportional control to adjust robot's velocity based on the error
        self.twist.linear.x = 0.1  # Adjust linear velocity as needed
        self.twist.angular.z = -float(error) / 100  # Adjust angular velocity as needed

        # Apply constraints on angular velocity
        self.twist.angular.z = max(self.min_angular_vel, min(self.max_angular_vel, self.twist.angular.z))

        # Reset the turning flag
        self.is_turning = False
        self.is_following_line = True
        print("Robot keeps going forward")

    def stop_robot(self):
        # Stop the robot
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.0  # Stop turning
        self.cmd_vel_pub.publish(self.twist)
        self.is_turning = True
        self.is_following_line = False
        print("Robot stopped")

    def turn(self, direction):
        # Set the angular velocity for turning
        self.twist.angular.z = self.turning_vel_90 if direction == 'left' else -self.turning_vel_90
        self.is_turning = True
        print("Robot turning", direction)


    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)

        # Crop the lower part of the image to focus only on the floor
        height, width, _ = cv_image.shape
        crop_height = height // 2  # Crop from the middle of the image downwards
        cropped_image = cv_image[crop_height:, :]

        # Convert cropped image from BGR to HSV
        hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)

        # Define lower and upper bounds for line color in HSV
        lower_color = np.array([0, 0, 200])
        upper_color = np.array([180, 30, 255])

        # Threshold the HSV image to get only white colors
        mask = cv2.inRange(hsv, lower_color, upper_color)

        # Calculate the center of mass of the white region
        M = cv2.moments(mask)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # Calculate error from the center of the image
            error = cx - cropped_image.shape[1] / 2

            self.move_forward(error)

            # Update the timestamp when the line was last detected
            self.last_line_detection_time = time()
        else:
            if self.is_following_line:
                # Check if it's been too long since the last line detection
                if time() - self.last_line_detection_time > self.max_time_without_line_detection:
                    self.stop_robot()

        # Publish velocity command
        self.cmd_vel_pub.publish(self.twist)

        # Display the processed image
        cv2.waitKey(3)

if __name__ == '__main__':
    try:
        follower = LineFollower()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


