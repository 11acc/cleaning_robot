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
        self.upper_black = np.array([180, 255, 50])

        # Directory to save images for debugging
        self.image_save_dir = '/local/student/catkin_ws/src/menelao_challenge/tmp/'
        if not os.path.exists(self.image_save_dir):
            os.makedirs(self.image_save_dir)

        # How often to save images (in seconds)
        self.save_interval = 5
        self.last_saved_time = rospy.Time.now()

        rospy.loginfo("Line follower node started, waiting for images...")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")  # Convert ROS image to OpenCV format
        except Exception as e:
            rospy.logerr("CV Bridge error: %s", e)
            return

        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        height, width, _ = cv_image.shape

        # Crop the image to the bottom 20%
        crop_img = hsv[int(height * 0.80):height, :]

        # Apply Gaussian blur to reduce image noise
        blurred = cv2.GaussianBlur(crop_img, (5, 5), 0)

        # Create a binary mask for black color
        mask = cv2.inRange(blurred, self.lower_black, self.upper_black)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 1000:
                M = cv2.moments(largest_contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    error = cx - (width // 2)

                    # Apply dead zone logic
                    if abs(error) < 10:
                        angular_z = 0.0
                    else:
                        angular_z = -float(error) / 250.0  # Adjust sensitivity

                    # Set movement commands
                    self.twist.linear.x = 0.15
                    self.twist.angular.z = angular_z

                    # Draw debug visuals
                    cv2.circle(mask, (cx, int(mask.shape[0] / 2)), 5, (255, 0, 0), -1)
                else:
                    self.twist.linear.x = 0.0
                    self.twist.angular.z = 0.3
            else:
                self.twist.linear.x = 0.0
                self.twist.angular.z = 0.3
        else:
            self.twist.linear.x = 0.0
            self.twist.angular.z = 0.3

        self.cmd_pub.publish(self.twist)

        # Show debug windows
        cv2.imshow("Cropped Image", crop_img)
        cv2.imshow("Mask", mask)
        cv2.waitKey(1)

        current_time = rospy.Time.now()
        if current_time - self.last_saved_time >= rospy.Duration(self.save_interval):
            self.save_image(cv_image)
            self.last_saved_time = current_time

    def save_image(self, image):
        # Generate a timestamped filename
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        image_filename = os.path.join(self.image_save_dir, f"image_{timestamp}.jpg")
        # Save the image to disk
        cv2.imwrite(image_filename, image)
        rospy.loginfo(f"Image saved: {image_filename}")

    def run(self):
        rospy.spin() # Keep the node running until shutdown

if __name__ == '__main__':
    try:
        node = LineFollower()
        node.run()
    except rospy.ROSInterruptException:
        pass
