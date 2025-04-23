#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

class LineFollower:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('line_follower', anonymous=True)
        
        self.bridge = CvBridge()
        self.twist = Twist()

        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)

        # HSV range for detecting the black line
        self.lower_black = np.array([0, 0, 0])
        self.upper_black = np.array([180, 255, 60])

        rospy.loginfo("Line follower node started, waiting for images...")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr("CV Bridge error: %s", e)
            return

        # Convert image to HSV color space for better color filtering
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        height, width, _ = cv_image.shape

        # Crop view
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

        # Display HSV image
        cv2.imshow("HSV", hsv)
        
        # Split the HSV channels
        h, s, v = cv2.split(hsv)
        
        # Get min and max values for each channel
        min_h, max_h = np.min(h), np.max(h)
        min_s, max_s = np.min(s), np.max(s)
        min_v, max_v = np.min(v), np.max(v)
        
        # Print the min and max values
        print(f"Hue: Min = {min_h}, Max = {max_h}")
        print(f"Saturation: Min = {min_s}, Max = {max_s}")
        print(f"Value: Min = {min_v}, Max = {max_v}")
        
        # Show cropped versions too (more useful for line following)
        h_crop, s_crop, v_crop = cv2.split(crop_img)
        min_h_crop, max_h_crop = np.min(h_crop), np.max(h_crop)
        min_s_crop, max_s_crop = np.min(s_crop), np.max(s_crop)
        min_v_crop, max_v_crop = np.min(v_crop), np.max(v_crop)
        
        print(f"Cropped Hue: Min = {min_h_crop}, Max = {max_h_crop}")
        print(f"Cropped Saturation: Min = {min_s_crop}, Max = {max_s_crop}")
        print(f"Cropped Value: Min = {min_v_crop}, Max = {max_v_crop}")
        
        # Display individual HSV channels
        cv2.imshow("Hue Channel", h)
        cv2.imshow("Saturation Channel", s)
        cv2.imshow("Value Channel", v)
        
        # Also show the original cropped and masked images
        cv2.imshow("Cropped HSV", crop_img)
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
