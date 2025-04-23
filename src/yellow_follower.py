#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import cv2
import numpy as np
from tf.transformations import euler_from_quaternion
import math

class YellowFollower:
    def __init__(self):
        rospy.init_node('yellow_follower')
        self.bridge = CvBridge()

        # Subscribers and publishers
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        self.twist = Twist()

        # State flags
        self.following = False  # True when actively following yellow
        self.last_yaw = 0.0  # Store the last known yaw direction

        # Parameters
        self.min_contour_area = 500
        self.max_speed = 0.25  # Adjust as needed
        self.angular_speed = 0.4  # Adjust as needed

        # HSV thresholds for yellow (widened for lighting variations)
        self.lower_yellow = np.array([15, 60, 100])
        self.upper_yellow = np.array([40, 255, 255])

        rospy.loginfo("YellowFollower node started, waiting for camera data...")

    def odom_callback(self, msg):
        # Get current orientation
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        self.last_yaw = yaw

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

            mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)

            contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                contour_area = cv2.contourArea(largest_contour)

                if contour_area > self.min_contour_area:
                    self.following = True
                    M = cv2.moments(largest_contour)
                    cx = int(M['m10'] / M['m00'])
                    height, width = cv_image.shape[:2]
                    error_x = float(cx - width / 2) / float(width / 2)  # Normalize

                    # Adjust heading based on error
                    if abs(error_x) > 0.05:
                        self.twist.angular.z = self.angular_speed * error_x  # Proportional control
                        self.twist.linear.x = 0.0  # Slow down while turning
                    else:
                        self.twist.linear.x = self.max_speed  # Move forward
                        self.twist.angular.z = 0.0

                else:  # Contour too small
                    if self.following:  # Stop if recently following something
                        rospy.loginfo("Lost small target - stopping")
                        self.twist.linear.x = 0.0
                        self.twist.angular.z = 0.0
                        self.following = False

                    else:  # Keep stopped
                        self.twist.linear.x = 0.0
                        self.twist.angular.z = 0.0

            else:  # No contours detected
                if self.following:  # Stop if was following something
                    rospy.loginfo("Lost target - stopping")
                    self.twist.linear.x = 0.0
                    self.twist.angular.z = 0.0
                    self.following = False
                else:  # Keep stopped
                    self.twist.linear.x = 0.0
                    self.twist.angular.z = 0.0

            self.cmd_vel_pub.publish(self.twist)

            # Visualization
            cv2.imshow("Camera View", cv_image)
            cv2.imshow("Processed Mask", mask)
            cv2.waitKey(3)

        except Exception as e:
            rospy.logerr(f"Error in image_callback: {e}")
            self.twist = Twist()
            self.cmd_vel_pub.publish(self.twist)

if __name__ == '__main__':
    try:
        YellowFollower()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
