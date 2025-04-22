#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np

class YellowFollower:
    def __init__(self):
        rospy.init_node('yellow_follower')
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.twist = Twist()

    def image_callback(self, msg):
        try:
            
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"CV Bridge error: {e}")
            return

        # Convert to HSV
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Define HSV range for pure yellow
        lower_yellow = np.array([60, 100, 100])
        upper_yellow = np.array([35, 255, 255])

        # Mask for yellow
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Calculate centroid of the yellow region
        M = cv2.moments(mask)
        height, width, _ = cv_image.shape
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            error = cx - width / 2

            # Move toward the yellow object
            self.twist.linear.x = 0.1
            self.twist.angular.z = -float(error) / 100

            # Draw a circle on the detected centroid
            cv2.circle(cv_image, (cx, int(height / 2)), 10, (0, 255, 0), -1)
        else:
            # Stop if nothing is detected
            self.twist.linear.x = 0.0
            self.twist.angular.z = 0.0

        # Publish movement command
        self.cmd_vel_pub.publish(self.twist)

        # Show the camera feed and the mask
        cv2.imshow("Camera View", cv_image)
        cv2.imshow("Yellow Mask", mask)
        cv2.waitKey(3)

if __name__ == '__main__':
    try:
        YellowFollower()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
