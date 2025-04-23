#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
from tf.transformations import euler_from_quaternion

class YellowFollower:
    def __init__(self):
        rospy.init_node('yellow_follower')
        self.bridge = CvBridge()

        # Publishers & Subscribers
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # Motion state
        self.twist = Twist()
        self.target_pose = None
        self.current_pose = None
        self.start_pose = None
        self.returning = False
        self.cycle_complete = False

        # Parameters
        self.focal_length_px = 554         # Camera focal length
        self.real_target_width = 0.2       # Width of yellow zone in meters
        self.stop_distance = 0.05          # How close to stop (meters)
        self.min_area = 500                # Minimum area to validate detection
        self.slow_down_radius = 0.4        # Start slowing when within this distance
        self.max_speed = 0.25
        self.min_speed = 0.05

        # Yellow detection range (tweak as needed)
        self.lower_yellow = np.array([15, 60, 100])
        self.upper_yellow = np.array([40, 255, 255])

        rospy.loginfo("YellowFollower node initialized.")

    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        orientation = [ori.x, ori.y, ori.z, ori.w]
        _, _, yaw = euler_from_quaternion(orientation)
        self.current_pose = (pos.x, pos.y, yaw)

        if self.start_pose is None:
            self.start_pose = self.current_pose
            rospy.loginfo(f"Start pose recorded: {self.start_pose}")

    def image_callback(self, msg):
        if self.cycle_complete:
            self.cmd_vel_pub.publish(Twist())
            return

        if self.returning:
            self.return_to_start()
            self.cmd_vel_pub.publish(self.twist)
            return

        image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > self.min_area:
                M = cv2.moments(largest)
                if M['m00'] == 0:
                    return
                cx = int(M['m10'] / M['m00'])
                x, y, w, h = cv2.boundingRect(largest)
                image_center = image.shape[1] / 2
                error_x = (cx - image_center) / image_center
                distance = (self.focal_length_px * self.real_target_width) / w

                if self.current_pose:
                    xr, yr, yaw = self.current_pose
                    lat_offset = error_x * self.real_target_width
                    tx = xr + distance * math.cos(yaw) - lat_offset * math.sin(yaw)
                    ty = yr + distance * math.sin(yaw) + lat_offset * math.cos(yaw)
                    self.target_pose = (tx, ty)

        if self.target_pose and self.current_pose:
            self.move_to_target()
            self.cmd_vel_pub.publish(self.twist)

    def move_to_target(self):
        x, y, yaw = self.current_pose
        tx, ty = self.target_pose

        dx = tx - x
        dy = ty - y
        distance = math.hypot(dx, dy)
        target_angle = math.atan2(dy, dx)
        angle_diff = (target_angle - yaw + math.pi) % (2 * math.pi) - math.pi

        if distance < self.stop_distance:
            rospy.loginfo("Arrived at target. Returning to start.")
            self.twist = Twist()
            self.returning = True
            return

        speed = self.min_speed + (self.max_speed - self.min_speed) * min(distance / self.slow_down_radius, 1.0)

        if abs(angle_diff) > 0.1:
            self.twist.linear.x = 0
            self.twist.angular.z = 0.5 if angle_diff > 0 else -0.5
        else:
            self.twist.linear.x = speed
            self.twist.angular.z = 0

    def return_to_start(self):
        if not self.start_pose or not self.current_pose:
            return

        x, y, yaw = self.current_pose
        sx, sy, syaw = self.start_pose

        dx = sx - x
        dy = sy - y
        distance = math.hypot(dx, dy)
        target_angle = math.atan2(dy, dx)
        angle_diff = (target_angle - yaw + math.pi) % (2 * math.pi) - math.pi

        if distance > self.stop_distance:
            if abs(angle_diff) > 0.1:
                self.twist.linear.x = 0
                self.twist.angular.z = 0.4 if angle_diff > 0 else -0.4
            else:
                self.twist.linear.x = 0.15
                self.twist.angular.z = 0
        else:
            # Final orientation adjustment
            yaw_diff = (syaw - yaw + math.pi) % (2 * math.pi) - math.pi
            if abs(yaw_diff) > 0.1:
                self.twist.linear.x = 0
                self.twist.angular.z = 0.3 if yaw_diff > 0 else -0.3
            else:
                rospy.loginfo("Back to start position and orientation.")
                self.twist = Twist()
                self.returning = False
                self.cycle_complete = True

if __name__ == '__main__':
    try:
        YellowFollower()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
