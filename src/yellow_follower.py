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

        # Publishers and Subscribers
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # Movement
        self.twist = Twist()

        # Flags
        self.target_pose = None
        self.current_pose = None
        self.start_pose = None
        self.returning = False
        self.cycle_complete = False

        # Detection Parameters
        self.min_contour_area = 500
        self.lower_yellow = np.array([15, 60, 100])
        self.upper_yellow = np.array([40, 255, 255])
        self.focal_length_px = 554
        self.real_yellow_width_m = 0.2
        self.stop_distance_m = 0.02  # Stop a bit earlier
        rospy.loginfo("YellowFollower node initialized.")

    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
        self.current_pose = (pos.x, pos.y, yaw)
        if self.start_pose is None:
            self.start_pose = self.current_pose
            rospy.loginfo("Start pose recorded.")

    def image_callback(self, msg):
        if self.cycle_complete:
            self.cmd_vel_pub.publish(Twist())
            return

        if self.returning:
            self.return_to_start()
            self.cmd_vel_pub.publish(self.twist)
            return

        if self.target_pose is None:
            self.detect_yellow_zone(msg)

        if self.target_pose and self.current_pose:
            self.navigate_to_target()
            self.cmd_vel_pub.publish(self.twist)

    def detect_yellow_zone(self, msg):
        cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            area = cv2.contourArea(c)
            if area > self.min_contour_area:
                M = cv2.moments(c)
                cx = int(M['m10'] / M['m00'])
                x, y, w, h = cv2.boundingRect(c)
                distance = (self.focal_length_px * self.real_yellow_width_m) / w
                error_x = (cx - cv_img.shape[1] / 2) / (cv_img.shape[1] / 2)
                if self.current_pose:
                    x_r, y_r, yaw = self.current_pose
                    offset = error_x * self.real_yellow_width_m
                    tx = x_r + distance * math.cos(yaw) - offset * math.sin(yaw)
                    ty = y_r + distance * math.sin(yaw) + offset * math.cos(yaw)
                    self.target_pose = (tx, ty)
                    rospy.loginfo(f"Target pose set: x={tx:.2f}, y={ty:.2f}")

    def navigate_to_target(self):
        x, y, yaw = self.current_pose
        tx, ty = self.target_pose
        dx, dy = tx - x, ty - y
        dist = math.hypot(dx, dy)
        angle_to = math.atan2(dy, dx)
        d_angle = (angle_to - yaw + math.pi) % (2 * math.pi) - math.pi

        if dist < self.stop_distance_m:
            rospy.loginfo("Reached target zone, starting return.")
            self.twist = Twist()
            self.returning = True
            return

        # Movement logic
        max_spd, min_spd = 0.3, 0.05
        slowdown_radius = 0.5
        spd = min_spd + (max_spd - min_spd) * min(dist / slowdown_radius, 1.0)
        spd = max(spd, min_spd)

        if abs(d_angle) > 0.1:
            self.twist.linear.x = 0
            self.twist.angular.z = 0.5 if d_angle > 0 else -0.5
        else:
            self.twist.linear.x = spd
            self.twist.angular.z = 0

    def return_to_start(self):
        x, y, yaw = self.current_pose
        sx, sy, syaw = self.start_pose
        dx, dy = sx - x, sy - y
        dist = math.hypot(dx, dy)
        angle_to = math.atan2(dy, dx)
        d_angle = (angle_to - yaw + math.pi) % (2 * math.pi) - math.pi

        if dist > self.stop_distance_m:
            if abs(d_angle) > 0.1:
                self.twist.linear.x = 0
                self.twist.angular.z = 0.4 if d_angle > 0 else -0.4
            else:
                self.twist.linear.x = 0.2
                self.twist.angular.z = 0
        else:
            yaw_diff = (syaw - yaw + math.pi) % (2 * math.pi) - math.pi
            if abs(yaw_diff) > 0.1:
                self.twist.linear.x = 0
                self.twist.angular.z = 0.4 if yaw_diff > 0 else -0.4
            else:
                rospy.loginfo("Returned to start pose. Cycle complete.")
                self.twist = Twist()
                self.cycle_complete = True

if __name__ == '__main__':
    try:
        YellowFollower()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
