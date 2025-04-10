#!/usr/bin/env python3

import rospy
import cv2
import cv_bridge
from sensor_msgs.msg import Image, CameraInfo, LaserScan
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import Twist, Point
from visualization_msgs.msg import Marker
from nav_msgs.msg import Odometry
import numpy as np
import time

class Follower:
    def __init__(self):
        self.node_name = "rosbot_black_line_follower"
        rospy.init_node(self.node_name)
        self.bridge = cv_bridge.CvBridge()

        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        self.rplidar_sub = rospy.Subscriber("/scan", LaserScan, self.lidar_callback)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.speed = Twist()
        self.last_seen_angular_speed = 0
        self.front_object = False
        self.object_detection = False
        self.rate = rospy.Rate(10)
        self.time_thres = 3

        # RViz Marker
        self.marker_pub = rospy.Publisher('/visualisation_marker', Marker, queue_size=1000)
        self.traveled_path_marker = self.create_marker('traveled_path')
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.current_position = Point()

    def odom_callback(self, msg):
        self.current_position.x = msg.pose.pose.position.x
        self.current_position.y = msg.pose.pose.position.y
        self.current_position.z = 0.2
        self.traveled_path_marker.points.append(self.current_position)
        self.marker_pub.publish(self.traveled_path_marker)
        self.rate.sleep()

    def create_marker(self, ns):
        marker = Marker()
        marker.header.frame_id = 'base_link'
        marker.type = marker.LINE_STRIP
        marker.action = marker.ADD
        marker.ns = ns
        marker.scale.x = 0.2
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.r = 0.4
        marker.color.g = 0.4
        marker.color.b = 0.4
        marker.color.a = 1.0
        marker.pose.orientation.w = 1.0
        return marker

    def lidar_callback(self, msg):
        current_time = rospy.Time.now()
        msg_age = current_time - msg.header.stamp
        if msg_age.to_sec() > self.time_thres:
            rospy.loginfo("Ignoring outdated LIDAR data")
            return

        ranges_len = len(msg.ranges)
        left_laser = msg.ranges[int(ranges_len*0.10):int(ranges_len*0.15)]
        right_laser = msg.ranges[int(ranges_len*0.85):int(ranges_len*0.90)]
        middle_laser1 = msg.ranges[int(ranges_len*0):int(ranges_len*0.05)]
        middle_laser2 = msg.ranges[int(ranges_len*0.95):int(ranges_len)]
        middle_laser_min = min(min(middle_laser1), min(middle_laser2))
        right_min = min(right_laser)
        left_min = min(left_laser)

        if right_min < 0.18 and left_min < 0.18:
            self.speed.angular.z = 0.0
            self.speed.linear.x = -0.1
            self.front_object = True
            self.object_detection = True
        elif left_min < 0.18:
            self.speed.angular.z = -0.3
            self.speed.linear.x = 0
            self.object_detection = True
        elif right_min < 0.18:
            self.speed.angular.z = 0.3
            self.speed.linear.x = 0
            self.object_detection = True
        elif 0.18 > middle_laser_min > 0:
            self.speed.angular.z = 0.0
            self.speed.linear.x = -0.1
            self.front_object = True
            self.object_detection = True
        elif 0.2 > middle_laser_min > 0.18:
            self.front_object = True
            self.object_detection = True
            self.speed.linear.x = 0.0
            self.cmd_vel_pub.publish(self.speed)
            time.sleep(1)
            self.speed.angular.z = 0.3
            self.speed.linear.x = 0.0
            count = 0
            while count < 100:
                count += 1
                self.cmd_vel_pub.publish(self.speed)
                self.rate.sleep()
        else:
            self.object_detection = False
            self.front_object = False

        self.cmd_vel_pub.publish(self.speed)

    def image_callback(self, msg):
        current_time = rospy.Time.now()
        msg_age = current_time - msg.header.stamp
        if msg_age.to_sec() > self.time_thres:
            rospy.loginfo("Ignoring outdated IMAGE data")
            return

        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Detect black line
        lower_black = np.array([0, 0, 0], dtype=np.uint8)
        upper_black = np.array([180, 255, 50], dtype=np.uint8)
        mask = cv2.inRange(hsv_image, lower_black, upper_black)
        res = cv2.bitwise_and(image, image, mask=mask)
        res_gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

        height, width = res_gray.shape
        res_gray[:int(height * 0.5), :] = 0  # Remove top half

        res_edges = cv2.Canny(res_gray, 70, 150)

        left_half = res_gray[:, :int(width * 0.60)]
        right_half = res_gray[:, int(width * 0.40):]
        left_black_pixels = cv2.countNonZero(left_half)
        right_black_pixels = cv2.countNonZero(right_half)
        total_black_pixels = left_black_pixels + right_black_pixels

        if total_black_pixels > 0:
            turn_ratio = ((left_black_pixels - right_black_pixels) / total_black_pixels) / 2
            max_turn_speed = 0.5
            turn_speed = max_turn_speed * turn_ratio

        if not self.object_detection and not self.front_object:
            if total_black_pixels > 2000:
                self.speed.angular.z = turn_speed
                self.last_seen_angular_speed = turn_speed
                self.speed.linear.x = 0.15
            else:
                self.speed.angular.z = 0.3 if self.last_seen_angular_speed > 0 else -0.3
                self.speed.linear.x = 0
            self.cmd_vel_pub.publish(self.speed)

        # 👁️ Visualization
        cv2.imshow("Original Image", image)
        cv2.imshow("Black Mask", mask)
        cv2.imshow("Masked Image (res)", res)
        cv2.imshow("Edge Detection", res_edges)
        cv2.waitKey(3)

if __name__ == '__main__':
    follower = Follower()
    rospy.spin()
