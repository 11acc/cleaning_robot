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

        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        self.twist = Twist()

        self.searching = True
        self.approaching = False
        self.stopped_at_target = False
        self.returning = False
        self.completed = False

        self.min_contour_area = 500
        self.focal_length_px = 554
        self.real_yellow_width_m = 0.2
        self.stop_distance_m = 0.05

        self.lower_yellow = np.array([15, 60, 100])
        self.upper_yellow = np.array([40, 255, 255])

        self.start_pose = None
        self.current_pose = None
        self.target_pose = None

        rospy.loginfo("YellowFollower node started")

    def odom_callback(self, msg):
        position = msg.pose.pose.position
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        self.current_pose = (position.x, position.y, yaw)
        if self.start_pose is None:
            self.start_pose = self.current_pose
            rospy.loginfo(f"Start pose recorded: x={position.x:.2f}, y={position.y:.2f}, yaw={yaw:.2f}")

    def image_callback(self, msg):
        try:
            if self.completed:
                self.twist.linear.x = 0
                self.twist.angular.z = 0
                self.cmd_vel_pub.publish(self.twist)
                return

            if self.stopped_at_target and self.returning:
                self.return_to_start()
                self.cmd_vel_pub.publish(self.twist)
                return

            if self.stopped_at_target:
                self.twist.linear.x = 0
                self.twist.angular.z = 0
                self.cmd_vel_pub.publish(self.twist)
                return

            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

            mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)

            contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours and self.target_pose is None:
                largest_contour = max(contours, key=cv2.contourArea)
                contour_area = cv2.contourArea(largest_contour)

                if contour_area > self.min_contour_area:
                    M = cv2.moments(largest_contour)
                    if M['m00'] == 0:
                        return
                    cx = int(M['m10'] / M['m00'])

                    height, width = cv_image.shape[:2]
                    error_x = (cx - width / 2) / (width / 2)

                    x, y, w, h = cv2.boundingRect(largest_contour)
                    perceived_width = max(w, h)
                    distance_to_center = (self.focal_length_px * self.real_yellow_width_m) / perceived_width

                    approach_distance = max(distance_to_center - 0.2, 0.2)

                    if self.current_pose is not None:
                        x_robot, y_robot, yaw_robot = self.current_pose
                        lateral_offset = error_x * self.real_yellow_width_m
                        target_x = x_robot + approach_distance * math.cos(yaw_robot) - lateral_offset * math.sin(yaw_robot)
                        target_y = y_robot + approach_distance * math.sin(yaw_robot) + lateral_offset * math.cos(yaw_robot)
                        self.target_pose = (target_x, target_y)

                        rospy.loginfo(f"Target set: x={target_x:.2f}, y={target_y:.2f}")
                        self.searching = False
                        self.approaching = True

            if self.target_pose is not None and self.current_pose is not None and not self.stopped_at_target:
                x_cur, y_cur, yaw_cur = self.current_pose
                target_x, target_y = self.target_pose
                dx = target_x - x_cur
                dy = target_y - y_cur
                dist_to_target = math.sqrt(dx * dx + dy * dy)
                angle_to_target = math.atan2(dy, dx)
                angle_diff = (angle_to_target - yaw_cur + math.pi) % (2 * math.pi) - math.pi

                if dist_to_target <= self.stop_distance_m:
                    rospy.loginfo("Reached target zone")
                    print("gripper open")
                    self.twist.linear.x = 0
                    self.twist.angular.z = 0
                    self.approaching = False
                    self.stopped_at_target = True
                    self.returning = True
                else:
                    max_speed = 0.15
                    min_speed = 0.02
                    slow_down_radius = 0.3
                    speed = min_speed + (max_speed - min_speed) * min(dist_to_target / slow_down_radius, 1.0)

                    if abs(angle_diff) > 0.05:
                        self.twist.linear.x = 0
                        self.twist.angular.z = 0.4 if angle_diff > 0 else -0.4
                    else:
                        self.twist.angular.z = 0
                        self.twist.linear.x = speed

            self.cmd_vel_pub.publish(self.twist)

            if cv_image is not None:
                if self.target_pose is not None:
                    cv2.putText(cv_image, "Target Set", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                elif self.searching:
                    cv2.putText(cv_image, "Searching...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow("Camera View", cv_image)
                cv2.imshow("Mask", mask)
                cv2.waitKey(3)

        except Exception as e:
            rospy.logerr(f"Error: {e}")
            self.twist = Twist()
            self.cmd_vel_pub.publish(self.twist)

    def return_to_start(self):
        if self.current_pose is None or self.start_pose is None:
            rospy.logwarn_throttle(5, "Waiting for odometry to return")
            self.twist.linear.x = 0
            self.twist.angular.z = 0
            return

        x_cur, y_cur, yaw_cur = self.current_pose
        x_start, y_start, yaw_start = self.start_pose

        dx = x_start - x_cur
        dy = y_start - y_cur
        distance = math.sqrt(dx * dx + dy * dy)

        angle_to_goal = math.atan2(dy, dx)
        angle_diff = (angle_to_goal - yaw_cur + math.pi) % (2 * math.pi) - math.pi

        if distance > self.stop_distance_m:
            if abs(angle_diff) > 0.05:
                self.twist.linear.x = 0
                self.twist.angular.z = 0.3 if angle_diff > 0 else -0.3
            else:
                self.twist.linear.x = 0.1
                self.twist.angular.z = 0
            return

        yaw_diff = (yaw_start - yaw_cur + math.pi) % (2 * math.pi) - math.pi

        if abs(yaw_diff) > 0.05:
            self.twist.linear.x = 0
            self.twist.angular.z = 0.3 if yaw_diff > 0 else -0.3
        else:
            rospy.loginfo("Returned to start. Done.")
            self.twist.linear.x = 0
            self.twist.angular.z = 0
            self.returning = False
            self.completed = True

if __name__ == '__main__':
    try:
        YellowFollower()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
