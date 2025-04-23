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
        self.searching = True
        self.approaching = False
        self.stopped_at_target = False
        self.returning = False

        # Parameters
        self.min_contour_area = 500

        # Camera and object parameters (adjust these to your setup)
        self.focal_length_px = 554  # Replace with your camera's focal length in pixels
        self.real_yellow_width_m = 0.2  # Real width of yellow zone in meters
        self.stop_distance_m = 0.05  # Stop distance in meters

        # HSV thresholds for yellow (widened for lighting variations)
        self.lower_yellow = np.array([15, 60, 100])
        self.upper_yellow = np.array([40, 255, 255])

        # Odometry poses
        self.start_pose = None  # (x, y, yaw)
        self.current_pose = None  # (x, y, yaw)

        # Fixed target pose (set once when yellow first detected)
        self.target_pose = None  # (x, y)

        rospy.loginfo("YellowFollower node started, waiting for odometry and camera data...")

    def odom_callback(self, msg):
        position = msg.pose.pose.position
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        self.current_pose = (position.x, position.y, yaw)
        if self.start_pose is None:
            self.start_pose = self.current_pose
            rospy.loginfo(f"Recorded start pose: x={position.x:.2f}, y={position.y:.2f}, yaw={yaw:.2f}")

    def image_callback(self, msg):
        try:
            # If stopped at target and returning, run return controller and skip image processing
            if self.stopped_at_target and self.returning:
                self.return_to_start()
                self.cmd_vel_pub.publish(self.twist)
                return

            # If stopped at target but not returning yet, do nothing (wait)
            if self.stopped_at_target and not self.returning:
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

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                contour_area = cv2.contourArea(largest_contour)

                if contour_area > self.min_contour_area:
                    M = cv2.moments(largest_contour)
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])

                    height, width = cv_image.shape[:2]
                    error_x = (cx - width / 2) / (width / 2)

                    x, y, w, h = cv2.boundingRect(largest_contour)
                    distance = (self.focal_length_px * self.real_yellow_width_m) / w

                    rospy.loginfo_throttle(1, f"Estimated distance to yellow zone: {distance:.2f} m")

                    # If target_pose not set yet, set it once here based on current robot position + estimated distance forward
                    if self.target_pose is None and self.current_pose is not None:
                        # Calculate target position relative to robot pose and camera view
                        # Approximate target position in world frame:
                        # Assume robot faces along yaw, target is distance meters ahead + some lateral offset from error_x
                        x_robot, y_robot, yaw_robot = self.current_pose
                        # Lateral offset from center, scale error_x by some factor (e.g. half yellow width)
                        lateral_offset = error_x * self.real_yellow_width_m
                        # Target position in world frame
                        target_x = x_robot + distance * math.cos(yaw_robot) - lateral_offset * math.sin(yaw_robot)
                        target_y = y_robot + distance * math.sin(yaw_robot) + lateral_offset * math.cos(yaw_robot)
                        self.target_pose = (target_x, target_y)
                        rospy.loginfo(f"Set fixed target pose at x={target_x:.2f}, y={target_y:.2f}")

                        # Switch state
                        self.searching = False
                        self.approaching = True

                    # Approach fixed target_pose if set
                    if self.target_pose is not None and self.current_pose is not None:
                        # Compute distance and angle to fixed target
                        x_cur, y_cur, yaw_cur = self.current_pose
                        target_x, target_y = self.target_pose
                        dx = target_x - x_cur
                        dy = target_y - y_cur
                        dist_to_target = math.sqrt(dx*dx + dy*dy)
                        angle_to_target = math.atan2(dy, dx)
                        angle_diff = angle_to_target - yaw_cur
                        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

                        if dist_to_target <= self.stop_distance_m:
                            if not self.stopped_at_target:
                                rospy.loginfo("Reached fixed target zone - stopping permanently")
                                print("gripper open")  # Your requested print
                                self.twist.linear.x = 0
                                self.twist.angular.z = 0
                                self.approaching = False
                                self.stopped_at_target = True
                                self.returning = True  # Start return after stopping
                        else:
                            # Control to approach target_pose
                            if abs(angle_diff) > 0.1:
                                self.twist.linear.x = 0
                                self.twist.angular.z = 0.4 if angle_diff > 0 else -0.4
                            else:
                                self.twist.angular.z = 0
                                self.twist.linear.x = 0.15

                    # Visualization
                    cv2.circle(cv_image, (cx, cy), 10, (0, 255, 0), -1)
                    cv2.putText(cv_image, "APPROACHING FIXED TARGET", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    self.handle_no_detection(cv_image)
            else:
                self.handle_no_detection(cv_image)

            self.cmd_vel_pub.publish(self.twist)

            cv2.imshow("Camera View", cv_image)
            cv2.imshow("Processed Mask", mask)
            cv2.waitKey(3)

        except Exception as e:
            rospy.logerr(f"Error in image_callback: {e}")
            self.twist = Twist()
            self.cmd_vel_pub.publish(self.twist)

    def handle_no_detection(self, frame):
        if self.stopped_at_target:
            # Already stopped at target - do nothing
            self.twist.linear.x = 0
            self.twist.angular.z = 0
            cv2.putText(frame, "STOPPED AT TARGET", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            return

        if self.approaching:
            rospy.loginfo("Lost target - returning to search")
            self.approaching = False
            self.searching = True

        if self.searching:
            rospy.loginfo("Searching for target...")
            self.twist.linear.x = 0
            self.twist.angular.z = 0.3

        cv2.putText(frame, "SEARCHING", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def return_to_start(self):
        if self.current_pose is None or self.start_pose is None:
            rospy.logwarn_throttle(5, "Waiting for odometry data to return to start")
            self.twist.linear.x = 0
            self.twist.angular.z = 0
            return

        x_cur, y_cur, yaw_cur = self.current_pose
        x_start, y_start, yaw_start = self.start_pose

        dx = x_start - x_cur
        dy = y_start - y_cur
        distance = math.sqrt(dx * dx + dy * dy)

        angle_to_goal = math.atan2(dy, dx)
        angle_diff = angle_to_goal - yaw_cur
        # Normalize angle_diff to [-pi, pi]
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

        if distance < 0.1:
            rospy.loginfo("Returned to start point - stopping")
            self.twist.linear.x = 0
            self.twist.angular.z = 0
            self.returning = False
            # Optionally, reset flags if you want to repeat
        else:
            # Rotate towards goal if angle difference is large
            if abs(angle_diff) > 0.1:
                self.twist.linear.x = 0
                self.twist.angular.z = 0.3 if angle_diff > 0 else -0.3
            else:
                self.twist.linear.x = 0.1
                self.twist.angular.z = 0

if __name__ == '__main__':
    try:
        YellowFollower()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
