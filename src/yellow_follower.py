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
        self.completed = False  # Entire cycle done flag

        # Parameters
        self.min_contour_area = 500
        self.focal_length_px = 554  # Replace with your camera's focal length in pixels
        self.real_yellow_width_m = 0.2  # Real width of yellow zone in meters
        self.stop_distance_m = 0.3  # Increased stop distance
        self.search_speed = 0.2
        self.approach_speed = 0.1

        # HSV thresholds for yellow (widened for lighting variations)
        self.lower_yellow = np.array([15, 60, 100])
        self.upper_yellow = np.array([40, 255, 255])

        # Odometry poses
        self.start_pose = None  # (x, y, yaw)
        self.current_pose = None  # (x, y, yaw)
        self.initial_yaw = None

        # Search Direction
        self.search_direction = -1 # Left

        rospy.loginfo("YellowFollower node started, waiting for odometry and camera data...")

    def odom_callback(self, msg):
        position = msg.pose.pose.position
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        self.current_pose = (position.x, position.y, yaw)

        if self.start_pose is None:
            self.start_pose = (position.x, position.y)
            self.initial_yaw = yaw
            rospy.loginfo(f"Recorded start pose: x={position.x:.2f}, y={position.y:.2f}, yaw={yaw:.2f}")

    def image_callback(self, msg):
        try:
            # If entire cycle completed, stop and do nothing
            if self.completed:
                self.twist.linear.x = 0
                self.twist.angular.z = 0
                self.cmd_vel_pub.publish(self.twist)
                return

            # If stopped at target and returning, run return controller and skip image processing
            if self.stopped_at_target and self.returning:
                self.return_to_start()
                self.cmd_vel_pub.publish(self.twist)
                return

            # If stopped at target but not returning yet, hold position
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
                # Only detect target once
                largest_contour = max(contours, key=cv2.contourArea)
                contour_area = cv2.contourArea(largest_contour)

                if contour_area > self.min_contour_area:
                    M = cv2.moments(largest_contour)
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])

                    height, width = cv_image.shape[:2]
                    error_x = (cx - width / 2) / (width / 2)

                    #x, y, w, h = cv2.boundingRect(largest_contour)
                    #distance = (self.focal_length_px * self.real_yellow_width_m) / w

                    rospy.loginfo(f"Yellow zone detected - approaching")

                    if self.current_pose is not None:
                        #x_robot, y_robot, yaw_robot = self.current_pose
                        #lateral_offset = error_x * self.real_yellow_width_m
                        #target_x = x_robot + distance * math.cos(yaw_robot) - lateral_offset * math.sin(yaw_robot)
                        #target_y = y_robot + distance * math.sin(yaw_robot) + lateral_offset * math.cos(yaw_robot)

                        #self.target_pose = (target_x, target_y)

                        self.searching = False
                        self.approaching = True

                        self.twist.linear.x = self.approach_speed
                        self.twist.angular.z = -error_x * 0.5  # Steering to center

                        if abs(error_x) < 0.1 and self.twist.linear.x <= self.stop_distance_m: #If error is small and close enough
                            rospy.loginfo("Reached fixed target zone")
                            print("open gripper")
                            self.twist.linear.x = 0.0
                            self.twist.angular.z = 0.0
                            self.approaching = False
                            self.stopped_at_target = True
                            self.returning = True  # Start return after stopping
                else:
                    self.twist.linear.x = 0.0
                    self.twist.angular.z = 0.0


            elif self.searching:
                rospy.loginfo("Searching for Yellow")
                self.twist.linear.x = 0.0
                self.twist.angular.z = self.search_speed * self.search_direction
            else:
                self.twist.linear.x = 0.0
                self.twist.angular.z = 0.0


            self.cmd_vel_pub.publish(self.twist)

            # Visualization
            if cv_image is not None:
                if self.approaching:
                    cv2.putText(cv_image, "Approaching Yellow", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                elif self.searching:
                    cv2.putText(cv_image, "Searching for Yellow", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow("Camera View", cv_image)
                cv2.imshow("Processed Mask", mask)
                cv2.waitKey(3)

        except Exception as e:
            rospy.logerr(f"Error in image_callback: {e}")
            self.twist = Twist()
            self.cmd_vel_pub.publish(self.twist)

    def return_to_start(self):
        if self.start_pose is None or self.initial_yaw is None or self.current_pose is None:
            rospy.logwarn_throttle(10, "Start pose or initial yaw not initialized.")
            return

        x_start, y_start = self.start_pose
        x_current, y_current, current_yaw = self.current_pose
        dx = x_start - x_current
        dy = y_start - y_current
        distance = math.sqrt(dx**2 + dy**2)
        rospy.loginfo(f"Distance to start: {distance}")
        angle_to_start = math.atan2(dy, dx)
        yaw_diff = angle_to_start - current_yaw

        # Normalize the angle difference
        yaw_diff = (yaw_diff + math.pi) % (2 * math.pi) - math.pi

        rospy.loginfo(f"Yaw diff: {yaw_diff}")
        # Prioritize rotation if the angle is significant
        if abs(yaw_diff) > 0.1:
            self.twist.linear.x = 0.0
            self.twist.angular.z = 0.3 if yaw_diff > 0 else -0.3  # Corrected direction
            rospy.loginfo("Rotating...")
        elif distance > 0.1:
            self.twist.linear.x = 0.1
            self.twist.angular.z = 0.0
            rospy.loginfo("Moving straight...")
        else:
            # Final rotation to initial yaw
            final_yaw_diff = self.initial_yaw - current_yaw
            final_yaw_diff = (final_yaw_diff + math.pi) % (2 * math.pi) - math.pi

            if abs(final_yaw_diff) > 0.1:
                self.twist.linear.x = 0.0
                self.twist.angular.z = 0.3 if final_yaw_diff > 0 else -0.3  # Corrected direction
                rospy.loginfo("Final rotation...")
            else:
                self.twist.linear.x = 0.0
                self.twist.angular.z = 0.0
                self.returning = False
                self.completed = True
                rospy.loginfo("Reached starting point and orientation.")

        self.cmd_vel_pub.publish(self.twist)

if __name__ == '__main__':
    try:
        YellowFollower()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
