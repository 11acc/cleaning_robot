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
        self.returning = False
        self.completed = False

        # Parameters
        self.min_contour_area = 500
        self.max_speed = 0.25  # Forward speed
        self.angular_speed = 0.4  # Turning speed
        self.max_approach_distance = 1.5  # meters to move forward while approaching

        # HSV thresholds for yellow (adjust if needed)
        self.lower_yellow = np.array([15, 60, 100])
        self.upper_yellow = np.array([40, 255, 255])

        # Odometry
        self.start_pose = None  # Robot pose when node starts
        self.approach_start_pose = None  # Robot pose when started approaching
        self.current_pose = None

        rospy.loginfo("YellowFollower node started, waiting for camera and odometry data...")

    def odom_callback(self, msg):
        position = msg.pose.pose.position
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (_, _, yaw) = euler_from_quaternion(orientation_list)
        self.current_pose = (position.x, position.y, yaw)

        if self.start_pose is None:
            self.start_pose = self.current_pose
            rospy.loginfo(f"Recorded start pose: x={position.x:.2f}, y={position.y:.2f}")

    def distance_moved(self, start, current):
        dx = current[0] - start[0]
        dy = current[1] - start[1]
        return math.sqrt(dx*dx + dy*dy)

    def image_callback(self, msg):
        try:
            if self.completed:
                # Stop robot permanently
                self.twist.linear.x = 0
                self.twist.angular.z = 0
                self.cmd_vel_pub.publish(self.twist)
                return

            if self.returning:
                # Return to start pose
                self.return_to_start()
                self.cmd_vel_pub.publish(self.twist)
                return

            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

            mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)

            contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            yellow_detected = False
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                contour_area = cv2.contourArea(largest_contour)

                if contour_area > self.min_contour_area:
                    yellow_detected = True
                    M = cv2.moments(largest_contour)
                    cx = int(M['m10'] / M['m00'])
                    height, width = cv_image.shape[:2]
                    error_x = float(cx - width / 2) / float(width / 2)  # Normalize

                    if self.searching:
                        rospy.loginfo("Yellow detected, starting approach")
                        self.searching = False
                        self.approaching = True
                        self.approach_start_pose = self.current_pose  # Record where approach started

                    if self.approaching:
                        # Check distance moved since start of approach
                        dist_moved = self.distance_moved(self.approach_start_pose, self.current_pose)
                        if dist_moved >= self.max_approach_distance:
                            rospy.loginfo(f"Approached max distance {self.max_approach_distance}m, stopping approach")
                            self.approaching = False
                            self.returning = True
                            self.twist.linear.x = 0
                            self.twist.angular.z = 0
                        else:
                            # Control robot to approach yellow
                            if abs(error_x) > 0.05:
                                self.twist.angular.z = -self.angular_speed * error_x  # Turn proportional to error_x
                                self.twist.linear.x = 0.0  # Slow down while turning
                            else:
                                self.twist.linear.x = self.max_speed
                                self.twist.angular.z = 0.0

                else:
                    # Contour too small
                    if self.approaching:
                        rospy.loginfo("Yellow lost during approach - stopping and returning")
                        self.approaching = False
                        self.returning = True
                        self.twist.linear.x = 0
                        self.twist.angular.z = 0
                    else:
                        self.twist.linear.x = 0
                        self.twist.angular.z = 0
            else:
                # No contours found
                if self.approaching:
                    rospy.loginfo("Yellow lost during approach - stopping and returning")
                    self.approaching = False
                    self.returning = True
                    self.twist.linear.x = 0
                    self.twist.angular.z = 0
                else:
                    self.twist.linear.x = 0
                    self.twist.angular.z = 0

            self.cmd_vel_pub.publish(self.twist)

            # Visualization
            cv2.imshow("Camera View", cv_image)
            cv2.imshow("Processed Mask", mask)
            cv2.waitKey(3)

        except Exception as e:
            rospy.logerr(f"Error in image_callback: {e}")
            self.twist = Twist()
            self.cmd_vel_pub.publish(self.twist)

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
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

        # Position control
        if distance > 0.05:  # 5 cm tolerance
            if abs(angle_diff) > 0.05:
                self.twist.linear.x = 0
                self.twist.angular.z = 0.3 if angle_diff > 0 else -0.3
            else:
                self.twist.linear.x = 0.1
                self.twist.angular.z = 0
            return

        # Orientation control once position reached
        yaw_diff = yaw_start - yaw_cur
        yaw_diff = (yaw_diff + math.pi) % (2 * math.pi) - math.pi

        if abs(yaw_diff) > 0.05:  # ~3 degrees tolerance
            self.twist.linear.x = 0
            self.twist.angular.z = 0.3 if yaw_diff > 0 else -0.3
        else:
            rospy.loginfo("Returned to start pose with correct orientation - stopping")
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
