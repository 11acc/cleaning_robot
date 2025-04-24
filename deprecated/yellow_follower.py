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
from std_msgs.msg import UInt16, Float64

class YellowFollower:
    def __init__(self):
        rospy.init_node('yellow_follower')
        self.bridge = CvBridge()

        # Subscribers and publishers
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.servo_pub = rospy.Publisher('/servo', UInt16, queue_size=10)
        self.servo_load_pub = rospy.Publisher('/servoLoad', Float64, queue_size=10)

        self.twist = Twist()

        # State flags
        self.searching = True
        self.approaching = False
        self.stopped_at_target = False
        self.returning = False
        self.completed = False
        self.recording_360 = False

        # Parameters
        self.min_contour_area = 500
        self.center_x = 320  # Center of the camera image
        self.center_y = 240
        self.fov_horizontal = 60  # Horizontal field of view in degrees
        self.search_rotation_speed = 0.2
        self.focal_length_px = 554
        self.real_yellow_width_m = 0.2
        self.stop_distance_m = 0.03
        self.max_distance_from_start_m = 1.5
        self.rotation_duration = 10.0

        # HSV thresholds for yellow
        self.lower_yellow = np.array([15, 60, 100])
        self.upper_yellow = np.array([40, 255, 255])

        # Odometry poses
        self.start_pose = None
        self.current_pose = None
        self.target_pose = None

        # 360 Scan Data
        self.scan_data = []  # List to store (yaw, mask_area) pairs
        self.start_rotation_time = None

        rospy.loginfo("YellowFollower node started...")

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
            if self.completed:
                self.twist.linear.x = 0
                self.twist.angular.z = 0
                self.cmd_vel_pub.publish(self.twist)
                return

            if self.stopped_at_target and self.returning:
                self.return_to_start()
                self.cmd_vel_pub.publish(self.twist)
                return
            
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
            mask_area = 0  # Total yellow area in the current mask
            
            # Get centroid of the largest contour
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX, cY = self.center_x, self.center_y  # Default to center if no moment
            else:
                cX, cY = self.center_x, self.center_y
            
            if contours:
                for contour in contours:
                    mask_area += cv2.contourArea(contour)  # Accumulate area for all yellow contours

            if self.searching:
                if not self.recording_360:
                    rospy.loginfo("Starting 360 degree scan...")
                    self.recording_360 = True
                    self.start_rotation_time = rospy.Time.now().to_sec()
                    self.scan_data = []  # Clear previous scan data
                else:
                    # Continue rotating and record mask area
                    self.twist.linear.x = 0
                    self.twist.angular.z = self.search_rotation_speed
                    self.cmd_vel_pub.publish(self.twist)

                    # Record the yaw and mask area
                    if self.current_pose is not None:
                        _, _, yaw = self.current_pose
                        self.scan_data.append((yaw, mask_area))  # Store current yaw and mask area

                    # Check if rotation duration is complete
                    time_now = rospy.Time.now().to_sec()
                    if (time_now - self.start_rotation_time) >= self.rotation_duration:
                        rospy.loginfo("360 degree scan complete, finding best yellow zone...")
                        self.recording_360 = False
                        self.twist.angular.z = 0.0
                        self.cmd_vel_pub.publish(self.twist)
                        self.find_and_set_best_target()  # Find the best target after scanning

            # Approaching logic (only runs if target is set and not searching)
            if self.approaching and self.target_pose is not None and self.current_pose is not None:
                x_cur, y_cur, yaw_cur = self.current_pose
                target_x, target_y = self.target_pose
                dx = target_x - x_cur
                dy = target_y - y_cur
                dist_to_target = math.sqrt(dx * dx + dy * dy)
                angle_to_target = math.atan2(dy, dx)
                angle_diff = angle_to_target - yaw_cur
                angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

                # Calculate angle error to center the yellow object
                err_x = float(cX - self.center_x) / self.center_x
                angle_adjust = -err_x * 0.3  # Adjust gain as needed

                if dist_to_target <= self.stop_distance_m +0.05:
                    rospy.loginfo("Reached target zone")
                    self.open_gripper()
                    self.twist.linear.x = 0
                    self.twist.angular.z = 0
                    self.approaching = False
                    self.stopped_at_target = True
                    self.returning = True
                else:
                    max_speed = 0.15
                    min_speed = 0.02
                    slow_down_radius = 0.3

                    if dist_to_target < slow_down_radius:
                        speed = min_speed + (max_speed - min_speed) * (dist_to_target / slow_down_radius)
                        speed = max(speed, min_speed)
                    else:
                        speed = max_speed

                    if abs(angle_diff) > 0.05:
                        self.twist.linear.x = 0
                        self.twist.angular.z = 0.4 if angle_diff > 0 else -0.4
                    else:
                        self.twist.angular.z = angle_adjust  # Use PID control to maintain center
                        self.twist.linear.x = speed

                self.cmd_vel_pub.publish(self.twist)

            # Visualization
            if cv_image is not None:
                if self.target_pose is not None:
                    cv2.putText(cv_image, "Target Set", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                elif self.searching:
                    cv2.putText(cv_image, "Searching for Yellow", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.circle(cv_image, (cX, cY), 5, (255, 255, 255), -1)  # Display centroid
                cv2.imshow("Camera View", cv_image)
                cv2.imshow("Processed Mask", mask)
                cv2.waitKey(3)

        except Exception as e:
            rospy.logerr(f"Error in image_callback: {e}")
            self.twist = Twist()
            self.cmd_vel_pub.publish(self.twist)

    def find_and_set_best_target(self):
        if not self.scan_data:
            rospy.logwarn("No scan data available to find best target.")
            return

        # Find the yaw with the maximum yellow mask area
        best_yaw = max(self.scan_data, key=lambda item: item[1])[0]

        # Calculate target pose based on the best yaw
        distance = 1.0  # Fixed distance for the target (adjust as needed)
        x_robot, y_robot, _ = self.current_pose
        target_x = x_robot + distance * math.cos(best_yaw)
        target_y = y_robot + distance * math.sin(best_yaw)

        self.target_pose = (target_x, target_y)
        self.searching = False
        self.approaching = True
        rospy.loginfo(f"Best yellow zone found at yaw {best_yaw:.2f}. Target set to x={target_x:.2f}, y={target_y:.2f}")

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

        # First, position control
        if distance > self.stop_distance_m:
            if abs(angle_diff) > 0.05:
                self.twist.linear.x = 0
                self.twist.angular.z = 0.3 if angle_diff > 0 else -0.3
            else:
                self.twist.linear.x = 0.1
                self.twist.angular.z = 0
            return

        # Then, orientation control once position is reached
        yaw_diff = yaw_start - yaw_cur
        yaw_diff = (yaw_diff + math.pi) % (2 * math.pi) - math.pi

        if abs(yaw_diff) > 0.05:
            self.twist.linear.x = 0
            self.twist.angular.z = 0.3 if yaw_diff > 0 else -0.3
        else:
            # Reached position and orientation
            rospy.loginfo("Returned to start pose with correct orientation - stopping")
            self.twist.linear.x = 0
            self.twist.angular.z = 0
            self.returning = False
            self.completed = True

    def open_gripper(self):
        self.servo_pub.publish(0)  # Open the gripper
        self.servo_load_pub.publish(0.0)
        rospy.loginfo("Gripper opened")

if __name__ == '__main__':
    try:
        YellowFollower()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
