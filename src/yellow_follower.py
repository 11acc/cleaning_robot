#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, Range
from geometry_msgs.msg import Twist
from std_msgs.msg import UInt16, Float64
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
import math

class CombinedNode:
    def __init__(self):
        rospy.init_node('combined_node')
        self.bridge = CvBridge()

        # --- Red Peg Grabber Parameters and Variables ---
        self.SEARCHING = 0
        self.APPROACHING = 1
        self.POSITIONING = 2
        self.GRABBING = 3
        self.GRABBED = 4
        self.MOVING_TO_YELLOW = 5  # New state to move to the yellow zone
        self.RELEASING = 6  # New state to release the peg
        self.RETURNING = 7 # New state to return to original pose

        self.state = self.SEARCHING
        self.rate = rospy.Rate(10)  # 10Hz

        self.state_start_time = rospy.Time.now()
        self.state_timeout = {
            self.SEARCHING: 20.0,
            self.APPROACHING: 15.0,
            self.POSITIONING: 10.0,
            self.GRABBING: 10.0,  # Increased grabbing timeout for load check
            self.GRABBED: 3.0,
            self.MOVING_TO_YELLOW: 30.0,  # Timeout for moving to yellow
            self.RELEASING: 5.0,  # Timeout for releasing
            self.RETURNING: 30.0
        }

        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        self.fl_sensor_sub = rospy.Subscriber('/range/fl', Range, self.fl_sensor_callback)
        self.fr_sensor_sub = rospy.Subscriber('/range/fr', Range, self.fr_sensor_callback)
        self.rl_sensor_sub = rospy.Subscriber('/range/rl', Range, self.rl_sensor_callback)
        self.rr_sensor_sub = rospy.Subscriber('/range/rr', Range, self.rr_sensor_callback)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)  # Odometry subscriber
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.servo_pub = rospy.Publisher('/servo', UInt16, queue_size=10)
        self.servo_load_pub = rospy.Publisher('/servoLoad', Float64, queue_size=10)

        self.current_image = None
        self.red_peg_detected = False
        self.red_peg_center_x = 0
        self.red_peg_area = 0
        self.fl_distance = float('inf')
        self.fr_distance = float('inf')
        self.rl_distance = float('inf')
        self.rr_distance = float('inf')
        self.front_distance = float('inf')
        self.left_distance = float('inf')
        self.right_distance = float('inf')
        self.current_servo_load = 0.0 # servo load
        self.peg_grabbed = False # Add a flag to track if the peg is grabbed
        self.servo_load_threshold = 5.0 # Adjust this value based on testing

        # --- Yellow Follower Parameters and Variables ---
        self.searching_yellow = True
        self.approaching_yellow = False
        self.stopped_at_yellow = False
        self.returning_to_start = False
        self.completed = False
        self.min_contour_area = 500
        self.focal_length_px = 554
        self.real_yellow_width_m = 0.2
        self.stop_distance_m = 0.05
        self.max_distance_from_red_m = 1.5
        self.lower_yellow = np.array([15, 60, 100])
        self.upper_yellow = np.array([40, 255, 255])
        self.start_pose = None
        self.current_pose = None
        self.target_pose = None

        # Subscribe to the servo load topic
        self.servo_load_sub = rospy.Subscriber('/servoLoad', Float64, self.servo_load_callback)

        rospy.loginfo("Combined Node initialized")

    # --- Odometry Callback ---
    def odom_callback(self, msg):
        position = msg.pose.pose.position
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        self.current_pose = (position.x, position.y, yaw)
        if self.start_pose is None:
            self.start_pose = self.current_pose
            rospy.loginfo(f"Recorded start pose: x={position.x:.2f}, y={position.y:.2f}, yaw={yaw:.2f}")
    
    def servo_load_callback(self, load):
        self.current_servo_load = load.data

    def image_callback(self, data):
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            if self.state in [self.SEARCHING, self.APPROACHING, self.POSITIONING]:
                self.detect_red_peg(self.current_image)
            elif self.state == self.MOVING_TO_YELLOW:
                self.process_yellow_detection(self.current_image)
        except CvBridgeError as e:
            rospy.logerr(e)

    def fl_sensor_callback(self, data):
        self.fl_distance = data.range
        self.left_distance = self.fl_distance
        self.update_front_distance()

    def fr_sensor_callback(self, data):
        self.fr_distance = data.range
        self.right_distance = self.fr_distance
        self.update_front_distance()

    def rl_sensor_callback(self, data):
        self.rl_distance = data.range

    def rr_sensor_callback(self, data):
        self.rr_distance = data.range

    def update_front_distance(self):
        self.front_distance = min(self.fl_distance, self.fr_distance)

    def detect_red_peg(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.red_peg_detected = False

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            if area > 100:
                self.red_peg_detected = True
                self.red_peg_area = area
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    self.red_peg_center_x = cx

    def process_yellow_detection(self, image):
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
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

                    height, width = image.shape[:2]
                    error_x = (cx - width / 2) / (width / 2)

                    x, y, w, h = cv2.boundingRect(largest_contour)
                    distance = (self.focal_length_px * self.real_yellow_width_m) / w

                    if self.current_pose is not None:
                        x_robot, y_robot, yaw_robot = self.current_pose
                        lateral_offset = error_x * self.real_yellow_width_m
                        target_x = x_robot + distance * math.cos(yaw_robot) - lateral_offset * math.sin(yaw_robot)
                        target_y = y_robot + distance * math.sin(yaw_robot) + lateral_offset * math.cos(yaw_robot)

                        self.target_pose = (target_x, target_y)
                        self.searching_yellow = False
                        self.approaching_yellow = True

            else:
                # No yellow detected
                if self.approaching_yellow:
                    rospy.loginfo("Yellow lost, stopping")
                    self.stop_and_transition(self.RELEASING)
                    return

            # If target_pose set, approach it
            if self.target_pose is not None and self.current_pose is not None:
                x_cur, y_cur, yaw_cur = self.current_pose
                target_x, target_y = self.target_pose
                dx = target_x - x_cur
                dy = target_y - y_cur
                dist_to_target = math.sqrt(dx * dx + dy * dy)
                angle_to_target = math.atan2(dy, dx)
                angle_diff = angle_to_target - yaw_cur
                angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

                if dist_to_target <= self.stop_distance_m:
                    rospy.loginfo("Reached target zone")
                    self.stop_and_transition(self.RELEASING)
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
                        self.move(0, 0.4 if angle_diff > 0 else -0.4)
                    else:
                        self.move(speed, 0)
        except Exception as e:
            rospy.logerr(f"Error in process_yellow_detection: {e}")
            self.move(0, 0)

    def run(self):
        while not rospy.is_shutdown():
            self.execute_state()
            self.rate.sleep()

    def execute_state(self):
        if self.state == self.SEARCHING:
            self.search()
        elif self.state == self.APPROACHING:
            self.approach()
        elif self.state == self.POSITIONING:
            self.position()
        elif self.state == self.GRABBING:
            self.grab()
        elif self.state == self.GRABBED:
            self.hold_grabbed()
        elif self.state == self.MOVING_TO_YELLOW:
            self.move_to_yellow()
        elif self.state == self.RELEASING:
            self.release()
        elif self.state == self.RETURNING:
            self.return_to_start()

        self.handle_state_timeout()

    def search(self):
        rospy.loginfo("Searching for red peg")
        self.move(0.1, 0.0)  # Example: move forward slowly while searching
        if self.red_peg_detected:
            rospy.loginfo("Red peg detected, approaching")
            self.transition_to(self.APPROACHING)

    def approach(self):
        rospy.loginfo("Approaching red peg")
        if self.red_peg_detected:
            error = self.current_image.shape[1] / 2 - self.red_peg_center_x
            angular_speed = -float(error) / 1000
            linear_speed = 0.1
            self.move(linear_speed, angular_speed)

            if self.front_distance < 0.3:
                rospy.loginfo("Close enough, positioning")
                self.transition_to(self.POSITIONING)
        else:
            rospy.loginfo("Red peg lost, searching")
            self.transition_to(self.SEARCHING)

    def position(self):
        rospy.loginfo("Positioning in front of red peg")
        if self.red_peg_detected:
            error = self.current_image.shape[1] / 2 - self.red_peg_center_x
            angular_speed = -float(error) / 1000
            linear_speed = 0.05
            self.move(linear_speed, angular_speed)

            # Check if well-aligned and close
            if abs(error) < 50 and self.front_distance < 0.2:
                rospy.loginfo("Ready to grab, grabbing")
                self.transition_to(self.GRABBING)
        else:
            rospy.loginfo("Red peg lost, searching")
            self.transition_to(self.SEARCHING)

    def grab(self):
        rospy.loginfo("Grabbing red peg")
        self.move(0, 0)  # Stop moving
        self.control_gripper(1000)  # Close the gripper
        grab_start_time = rospy.Time.now()

        while (rospy.Time.now() - grab_start_time).to_sec() < self.state_timeout[self.GRABBING]:
            rospy.sleep(0.1)  # Check servo load frequently

            if self.current_servo_load > self.servo_load_threshold:
                rospy.loginfo("Peg grabbed successfully!")
                rospy.loginfo(f"Servo Load: {self.current_servo_load}")
                self.peg_grabbed = True
                self.transition_to(self.MOVING_TO_YELLOW)
                return  # Exit the GRABBING state immediately

        # If servo load doesn't reach threshold
        rospy.logwarn("Failed to grab peg, servo load insufficient!")
        self.control_gripper(0)  # Open gripper
        self.transition_to(self.SEARCHING)

    def hold_grabbed(self):
        rospy.loginfo("Holding grabbed peg")
        self.move(0, 0)  # Keep holding position
        rospy.sleep(self.state_timeout[self.GRABBED])
        self.transition_to(self.MOVING_TO_YELLOW)

    def move_to_yellow(self):
         rospy.loginfo("Moving to yellow zone")
         self.searching_yellow = True
         self.approaching_yellow = False

         while not rospy.is_shutdown() and self.searching_yellow:
            self.process_yellow_detection(self.current_image)
            self.rate.sleep()

         if not self.searching_yellow:
            rospy.loginfo("Yellow zone found, moving towards it")

    def release(self):
        rospy.loginfo("Releasing red peg")
        self.move(0, 0)  # Stop moving
        self.control_gripper(0)  # Open the gripper
        rospy.sleep(2)  # Simulate releasing

        rospy.loginfo("Peg released, transitioning to RETURNING")
        self.transition_to(self.RETURNING)

    def return_to_start(self):
        if self.current_pose is None or self.start_pose is None:
            rospy.logwarn_throttle(5, "Waiting for odometry data to return to start")
            self.move(0,0)
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
                self.move(0, 0.3 if angle_diff > 0 else -0.3)
            else:
                self.move(0.1, 0)
            return

        # Then, orientation control once position is reached
        yaw_diff = yaw_start - yaw_cur
        yaw_diff = (yaw_diff + math.pi) % (2 * math.pi) - math.pi

        if abs(yaw_diff) > 0.05:
            self.move(0, 0.3 if yaw_diff > 0 else -0.3)
        else:
            # Reached position and orientation
            rospy.loginfo("Returned to start pose with correct orientation - stopping")
            self.move(0, 0)
            self.returning_to_start = False
            self.completed = True  # Mark entire cycle done
            self.transition_to(self.SEARCHING)

    def handle_state_timeout(self):
        if rospy.Time.now() - self.state_start_time > rospy.Duration(self.state_timeout[self.state]):
            rospy.logwarn(f"State {self.state} timed out, transitioning to SEARCHING")
            self.transition_to(self.SEARCHING)  # Go back to searching

    def transition_to(self, new_state):
        rospy.loginfo(f"Transitioning from {self.state} to {new_state}")
        self.state = new_state
        self.state_start_time = rospy.Time.now()

    def stop_and_transition(self, next_state):
        self.move(0, 0)
        self.transition_to(next_state)

    def move(self, linear_speed, angular_speed):
        twist = Twist()
        twist.linear.x = linear_speed
        twist.angular.z = angular_speed
        self.cmd_vel_pub.publish(twist)

    def control_gripper(self, position):
        gripper_pos = UInt16()
        gripper_pos.data = position
        self.servo_pub.publish(gripper_pos)

if __name__ == '__main__':
    try:
        node = CombinedNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
