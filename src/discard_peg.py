#!/usr/bin/env python3
"""
SCRIPT EXECUTION FLOW
=====================

Callback-Driven Execution:
    PegGrabberDiscarder class is instantiated
    └── image_callback() - Main driver, triggered by camera frames
        ├── Converts image and detects red pegs
        └── Based on current state, calls one of these methods:
            │
            ├── If approaching_peg: approach_peg()
            │   └── Calculates distance and angle to peg
            │      └── If close enough, transitions to grabbing_peg state
            │
            ├── If grabbing_peg: close_gripper()
            │   └── Transitions to turning_to_discard state
            │
            ├── If turning_to_discard: execute_turn_to_discard()
            │   └── When turn complete, transitions to discarding state
            │
            ├── If discarding: execute_discard()
            │   └── Opens gripper and transitions to turning_back state
            │
            └── If turning_back: execute_turn_back()
                └── When turn complete, transitions to completed state

State Transitions:
    approaching_peg → grabbing_peg → turning_to_discard → discarding → turning_back → completed

Helper Methods:
    - stop_robot() - Stops all movement
    - visualize() - Displays camera view and robot state
"""

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
from tf.transformations import euler_from_quaternion
import math
from std_msgs.msg import UInt16, Float64


class PegGrabberDiscarder:
    def __init__(self):
        rospy.init_node('discard_peg')
        self.bridge = CvBridge()
        self.twist = Twist()

        # Set up ROS subscribers and publishers
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.servo_pub = rospy.Publisher('/servo', UInt16, queue_size=10)
        self.servo_load_pub = rospy.Publisher('/servoLoad', Float64, queue_size=10)

        # ---------- State Machine Flags ----------
        self.approaching_peg = True      # Initial state: Moving toward the detected peg
        self.grabbing_peg = False        # Closing the gripper to grab the peg
        self.turning_to_discard = False  # Turning 90° away from the deploy zone
        self.discarding = False          # Opening the gripper to drop the peg
        self.turning_back = False        # Turning back 90° to original orientation
        self.completed = False           # Task finished flag
        self.gripper_closed = False      # Flag to know gripper state

        # ---------- Parameter Configuration ----------
        # Vision parameters
        self.min_contour_area = 500      # Minimum size for detecting peg in pixels
        self.center_x = 320              # Center of the camera image (horizontal)
        self.center_y = 240              # Center of the camera image (vertical)
        self.fov_horizontal = 60         # Camera's horizontal field of view in degrees
        self.focal_length_px = 554       # Camera focal length in pixels - used for distance calculation

        # Physical parameters
        self.real_peg_width_m = 0.2      # Actual width of the peg in meters
        self.stop_distance_m = 0.03      # Distance to stop from the peg when grabbing

        # Side of the track where the deploy zone is located
        self.deploy_zone_position = 'left'

        # HSV color thresholds for detecting red pegs // to be changed probably received from global script
        self.lower_red1 = np.array([0, 70, 50])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 70, 50])
        self.upper_red2 = np.array([180, 255, 255])

        # Movement parameters
        self.turn_speed = 0.5            # Angular velocity for turning (rad/s)
        self.turn_duration = 2.5         # Time to complete a 90-degree turn (seconds)

        # ---------- State Tracking Variables ----------
        self.current_pose = None         # Current robot position and orientation
        self.current_servo_load = 0.0    # Current gripper servo load
        self.turn_start_time = None      # Time when turning began
        self.discard_start_time = None   # Time when discarding began

        # Initialization message
        rospy.loginfo("PegGrabberDiscarder node started...")


    # ---------- Callback Methods -----------------------------------------------------------------
    def servo_load_callback(self, msg):
        """
        Tracks the gripper servo load to monitor gripping force
        Called automatically when new servo load data is published
        """
        self.current_servo_load = msg.data


    # ---------- Gripper Helpers ------------------------------------------------------------------
    def close_gripper(self):
        """
        Commands the gripper to close (grab the peg)
        """
        self.servo_pub.publish(170)
        self.gripper_closed = True
        rospy.loginfo("Gripper closed")

    def open_gripper(self):
        """
        Commands the gripper to open (release the peg).
        """
        self.servo_pub.publish(0)
        self.gripper_closed = False
        rospy.loginfo("Gripper opened")


    # ---------- Main Method ----------------------------------------------------------------------
    def image_callback(self, msg):
        """
        Analyses camera images to detect pegs and drives the robot's state machine
        Called automatically when new camera frames arrive
        """
        try:
            # Skip processing if task is completed
            if self.completed:
                self.stop_robot()
                return

            # Convert ROS image to OpenCV format and process
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

            # Create mask for red color (handles HSV color space wraparound for red)
            mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
            mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)

            # Clean up the mask with erosion and dilation
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)

            # Find contours in the mask (potential peg objects)
            contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Execute the appropriate action based on current state
            if self.approaching_peg:
                # Find and move toward the peg
                self.approach_peg(contours, cv_image)
            elif self.grabbing_peg:
                # Close gripper and transition to turning state
                self.close_gripper()
                self.grabbing_peg = False
                self.turning_to_discard = True
                self.turn_start_time = rospy.Time.now().to_sec()
                rospy.loginfo("Peg grabbed, beginning turn to discard")
            elif self.turning_to_discard:
                # Execute the turn away from deploy zone
                self.execute_turn_to_discard()
            elif self.discarding:
                # Open gripper to release peg
                self.execute_discard()
            elif self.turning_back:
                # Turn back to original orientation
                self.execute_turn_back()

            # Display visualization if available
            if cv_image is not None:
                self.visualize(cv_image, mask)

        except Exception as e:
            rospy.logerr(f"Error in image_callback: {e}")
            self.stop_robot()


    # ---------- State Methods --------------------------------------------------------------------
    def approach_peg(self, contours, cv_image):
        """
        Detects and approaches the peg, centering it in the camera view
        Transitions to grabbing state when close enough
        """
        # If no contours (peg) detected, do a small rotation to search
        if not contours:
            self.twist.linear.x = 0.0
            self.twist.angular.z = 0.1
            self.cmd_vel_pub.publish(self.twist)
            return

        # Get the largest contour (assuming it's the peg)
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)

        # Just in case its 0 when it shouldn't be
        if M["m00"] == 0:
            return

        # Calculate centroid of the peg
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # Calculate contour width to estimate distance
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Calculate distance to peg
        # distance = (actual_width * focal_length) / perceived_width
        if w > 0:
            distance = (self.real_peg_width_m * self.focal_length_px) / w
        else:
            distance = float('inf')

        # Calculate how far peg is from center of image (-1 to 1 scale)
        err_x = float(cX - self.center_x) / self.center_x
        angle_adjust = -err_x * 0.3  # Proportional control coefficient

        # If we're close enough to the peg, stop and grab it
        if distance <= self.stop_distance_m + 0.05:
            rospy.loginfo(f"Reached peg (distance: {distance:.2f}m)")
            self.stop_robot()
            self.approaching_peg = False
            self.grabbing_peg = True
        else:
            # Move toward the peg with adaptive speed based on distance
            max_speed = 0.15
            min_speed = 0.05
            slow_down_radius = 0.3  # Distance at which to start slowing down

            # Calculate speed based on distance (slow down as we get closer)
            if distance < slow_down_radius:
                speed = min_speed + (max_speed - min_speed) * (distance / slow_down_radius)
                speed = max(speed, min_speed)  # Ensure minimum speed
            else:
                speed = max_speed

            # Movement control logic
            if abs(err_x) > 0.2:  # If peg is significantly off-center
                # Turn in place to center the peg
                self.twist.linear.x = 0.0
                self.twist.angular.z = angle_adjust * 2.0  # Stronger turning response
            else:
                # Move forward while making small steering adjustments
                self.twist.linear.x = speed
                self.twist.angular.z = angle_adjust  # Gentle steering correction

            # Send movement command
            self.cmd_vel_pub.publish(self.twist)

    def execute_turn_to_discard(self):
        """
        Turns the robot 90 degrees away from the deploy zone
        Uses timing to determine when turn is complete
        """
        # Calculate elapsed time since turn started
        current_time = rospy.Time.now().to_sec()
        if current_time - self.turn_start_time < self.turn_duration:
            # Turn direction depends on deploy zone position
            # Positive = counterclockwise, Negative = clockwise
            turn_direction = 1.0 if self.deploy_zone_position == 'left' else -1.0
            self.twist.linear.x = 0.0
            self.twist.angular.z = self.turn_speed * turn_direction
            self.cmd_vel_pub.publish(self.twist)
        else:
            # Turn completed, stop and transition to discard state
            self.stop_robot()
            self.turning_to_discard = False
            self.discarding = True
            self.discard_start_time = rospy.Time.now().to_sec()
            rospy.loginfo("Turn completed, now discarding peg")

    def execute_discard(self):
        """
        Waits briefly, then opens the gripper to release the peg
        Transitions to turning back state after discard
        """
        # Wait a short time before opening gripper
        current_time = rospy.Time.now().to_sec()
        if current_time - self.discard_start_time > 1.0:  # 1 second delay
            self.open_gripper()
            self.discarding = False
            self.turning_back = True
            self.turn_start_time = rospy.Time.now().to_sec()
            rospy.loginfo("Peg discarded, turning back")

    def execute_turn_back(self):
        """
        Turns the robot back 90 degrees to its original orientation
        Completes the task when finished
        """
        # Calculate elapsed time since turn started
        current_time = rospy.Time.now().to_sec()
        if current_time - self.turn_start_time < self.turn_duration:
            # Turn in opposite direction of the first turn
            turn_direction = -1.0 if self.deploy_zone_position == 'left' else 1.0
            self.twist.linear.x = 0.0
            self.twist.angular.z = self.turn_speed * turn_direction
            self.cmd_vel_pub.publish(self.twist)
        else:
            # Turn completed, stop and mark task as completed
            self.stop_robot()
            self.turning_back = False
            self.completed = True
            rospy.loginfo("Turn back completed, task finished")


    # ---------- Helper Methods -------------------------------------------------------------------
    def stop_robot(self):
        """
        Stops all robot movement
        """
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.0
        self.cmd_vel_pub.publish(self.twist)

    def visualize(self, cv_image, mask):
        """
        Creates visualization windows showing the camera view and mask
        Displays current state and gripper status on the image
        """
        # Determine status text based on current state
        status = "Unknown"
        if self.approaching_peg:
            status = "Approaching Peg"
        elif self.grabbing_peg:
            status = "Grabbing Peg"
        elif self.turning_to_discard:
            status = "Turning to Discard"
        elif self.discarding:
            status = "Discarding Peg"
        elif self.turning_back:
            status = "Turning Back"
        elif self.completed:
            status = "Task Completed"

        # Draw status text on image
        cv2.putText(cv_image, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw gripper state on image
        gripper_status = "Gripper: Closed" if self.gripper_closed else "Gripper: Open"
        cv2.putText(cv_image, gripper_status, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display the camera view and the processed mask
        cv2.imshow("Camera View", cv_image)
        cv2.imshow("Processed Mask", mask)
        cv2.waitKey(3)



if __name__ == '__main__':
    try:
        PegGrabberDiscarder()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
