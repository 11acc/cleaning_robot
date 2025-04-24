#!/usr/bin/env python3
"""
SCRIPT EXECUTION FLOW
=====================

Callback-Driven Execution:
    PegGrabberDiscarder class is instantiated
    ├── odom_callback() - Triggered by odometry data
    │   └── Updates current_pose with position and orientation
    │
    └── image_callback() - Main driver, triggered by camera frames
        ├── Converts image and detects red pegs
        └── Based on current state, calls one of these methods:
            │
            ├── If approaching: approach_target()
            │   └── Uses target_pose and visual feedback for navigation
            │      └── If close enough, transitions to grabbed_peg state
            │
            ├── If grabbed_peg: Initiates turning_to_discard state
            │   └── Transitions to execute_turn_to_discard()
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
    approaching → grabbed_peg → turning_to_discard → discarding → turning_back → completed

Helper Methods:
    - stop_robot() - Stops all movement
    - visualize() - Displays camera view and robot state
"""

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


class PegGrabberDiscarder:
    def __init__(self):
        rospy.init_node('discard_peg_like_yellow')
        self.bridge = CvBridge()
        self.twist = Twist()

        # Subscribers and publishers
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.servo_pub = rospy.Publisher('/servo', UInt16, queue_size=10)
        self.servo_load_pub = rospy.Publisher('/servoLoad', Float64, queue_size=10)

        # ---------- State Machine Flags ----------
        self.approaching = True          # Initial state: Using target-based approach to reach peg
        self.grabbed_peg = False         # State after successfully grabbing the peg
        self.turning_to_discard = False  # Turning 90° away from deploy zone
        self.discarding = False          # Opening gripper to discard the peg
        self.turning_back = False        # Turning back 90° to original orientation
        self.completed = False           # Task completed flag
        self.gripper_closed = False      # Flag for gripper state

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
        self.target_pose = None          # Target position for approach (x, y)
        self.current_pose = None         # Current robot position and orientation
        self.current_servo_load = 0.0    # Current load on gripper servo
        self.turn_start_time = None      # Time when turning began
        self.discard_start_time = None   # Time when discarding began

        # Initialization message
        rospy.loginfo("PegGrabberDiscarder node started...")


    # ---------- Callback Methods -----------------------------------------------------------------
    def odom_callback(self, msg):
        """
        Processes robot odometry data to track position and orientation
        Called automatically when new odometry data is published
        """
        position = msg.pose.pose.position
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        self.current_pose = (position.x, position.y, yaw)

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

            # Get centroid of the largest contour (if any)
            cX, cY = self.center_x, self.center_y  # Default to center
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])

                    # Estimate target distance using contour width
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    if w > 0:
                        # Set target distance based on contour size
                        distance = (self.real_peg_width_m * self.focal_length_px) / w

                        # Set the target pose if not already set
                        if self.target_pose is None and self.approaching:
                            # Set the target pose directly in front of robot
                            if self.current_pose is not None:
                                x_cur, y_cur, yaw_cur = self.current_pose
                                # Target is straight ahead at calculated distance
                                target_x = x_cur + distance * math.cos(yaw_cur)
                                target_y = y_cur + distance * math.sin(yaw_cur)
                                self.target_pose = (target_x, target_y)
                                rospy.loginfo(f"Target set to x={target_x:.2f}, y={target_y:.2f}")

            # ---------- State Machine Execution ----------
            # Execute actions based on current state
            if self.approaching and self.target_pose is not None and self.current_pose is not None:
                self.approach_target(cX)
            elif self.grabbed_peg:
                if not self.turning_to_discard:
                    # Initiate turn
                    self.turning_to_discard = True
                    self.turn_start_time = rospy.Time.now().to_sec()
                    rospy.loginfo("Beginning turn to discard")
                self.execute_turn_to_discard()
            elif self.turning_to_discard:
                self.execute_turn_to_discard()
            elif self.discarding:
                self.execute_discard()
            elif self.turning_back:
                self.execute_turn_back()

            # Visualization
            if cv_image is not None:
                self.visualize(cv_image, mask, cX, cY)

        except Exception as e:
            rospy.logerr(f"Error in image_callback: {e}")
            self.stop_robot()


    # ---------- State Machine Methods ------------------------------------------------------------
    def approach_target(self, cX):
        """
        Approaches the target using a combination of position-based navigation and visual feedback
        Similar to the original yellow follower approach method
        """
        # Get current pose information
        x_cur, y_cur, yaw_cur = self.current_pose
        target_x, target_y = self.target_pose

        # Calculate distance and angle to target
        dx = target_x - x_cur
        dy = target_y - y_cur
        dist_to_target = math.sqrt(dx * dx + dy * dy)
        angle_to_target = math.atan2(dy, dx)
        angle_diff = angle_to_target - yaw_cur

        # Normalize angle to range [-pi, pi]
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

        # Calculate angle error to center the peg in image
        err_x = float(cX - self.center_x) / self.center_x
        angle_adjust = -err_x * 0.3  # Proportional control coefficient

        # Check if we've reached the target
        if dist_to_target <= self.stop_distance_m + 0.05:
            rospy.loginfo(f"Reached target zone at distance {dist_to_target:.2f}m")
            self.stop_robot()
            self.approaching = False
            self.close_gripper()
            self.grabbed_peg = True
        else:
            # Adaptive speed control based on distance to target
            max_speed = 0.15
            min_speed = 0.02
            slow_down_radius = 0.3  # Begin slowing down within this radius

            # Calculate speed - slower as we get closer to target
            if dist_to_target < slow_down_radius:
                speed = min_speed + (max_speed - min_speed) * (dist_to_target / slow_down_radius)
                speed = max(speed, min_speed)  # Ensure minimum speed
            else:
                speed = max_speed

            # Motion control logic
            if abs(angle_diff) > 0.05:
                # If angle difference is significant, turn in place first
                self.twist.linear.x = 0
                self.twist.angular.z = 0.4 if angle_diff > 0 else -0.4
            else:
                # Otherwise move forward with visual feedback for centering
                self.twist.angular.z = angle_adjust
                self.twist.linear.x = speed

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
    def visualize(self, cv_image, mask, cX, cY):
        """
        Creates visualization windows showing the camera view and mask
        Displays current state and gripper status on the image
        """
        # Determine status text based on current state
        status = "Unknown"
        if self.approaching:
            status = "Approaching Peg"
        elif self.grabbed_peg:
            status = "Grabbed Peg"
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

        # Display centroid of detected peg
        cv2.circle(cv_image, (cX, cY), 5, (255, 255, 255), -1)

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
