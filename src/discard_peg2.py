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
        ├── Converts image and detects pegs (red, blue, or green)
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

Special Behavior:
    - Red/Blue pegs: Turn AWAY from deploy zone when discarding
    - Green pegs: Turn TOWARD deploy zone when discarding

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
        self.detected_peg_color = None   # Track which color peg was detected and grabbed

        # ---------- Parameter Configuration ----------
        # Vision parameters
        self.min_contour_area = 500      # Minimum size for detecting peg in pixels
        self.image_width = 640           # Full image width
        self.image_height = 480          # Full image height
        self.center_x = 320              # Center of the camera image (horizontal)
        self.center_y = 120              # Center of the cropped image (vertical - adjusted for cropping)
        self.fov_horizontal = 60         # Camera's horizontal field of view in degrees
        self.focal_length_px = 554       # Camera focal length in pixels - used for distance calculation

        # Physical parameters
        self.real_peg_width_m = 0.2      # Actual width of the peg in meters
        self.stop_distance_m = 0.89      # Distance to stop from the peg when grabbing

        # Side of the track where the deploy zone is located
        self.deploy_zone_position = 'left'
        
        # HSV color thresholds for detecting red pegs
        self.lower_red1 = np.array([0, 70, 50])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 70, 50])
        self.upper_red2 = np.array([180, 255, 255])

        # HSV color thresholds for detecting dark blue pegs
        self.lower_blue = np.array([90, 50, 50])
        self.upper_blue = np.array([130, 255, 155])

        # HSV color thresholds for detecting green pegs
        self.lower_green = np.array([35, 45, 40])
        self.upper_green = np.array([85, 255, 155])

        # Movement parameters
        self.turn_speed = 0.65             # Angular velocity for turning (rad/s)
        self.turn_duration = 2.4           # Time to complete a 90-degree turn (seconds)
        
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
        Only processes the bottom 50% of the image to reduce noise
        """
        try:
            # Skip processing if task is completed
            if self.completed:
                self.stop_robot()
                return

            # Convert ROS image to OpenCV format
            full_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Get image dimensions
            height, width = full_image.shape[:2]
            
            # Crop to bottom 50% of the image
            crop_height = height // 2
            cv_image = full_image[crop_height:height, 0:width]
            
            # Process the cropped image
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

            # Create mask for red color (handles HSV color space wraparound for red)
            mask_red1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
            mask_red2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)

            # Create mask for blue color
            mask_blue = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
            
            # Create mask for green color
            mask_green = cv2.inRange(hsv, self.lower_green, self.upper_green)

            # Combine masks for overall detection
            combined_mask = cv2.bitwise_or(cv2.bitwise_or(mask_red, mask_blue), mask_green)

            # Clean up the combined mask with erosion and dilation
            processed_mask = cv2.erode(combined_mask, None, iterations=2)
            processed_mask = cv2.dilate(processed_mask, None, iterations=2)

            # Find contours in the mask (potential peg objects)
            contours, _ = cv2.findContours(processed_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Get centroid of the largest contour (if any)
            cX, cY = self.center_x, self.center_y  # Default to center
            if contours and self.approaching:  # Only process colors during approach phase
                largest_contour = max(contours, key=cv2.contourArea)
                contour_area = cv2.contourArea(largest_contour)
                
                # Only process if contour is large enough
                if contour_area > self.min_contour_area:
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        
                        # Create mask of just this contour to determine color
                        contour_mask = np.zeros_like(processed_mask)
                        cv2.drawContours(contour_mask, [largest_contour], -1, 255, -1)
                        
                        # Check which color is most prevalent in the contour
                        red_pixels = cv2.bitwise_and(mask_red, contour_mask)
                        blue_pixels = cv2.bitwise_and(mask_blue, contour_mask)
                        green_pixels = cv2.bitwise_and(mask_green, contour_mask)
                        
                        red_count = cv2.countNonZero(red_pixels)
                        blue_count = cv2.countNonZero(blue_pixels)
                        green_count = cv2.countNonZero(green_pixels)
                        
                        # Determine color based on highest pixel count
                        if red_count > blue_count and red_count > green_count:
                            self.detected_peg_color = "red"
                            rospy.loginfo_throttle(1, "Red peg detected")
                        elif blue_count > red_count and blue_count > green_count:
                            self.detected_peg_color = "blue"
                            rospy.loginfo_throttle(1, "Blue peg detected")
                        elif green_count > red_count and green_count > blue_count:
                            self.detected_peg_color = "green"
                            rospy.loginfo_throttle(1, "Green peg detected")

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
                                    rospy.loginfo(f"Target set to x={target_x:.2f}, y={target_y:.2f}, color={self.detected_peg_color}")

            # ---------- State Machine Execution ----------
            # Execute actions based on current state
            if self.approaching and self.target_pose is not None and self.current_pose is not None:
                self.approach_target(cX)
            # FIX: Added this condition to ensure we don't re-enter grabbed_peg state when already in discarding process
            elif self.grabbed_peg and not self.turning_to_discard and not self.discarding and not self.turning_back:
                # Initiate turn
                self.turning_to_discard = True
                self.turn_start_time = rospy.Time.now().to_sec()
                rospy.loginfo(f"Beginning turn to discard {self.detected_peg_color} peg")
                self.execute_turn_to_discard()
            elif self.turning_to_discard:
                self.execute_turn_to_discard()
            elif self.discarding:
                self.execute_discard()
            elif self.turning_back:
                self.execute_turn_back()

            # Visualization
            if cv_image is not None:
                self.visualize(cv_image, processed_mask, cX, cY)

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

        print(f"//DEBUG//  Distance: {dist_to_target}")

        # Check if we've reached the target
        if dist_to_target <= self.stop_distance_m:
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
        Turns the robot 90 degrees - direction depends on peg color
        - For red/blue pegs: Turn AWAY from deploy zone
        - For green pegs: Turn TOWARD deploy zone
        Uses timing to determine when turn is complete
        """
        # Calculate elapsed time since turn started
        current_time = rospy.Time.now().to_sec()
        if current_time - self.turn_start_time < self.turn_duration:
            # Determine turn direction based on peg color and deploy zone position
            if self.detected_peg_color == "green":
                # For green pegs: Turn TOWARD deploy zone (opposite of normal behavior)
                turn_direction = -1.0 if self.deploy_zone_position == 'left' else 1.0
            else:
                # For red/blue pegs: Turn AWAY from deploy zone (normal behavior)
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
        Opens the gripper to release the peg, backs up slightly to clear the peg,
        then transitions to turning back state
        """
        current_time = rospy.Time.now().to_sec()

        # If we haven't defined our phase tracking attributes, initialize them
        if not hasattr(self, 'discard_phase'):
            self.discard_phase = 1
            rospy.loginfo("Starting discard sequence")

        # Phase 1: Wait before opening gripper (robot stabilization)
        if self.discard_phase == 1:
            if current_time - self.discard_start_time > 1.0:  # 1 second stabilization delay
                self.open_gripper()
                self.discard_phase = 2
                self.gripper_opened_time = rospy.Time.now().to_sec()
                rospy.loginfo("Opening gripper to release peg")

        # Phase 2: Wait for gripper to fully open
        elif self.discard_phase == 2:
            if current_time - self.gripper_opened_time > 1.5:  # 1.5 second delay for gripper to open
                self.discard_phase = 3
                self.backup_start_time = rospy.Time.now().to_sec()
                rospy.loginfo("Moving backward to clear peg")

        # Phase 3: Back up slightly to clear the peg
        elif self.discard_phase == 3:
            # Apply reverse movement for a short duration
            backup_duration = 2.0  # seconds to back up
            backup_speed = -0.1    # negative value for backwards movement

            if current_time - self.backup_start_time < backup_duration:
                # Apply backward movement
                self.twist.linear.x = backup_speed
                self.twist.angular.z = 0.0
                self.cmd_vel_pub.publish(self.twist)
            else:
                # Stop after backing up
                self.stop_robot()
                self.discard_phase = 4
                rospy.loginfo("Backup complete, preparing to turn")

        # Phase 4: Prepare for turning back
        elif self.discard_phase == 4:
            # Clean up our phase tracking attributes
            delattr(self, 'discard_phase')
            if hasattr(self, 'gripper_opened_time'):
                delattr(self, 'gripper_opened_time')
            if hasattr(self, 'backup_start_time'):
                delattr(self, 'backup_start_time')

            # Transition to turning back state
            self.discarding = False
            self.turning_back = True
            self.turn_start_time = rospy.Time.now().to_sec()
            rospy.loginfo("Peg fully released, turning back")

    def execute_turn_back(self):
        """
        Turns the robot back to its original orientation with improved parameters
        to ensure a complete turn. Direction depends on which way we initially turned.
        """
        # Calculate elapsed time since turn started
        current_time = rospy.Time.now().to_sec()

        # Increase turn duration by 25% to ensure full rotation back
        # This compensates for potential mechanical variances or slippage
        extended_turn_duration = self.turn_duration * 1.25

        if current_time - self.turn_start_time < extended_turn_duration:
            # Turn in opposite direction based on peg color and deploy zone
            if self.detected_peg_color == "green":
                # For green pegs: Turn opposite of the towards-deploy-zone direction
                turn_direction = 1.0 if self.deploy_zone_position == 'left' else -1.0
            else:
                # For red/blue pegs: Turn opposite of the away-from-deploy-zone direction
                turn_direction = -1.0 if self.deploy_zone_position == 'left' else 1.0

            # Use the same turn speed for consistency
            self.twist.linear.x = 0.0
            self.twist.angular.z = self.turn_speed * turn_direction
            self.cmd_vel_pub.publish(self.twist)
        else:
            # Turn completed, stop and mark task as completed
            self.stop_robot()
            self.turning_back = False
            self.completed = True
            rospy.loginfo(f"Turn back completed, {self.detected_peg_color} peg task finished")


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
        Shows both the cropped view (processing) and the full image (context)
        """
        # Create a blank full-size image to show the context of where we're looking
        full_context = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        # Place the cropped image in the bottom half
        crop_height = self.image_height // 2
        full_context[crop_height:, :] = cv_image
        # Draw a line showing the crop boundary
        cv2.line(full_context, (0, crop_height), (self.image_width, crop_height), (0, 255, 255), 2)
        
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

        # Draw status text on images
        cv2.putText(cv_image, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(full_context, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw gripper state on images
        gripper_status = "Gripper: Closed" if self.gripper_closed else "Gripper: Open"
        cv2.putText(cv_image, gripper_status, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(full_context, gripper_status, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw detected peg color
        if self.detected_peg_color:
            color_text = f"Peg Color: {self.detected_peg_color}"
            cv2.putText(cv_image, color_text, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(full_context, color_text, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display "CROPPED VIEW - PROCESSING AREA" text 
        cv2.putText(full_context, "CROPPED VIEW - PROCESSING AREA", (10, crop_height + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Display centroid of detected peg (in the cropped image coordinates)
        cv2.circle(cv_image, (cX, cY), 5, (255, 255, 255), -1)
        # Also display in full context (need to adjust Y coordinate)
        if cY >= 0 and cX >= 0:  # Only if valid centroid
            cv2.circle(full_context, (cX, cY + crop_height), 5, (255, 255, 255), -1)

        # Display all the views
        #cv2.imshow("Camera View (Full Context)", full_context)
        cv2.imshow("Processing View (Cropped)", cv_image)
        #cv2.imshow("Processed Mask", mask)
        cv2.waitKey(3)


if __name__ == '__main__':
    try:
        PegGrabberDiscarder()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()