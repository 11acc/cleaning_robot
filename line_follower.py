#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from geometry_msgs.msg import Twist
from time import time
import sys
import termios
import tty
import select
import threading

class LineFollower:
    def __init__(self):
        rospy.init_node('line_follower')

        # ──────────────── parameters you can tune on the fly ────────────────
        # HSV values optimized for black line detection
        self.lower_black     = np.array([0, 0, 0])        # Black lower bound
        self.upper_black     = np.array([180, 255, 100])  # Black upper bound
        self.min_contour_area = 20  # Reduced to detect thinner lines
        
        # Increase field of view by using larger portion of image
        self.roi_height_frac  = 0.40  # bottom 40% 
        self.roi_width_frac   = 0.80  # center 80%
        
        # Speed and control parameters
        self.linear_vel       = 0.08  # Slower default speed for better control
        self.kp               = 0.012  # P gain
        
        # Turn control parameters
        self.turn_delay       = 0.5   # Seconds to delay turning after detecting horizontal line
        self.turn_threshold   = 0.25  # Position threshold for turn (% of ROI height from bottom)
        # ────────────────────────────────────────────────────────────────────

        # ROS setup
        self.bridge    = CvBridge()
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', 
                                         Image, self.image_callback, queue_size=1)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
        self.twist    = Twist()
        
        # Movement parameters
        self.min_angular_vel = -0.3
        self.max_angular_vel = 0.3
        
        # State tracking
        self.state = "FOLLOWING"  # States: FOLLOWING, APPROACHING_TURN, TURNING, LOST
        self.last_line_detection_time = time()
        self.max_time_without_line_detection = 1.5
        self.robot_enabled = False    # Start with robot disabled
        self.state_change_time = time()
        self.horizontal_line_detected = False
        self.horizontal_line_first_seen = 0
        self.next_turn = None
        
        # Start keyboard listener thread
        self.keyboard_thread = threading.Thread(target=self.keyboard_listener)
        self.keyboard_thread.daemon = True
        self.keyboard_thread.start()
        
        rospy.loginfo("Press 'r' to toggle robot movement. Currently DISABLED.")
        rospy.loginfo("Press 'q' to quit the program.")
        rospy.loginfo("Line follower node started, waiting for images...")

    # ─────────────── keyboard control functions ───────────────
    def keyboard_listener(self):
        """Listen for keyboard commands in a separate thread"""
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while not rospy.is_shutdown():
                if select.select([sys.stdin], [], [], 0)[0]:
                    key = sys.stdin.read(1)
                    if key == 'r':
                        self.robot_enabled = not self.robot_enabled
                        rospy.loginfo("Robot movement %s", 
                                    "ENABLED" if self.robot_enabled else "DISABLED")
                        if not self.robot_enabled:
                            self.stop_robot()
                    elif key == 'q':
                        rospy.loginfo("Quitting...")
                        self.stop_robot()
                        rospy.signal_shutdown("User requested shutdown")
                rospy.sleep(0.1)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    # ─────────────── line detection helpers ───────────────
    def detect_line_orientation(self, contour, roi):
        """Determine if line is vertical, horizontal, or turning"""
        # Fit a line to the contour
        [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
        
        # Calculate the angle in degrees - convert numpy values to float
        angle = float(np.arctan2(vy[0], vx[0]) * 180 / np.pi)
        
        # Get bounding rect to check aspect ratio
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 999
        
        # Draw line direction for visualization
        lefty = int((-x * vy[0] / vx[0]) + y) if vx[0] != 0 else y
        righty = int(((roi.shape[1] - x) * vy[0] / vx[0]) + y) if vx[0] != 0 else y
        cv2.line(roi, (roi.shape[1]-1, righty), (0, lefty), (255, 0, 255), 2)
        
        # Display angle and aspect ratio for debugging
        cv2.putText(roi, f"Angle: {angle:.1f}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(roi, f"Ratio: {aspect_ratio:.1f}", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Check orientation based on angle and aspect ratio
        if aspect_ratio > 1.5:  # Wider than tall - likely horizontal
            orientation = "HORIZONTAL"
        elif aspect_ratio < 0.67:  # Taller than wide - likely vertical
            orientation = "VERTICAL"
        else:
            # For square-ish contours, check the angle
            if -30 <= angle <= 30 or 150 <= abs(angle) <= 180:
                orientation = "HORIZONTAL"
            elif 60 <= abs(angle) <= 120:
                orientation = "VERTICAL"
            else:
                orientation = "TURNING"
        
        return orientation, angle

    # ─────────────── movement control functions ───────────────
    def calculate_movement(self, contour, cx, cy, roi_center, roi):
        """Calculate movement parameters based on line analysis"""
        roi_height = roi.shape[0]
        
        # Detect line orientation
        orientation, angle = self.detect_line_orientation(contour, roi)
        
        # Calculate basic error (distance from center)
        error = cx - roi_center
        
        # Keep track of vertical position in ROI (0 = top, 1 = bottom)
        relative_y_pos = cy / roi_height
        
        # State machine for different line following behaviors
        if self.state == "FOLLOWING":
            # Normal line following behavior
            if orientation == "VERTICAL":
                # Standard P controller for vertical lines
                angular_z = -float(error) * self.kp
                cv2.putText(roi, "VERTICAL LINE", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Reset horizontal line detection
                self.horizontal_line_detected = False
                
            elif orientation == "HORIZONTAL":
                # We've found a horizontal line
                cv2.putText(roi, "HORIZONTAL LINE", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Record when we first see the horizontal line
                if not self.horizontal_line_detected:
                    self.horizontal_line_detected = True
                    self.horizontal_line_first_seen = time()
                    # Determine turn direction based on where the line extends
                    if cx < roi_center:
                        self.next_turn = "LEFT"
                    else:
                        self.next_turn = "RIGHT"
                
                # Continue normal following until we've seen the horizontal line long enough
                # and we're close enough to it
                if (time() - self.horizontal_line_first_seen > self.turn_delay and 
                    relative_y_pos < self.turn_threshold):
                    # Now transition to approaching turn
                    self.state = "APPROACHING_TURN"
                    self.state_change_time = time()
                    rospy.loginfo(f"Approaching {self.next_turn} turn")
                
                # For now just correct normally
                angular_z = -float(error) * self.kp
                
            elif orientation == "TURNING":
                # We're on a curve - adjust based on angle
                if angle > 0:
                    cv2.putText(roi, "TURNING RIGHT", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    # More aggressive correction for right turns
                    angular_z = self.min_angular_vel * 0.7
                else:
                    cv2.putText(roi, "TURNING LEFT", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    # More aggressive correction for left turns
                    angular_z = self.max_angular_vel * 0.7
            
            else:  # Default fallback
                angular_z = -float(error) * self.kp
        
        elif self.state == "APPROACHING_TURN":
            # We're getting close to the turn - slow down
            linear_speed = self.linear_vel * 0.6
            
            # Continue using P controller but be more responsive
            angular_z = -float(error) * (self.kp * 1.5)
            
            # Show state
            cv2.putText(roi, f"APPROACHING {self.next_turn}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Check if we're at the position to start turning
            # Make sure we've moved sufficiently forward after detecting the line
            if relative_y_pos < 0.15:  # Very close to the top of ROI
                self.state = "TURNING"
                self.state_change_time = time()
                rospy.loginfo(f"Executing {self.next_turn} turn")
            
            # Timeout to prevent getting stuck
            if time() - self.state_change_time > 2.0:
                self.state = "FOLLOWING"
                self.horizontal_line_detected = False
        
        elif self.state == "TURNING":
            # Execute the turn more decisively and don't try to follow the line
            # This avoids the robot trying to follow into the corner
            linear_speed = self.linear_vel * 0.5  # Even slower while turning
            
            if self.next_turn == "LEFT":
                angular_z = self.max_angular_vel  # Full left turn
                cv2.putText(roi, "TURNING LEFT", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                angular_z = self.min_angular_vel  # Full right turn
                cv2.putText(roi, "TURNING RIGHT", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Only check for vertical lines when we've been turning a bit
            # to avoid immediately going back to FOLLOWING
            if time() - self.state_change_time > 0.5:
                # Check if we've found a vertical line to transition back to following
                if orientation == "VERTICAL" and abs(error) < roi.shape[1] / 4:
                    self.state = "FOLLOWING"
                    self.horizontal_line_detected = False
                    rospy.loginfo("Turn complete, resuming line following")
            
            # Timeout to prevent getting stuck in turn
            if time() - self.state_change_time > 2.5:
                self.state = "FOLLOWING"
                self.horizontal_line_detected = False
        
        elif self.state == "LOST":
            # We've lost the line - execute recovery behavior
            if time() - self.state_change_time < 1.0:
                # First try backing up slightly
                linear_speed = -0.05
                angular_z = 0
                cv2.putText(roi, "LOST - BACKING UP", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                # Then try turning to find the line
                linear_speed = 0
                # Turn in the opposite direction of last known turn
                if self.next_turn == "LEFT":
                    angular_z = self.min_angular_vel * 0.7  # Search right
                else:
                    angular_z = self.max_angular_vel * 0.7  # Search left
                    
                cv2.putText(roi, "LOST - SEARCHING", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # If we detected the line again, go back to following
            if orientation in ["VERTICAL", "HORIZONTAL"]:
                self.state = "FOLLOWING"
                self.horizontal_line_detected = False
                rospy.loginfo("Line recovered, resuming following")
            
            # Timeout for full recovery attempt
            if time() - self.state_change_time > 5.0:
                self.stop_robot()
                cv2.putText(roi, "RECOVERY FAILED - STOPPED", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Clip angular velocity
        angular_z = max(self.min_angular_vel, min(self.max_angular_vel, angular_z))
        
        # Debug output
        rospy.loginfo("[DEBUG] Line Center: %d, Image Center: %d, Error: %.2f, State: %s, Orientation: %s",
                     cx, roi_center, error, self.state, orientation)
        
        # Only update movement commands if robot is enabled
        if self.robot_enabled:
            if self.state == "FOLLOWING":
                self.twist.linear.x = self.linear_vel
            elif self.state == "APPROACHING_TURN":
                self.twist.linear.x = self.linear_vel * 0.6  # Slow down approaching turn
            elif self.state == "TURNING":
                self.twist.linear.x = self.linear_vel * 0.5  # Even slower during turn
            elif self.state == "LOST":
                # Linear speed determined in LOST state logic
                self.twist.linear.x = linear_speed if 'linear_speed' in locals() else 0.0
            else:
                self.twist.linear.x = 0.0
                
            self.twist.angular.z = angular_z
        else:
            self.twist.linear.x = 0.0
            self.twist.angular.z = 0.0
        
        # Update tracking
        self.last_line_detection_time = time()

    def stop_robot(self):
        """Stop the robot completely"""
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.0
        self.cmd_vel_pub.publish(self.twist)
        rospy.loginfo("[DEBUG] Robot stopped")

    # ─────────────── image processing callback ───────────────
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        # ••• extract image dimensions •••
        height, width, _ = cv_image.shape

        # ••• region of interest - bottom portion •••
        crop_start = int(height * (1 - self.roi_height_frac))
        roi = cv_image[crop_start:, :]

        # ••• center horizontal crop •••
        roi_width = int(width * self.roi_width_frac)
        center_x_full = width // 2
        left_bound = center_x_full - roi_width // 2
        right_bound = center_x_full + roi_width // 2
        roi = roi[:, left_bound:right_bound]

        # ••• apply gaussian blur to reduce noise (enhanced from final_follower.py) •••
        # First convert to HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # Then apply blur (using same kernel size as final_follower.py)
        blurred = cv2.GaussianBlur(hsv, (5, 5), 0)

        # ••• create mask for black line (now using blurred HSV) •••
        mask = cv2.inRange(blurred, self.lower_black, self.upper_black)
        
        # ••• clean up noise with morphological operations •••
        kernel = np.ones((3, 3), np.uint8)  # Smaller kernel
        mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # ••• find contours in the mask •••
        contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # ••• process line contours if found •••
        if contours:
            # Filter out small contours
            valid_contours = [c for c in contours if cv2.contourArea(c) > self.min_contour_area]
            
            if valid_contours:
                # Find the largest contour (assumed to be the line)
                largest_contour = max(valid_contours, key=cv2.contourArea)
                
                # Calculate moments to find centroid
                M = cv2.moments(largest_contour)
                
                if M['m00'] > 0:  # Avoid division by zero
                    # Get centroid coordinates
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    
                    # Draw bounding box and center
                    cv2.drawContours(roi, [largest_contour], -1, (0, 255, 0), 2)
                    cv2.circle(roi, (cx, cy), 5, (0, 0, 255), -1)

                    # Draw the center of the cropped image
                    roi_center = roi.shape[1] // 2
                    cv2.line(roi, (roi_center, 0), (roi_center, roi.shape[0]), (255, 0, 0), 2)

                    # Calculate movement based on contour analysis
                    self.calculate_movement(largest_contour, cx, cy, roi_center, roi)
        else:
            # No valid contours found - handle line loss
            if time() - self.last_line_detection_time > self.max_time_without_line_detection:
                if self.state != "LOST" and self.robot_enabled:
                    self.state = "LOST"
                    self.state_change_time = time()
                    rospy.loginfo("Line lost - entering recovery mode")
                elif self.state == "LOST" and time() - self.state_change_time > 5.0:
                    self.stop_robot()
                    rospy.loginfo("Recovery failed - stopping robot")

        # ••• add status indicators on the image •••
        # Movement status
        status_text = "MOVEMENT: ON" if self.robot_enabled else "MOVEMENT: OFF"
        cv2.putText(roi, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                    (0, 255, 0) if self.robot_enabled else (0, 0, 255), 2)
        
        # State and turn direction
        turn_info = f" {self.next_turn}" if self.next_turn else ""
        cv2.putText(roi, f"STATE: {self.state}{turn_info}", (10, roi.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Horizontal line detection timer
        if self.horizontal_line_detected:
            time_since = time() - self.horizontal_line_first_seen
            cv2.putText(roi, f"H-Line: {time_since:.1f}s", (roi.shape[1] - 120, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Publish movement commands
        self.cmd_vel_pub.publish(self.twist)

        # ••• display processed images •••
        cv2.imshow("Line Tracking", roi)
        cv2.imshow("HSV Mask", mask)
        cv2.imshow("Cleaned Mask", mask_cleaned)
        cv2.waitKey(3)

# ─────────────── main program entry point ───────────────
if __name__ == '__main__':
    try:
        follower = LineFollower()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        # Clean up
        cv2.destroyAllWindows()