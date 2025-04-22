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
        self.upper_black     = np.array([180, 255, 70])   # Black upper bound
        #self.min_contour_area = rospy.get_param('~min_contour_area', 100)   # Min area to consider
        self.min_contour_area = 50

        self.roi_height_frac  = rospy.get_param('~roi_height_fraction', 0.25)  # bottom 25%
        self.roi_width_frac   = rospy.get_param('~roi_width_fraction', 0.50)   # center 50%
        self.linear_vel       = rospy.get_param('~linear_velocity', 0.1)
        self.kp               = rospy.get_param('~kp', 0.01)                   # P gain
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
        self.is_turning = False
        self.is_following_line = False
        self.last_line_detection_time = time()
        self.max_time_without_line_detection = 1.5
        self.robot_enabled = False    # Start with robot disabled
        
        # Start keyboard listener thread
        self.keyboard_thread = threading.Thread(target=self.keyboard_listener)
        self.keyboard_thread.daemon = True
        self.keyboard_thread.start()
        
        rospy.loginfo("Press 'r' to toggle robot movement. Currently DISABLED.")
        rospy.loginfo("Press 'q' to quit the program.")

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

    # ─────────────── movement control functions ───────────────
    def calculate_movement(self, error, center_x, image_center):
        """Calculate movement parameters based on line position"""
        angular_z = -float(error) * self.kp
        angular_z = max(self.min_angular_vel, min(self.max_angular_vel, angular_z))

        direction = "centered"
        if error > 20:
            direction = "tilting right — correcting left"
        elif error < -20:
            direction = "tilting left — correcting right"

        rospy.loginfo("[DEBUG] True Line Center: %d, Image Center: %d, Error: %.2f, Turning: %s, Angular z: %.2f",
                     center_x, image_center, error, direction, angular_z)
        
        # Only update movement commands if robot is enabled
        if self.robot_enabled:
            self.twist.linear.x = self.linear_vel
            self.twist.angular.z = angular_z
        else:
            self.twist.linear.x = 0.0
            self.twist.angular.z = 0.0
            
        self.is_turning = False
        self.is_following_line = True

    def stop_robot(self):
        """Stop the robot completely"""
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.0
        self.cmd_vel_pub.publish(self.twist)
        self.is_turning = True
        self.is_following_line = False
        rospy.loginfo("[DEBUG] Robot stopped — line lost")

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

        # ••• convert to HSV and create mask for black line •••
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_black, self.upper_black)
        
        # ••• clean up noise with morphological operations •••
        kernel = np.ones((5, 5), np.uint8)

        # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # ••• find contours in the mask •••
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

                    # Calculate error and movement
                    error = cx - roi_center
                    self.calculate_movement(error, cx, roi_center)
                    self.last_line_detection_time = time()
        else:
            # Stop if line lost for too long
            if self.is_following_line and (time() - self.last_line_detection_time > self.max_time_without_line_detection):
                self.stop_robot()

        # ••• add movement status indicator on the image •••
        status_text = "MOVEMENT: ON" if self.robot_enabled else "MOVEMENT: OFF"
        cv2.putText(roi, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                    (0, 255, 0) if self.robot_enabled else (0, 0, 255), 2)

        self.cmd_vel_pub.publish(self.twist)

        # ••• display processed images •••
        cv2.imshow("Line Tracking", roi)
        cv2.imshow("Mask", mask)
        cv2.imshow("Raw Mask", mask.copy())
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