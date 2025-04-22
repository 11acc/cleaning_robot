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
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.twist = Twist()

        self.min_angular_vel = -0.3
        self.max_angular_vel = 0.3
        self.turning_vel_90 = 0.5
        self.is_turning = False
        self.is_following_line = False
        self.last_line_detection_time = time()
        self.max_time_without_line_detection = 1.5
        
        # Add movement control flag
        self.robot_enabled = False
        
        # Start keyboard listener thread
        self.keyboard_thread = threading.Thread(target=self.keyboard_listener)
        self.keyboard_thread.daemon = True
        self.keyboard_thread.start()
        
        print("Press 'r' to toggle robot movement. Currently DISABLED.")
        print("Press 'q' to quit the program.")
        print("Line detection debug info will show regardless of robot movement status.")

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
                        if self.robot_enabled:
                            print("Robot movement ENABLED")
                        else:
                            print("Robot movement DISABLED")
                            # Stop the robot when disabling
                            self.stop_robot()
                    elif key == 'q':
                        print("Quitting...")
                        self.stop_robot()
                        rospy.signal_shutdown("User requested shutdown")
                rospy.sleep(0.1)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def calculate_movement(self, error, center_x, image_center):
        """Calculate movement but don't apply it unless enabled"""
        angular_z = -float(error) / 100
        angular_z = max(self.min_angular_vel, min(self.max_angular_vel, angular_z))

        direction = "centered"
        if error > 20:
            direction = "tilting right — correcting left"
        elif error < -20:
            direction = "tilting left — correcting right"

        print(f"[DEBUG] True Line Center: {center_x}, Image Center: {image_center}, Error: {error:.2f}, Turning: {direction}, Angular z: {angular_z:.2f}")
        
        # Only update movement commands if robot is enabled
        if self.robot_enabled:
            self.twist.linear.x = 0.1
            self.twist.angular.z = angular_z
        else:
            self.twist.linear.x = 0.0
            self.twist.angular.z = 0.0
            
        self.is_turning = False
        self.is_following_line = True

    def stop_robot(self):
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.0
        self.cmd_vel_pub.publish(self.twist)
        self.is_turning = True
        self.is_following_line = False
        print("[DEBUG] Robot stopped — line lost")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)
            return

        height, width, _ = cv_image.shape

        # Crop bottom quarter
        crop_start = int(height * 0.75)
        cropped_image = cv_image[crop_start:, :]

        # Center horizontal crop
        half_width = width // 4
        center_x_full = width // 2
        left_bound = center_x_full - half_width
        right_bound = center_x_full + half_width
        cropped_image = cropped_image[:, left_bound:right_bound]

        # Convert to HSV and mask black
        hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 50])
        mask = cv2.inRange(hsv, lower_black, upper_black)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest contour (assumed to be the line)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            line_center_x = x + w // 2
            line_center_y = y + h // 2

            # Draw bounding box and center
            cv2.rectangle(cropped_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(cropped_image, (line_center_x, line_center_y), 5, (0, 0, 255), -1)

            # Draw the center of the cropped image
            cropped_center = cropped_image.shape[1] // 2
            cv2.line(cropped_image, (cropped_center, 0), (cropped_center, cropped_image.shape[0]), (255, 0, 0), 2)

            # Calculate error and movement - always calculate and print, but only move if enabled
            error = line_center_x - cropped_center
            self.calculate_movement(error, line_center_x, cropped_center)
            self.last_line_detection_time = time()
        else:
            if self.is_following_line and (time() - self.last_line_detection_time > self.max_time_without_line_detection):
                self.stop_robot()

        # Add movement status indicator on the image
        status_text = "MOVEMENT: ON" if self.robot_enabled else "MOVEMENT: OFF"
        cv2.putText(cropped_image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                    (0, 255, 0) if self.robot_enabled else (0, 0, 255), 2)

        self.cmd_vel_pub.publish(self.twist)

        # Show windows
        cv2.imshow("Line Tracking", cropped_image)
        cv2.imshow("Mask", mask)
        cv2.waitKey(3)

if __name__ == '__main__':
    try:
        follower = LineFollower()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        # Restore terminal settings
        cv2.destroyAllWindows()