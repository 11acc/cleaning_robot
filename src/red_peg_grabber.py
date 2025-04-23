#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, Range
from geometry_msgs.msg import Twist
from std_msgs.msg import UInt16, Float64
from cv_bridge import CvBridge, CvBridgeError

class RedPegGrabber:
    def __init__(self):
        rospy.init_node('red_peg_grabber')
        self.bridge = CvBridge()
        
        # State definitions
        self.WAITING = 0
        self.GRABBING = 1
        self.MOVING_RIGHT = 2  # New state to move right
        self.RELEASING = 3
        
        self.state = self.WAITING
        self.rate = rospy.Rate(10)  # 10Hz
        
        # Add timestamp for logging
        self.last_state_change = rospy.Time.now()
        
        # Add detection tracking
        self.detection_count = 0
        self.centered_count = 0
        self.last_log_time = rospy.Time.now()
        
        # Camera subscriber
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        
        # Distance sensors for safety
        self.fl_sensor_sub = rospy.Subscriber('/range/fl', Range, self.fl_sensor_callback)
        self.fr_sensor_sub = rospy.Subscriber('/range/fr', Range, self.fr_sensor_callback)
        
        # Motion control publisher
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        # Gripper control publishers
        self.servo_pub = rospy.Publisher('/servo', UInt16, queue_size=10)
        self.servo_load_pub = rospy.Publisher('/servoLoad', Float64, queue_size=10)
        
        # Initialize variables
        self.current_image = None
        self.red_peg_detected = False
        self.red_peg_centered = False
        self.red_peg_center_x = 0
        self.red_peg_area = 0
        
        # Initialize distances
        self.fl_distance = float('inf')
        self.fr_distance = float('inf')
        self.front_distance = float('inf')
        
        # Movement parameters
        self.move_speed = 0.2  # Adjust as needed
        self.move_duration = 2.0  # Time to move right in seconds
        self.move_start_time = None
        
        # Thresholds
        self.min_area = 1000  # Minimum area to consider a valid red peg
        self.center_threshold = 50  # Pixels from center to consider centered
        
        rospy.loginfo("Simple Red Peg Grabber initialized")
    
    def fl_sensor_callback(self, data):
        self.fl_distance = data.range
        self.update_front_distance()
    
    def fr_sensor_callback(self, data):
        self.fr_distance = data.range
        self.update_front_distance()
    
    def update_front_distance(self):
        self.front_distance = min(self.fl_distance, self.fr_distance)
    
    def image_callback(self, data):
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.detect_red_peg(self.current_image)
        except CvBridgeError as e:
            rospy.logerr(e)
    
    def detect_red_peg(self, image):
        if image is None:
            rospy.logwarn_throttle(5.0, "Received empty image")
            return
            
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define range of red color in HSV
        # Red wraps around in HSV, so we need two masks
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        # Create masks for red color
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # Apply morphology operations to remove noise
        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Reset detection flags
        self.red_peg_detected = False
        self.red_peg_centered = False
        
        if contours and len(contours) > 0:
            # Find the largest contour (assuming it's the red peg)
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            # Only consider it if the area is large enough
            if area > self.min_area:
                self.red_peg_detected = True
                self.red_peg_area = area
                self.detection_count += 1
                
                # Get the center of the contour
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    self.red_peg_center_x = cx
                    
                    # Check if the peg is centered
                    img_center_x = image.shape[1] // 2
                    x_offset = abs(cx - img_center_x)
                    
                    if x_offset < self.center_threshold:
                        self.red_peg_centered = True
                        self.centered_count += 1
                    
                    # Draw contour and center for visualization
                    cv2.drawContours(image, [largest_contour], 0, (0, 255, 0), 2)
                    cv2.circle(image, (cx, cy), 5, (255, 0, 0), -1)
                    cv2.line(image, (img_center_x, 0), (img_center_x, image.shape[0]), (0, 0, 255), 1)
                    
                    # Add information overlay
                    cv2.putText(image, f"Area: {area}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(image, f"Centered: {self.red_peg_centered}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(image, f"X offset: {x_offset}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    if self.state == self.WAITING:
                        state_str = "WAITING"
                    elif self.state == self.GRABBING:
                        state_str = "GRABBING"
                    elif self.state == self.MOVING_RIGHT:
                        state_str = "MOVING_RIGHT"
                    else:
                        state_str = "RELEASING"
                    cv2.putText(image, f"State: {state_str}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Log detection statistics every second
                    if (rospy.Time.now() - self.last_log_time).to_sec() >= 1.0:
                        self.last_log_time = rospy.Time.now()
                        rospy.loginfo(f"Detection stats: Area={area}, X-offset={x_offset}, " +
                                      f"Detected={self.detection_count}, Centered={self.centered_count} in last second")
                        # Reset counters
                        self.detection_count = 0
                        self.centered_count = 0
            else:
                rospy.logwarn_throttle(3.0, f"Contour found but area {area} is below minimum threshold {self.min_area}")
        else:
            rospy.logwarn_throttle(3.0, "No contours found in the image")
        
        # Display the result
        cv2.imshow("Red Peg Detection", image)
        cv2.waitKey(1)
    
    def open_gripper(self):
        # Open the gripper - set servo to 0 degrees
        servo_angle = UInt16()
        servo_angle.data = 0  # 0 degrees to open
        self.servo_pub.publish(servo_angle)
        
        # Set servo load
        servo_load = Float64()
        servo_load.data = 0.5  # moderate load
        self.servo_load_pub.publish(servo_load)
        
        rospy.loginfo("Opening gripper")
        rospy.sleep(1.5)  # Wait for gripper to open
    
    def close_gripper(self):
        # Close the gripper - set servo to 170 degrees
        servo_angle = UInt16()
        servo_angle.data = 170  # 170 degrees to close
        self.servo_pub.publish(servo_angle)
        
        # Set servo load for a firm grip
        servo_load = Float64()
        servo_load.data = 0.8  # stronger load
        self.servo_load_pub.publish(servo_load)
        
        rospy.loginfo("Closing gripper")
        rospy.sleep(1.5)  # Wait for gripper to close
    
    def move_right(self):
        cmd = Twist()
        cmd.linear.x = self.move_speed  # Move right (or left, adjust sign if needed)
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)
    
    def stop_robot(self):
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)
    
    def run(self):
        # Start with gripper open
        self.open_gripper()
        rospy.loginfo("=== Red Peg Grabber Started ===")
        
        while not rospy.is_shutdown():
            # Log the current state duration
            state_duration = (rospy.Time.now() - self.last_state_change).to_sec()
            
            # State machine logic
            if self.state == self.WAITING:
                # Log if waiting too long
                if state_duration > 10.0:
                    rospy.logwarn_throttle(5.0, f"Still in WAITING state after {state_duration:.1f} seconds")
                    rospy.loginfo_throttle(5.0, f"Detection status: detected={self.red_peg_detected}, centered={self.red_peg_centered}")
                
                # Check if red peg is detected and centered
                if self.red_peg_detected and self.red_peg_centered:
                    rospy.loginfo(f"Red peg detected and centered! Area: {self.red_peg_area} Transitioning to GRABBING")
                    self.stop_robot()
                    self.state = self.GRABBING
                    self.last_state_change = rospy.Time.now()
            
            elif self.state == self.GRABBING:
                # Log if grabbing is taking too long
                if state_duration > 5.0:
                    rospy.logwarn(f"GRABBING state has been active for {state_duration:.1f} seconds!")
                
                # Close the gripper to grab the peg
                self.close_gripper()
                
                # Transition to MOVING_RIGHT state
                rospy.loginfo("Peg grabbed! Transitioning to MOVING_RIGHT state")
                self.state = self.MOVING_RIGHT
                self.last_state_change = rospy.Time.now()
                self.move_start_time = rospy.Time.now()  # Record when movement started
            
            elif self.state == self.MOVING_RIGHT:
                # Check if we've moved right long enough
                current_time = rospy.Time.now()
                time_moved = (current_time - self.move_start_time).to_sec()
                
                # Move right
                self.move_right()
                
                if time_moved >= self.move_duration:
                    rospy.loginfo(f"Move time complete ({time_moved:.1f} seconds). Transitioning to RELEASING")
                    self.stop_robot()
                    self.state = self.RELEASING
                    self.last_state_change = rospy.Time.now()
            
            elif self.state == self.RELEASING:
                # Log if releasing is taking too long
                if state_duration > 3.0:
                    rospy.logwarn(f"RELEASING state has been active for {state_duration:.1f} seconds!")
                
                # Open the gripper to release the peg
                self.open_gripper()
                
                # Go back to waiting
                rospy.loginfo("Peg released. Transitioning back to WAITING")
                self.state = self.WAITING
                self.last_state_change = rospy.Time.now()
            
            self.rate.sleep()

if __name__ == '__main__':
    try:
        grabber = RedPegGrabber()
        grabber.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Unexpected error: {str(e)}")
        import traceback
        rospy.logerr(traceback.format_exc())
    finally:
        rospy.loginfo("Red Peg Grabber node is shutting down.")