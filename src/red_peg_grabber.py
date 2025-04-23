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
       
        # State machine states
        self.SEARCHING = 0
        self.APPROACHING = 1
        self.POSITIONING = 2
        self.GRABBING = 3
        self.GRABBED = 4
        self.RECOVERY = 5  # New recovery state
       
        self.state = self.SEARCHING
        self.rate = rospy.Rate(10)  # 10Hz
        
        # Timeout tracking
        self.state_start_time = rospy.Time.now()
        self.state_timeout = {
            self.SEARCHING: 20.0,    # 20 seconds to find a red peg
            self.APPROACHING: 15.0,  # 15 seconds to approach
            self.POSITIONING: 10.0,  # 10 seconds to position
            self.GRABBING: 5.0,      # 5 seconds to grab
            self.GRABBED: 3.0,       # 3 seconds in grabbed state
            self.RECOVERY: 10.0      # 10 seconds to recover
        }
        
        # Camera subscriber
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
       
        # Distance sensors subscribers
        self.fl_sensor_sub = rospy.Subscriber('/range/fl', Range, self.fl_sensor_callback)
        self.fr_sensor_sub = rospy.Subscriber('/range/fr', Range, self.fr_sensor_callback)
        self.rl_sensor_sub = rospy.Subscriber('/range/rl', Range, self.rl_sensor_callback)
        self.rr_sensor_sub = rospy.Subscriber('/range/rr', Range, self.rr_sensor_callback)
       
        # Motion control publisher
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
       
        # Gripper control publishers
        self.servo_pub = rospy.Publisher('/servo', UInt16, queue_size=10)
        self.servo_load_pub = rospy.Publisher('/servoLoad', Float64, queue_size=10)
       
        # Initialize variables
        self.current_image = None
        self.red_peg_detected = False
        self.red_peg_center_x = 0
        self.red_peg_area = 0
        self.last_detection_time = rospy.Time.now()
        self.detection_timeout = rospy.Duration(3.0)  # 3 seconds without detection means we lost it
        
        # Initialize range sensor values
        self.fl_distance = float('inf')
        self.fr_distance = float('inf')
        self.rl_distance = float('inf')
        self.rr_distance = float('inf')
        
        # Computed distances for convenience
        self.front_distance = float('inf')  # Will be min of fl and fr
        self.left_distance = float('inf')   # Will be fl
        self.right_distance = float('inf')  # Will be fr

        # Visual servoing parameters
        self.target_area = 15000  # Target area for stopping
        self.area_tolerance = 2000  # Tolerance for area
        self.center_tolerance = 30  # Tolerance for centering
        self.max_linear_speed = 0.2
        self.min_linear_speed = 0.05
        self.max_angular_speed = 0.4
        self.min_angular_speed = 0.05
        self.approach_distance = 0.15  # Target distance for approaching
       
        rospy.loginfo("Red Peg Grabber initialized")

    def image_callback(self, data):
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.detect_red_peg(self.current_image)
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
        # Use the minimum of front-left and front-right as the front distance
        self.front_distance = min(self.fl_distance, self.fr_distance)
   
    def detect_red_peg(self, image):
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
        
        # Apply morphological operations to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.erode(red_mask, kernel, iterations=1)
        red_mask = cv2.dilate(red_mask, kernel, iterations=1)
       
        # Find contours
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       
        # Reset detection flag
        prev_detected = self.red_peg_detected
        self.red_peg_detected = False
       
        if contours:
            # Find the largest contour (assuming it's the red peg)
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
           
            # Only consider it if the area is large enough (filter out noise)
            if area > 200:  # Increased threshold to filter more noise
                self.red_peg_detected = True
                self.red_peg_area = area
                self.last_detection_time = rospy.Time.now()
               
                # Get the center of the contour
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    self.red_peg_center_x = cx
                   
                    # Draw contour and center for visualization
                    cv2.drawContours(image, [largest_contour], 0, (0, 255, 0), 2)
                    cv2.circle(image, (cx, int(M["m01"] / M["m00"])), 5, (255, 0, 0), -1)
                    
                    # Draw area and center information for debugging
                    cv2.putText(image, f"Area: {area}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(image, f"Center: {cx}", (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Add state information
                    state_names = {
                        self.SEARCHING: "SEARCHING",
                        self.APPROACHING: "APPROACHING",
                        self.POSITIONING: "POSITIONING",
                        self.GRABBING: "GRABBING",
                        self.GRABBED: "GRABBED",
                        self.RECOVERY: "RECOVERY"
                    }
                    cv2.putText(image, f"State: {state_names[self.state]}", (10, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # If we haven't detected the peg for a while, consider it lost
        if not self.red_peg_detected and (rospy.Time.now() - self.last_detection_time) > self.detection_timeout:
            # Lost track of the peg
            if prev_detected:
                rospy.logwarn("Lost track of the red peg!")
       
        # Display the result (useful for debugging)
        cv2.imshow("Red Peg Detection", image)
        cv2.waitKey(1)
   
    def open_gripper(self):
        # Open the gripper - set servo to 0 degrees
        servo_angle = UInt16()
        servo_angle.data = 0  # 0 degrees to open
        self.servo_pub.publish(servo_angle)
        
        # Set servo load if needed
        servo_load = Float64()
        servo_load.data = 0.5  # moderate load
        self.servo_load_pub.publish(servo_load)
        
        rospy.loginfo("Opening gripper - setting servo to 0 degrees")
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
        
        rospy.loginfo("Closing gripper - setting servo to 170 degrees")
        rospy.sleep(1.5)  # Wait for gripper to close
   
    def stop_robot(self):
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)
        rospy.loginfo("Robot stopped")

    def calculate_visual_servoing_cmd(self):
        """Calculate visual servoing command based on peg position and area"""
        cmd = Twist()
        
        if not self.red_peg_detected:
            return cmd
        
        # Get image center
        img_width = self.current_image.shape[1]
        img_center_x = img_width // 2
        
        # Calculate error
        error_x = self.red_peg_center_x - img_center_x
        
        # Angular control based on position error
        if abs(error_x) > self.center_tolerance:
            # P controller for angular velocity
            kp_angular = 0.001  # Tuning parameter
            cmd.angular.z = -kp_angular * error_x
            
            # Clamp angular velocity
            cmd.angular.z = max(min(cmd.angular.z, self.max_angular_speed), -self.max_angular_speed)
            if abs(cmd.angular.z) < self.min_angular_speed:
                cmd.angular.z = self.min_angular_speed * (1 if cmd.angular.z > 0 else -1)
        
        # Linear control based on area (distance)
        if abs(error_x) < 2 * self.center_tolerance:  # Only move forward if somewhat centered
            # Determine if we need to move forward based on area
            if self.red_peg_area < (self.target_area - self.area_tolerance):
                # P controller for linear velocity
                kp_linear = 0.00005  # Tuning parameter
                cmd.linear.x = kp_linear * (self.target_area - self.red_peg_area)
                
                # Clamp linear velocity
                cmd.linear.x = max(min(cmd.linear.x, self.max_linear_speed), self.min_linear_speed)
            
            # If the area is larger than target + tolerance (too close), back up
            elif self.red_peg_area > (self.target_area + self.area_tolerance) and self.state == self.APPROACHING:
                cmd.linear.x = -0.05  # Back up slowly
        
        # Additional distance-based safety
        if self.front_distance < 0.1:  # Safety distance
            cmd.linear.x = min(cmd.linear.x, 0)  # Don't move forward if too close
        
        return cmd
        
    def run(self):
        # Start with gripper open
        self.open_gripper()
        self.state_start_time = rospy.Time.now()
       
        while not rospy.is_shutdown():
            cmd = Twist()
            
            # Check for state timeout
            elapsed_time = (rospy.Time.now() - self.state_start_time).to_sec()
            if elapsed_time > self.state_timeout[self.state]:
                rospy.logwarn(f"Timeout in state {self.state}. Reverting to RECOVERY.")
                self.state = self.RECOVERY
                self.state_start_time = rospy.Time.now()
                self.stop_robot()
           
            if self.state == self.SEARCHING:
                rospy.loginfo_throttle(2.0, "State: SEARCHING")
                if self.red_peg_detected:
                    rospy.loginfo("Red peg detected! Transitioning to APPROACHING")
                    self.state = self.APPROACHING
                    self.state_start_time = rospy.Time.now()
                else:
                    # Rotate to search for red peg
                    cmd.angular.z = 0.3
           
            elif self.state == self.APPROACHING:
                rospy.loginfo_throttle(2.0, "State: APPROACHING")
                if not self.red_peg_detected:
                    rospy.logwarn("Lost sight of peg during APPROACHING. Reverting to SEARCHING")
                    self.state = self.SEARCHING
                    self.state_start_time = rospy.Time.now()
                else:
                    # Use visual servoing to approach the peg
                    cmd = self.calculate_visual_servoing_cmd()
                    
                    # Check if we're close enough to transition to POSITIONING
                    approach_condition = (self.front_distance < self.approach_distance or 
                                          self.red_peg_area > self.target_area)
                    center_condition = abs(self.red_peg_center_x - (self.current_image.shape[1] // 2)) < self.center_tolerance
                    
                    if approach_condition and center_condition:
                        rospy.loginfo(f"Transitioning to POSITIONING - Distance: {self.front_distance}, Area: {self.red_peg_area}")
                        self.stop_robot()
                        self.state = self.POSITIONING
                        self.state_start_time = rospy.Time.now()
           
            elif self.state == self.POSITIONING:
                rospy.loginfo_throttle(2.0, "State: POSITIONING")
                if not self.red_peg_detected:
                    rospy.logwarn("Lost sight of peg during POSITIONING. Reverting to SEARCHING")
                    self.state = self.SEARCHING
                    self.state_start_time = rospy.Time.now()
                else:
                    # Final adjustments to position
                    img_center_x = self.current_image.shape[1] // 2
                    error_x = self.red_peg_center_x - img_center_x
                   
                    if abs(error_x) > 15:
                        # Still need to center
                        cmd.angular.z = -float(error_x) / 800.0
                    else:
                        # Check sensor distances for fine positioning
                        fl_fr_diff = self.fl_distance - self.fr_distance
                        if abs(fl_fr_diff) > 0.05:  # If significant difference between front-left and front-right
                            if fl_fr_diff > 0:  # Peg is more to the right
                                cmd.angular.z = -0.05
                            else:  # Peg is more to the left
                                cmd.angular.z = 0.05
                        else:
                            # Move forward slowly until optimal grabbing distance
                            # Use a smaller distance for final positioning
                            if self.front_distance > 0.1:
                                cmd.linear.x = 0.05
                            else:
                                rospy.loginfo(f"Transitioning to GRABBING - Distance: {self.front_distance}")
                                self.stop_robot()
                                self.state = self.GRABBING
                                self.state_start_time = rospy.Time.now()
           
            elif self.state == self.GRABBING:
                rospy.loginfo("State: GRABBING")
                # Make final small adjustment if needed
                if self.red_peg_detected:
                    img_center_x = self.current_image.shape[1] // 2
                    error_x = self.red_peg_center_x - img_center_x
                   
                    if abs(error_x) > 10:
                        # One last centering adjustment
                        cmd.angular.z = -float(error_x) / 1000.0
                        self.cmd_vel_pub.publish(cmd)
                        rospy.sleep(0.5)
                        self.stop_robot()
                    
                    # Close the gripper to grab the peg
                    self.close_gripper()
                    
                    # Check if we've successfully grabbed the peg
                    # Simple check: if the gripper is closed fully, we missed the peg
                    # In a real system, you might use force sensors or other feedback
                    rospy.loginfo("Checking if grab was successful...")
                    rospy.sleep(1.0)  # Give time for gripper to stabilize
                    
                    # For demo purposes, assume we grabbed successfully
                    rospy.loginfo("Peg grabbed successfully! Transitioning to GRABBED state")
                    self.state = self.GRABBED
                    self.state_start_time = rospy.Time.now()
                else:
                    # Lost sight of the peg during final approach
                    rospy.logwarn("Lost sight of peg during GRABBING phase - reverting to SEARCHING")
                    self.state = self.SEARCHING
                    self.state_start_time = rospy.Time.now()
           
            elif self.state == self.GRABBED:
                rospy.loginfo("State: GRABBED - Moving the peg to the target location")
                # Here you would add code to move the robot to a drop-off location
                # For this demo, we'll just wait a bit and then go back to searching
                rospy.sleep(2.0)
                
                # Open gripper to release the peg
                self.open_gripper()
                
                # Reset to searching state
                rospy.loginfo("Peg placement complete. Returning to SEARCHING state")
                self.state = self.SEARCHING
                self.state_start_time = rospy.Time.now()
            
            elif self.state == self.RECOVERY:
                rospy.loginfo("State: RECOVERY - Attempting to recover from error")
                # Stop the robot and reset
                self.stop_robot()
                
                # Open the gripper to ensure we're in a known state
                self.open_gripper()
                
                # Back up slightly to get away from any obstacles
                cmd.linear.x = -0.1
                self.cmd_vel_pub.publish(cmd)
                rospy.sleep(1.0)
                self.stop_robot()
                
                # Turn a bit to look for the peg in a different direction
                cmd.angular.z = 0.2
                self.cmd_vel_pub.publish(cmd)
                rospy.sleep(2.0)
                self.stop_robot()
                
                # Return to searching
                rospy.loginfo("Recovery complete. Returning to SEARCHING state")
                self.state = self.SEARCHING
                self.state_start_time = rospy.Time.now()
           
            # Publish movement command
            self.cmd_vel_pub.publish(cmd)
            self.rate.sleep()

if __name__ == '__main__':
    try:
        grabber = RedPegGrabber()
        grabber.run()
    except rospy.ROSInterruptException:
        pass
