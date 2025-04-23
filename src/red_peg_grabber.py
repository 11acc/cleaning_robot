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
       
        self.state = self.SEARCHING
        self.rate = rospy.Rate(10)  # 10Hz
        
        # Timeout tracking
        self.state_start_time = rospy.Time.now()
        self.state_timeout = {
            self.SEARCHING: 20.0,    # 20 seconds to find a red peg
            self.APPROACHING: 15.0,  # 15 seconds to approach
            self.POSITIONING: 10.0,  # 10 seconds to position
            self.GRABBING: 5.0,      # 5 seconds to grab
            self.GRABBED: 3.0        # 3 seconds in grabbed state
        }
       
        # Camera subscriber
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
       
        # Distance sensors subscribers based on your rostopics
        self.fl_sensor_sub = rospy.Subscriber('/range/fl', Range, self.fl_sensor_callback)
        self.fr_sensor_sub = rospy.Subscriber('/range/fr', Range, self.fr_sensor_callback)
        self.rl_sensor_sub = rospy.Subscriber('/range/rl', Range, self.rl_sensor_callback)
        self.rr_sensor_sub = rospy.Subscriber('/range/rr', Range, self.rr_sensor_callback)
       
        # Motion control publisher
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
       
        # Gripper control publishers - updated with correct message type
        self.servo_pub = rospy.Publisher('/servo', UInt16, queue_size=10)
        self.servo_load_pub = rospy.Publisher('/servoLoad', Float64, queue_size=10)
       
        # Initialize variables
        self.current_image = None
        self.red_peg_detected = False
        self.red_peg_center_x = 0
        self.red_peg_area = 0
        
        # Initialize range sensor values
        self.fl_distance = float('inf')
        self.fr_distance = float('inf')
        self.rl_distance = float('inf')
        self.rr_distance = float('inf')
        
        # Computed distances for convenience
        self.front_distance = float('inf')  # Will be min of fl and fr
        self.left_distance = float('inf')   # Will be fl
        self.right_distance = float('inf')  # Will be fr
       
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
       
        # Find contours
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       
        # Reset detection flag
        self.red_peg_detected = False
       
        if contours:
            # Find the largest contour (assuming it's the red peg)
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
           
            # Only consider it if the area is large enough (filter out noise)
            if area > 100:
                self.red_peg_detected = True
                self.red_peg_area = area
               
                # Get the center of the contour
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    self.red_peg_center_x = cx
                   
                    # Draw contour and center for visualization
                    cv2.drawContours(image, [largest_contour], 0, (0, 255, 0), 2)
                    cv2.circle(image, (cx, int(M["m01"] / M["m00"])), 5, (255, 0, 0), -1)
       
        # Display the result (optional, useful for debugging)
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
   
    def run(self):
        # Start with gripper open
        self.open_gripper()
       
        while not rospy.is_shutdown():
            cmd = Twist()
           
            if self.state == self.SEARCHING:
                rospy.loginfo("State: SEARCHING")
                if self.red_peg_detected:
                    self.state = self.APPROACHING
                else:
                    # Rotate to search for red peg
                    cmd.angular.z = 0.3
           
            elif self.state == self.APPROACHING:
                rospy.loginfo("State: APPROACHING")
                if not self.red_peg_detected:
                    self.state = self.SEARCHING
                else:
                    # Center the red peg in the camera view
                    img_center_x = self.current_image.shape[1] // 2
                    error_x = self.red_peg_center_x - img_center_x
                   
                    # Adjust orientation to center the peg
                    cmd.angular.z = -float(error_x) / 500.0
                   
                    # If peg is roughly centered, move forward
                    if abs(error_x) < 30:
                        cmd.linear.x = 0.1
                   
                    # If close enough to the peg, proceed to positioning
                    # Using front distance (min of fl and fr) OR a large enough area of red
                    if self.front_distance < 0.2 or self.red_peg_area > 10000:
                        rospy.loginfo(f"Transitioning to POSITIONING - Distance: {self.front_distance}, Area: {self.red_peg_area}")
                        self.stop_robot()
                        self.state = self.POSITIONING
           
            elif self.state == self.POSITIONING:
                rospy.loginfo("State: POSITIONING")
                if not self.red_peg_detected:
                    self.state = self.SEARCHING
                else:
                    # Fine positioning - make sure peg is centered
                    img_center_x = self.current_image.shape[1] // 2
                    error_x = self.red_peg_center_x - img_center_x
                   
                    if abs(error_x) > 15:
                        # Still need to center
                        cmd.angular.z = -float(error_x) / 800.0
                    else:
                        # Check if the peg is more to the left or right to align properly
                        fl_fr_diff = self.fl_distance - self.fr_distance
                        if abs(fl_fr_diff) > 0.05:  # If significant difference between front-left and front-right
                            if fl_fr_diff > 0:  # Peg is more to the right
                                cmd.angular.z = -0.05
                            else:  # Peg is more to the left
                                cmd.angular.z = 0.05
                        else:
                            # Move forward slowly until optimal grabbing distance
                            if self.front_distance > 0.1:
                                cmd.linear.x = 0.05
                            else:
                                rospy.loginfo(f"Transitioning to GRABBING - Distance: {self.front_distance}")
                                self.stop_robot()
                                self.state = self.GRABBING
           
            elif self.state == self.GRABBING:
                rospy.loginfo("State: GRABBING")
                # Make final small adjustment if needed
                if self.red_peg_detected:
                    img_center_x = self.current_image.shape[1] // 2
                    error_x = self.red_peg_center_x - img_center_x
                   
                    if abs(error_x) > 10:
                        cmd.angular.z = -float(error_x) / 1000.0
                    else:
                        # Close the gripper to grab the peg
                        self.close_gripper()
                        rospy.loginfo("Peg grabbed successfully! Transitioning to GRABBED state")
                        self.state = self.GRABBED
                else:
                    # Lost sight of the peg during final approach
                    rospy.logwarn("Lost sight of peg during GRABBING phase - reverting to SEARCHING")
                    self.state = self.SEARCHING
           
            elif self.state == self.GRABBED:
                rospy.loginfo("State: GRABBED - Peg has been successfully grabbed!")
                # Mission accomplished - you could add what to do next here
                # For example, move to a specific location to drop the peg
                rospy.sleep(2.0)
                # Reset to searching state after success (or define a new state)
                self.state = self.SEARCHING
           
            # Publish movement command
            self.cmd_vel_pub.publish(cmd)
            self.rate.sleep()

if __name__ == '__main__':
    try:
        grabber = RedPegGrabber()
        grabber.run()
    except rospy.ROSInterruptException:
        pass
