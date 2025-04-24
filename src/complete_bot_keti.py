#!/usr/bin/env python3
import rospy
import cv2
import enum
import numpy as np
from sensor_msgs.msg import Image, Range
from geometry_msgs.msg import Twist
from std_msgs.msg import UInt16, Float64
from cv_bridge import CvBridge, CvBridgeError
import datetime
import os

BLACK_COLOR_LOWER_THRESHOLD = np.array([0, 0, 0])
BLACK_COLOR_UPPER_THRESHOLD = np.array([180, 255, 60])

RED_COLOR_LOWER_THRESHOLD1 = np.array([0, 0, 100])
RED_COLOR_UPPER_THRESHOLD1	 = np.array([10, 255, 255])
RED_COLOR_LOWER_THRESHOLD2 = np.array([160, 100, 100])
RED_COLOR_UPPER_THRESHOLD2 = np.array([180, 255, 255])

GREEN_COLOR_LOWER_THRESHOLD = np.array([0, 100, 100])
GREEN_COLOR_UPPER_THRESHOLD = np.array([10, 255, 255])

BLUE_COLOR_LOWER_THRESHOLD = np.array([180, 50, 50])
BLUE_COLOR_UPPER_THRESHOLD = np.array([250, 255, 255])

YELLOW_COLOR_LOWER_THRESHOLD = np.array([15, 60, 100])
YELLOW_COLOR_UPPER_THRESHOLD = np.array([40, 255, 255])

LINE_FOLLOW_SPEED = 0.1        # m/s
TURN_SPEED = 0.1               # rad/s
GRIPPER_CLOSE_POSITION = 170   # Gripper closed position
GRIPPER_OPEN_POSITION = 0    # Gripper open position

class State(enum.Enum) :
    FOLLOW_LINE = 1,
    APPROACH_OBJECT = 2,
    GRAB_OBJECT = 3,
    YELLOW_ZONE = 4,
    DISCARD_ZONE = 5


class ObjectType(enum.Enum) :
    RED = 1,
    GREEN = 2,
    BLUE = 3,
    UNKNOWN = 4


color_to_zone = {
    ObjectType.RED: "yellow zone",
    ObjectType.GREEN: "discard zone",
    ObjectType.BLUE: "discard zone",
}

# a class for a complete bot that can follow a black line and grab red objects if they are on the way.
class CompleteBot:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('complete_bot', anonymous=True)

        self.bridge = CvBridge()
        self.speed = Twist()

        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        self.current_image = None
        # Gripper
        self.servo_pub = rospy.Publisher('/servo', UInt16, queue_size=10)
        self.servo_load_pub = rospy.Publisher('/servoLoad', Float64, queue_size=10)
        # Distance sensors for safety
        self.fl_sensor_sub = rospy.Subscriber('/range/fl', Range, self.fl_sensor_callback)
        self.fr_sensor_sub = rospy.Subscriber('/range/fr', Range, self.fr_sensor_callback)

        # depth sensor
        self.depth_sub = rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_callback, queue_size=1)
        self.latest_depth_image = None
        
        self.gripper_position = 170
        self.rate = rospy.Rate(10)

        self.top_roi = (0.55, 0.70)       # 55 % – 70 % of rows
        self.bot_roi = (0.88, 1.00)       # 88 % – 100 %

        self.collision_risk = False

        self.state = State.APPROACH_OBJECT
        self.object_type = ObjectType.UNKNOWN

    def image_callback(self, msg):
        rospy.loginfo(self.state)
        if self.state == State.FOLLOW_LINE:
            self.follow_line(msg)
        elif self.state == State.APPROACH_OBJECT:
            rospy.loginfo("Approaching")
            self.approach_object(msg)
        elif self.state == State.GRAB_OBJECT:
            self.grab_object(msg)
        elif self.state == State.YELLOW_ZONE:
            self.yellow_zone(msg)
        elif self.state == State.DISCARD_ZONE:
            self.discard_zone(msg)
    
    def fl_sensor_callback(self, msg):
        # if we are nbot approaching a peg and it's an obstacle
        if self.state != State.APPROACH_OBJECT:
            # if range is less than 0.15m, stop the robot and turn it left
            if msg.range < 0.15:
                rospy.loginfo("Collision risk detected. Turning.")
                self.collision_risk = True
                self.speed.linear.x = 0
                self.speed.angular.z = TURN_SPEED
                # publish velocity
                self.cmd_vel_pub.publish(self.speed)
                rospy.sleep(0.5)
                self.speed.angular.z = 0
                self.cmd_vel_pub.publish(self.speed)
            else:
                self.collision_risk = False


    def fr_sensor_callback(self, msg):
        if self.state != State.APPROACH_OBJECT:
            # if range is less than 0.15m, stop the robot and turn it left
            if msg.range < 0.15:
                rospy.loginfo("Collision risk detected. Turning.")
                self.collision_risk = True
                self.speed.linear.x = 0
                self.speed.angular.z = TURN_SPEED
                # publish velocity
                self.cmd_vel_pub.publish(self.speed)
                rospy.sleep(0.5)
                self.speed.angular.z = 0
                self.cmd_vel_pub.publish(self.speed)
            else:
                self.collision_risk = False
    
    def depth_callback(self, msg):
        self.latest_depth_image = msg

# ============================================================
# ======================= STATE FUNCTIONS =====================
# ============================================================

    # finding the line and following it goes here. 
    # It should have a state change if object is detected on the way while following the line.
    def follow_line(self, msg):
        # if object detected:
        self.state = State.APPROACH_OBJECT
    # approach the object goes here.
    def approach_object(self, msg):
        try:
            rospy.loginfo("Approaching object")
            self.current_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Create a copy of the original image for display
            display_image = self.current_image.copy()
            
            # Crop the bottom 50% of the image for better detection
            height, width, _ = self.current_image.shape
            crop_start_y = int(height * 0.5)  # Take only the bottom 50%
            cropped_image = self.current_image[crop_start_y:, :]
            
            # Draw a line showing the crop boundary
            cv2.line(display_image, (0, crop_start_y), (width, crop_start_y), (255, 0, 0), 2)
            
            # Convert cropped image to HSV color space for better color detection
            hsv_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
            
            # detect whether object in front is red, green or blue by counting how 
            # many pixels in the image are within each threshold
            red_pixels1 = cv2.inRange(hsv_image, RED_COLOR_LOWER_THRESHOLD1, RED_COLOR_UPPER_THRESHOLD1)
            red_pixels2 = cv2.inRange(hsv_image, RED_COLOR_LOWER_THRESHOLD2, RED_COLOR_UPPER_THRESHOLD2)
            green_pixels = cv2.inRange(hsv_image, GREEN_COLOR_LOWER_THRESHOLD, GREEN_COLOR_UPPER_THRESHOLD)
            blue_pixels = cv2.inRange(hsv_image, BLUE_COLOR_LOWER_THRESHOLD, BLUE_COLOR_UPPER_THRESHOLD)

            # compare the number of pixels between the three and see which one is more major
            if (np.sum(red_pixels1)+np.sum(red_pixels2)) > np.sum(green_pixels) and (np.sum(red_pixels1)+np.sum(red_pixels2)) > np.sum(blue_pixels):
                self.object_type = ObjectType.RED
                # Create combined red mask
                color_mask = cv2.bitwise_or(red_pixels1, red_pixels2)
            elif np.sum(green_pixels) > (np.sum(red_pixels1)+np.sum(red_pixels2)) and np.sum(green_pixels) > np.sum(blue_pixels):
                self.object_type = ObjectType.GREEN
                color_mask = green_pixels
            elif np.sum(blue_pixels) > (np.sum(red_pixels1)+np.sum(red_pixels2)) and np.sum(blue_pixels) > np.sum(green_pixels):
                self.object_type = ObjectType.BLUE
                color_mask = blue_pixels
            else:
                self.object_type = ObjectType.UNKNOWN
                color_mask = None
            
            rospy.loginfo("Object type: %s", self.object_type)

            # Apply morphology operations to clean up the mask
            if color_mask is not None:
                kernel = np.ones((5, 5), np.uint8)
                color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
                color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
                
                # Display the mask
                cv2.imshow("Color Mask", color_mask)
                
                # Find contours for the detected color
                contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # If contours found, proceed with object approach
                if contours and len(contours) > 0:
                    # Find the largest contour
                    largest_contour = max(contours, key=cv2.contourArea)
                    contour_area = cv2.contourArea(largest_contour)
                    
                    # Only proceed if the contour is large enough (to avoid noise)
                    if contour_area > 100:  # Adjust this threshold as needed
                        # Get the bounding box for the largest contour
                        x, y, w, h = cv2.boundingRect(largest_contour)
                        
                        # Adjust y coordinate to account for cropping
                        y_display = y + crop_start_y
                        
                        # Draw rectangle around the detected object
                        cv2.rectangle(display_image, (x, y_display), (x + w, y_display + h), (0, 255, 0), 2)
                        
                        # Calculate center of the contour
                        center_x = x + w//2
                        center_y = y + h//2
                        
                        # Adjust center_y for display
                        center_y_display = center_y + crop_start_y
                        
                        # Draw center point
                        cv2.circle(display_image, (center_x, center_y_display), 5, (0, 0, 255), -1)
                        
                        # Add text labels
                        cv2.putText(display_image, f"Object: {self.object_type}", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(display_image, f"Size: {w}x{h}", (10, 60), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Centered text showing pixel count
                        total_pixels = np.sum(color_mask > 0)
                        cv2.putText(display_image, f"Pixels: {total_pixels}", (10, 120), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Center of image crosshair
                        img_center_x = width // 2
                        img_center_y = height // 2
                        cv2.line(display_image, (img_center_x - 20, img_center_y), (img_center_x + 20, img_center_y), (255, 0, 0), 2)
                        cv2.line(display_image, (img_center_x, img_center_y - 20), (img_center_x, img_center_y + 20), (255, 0, 0), 2)
                        
                        # Calculate error from center 
                        error_x = center_x - img_center_x
                        error_y = center_y_display - img_center_y
                        cv2.putText(display_image, f"Error X: {error_x}", (10, 150), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Calculate proportional control
                        turn_p = 0.003  # Adjust this value based on testing
                        turn_speed = -turn_p * error_x
                        
                        # Show turn speed
                        cv2.putText(display_image, f"Turn: {turn_speed:.2f}", (10, 180), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # grab depth (distance) of the object
                        depth = None
                        try:
                            if self.latest_depth_image is not None:
                                depth_image = self.bridge.imgmsg_to_cv2(self.latest_depth_image, desired_encoding='passthrough')
                                # Use the center of the detected object for depth
                                depth = depth_image[center_y_display, center_x]
                                rospy.loginfo("Depth: %f", depth)
                                cv2.putText(display_image, f"Depth: {depth:.2f}m", (10, 90), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        except Exception as e:
                            rospy.logwarn("Could not get depth information: %s", str(e))
                        
                        # Control logic based on object detection and depth
                        if depth is not None and 0 < depth < 0.5:
                            # Object is close enough to grab
                            rospy.loginfo("Object in grabbing range. Grabbing...")
                            
                            # Stop the robot
                            self.speed.linear.x = 0.0
                            self.speed.angular.z = 0.0
                            self.cmd_vel_pub.publish(self.speed)
                            
                            # Close the gripper (value of 130)
                            self.servo_pub.publish(UInt16(130))
                            
                            # Wait for gripper to close
                            rospy.sleep(1.0)
                            
                            # Transition to the next state based on object type
                            if self.object_type == ObjectType.RED:
                                self.state = State.YELLOW_ZONE
                            else:
                                self.state = State.DISCARD_ZONE
                        else:
                            # Object is still far, approach it
                            # Calculate speed based on how far off-center the object is
                            if abs(error_x) > 50:  # Object is significantly off-center
                                # Turn more, move slower
                                self.speed.angular.z = turn_speed
                                self.speed.linear.x = 0.05  # Slow forward speed
                            else:
                                # Object is centered, move faster
                                self.speed.angular.z = turn_speed * 0.2  # Gentler turning
                                self.speed.linear.x = 0.15  # Faster forward speed
                                
                            self.cmd_vel_pub.publish(self.speed)
                    else:
                        # Contour too small, probably noise
                        self.speed.angular.z = 0.0
                        self.speed.linear.x = 0.0
                        self.cmd_vel_pub.publish(self.speed)
                else:
                    # No contours found
                    cv2.putText(display_image, "No contours found", (10, 210), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    self.speed.angular.z = 0.0
                    self.speed.linear.x = 0.0
                    self.cmd_vel_pub.publish(self.speed)
            else:
                # No color mask
                cv2.putText(display_image, "No color detected", (10, 210), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                self.speed.angular.z = 0.0
                self.speed.linear.x = 0.0
                self.cmd_vel_pub.publish(self.speed)
            
            # Display the camera view with annotations
            cv2.imshow("Robot Vision", display_image)
            cv2.waitKey(1)  # Wait 1ms to update display

        except CvBridgeError as e:
            rospy.logerr(e)
            # Safety - stop the robot
            self.speed.angular.z = 0.0
            self.speed.linear.x = 0.0
            self.cmd_vel_pub.publish(self.speed)
    # grab the object goes here.
    def grab_object(self, msg):
        pass

    def yellow_zone(self, msg):
        pass

    def discard_zone(self, msg):
        pass

if __name__ == "__main__":
    try:
        CompleteBot()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
        