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
            
            # Get the center of the image
            height, width, _ = self.current_image.shape
            center_x = width // 2
            center_y = height // 2
            
            # Create a copy of the image for display
            display_image = self.current_image.copy()
            
            # Draw a dot at the center
            cv2.circle(display_image, (center_x, center_y), 5, (0, 0, 255), -1)
            
            # Get the HSV color at the center point
            hsv_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2HSV)
            center_hsv = hsv_image[center_y, center_x]
            
            # Log the HSV color at the center
            rospy.loginfo(f"Center HSV color: {center_hsv}")
            
            # Get depth at the center point
            depth = None
            try:
                if self.latest_depth_image is not None:
                    depth_image = self.bridge.imgmsg_to_cv2(self.latest_depth_image, desired_encoding='passthrough')
                    depth = depth_image[center_y, center_x]
                    rospy.loginfo(f"Center depth: {depth} meters")
            except Exception as e:
                rospy.logwarn(f"Could not get depth information: {str(e)}")
            
            # Add text showing HSV and depth values
            cv2.putText(display_image, f"HSV: {center_hsv}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if depth is not None:
                cv2.putText(display_image, f"Depth: {depth:.3f}m", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display_image, "Depth: N/A", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display the image
            cv2.imshow("Center Point", display_image)
            

        except CvBridgeError as e:
            rospy.logerr(e)
            
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
        
