#!/usr/bin/env python3
import rospy
import cv2
import enum
import numpy as np
from sensor_msgs.msg import Image, Range
from geometry_msgs.msg import Twist
from std_msgs.msg import UInt16, Float64
from cv_bridge import CvBridge
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

BLUE_COLOR_LOWER_THRESHOLD = np.array([100, 0, 0])
BLUE_COLOR_UPPER_THRESHOLD = np.array([255, 10, 10])

YELLOW_COLOR_LOWER_THRESHOLD = np.array([0, 200, 200])
YELLOW_COLOR_UPPER_THRESHOLD = np.array([10, 255, 255])

LINE_FOLLOW_SPEED = 0.1        # m/s
TURN_SPEED = 0.1               # rad/s
GRIPPER_CLOSE_POSITION = 170   # Gripper closed position
GRIPPER_OPEN_POSITION = 0    # Gripper open position

enum State {
    FOLLOW_LINE,
    APPROACH_OBJECT,
    GRAB_OBJECT,
    YELLOW_ZONE,
    DISCARD_ZONE
}

enum ObjectType {
    RED,
    GREEN,
    BLUE,
    UNKNOWN
}

# a class for a complete bot that can follow a black line and grab red objects if they are on the way.
class CompleteBot:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('complete_bot', anonymous=True)

        self.bridge = CvBridge()
        self.twist = Twist()

        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        # Gripper
        self.servo_pub = rospy.Publisher('/servo', UInt16, queue_size=10)
        self.servo_load_pub = rospy.Publisher('/servoLoad', Float64, queue_size=10)
        # Distance sensors for safety
        self.fl_sensor_sub = rospy.Subscriber('/range/fl', Range, self.fl_sensor_callback)
        self.fr_sensor_sub = rospy.Subscriber('/range/fr', Range, self.fr_sensor_callback)
        
        self.gripper_position = 170
        self.rate = rospy.Rate(10)

        self.top_roi = (0.55, 0.70)       # 55 % – 70 % of rows
        self.bot_roi = (0.88, 1.00)       # 88 % – 100 %

        self.collision_risk = False

        self.state = State.FOLLOW_LINE

    def image_callback(self, msg):
        if self.state == State.FOLLOW_LINE:
            self.follow_line(msg)
        elif self.state == State.APPROACH_OBJECT:
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

    def follow_line(self, msg):

    def approach_object(self, msg):


    def grab_object(self, msg):

        
        
        
        
        