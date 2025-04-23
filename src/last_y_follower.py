#!/usr/bin/env python3

import rospy
import cv2
import cv_bridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import numpy as np

class Follower:
  def __init__(self):
    self.node_name = "rosbot_line_follower"
    rospy.init_node(self.node_name)
    self.bridge = cv_bridge.CvBridge()

    self.image_sub = rospy.Subscriber(
       '/camera/color/image_raw', 
      Image, self.image_callback)
    self.cmd_vel_pub = rospy.Publisher(
       '/cmd_vel',
       Twist,
       queue_size=1)
    self.speed = Twist()
    self.last_seen_angular_speed = 0
    self.rate = rospy.Rate(10)

  def image_callback(self, msg):
    image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define black color range in HSV
    lower_black = np.array([0, 0, 0], dtype=np.uint8)
    upper_black = np.array([180, 255, 50], dtype=np.uint8)
    mask = cv2.inRange(hsv_image, lower_black, upper_black)
    
    # Apply the mask to the original image
    res = cv2.bitwise_and(image, image, mask=mask)
    
    # Convert to grayscale
    res_gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    
    # Crop the top portion of the image to focus on the line
    height, width = res_gray.shape
    res_gray[:int(height*0.5), :] = 0  # Cut top half of the image
    
    # Apply edge detection
    res_edges = cv2.Canny(res_gray, 70, 150)
    
    # Analyze left and right halves to determine turn direction
    left_half = res_gray[:, :int(width*0.60)]
    right_half = res_gray[:, int(width*0.40):]
    left_black_pixels = cv2.countNonZero(left_half)
    right_black_pixels = cv2.countNonZero(right_half)
    total_black_pixels = left_black_pixels + right_black_pixels

    if total_black_pixels > 0:
      # Calculate turn ratio based on line position
      turn_ratio = ((left_black_pixels - right_black_pixels) / total_black_pixels) / 2
      # Adjust turn speed based on the ratio
      max_turn_speed = 0.5  # Maximum angular speed
      turn_speed = max_turn_speed * turn_ratio
      
      # Set robot movement
      if total_black_pixels > 200:  # Ensure we're actually seeing the line
        self.speed.angular.z = turn_speed
        self.last_seen_angular_speed = turn_speed
        self.speed.linear.x = 0.15  # Forward speed
      else:
        # If line is lost, turn in the direction it was last seen
        if self.last_seen_angular_speed > 0:
          self.speed.angular.z = 0.3
        else:
          self.speed.angular.z = -0.3
        self.speed.linear.x = 0  # Stop forward movement while searching
    else:
      # Line completely lost, search for it
      if self.last_seen_angular_speed > 0:
        self.speed.angular.z = 0.3
      else:
        self.speed.angular.z = -0.3
      self.speed.linear.x = 0
    
    # Publish the movement command
    self.cmd_vel_pub.publish(self.speed)
    self.rate.sleep()

if __name__ == '__main__':
  follower = Follower()
  rospy.spin()
