#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np

class YellowFollower:
    def __init__(self):
        rospy.init_node('yellow_follower')
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.twist = Twist()
        
        self.searching = True
        self.approaching = False
        
        self.min_contour_area = 500
        
        # Camera and object parameters (adjust these to your setup)
        self.focal_length_px = 554  # example focal length in pixels (replace with your camera's)
        self.real_yellow_width_m = 0.2  # real width of yellow zone in meters
        self.stop_distance_m = 0.15  # stop distance in meters
        
        # HSV thresholds for yellow (widened for lighting variations)
        self.lower_yellow = np.array([15, 60, 100])
        self.upper_yellow = np.array([40, 255, 255])
        
        # Uncomment below to enable interactive HSV tuning with trackbars
        # self.init_trackbars()

    def init_trackbars(self):
        def nothing(x):
            pass
        cv2.namedWindow('mask')
        cv2.createTrackbar('H Lower', 'mask', 15, 179, nothing)
        cv2.createTrackbar('H Upper', 'mask', 40, 179, nothing)
        cv2.createTrackbar('S Lower', 'mask', 60, 255, nothing)
        cv2.createTrackbar('S Upper', 'mask', 255, 255, nothing)
        cv2.createTrackbar('V Lower', 'mask', 100, 255, nothing)
        cv2.createTrackbar('V Upper', 'mask', 255, 255, nothing)

    def update_hsv_from_trackbars(self):
        h_lower = cv2.getTrackbarPos('H Lower', 'mask')
        h_upper = cv2.getTrackbarPos('H Upper', 'mask')
        s_lower = cv2.getTrackbarPos('S Lower', 'mask')
        s_upper = cv2.getTrackbarPos('S Upper', 'mask')
        v_lower = cv2.getTrackbarPos('V Lower', 'mask')
        v_upper = cv2.getTrackbarPos('V Upper', 'mask')
        self.lower_yellow = np.array([h_lower, s_lower, v_lower])
        self.upper_yellow = np.array([h_upper, s_upper, v_upper])

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            
            # Uncomment below to update thresholds dynamically from trackbars
            # self.update_hsv_from_trackbars()
            
            mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            
            contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                contour_area = cv2.contourArea(largest_contour)
                
                if contour_area > self.min_contour_area:
                    M = cv2.moments(largest_contour)
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    
                    height, width = cv_image.shape[:2]
                    error_x = (cx - width/2) / (width/2)
                    
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    distance = (self.focal_length_px * self.real_yellow_width_m) / w
                    
                    rospy.loginfo(f"Estimated distance to yellow zone: {distance:.2f} m")
                    
                    if self.searching:
                        rospy.loginfo("Found yellow - switching to approach")
                        self.searching = False
                        self.approaching = True
                    
                    if self.approaching:
                        if distance <= self.stop_distance_m:
                            rospy.loginfo("Reached target zone - stopping")
                            self.twist.linear.x = 0
                            self.twist.angular.z = 0
                            self.approaching = False
                        else:
                            if abs(error_x) > 0.05:
                                self.twist.angular.z = -error_x * 0.5
                            else:
                                self.twist.angular.z = 0
                            self.twist.linear.x = 0.1 * (1 - abs(error_x))
                            
                            cv2.circle(cv_image, (cx, cy), 10, (0, 255, 0), -1)
                            cv2.putText(cv_image, "APPROACHING", (10, 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    self.handle_no_detection(cv_image)
            else:
                self.handle_no_detection(cv_image)
            
            self.cmd_vel_pub.publish(self.twist)
            
            cv2.imshow("Camera View", cv_image)
            cv2.imshow("Processed Mask", mask)
            cv2.waitKey(3)
            
        except Exception as e:
            rospy.logerr(f"Error: {e}")
            self.twist = Twist()
            self.cmd_vel_pub.publish(self.twist)

    def handle_no_detection(self, frame):
        if self.approaching:
            rospy.loginfo("Lost target - returning to search")
            self.approaching = False
            self.searching = True
            
        if self.searching:
            rospy.loginfo("Searching for target...")
            self.twist.linear.x = 0
            self.twist.angular.z = 0.3
            
        cv2.putText(frame, "SEARCHING", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

if __name__ == '__main__':
    try:
        YellowFollower()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
