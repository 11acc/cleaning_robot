#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import math
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import UInt16, Float64
from cv_bridge import CvBridge
from tf.transformations import euler_from_quaternion

class RobotController:
    def __init__(self):
        rospy.init_node('robot_controller')

        # --- Publishers & Subscribers ---
        self.cmd_vel_pub     = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.servo_pub       = rospy.Publisher('/servo', UInt16, queue_size=1)
        self.servo_load_pub  = rospy.Publisher('/servoLoad', Float64, queue_size=1)

        self.image_sub       = rospy.Subscriber('/camera/color/image_raw', Image, self.image_cb)
        self.odom_sub        = rospy.Subscriber('/odom', Odometry, self.odom_cb)
        self.bridge          = CvBridge()

        # Create OpenCV windows
        cv2.namedWindow("Camera View", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Mask View", cv2.WINDOW_NORMAL)

        # --- State & timing ---
        self.state           = 'LINE'   # LINE → GRAB → HOLD → YELLOW_SEARCH → YELLOW_APPROACH → RETURN → LINE …
        self.last_ts         = rospy.Time.now()
        self.hold_time       = 5.0      # seconds to hold peg
        self.hold_start      = None

        # --- Odometry & poses ---
        self.current_pose    = None     # (x, y, yaw)
        self.start_pose      = None
        self.target_pose     = None     # for yellow approach

        # --- Image buffers & flags ---
        self.latest_image    = None

        # --- Vision parameters ---
        # black line
        self.l_black = np.array([0, 0, 0])
        self.u_black = np.array([180, 255, 50])
        # red peg
        self.l_r1, self.u_r1 = np.array([0,100,100]), np.array([10,255,255])
        self.l_r2, self.u_r2 = np.array([160,100,100]), np.array([180,255,255])
        self.min_red_area    = 1000
        self.red_center_th   = 50
        # yellow zone
        self.l_yellow        = np.array([15,60,100])
        self.u_yellow        = np.array([40,255,255])
        self.min_yellow_area = 500
        self.focal_px        = 554        # adjust for your camera
        self.real_y_w        = 0.2        # 20 cm
        self.stop_dist       = 0.05       # 5 cm

        rospy.loginfo("Controller initialized, opening gripper and starting LINE state.")
        self.open_gripper()
        rospy.sleep(1.5)
        self.run()

    def odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.current_pose = (p.x, p.y, yaw)
        if self.start_pose is None:
            self.start_pose = self.current_pose

    def image_cb(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except:
            pass

    def open_gripper(self):
        a = UInt16(); a.data = 0
        l = Float64(); l.data = 0.5
        self.servo_pub.publish(a)
        self.servo_load_pub.publish(l)

    def close_gripper(self):
        a = UInt16(); a.data = 170
        l = Float64(); l.data = 0.8
        self.servo_pub.publish(a)
        self.servo_load_pub.publish(l)

    def detect_red(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, self.l_r1, self.u_r1)
        m2 = cv2.inRange(hsv, self.l_r2, self.u_r2)
        mask = cv2.morphologyEx(m1 | m2, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centered = False
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            area = cv2.contourArea(c)
            if area >= self.min_red_area:
                M = cv2.moments(c)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    off = abs(cx - img.shape[1]//2)
                    centered = (off < self.red_center_th)
        return centered, mask

    def detect_yellow_and_set_target(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.l_yellow, self.u_yellow)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts or self.current_pose is None:
            return False, mask
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < self.min_yellow_area:
            return False, mask
        x,y,w,_ = cv2.boundingRect(c)
        dist = (self.focal_px * self.real_y_w) / w
        x_r, y_r, yaw = self.current_pose
        err_x = ((x + w/2) - img.shape[1]/2) / (img.shape[1]/2)
        lat = err_x * self.real_y_w
        tx = x_r + dist * math.cos(yaw) - lat * math.sin(yaw)
        ty = y_r + dist * math.sin(yaw) + lat * math.cos(yaw)
        self.target_pose = (tx, ty)
        return True, mask

    def line_follow(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, w = img.shape[:2]
        crop = hsv[int(h*0.8):h, :]
        blur = cv2.GaussianBlur(crop, (5,5), 0)
        mask = cv2.inRange(blur, self.l_black, self.u_black)
        cnts, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cmd = Twist()
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c) > 1000 and cv2.moments(c)['m00'] > 0:
                cx = int(cv2.moments(c)['m10'] / cv2.moments(c)['m00'])
                err = cx - (w // 2)
                cmd.linear.x = 0.15
                cmd.angular.z = -float(err) / 250.0
            else:
                cmd.angular.z = 0.3
        else:
            cmd.angular.z = 0.3
        self.cmd_vel_pub.publish(cmd)
        return mask

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            img = self.latest_image
            mask = None

            if img is not None:
                if self.state == 'LINE':
                    centered, red_mask = self.detect_red(img)
                    if centered:
                        rospy.loginfo("→ RED detected, grabbing")
                        self.cmd_vel_pub.publish(Twist())
                        self.state = 'GRAB'
                        continue
                    mask = self.line_follow(img)

                elif self.state == 'GRAB':
                    self.close_gripper()
                    self.hold_start = rospy.Time.now()
                    rospy.loginfo("Grabbed: holding for %.1f s", self.hold_time)
                    self.state = 'HOLD'

                elif self.state == 'HOLD':
                    if (rospy.Time.now() - self.hold_start).to_sec() >= self.hold_time:
                        rospy.loginfo("Hold done, releasing and searching YELLOW")
                        self.open_gripper()
                        self.state = 'YELLOW_SEARCH'

                elif self.state == 'YELLOW_SEARCH':
                    found, y_mask = self.detect_yellow_and_set_target(img)
                    if found:
                        rospy.loginfo("→ YELLOW found, approach at (%.2f,%.2f)", *self.target_pose)
                        mask = y_mask
                        self.state = 'YELLOW_APPROACH'
                    else:
                        cmd = Twist(); cmd.angular.z = 0.3
                        self.cmd_vel_pub.publish(cmd)
                        mask = y_mask

                elif self.state == 'YELLOW_APPROACH':
                    # approach logic (same as before) ...
                    found, y_mask = False, None  # placeholder; you can keep existing approach code
                    mask = y_mask

                elif self.state == 'RETURN':
                    # return logic (same as before) ...
                    pass

                # Show windows
                cv2.imshow("Camera View", img)
                if mask is not None:
                    cv2.imshow("Mask View", mask)
                cv2.waitKey(1)

            rate.sleep()

if __name__ == '__main__':
    try:
        RobotController()
    except rospy.ROSInterruptException:
        pass
