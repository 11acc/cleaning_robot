#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import os
import datetime
from time import time

class RobustLineFollower:
    def __init__(self):
        rospy.init_node('robust_line_follower')

        self.bridge = CvBridge()
        self.twist = Twist()

        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.mask_pub = rospy.Publisher('/line_mask/image_raw', Image, queue_size=1)
        rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)

        # Parameters
        self.forward_speed    = rospy.get_param('~forward_speed', 0.10)
        self.search_speed     = rospy.get_param('~search_speed', 0.3)
        self.crop_fraction    = rospy.get_param('~crop_fraction', 0.2)
        self.kp               = rospy.get_param('~kp', 0.004)
        self.kd               = rospy.get_param('~kd', 0.002)
        self.max_ang_vel      = rospy.get_param('~max_ang_vel', 0.3)
        self.min_contour_area = rospy.get_param('~min_contour_area', 500)
        self.delta_floor      = rospy.get_param('~delta_floor', 50)
        self.max_no_line_time = rospy.get_param('~max_no_line_time', 0.5)

        # States
        self.STATE_TRACK = 0
        self.STATE_SEARCH = 1
        self.state = self.STATE_SEARCH
        self.last_seen = time()
        self.prev_error = 0.0

        # Image saving
        self.image_save_dir = rospy.get_param('~image_save_dir', '/tmp/line_debug/')
        os.makedirs(self.image_save_dir, exist_ok=True)
        self.save_interval = rospy.get_param('~save_interval', 5)
        self.last_saved_time = rospy.Time.now()

        rospy.loginfo("RobustLineFollower initialized. Starting in SEARCH mode.")

    def image_callback(self, msg):
        now = time()
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr("CV Bridge error: %s", e)
            return

        h, w = img.shape[:2]
        crop_h = int(h * self.crop_fraction)
        crop = img[h - crop_h:, :]

        # Convert to grayscale and blur
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)

        # Dynamic thresholding
        ph, pw = max(1, int(crop_h * 0.1)), max(1, int(w * 0.1))
        left_patch = blur[crop_h - ph:crop_h, :pw]
        right_patch = blur[crop_h - ph:crop_h, -pw:]
        floor_median = np.median(np.concatenate((left_patch.flatten(), right_patch.flatten())))
        thresh_val = max(floor_median - self.delta_floor, 0)
        _, mask = cv2.threshold(blur, thresh_val, 255, cv2.THRESH_BINARY_INV)

        # Morphological filtering
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Publish mask for debugging
        self.mask_pub.publish(self.bridge.cv2_to_imgmsg(mask, encoding='mono8'))

        # Contour analysis
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_contour = None
        best_score = -1
        for cnt in contours:
            if cv2.contourArea(cnt) < self.min_contour_area:
                continue
            bottom_y = cnt[:, :, 1].max()
            if bottom_y > best_score:
                best_score = bottom_y
                best_contour = cnt

        if best_contour is not None:
            M = cv2.moments(best_contour)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                error = cx - (w // 2)
                self._track(error)
                self.last_seen = now
                self._switch_state(self.STATE_TRACK)
        else:
            if now - self.last_seen > self.max_no_line_time:
                self._switch_state(self.STATE_SEARCH)

        if self.state == self.STATE_SEARCH:
            self._search()

        self.cmd_pub.publish(self.twist)

        # Debug windows
        cv2.imshow("Crop View", crop)
        cv2.imshow("Line Mask", mask)
        cv2.waitKey(1)

        # Save debug image
        current_time = rospy.Time.now()
        if current_time - self.last_saved_time >= rospy.Duration(self.save_interval):
            self._save_image(img)
            self.last_saved_time = current_time

    def _track(self, error):
        d_error = error - self.prev_error
        self.prev_error = error
        ang = -(self.kp * error + self.kd * d_error)
        ang = np.clip(ang, -self.max_ang_vel, self.max_ang_vel)
        self.twist.linear.x = self.forward_speed
        self.twist.angular.z = ang

    def _search(self):
        self.twist.linear.x = 0.0
        self.twist.angular.z = self.search_speed

    def _switch_state(self, new_state):
        if self.state != new_state:
            self.state = new_state
            self.twist = Twist()
            name = "TRACK" if new_state == self.STATE_TRACK else "SEARCH"
            rospy.loginfo(f"Switched to {name} mode")

    def _save_image(self, image):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        path = os.path.join(self.image_save_dir, f"image_{timestamp}.jpg")
        cv2.imwrite(path, image)
        rospy.loginfo(f"Image saved: {path}")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = RobustLineFollower()
        node.run()
    except rospy.ROSInterruptException:
        pass
