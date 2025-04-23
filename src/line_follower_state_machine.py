#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

class LineFollowerSM:
    def __init__(self):
        rospy.init_node("line_follower_sm", anonymous=True)

        # ---------------- Parameters --------------------------------------
        # HSV thresholds for a dark line on a bright floor
        self.lower_black = np.array([0, 0, 0])
        self.upper_black = np.array([180, 255, 60])

        self.kp           = 1.0 / 450.0   # steering gain
        self.linear_speed = 0.06          # m/s

        self.turn_speed   = 0.35          # rad/s
        self.kp_turn      = 1.0 / 200.0   # tighter → more accurate, looser → faster
        self.center_tol   = 100           # larger  → crisper centring response

        # Visual text stuff
        self.bar_px_thresh  = 2000
        self.line_px_thresh = 1000

        # ROI limits
        self.top_roi = (0.55, 0.70)
        self.bot_roi = (0.94, 1.00)

        # How long to keep going forward after the line vanishes
        self.wait_after_loss = 8          # frames (≈0.8 s @ 10 fps)
        self.loss_counter    = 0

        self.debug = rospy.get_param("~debug", True)

        # ---------------- State machine -----------------------------------
        self.state          = "FOLLOW"
        self.turn_direction = "LEFT"

        # ---------------- ROS wiring --------------------------------------
        self.bridge  = CvBridge()
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        rospy.Subscriber("/camera/color/image_raw", Image, self.image_cb)

        self.twist = Twist()
        rospy.loginfo("lv: Line-follower state-machine node started")

    @staticmethod
    def choose_turn_direction(bar_mask):
        # Return 'LEFT' if more bar pixels are on the left half, else 'RIGHT'
        h, w = bar_mask.shape
        left  = cv2.countNonZero(bar_mask[:, : w // 2])
        right = cv2.countNonZero(bar_mask[:, w // 2 :])
        return "RIGHT" if right > left else "LEFT"

    @staticmethod
    def centroid_x(mask):
        # Return centroid x of a binary mask, or None if empty
        M = cv2.moments(mask)
        return int(M["m10"] / M["m00"]) if M["m00"] else None

    def follow_centre_line(self, line_mask, width):
        # Proportional steering on bottom-ROI mask
        cx = self.centroid_x(line_mask)
        # no centroid → creep straight
        if cx is None:
            self.twist.linear.x  = 0.02
            self.twist.angular.z = 0.0
            return

        error = cx - width // 2
        self.twist.linear.x  = self.linear_speed
        self.twist.angular.z = -self.kp * float(error)

    def image_cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr("CV Bridge: %s", e)
            return

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w, _ = hsv.shape

        # -------- Build masks for both ROIs -------------------------------
        top_start, top_end = int(h * self.top_roi[0]), int(h * self.top_roi[1])
        bot_start, bot_end = int(h * self.bot_roi[0]), int(h * self.bot_roi[1])

        top_mask = cv2.inRange(
            hsv[top_start:top_end, :], self.lower_black, self.upper_black
        )
        bot_mask = cv2.inRange(
            hsv[bot_start:bot_end, :], self.lower_black, self.upper_black
        )

        has_bar  = cv2.countNonZero(top_mask) > self.bar_px_thresh
        has_line = cv2.countNonZero(bot_mask) > self.line_px_thresh

        # ---------------- STATE MACHINE -----------------------------------
        if self.state == "FOLLOW":
            if has_bar:
                self.state = "PREPARE"
                self.loss_counter = 0
                rospy.loginfo_once("FOLLOW → PREPARE")
            self.follow_centre_line(bot_mask, w)

        elif self.state == "PREPARE":
            # if centre line disappeared
            if not has_line:
                self.loss_counter += 1
                # go forward or a few frames
                if self.loss_counter < self.wait_after_loss:
                    self.twist.linear.x = self.linear_speed
                    self.twist.angular.z = 0.0
                else:
                    self.turn_direction = self.choose_turn_direction(top_mask)
                    self.state = "TURN"
                    rospy.loginfo_once(f"PREPARE → TURN ({self.turn_direction})")
            else:
                self.loss_counter = 0
                self.follow_centre_line(bot_mask, w)

        elif self.state == "TURN":
            # keep turning in chosen direction
            self.twist.linear.x = 0.0
            cx = self.centroid_x(bot_mask) if has_line else None
            if cx is not None:
                # use proportional centring once the line is visible again
                error = cx - w // 2
                base = self.turn_speed if self.turn_direction == "LEFT" else -self.turn_speed
                self.twist.angular_z = base - self.kp_turn * float(error)
                if abs(error) < self.center_tol:
                    # lock on: re‑enter FOLLOW
                    self.state = "FOLLOW"
                    rospy.loginfo_once("TURN → FOLLOW (centred)")
            else:
                # line not visible yet – spin at nominal rate
                self.twist.angular.z = self.turn_speed if self.turn_direction == "LEFT" else -self.turn_speed

        # ---------------- Publish command ---------------------------------
        self.cmd_pub.publish(self.twist)

        # ---------------- Visualisation -----------------------------------
        if self.debug:
            # 1. Raw camera with overlays
            vis = frame.copy()
            cv2.rectangle(vis, (0, top_start), (w, top_end), (255, 0, 0), 2)
            cv2.rectangle(vis, (0, bot_start), (w, bot_end), (0, 255, 0), 2)
            cv2.putText(vis, self.state, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow("camera", vis)

            # 2. Binary mask (top + bottom ROIs)
            mask_full = np.zeros((h, w), dtype=np.uint8)
            mask_full[top_start:top_end, :] = top_mask
            mask_full[bot_start:bot_end, :] = bot_mask
            cv2.imshow("mask", mask_full)
            cv2.waitKey(1)

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    try:
        LineFollowerSM().run()
    except rospy.ROSInterruptException:
        pass
