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

        # --- Parameters ----------------------------------------------------
        self.lower_black = np.array([0, 0, 0])     # HSV thresholds
        self.upper_black = np.array([180, 255, 60])

        self.kp             = 1.0 / 450.0          # steering gain
        self.linear_speed   = 0.06                 # m/s while following
        self.turn_speed     = 0.35                 # rad/s while turning

        self.bar_px_thresh  = 2000                 # pixels in TOP ROI to say "I see a bar"
        self.line_px_thresh = 1000                 # pixels in BOTTOM ROI to say "I see the centre line"

        # Region‑of‑interest limits as fractions of image height
        self.top_roi = (0.55, 0.70)   # 55 % – 70 %
        self.bot_roi = (0.88, 1.00)   # 88 % – 100 %

        # --- State machine -------------------------------------------------
        self.state          = "FOLLOW"            # initial state
        self.turn_direction = "LEFT"             # updated when entering TURN

        # --- ROS wiring ----------------------------------------------------
        self.bridge  = CvBridge()
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        rospy.Subscriber("/camera/color/image_raw", Image, self.image_cb)

        self.twist   = Twist()
        rospy.loginfo("Line-follower state-machine node started")

    # ---------------------------------------------------------------------
    def choose_turn_direction(self, bar_mask):
        """Return 'LEFT' if more bar pixels are on the left half, else 'RIGHT'."""
        h, w = bar_mask.shape
        left  = cv2.countNonZero(bar_mask[:, :w//2])
        right = cv2.countNonZero(bar_mask[:, w//2:])
        return "RIGHT" if right > left else "LEFT"

    # ---------------------------------------------------------------------
    def follow_centre_line(self, line_mask, width):
        """Proportional steering on bottom‑ROI mask."""
        M = cv2.moments(line_mask)
        if M["m00"] == 0:
            # No centroid – keep current heading slowly
            self.twist.linear.x  = 0.02
            self.twist.angular.z = 0.0
            return

        cx = int(M["m10"] / M["m00"])
        error = cx - width // 2

        self.twist.linear.x  = self.linear_speed
        self.twist.angular.z = -self.kp * float(error)

    # ---------------------------------------------------------------------
    def image_cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr("CV Bridge: %s", e)
            return

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w, _ = hsv.shape

        # -------- build masks for both ROIs --------------------------------
        top_start, top_end = int(h * self.top_roi[0]), int(h * self.top_roi[1])
        bot_start, bot_end = int(h * self.bot_roi[0]), int(h * self.bot_roi[1])

        top_mask = cv2.inRange(hsv[top_start:top_end, :], self.lower_black, self.upper_black)
        bot_mask = cv2.inRange(hsv[bot_start:bot_end, :], self.lower_black, self.upper_black)

        has_bar  = cv2.countNonZero(top_mask) > self.bar_px_thresh
        has_line = cv2.countNonZero(bot_mask) > self.line_px_thresh

        # -------- STATE MACHINE -------------------------------------------
        if self.state == "FOLLOW":
            if has_bar:
                self.state = "PREPARE"
                rospy.loginfo_once("FOLLOW → PREPARE")
            self.follow_centre_line(bot_mask, w)

        elif self.state == "PREPARE":
            self.follow_centre_line(bot_mask, w)
            if not has_line:   # centre line disappeared under the robot
                self.turn_direction = self.choose_turn_direction(top_mask)
                self.state = "TURN"
                rospy.loginfo_once(f"PREPARE → TURN ({self.turn_direction})")

        elif self.state == "TURN":
            # constant in‑place rotation
            self.twist.linear.x  = 0.0
            self.twist.angular.z =  self.turn_speed if self.turn_direction == "LEFT" else -self.turn_speed
            if has_line:        # new line found
                self.state = "FOLLOW"
                rospy.loginfo_once("TURN → FOLLOW")

        # Publish command ---------------------------------------------------
        self.cmd_pub.publish(self.twist)

        # ------------- optional visualisation -----------------------------
        if rospy.get_param("~debug", False):
            vis = frame.copy()
            cv2.rectangle(vis, (0, top_start), (w, top_end), (255,0,0), 2)
            cv2.rectangle(vis, (0, bot_start), (w, bot_end), (0,255,0), 2)
            cv2.putText(vis, self.state, (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,(0,0,255), 2, cv2.LINE_AA)
            cv2.imshow("line follower", vis)
            cv2.waitKey(1)

    # ---------------------------------------------------------------------
    def run(self):
        rospy.spin()

# -------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        LineFollowerSM().run()
    except rospy.ROSInterruptException:
        pass
