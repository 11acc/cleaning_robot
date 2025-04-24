#!/usr/bin/env python3
import rospy
import cv2
import enum
import numpy as np
import math
from sensor_msgs.msg import Image, Range
from geometry_msgs.msg import Twist
from std_msgs.msg import UInt16, Float64
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion

# Constants and thresholds
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
GRIPPER_OPEN_POSITION = 0      # Gripper open position

class State(enum.Enum):
    FOLLOW_LINE = 1
    APPROACH_OBJECT = 2
    GRAB_OBJECT = 3
    YELLOW_ZONE = 4
    DISCARD_ZONE = 5

class ObjectType(enum.Enum):
    RED = 1
    GREEN = 2
    BLUE = 3
    UNKNOWN = 4

class CompleteBot:
    def __init__(self):
        rospy.init_node('complete_bot', anonymous=True)

        self.bridge = CvBridge()
        self.twist = Twist()
        self.speed = Twist()  # Used in sensor callbacks

        # Subscribers and publishers
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.servo_pub = rospy.Publisher('/servo', UInt16, queue_size=10)
        self.servo_load_pub = rospy.Publisher('/servoLoad', Float64, queue_size=10)
        self.fl_sensor_sub = rospy.Subscriber('/range/fl', Range, self.fl_sensor_callback)
        self.fr_sensor_sub = rospy.Subscriber('/range/fr', Range, self.fr_sensor_callback)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)

        self.gripper_position = GRIPPER_CLOSE_POSITION
        self.rate = rospy.Rate(10)

        self.top_roi = (0.55, 0.70)       # 55% – 70% of rows
        self.bot_roi = (0.88, 1.00)       # 88% – 100%

        self.collision_risk = False

        self.state = State.FOLLOW_LINE

        # Yellow zone related variables
        self.searching = True
        self.approaching = False
        self.stopped_at_target = False
        self.returning = False
        self.completed = False
        self.recording_360 = False

        self.min_contour_area = 500
        self.center_x = 320  # Assuming 640x480 image resolution
        self.center_y = 240
        self.fov_horizontal = 60
        self.search_rotation_speed = 0.2
        self.focal_length_px = 554
        self.real_yellow_width_m = 0.2
        self.stop_distance_m = 0.03
        self.max_distance_from_start_m = 1.5
        self.rotation_duration = 10.0

        self.lower_yellow = np.array([15, 60, 100])
        self.upper_yellow = np.array([40, 255, 255])

        self.start_pose = None
        self.current_pose = None
        self.target_pose = None

        self.scan_data = []
        self.start_rotation_time = None

        rospy.loginfo("CompleteBot initialized.")

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
        if self.state != State.APPROACH_OBJECT:
            if msg.range < 0.15:
                rospy.loginfo("Collision risk detected (front-left). Turning.")
                self.collision_risk = True
                self.speed.linear.x = 0
                self.speed.angular.z = TURN_SPEED
                self.cmd_vel_pub.publish(self.speed)
                rospy.sleep(0.5)
                self.speed.angular.z = 0
                self.cmd_vel_pub.publish(self.speed)
            else:
                self.collision_risk = False

    def fr_sensor_callback(self, msg):
        if self.state != State.APPROACH_OBJECT:
            if msg.range < 0.15:
                rospy.loginfo("Collision risk detected (front-right). Turning.")
                self.collision_risk = True
                self.speed.linear.x = 0
                self.speed.angular.z = TURN_SPEED
                self.cmd_vel_pub.publish(self.speed)
                rospy.sleep(0.5)
                self.speed.angular.z = 0
                self.cmd_vel_pub.publish(self.speed)
            else:
                self.collision_risk = False

    def odom_callback(self, msg):
        position = msg.pose.pose.position
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        self.current_pose = (position.x, position.y, yaw)
        if self.start_pose is None:
            self.start_pose = self.current_pose
            rospy.loginfo(f"Recorded start pose: x={position.x:.2f}, y={position.y:.2f}, yaw={yaw:.2f}")

    def follow_line(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr("CV Bridge Error: %s", e)
            return

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w, _ = hsv.shape

        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 60])
        top_start, top_end = int(h * self.top_roi[0]), int(h * self.top_roi[1])
        bot_start, bot_end = int(h * self.bot_roi[0]), int(h * self.bot_roi[1])

        top_mask = cv2.inRange(hsv[top_start:top_end, :], lower_black, upper_black)
        bot_mask = cv2.inRange(hsv[bot_start:bot_end, :], lower_black, upper_black)

        # NEW CODE: Detect colored objects
        # Define a region of interest for object detection (middle section of the image)
        object_roi_start = int(h * 0.3)  # Top 30% of the image
        object_roi_end = int(h * 0.7)    # Bottom 70% of the image
        
        # Create masks for each color
        red_mask1 = cv2.inRange(hsv[object_roi_start:object_roi_end, :], RED_COLOR_LOWER_THRESHOLD1, RED_COLOR_UPPER_THRESHOLD1)
        red_mask2 = cv2.inRange(hsv[object_roi_start:object_roi_end, :], RED_COLOR_LOWER_THRESHOLD2, RED_COLOR_UPPER_THRESHOLD2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        green_mask = cv2.inRange(hsv[object_roi_start:object_roi_end, :], GREEN_COLOR_LOWER_THRESHOLD, GREEN_COLOR_UPPER_THRESHOLD)
        blue_mask = cv2.inRange(hsv[object_roi_start:object_roi_end, :], BLUE_COLOR_LOWER_THRESHOLD, BLUE_COLOR_UPPER_THRESHOLD)
        
        # Count pixels of each color
        red_pixels = cv2.countNonZero(red_mask)
        green_pixels = cv2.countNonZero(green_mask)
        blue_pixels = cv2.countNonZero(blue_mask)
        
        # Define thresholds for detection
        color_threshold = 100  # Minimum number of colored pixels to trigger approach
        
        # Check if any color exceeds the threshold
        if red_pixels > color_threshold:
            rospy.loginfo(f"Red object detected ({red_pixels} pixels). Switching to APPROACH_OBJECT state.")
            self.object_type = ObjectType.RED
            self.state = State.APPROACH_OBJECT
            return
        elif green_pixels > color_threshold:
            rospy.loginfo(f"Green object detected ({green_pixels} pixels). Switching to APPROACH_OBJECT state.")
            self.object_type = ObjectType.GREEN
            self.state = State.APPROACH_OBJECT
            return
        elif blue_pixels > color_threshold:
            rospy.loginfo(f"Blue object detected ({blue_pixels} pixels). Switching to APPROACH_OBJECT state.")
            self.object_type = ObjectType.BLUE
            self.state = State.APPROACH_OBJECT
            return
        
         # Continue with regular line following if no object detected
        # State machine logic (existing line following code)
        if not hasattr(self, 'line_state'):
            self.line_state = "FOLLOW"
            self.turn_direction = "LEFT"

        bar_px_thresh = 2000
        line_px_thresh = 1000
        has_bar = cv2.countNonZero(top_mask) > bar_px_thresh
        has_line = cv2.countNonZero(bot_mask) > line_px_thresh

        # State machine logic
        if not hasattr(self, 'line_state'):
            self.line_state = "FOLLOW"
            self.turn_direction = "LEFT"

        if self.line_state == "FOLLOW":
            if has_bar:
                self.line_state = "PREPARE"
                rospy.loginfo_once("FOLLOW → PREPARE")
            self.follow_centre_line(bot_mask, w)

        elif self.line_state == "PREPARE":
            self.follow_centre_line(bot_mask, w)
            if not has_line:
                self.turn_direction = "RIGHT" if cv2.countNonZero(top_mask[:, w // 2:]) > cv2.countNonZero(top_mask[:, :w // 2]) else "LEFT"
                self.line_state = "TURN"
                rospy.loginfo_once(f"PREPARE → TURN ({self.turn_direction})")

        elif self.line_state == "TURN":
            cx = self.centroid_x(bot_mask) if has_line else None
            if cx is not None:
                error = cx - w // 2
                turn_speed = 0.35
                kp_turn = 1.0 / 200.0
                center_tol = 100
                self.twist.angular.z = (turn_speed if self.turn_direction == "LEFT" else -turn_speed) - kp_turn * float(error)
                self.twist.linear.x = 0.0
                if abs(error) < center_tol:
                    self.line_state = "FOLLOW"
                    rospy.loginfo_once("TURN → FOLLOW (centred)")
            else:
                self.twist.angular.z = 0.35 if self.turn_direction == "LEFT" else -0.35
                self.twist.linear.x = 0.0

        self.cmd_vel_pub.publish(self.twist)

    def follow_centre_line(self, line_mask, width):
        cx = self.centroid_x(line_mask)
        if cx is None:
            self.twist.linear.x = 0.02
            self.twist.angular.z = 0.0
        else:
            error = cx - width // 2
            kp = 1.0 / 450.0
            self.twist.linear.x = LINE_FOLLOW_SPEED
            self.twist.angular.z = -kp * float(error)

    def centroid_x(self, mask):
        M = cv2.moments(mask)
        if M["m00"] == 0:
            return None
        return int(M["m10"] / M["m00"])


    def approach_object(self, msg):
        rospy.loginfo("Approaching object... (not implemented)")
        # TODO: Implement object approach

    def grab_object(self, msg):
        rospy.loginfo("Grabbing object... (not implemented)")
        # TODO: Implement grabbing

    def discard_zone(self, msg):
        rospy.loginfo("Executing discard zone routine")

        def stop():
            twist = Twist()
            self.cmd_vel_pub.publish(twist)
            rospy.sleep(0.5)

        def turn(angular_speed, duration):
            direction = "right" if angular_speed < 0 else "left"
            rospy.loginfo(f"Turning {direction} with speed {angular_speed} for {duration} seconds")
            twist = Twist()
            twist.angular.z = angular_speed
            rate = rospy.Rate(10)
            start_time = rospy.Time.now()
            while (rospy.Time.now() - start_time).to_sec() < duration:
                self.cmd_vel_pub.publish(twist)
                rate.sleep()
            stop()

        def move(speed, duration):
            direction = "forward" if speed > 0 else "backward"
            rospy.loginfo(f"Moving {direction} at speed {speed} for {duration} seconds")
            twist = Twist()
            twist.linear.x = speed
            rate = rospy.Rate(10)
            start_time = rospy.Time.now()
            while (rospy.Time.now() - start_time).to_sec() < duration:
                self.cmd_vel_pub.publish(twist)
                rate.sleep()
            stop()

        def open_gripper():
            rospy.loginfo("Opening gripper")
            self.servo_pub.publish(UInt16(data=0))
            self.servo_load_pub.publish(Float64(data=0.5))
            rospy.sleep(1.5)

        # Actual routine steps
        turn(angular_speed=-0.5, duration=3.5)   # Turn right 90°
        move(speed=0.1, duration=3)            # Move forward
        open_gripper()                           # Drop object
        move(speed=-0.1, duration=3)           # Move backward
        turn(angular_speed=0.5, duration=3.5)    # Turn left 90°

        rospy.loginfo("Discard routine complete")

    def yellow_zone(self, msg):
        def open_gripper():
            rospy.loginfo("Opening gripper")
            self.servo_pub.publish(UInt16(data=0))
            self.servo_load_pub.publish(Float64(data=0.5))
            rospy.sleep(1.5)

        def find_and_set_best_target():
            if not self.scan_data:
                rospy.logwarn("No scan data available to find best target.")
                return

            best_yaw = max(self.scan_data, key=lambda item: item[1])[0]

            distance = 1.0
            x_robot, y_robot, _ = self.current_pose
            target_x = x_robot + distance * math.cos(best_yaw)
            target_y = y_robot + distance * math.sin(best_yaw)

            nonlocal target_pose
            target_pose = (target_x, target_y)
            nonlocal searching, approaching
            searching = False
            approaching = True
            rospy.loginfo(f"Best yellow zone found at yaw {best_yaw:.2f}. Target set to x={target_x:.2f}, y={target_y:.2f}")

        def return_to_start():
            if self.current_pose is None or self.start_pose is None:
                rospy.logwarn_throttle(5, "Waiting for odometry data to return to start")
                self.twist.linear.x = 0
                self.twist.angular.z = 0
                self.cmd_vel_pub.publish(self.twist)
                return

            x_cur, y_cur, yaw_cur = self.current_pose
            x_start, y_start, yaw_start = self.start_pose
            dx = x_start - x_cur
            dy = y_start - y_cur
            distance = math.sqrt(dx * dx + dy * dy)

            angle_to_goal = math.atan2(dy, dx)
            angle_diff = angle_to_goal - yaw_cur
            angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

            if distance > self.stop_distance_m:
                if abs(angle_diff) > 0.05:
                    self.twist.linear.x = 0
                    self.twist.angular.z = 0.3 if angle_diff > 0 else -0.3
                else:
                    self.twist.linear.x = 0.1
                    self.twist.angular.z = 0
                self.cmd_vel_pub.publish(self.twist)
                return

            yaw_diff = yaw_start - yaw_cur
            yaw_diff = (yaw_diff + math.pi) % (2 * math.pi) - math.pi

            if abs(yaw_diff) > 0.05:
                self.twist.linear.x = 0
                self.twist.angular.z = 0.3 if yaw_diff > 0 else -0.3
            else:
                rospy.loginfo("Returned to start pose with correct orientation - stopping")
                self.twist.linear.x = 0
                self.twist.angular.z = 0
                nonlocal returning, completed
                returning = False
                completed = True
            self.cmd_vel_pub.publish(self.twist)

        # Use nonlocal to modify outer scope variables inside nested functions
        searching = self.searching
        approaching = self.approaching
        stopped_at_target = self.stopped_at_target
        returning = self.returning
        completed = self.completed
        target_pose = self.target_pose

        try:
            if completed:
                self.twist.linear.x = 0
                self.twist.angular.z = 0
                self.cmd_vel_pub.publish(self.twist)
                return

            if stopped_at_target and returning:
                return_to_start()
                # Update flags back to self
                self.returning = returning
                self.completed = completed
                return
            
            if stopped_at_target and not returning:
                self.twist.linear.x = 0
                self.twist.angular.z = 0
                self.cmd_vel_pub.publish(self.twist)
                return

            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)

            contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask_area = 0
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX, cY = self.center_x, self.center_y
            else:
                cX, cY = self.center_x, self.center_y
            
            for contour in contours:
                mask_area += cv2.contourArea(contour)

            if searching:
                if not self.recording_360:
                    rospy.loginfo("Starting 360 degree scan...")
                    self.recording_360 = True
                    self.start_rotation_time = rospy.Time.now().to_sec()
                    self.scan_data = []
                else:
                    self.twist.linear.x = 0
                    self.twist.angular.z = self.search_rotation_speed
                    self.cmd_vel_pub.publish(self.twist)

                    if self.current_pose is not None:
                        _, _, yaw = self.current_pose
                        self.scan_data.append((yaw, mask_area))

                    time_now = rospy.Time.now().to_sec()
                    if (time_now - self.start_rotation_time) >= self.rotation_duration:
                        rospy.loginfo("360 degree scan complete, finding best yellow zone...")
                        self.recording_360 = False
                        self.twist.angular.z = 0.0
                        self.cmd_vel_pub.publish(self.twist)
                        find_and_set_best_target()

            if approaching and target_pose is not None and self.current_pose is not None:
                x_cur, y_cur, yaw_cur = self.current_pose
                target_x, target_y = target_pose
                dx = target_x - x_cur
                dy = target_y - y_cur
                dist_to_target = math.sqrt(dx * dx + dy * dy)
                angle_to_target = math.atan2(dy, dx)
                angle_diff = angle_to_target - yaw_cur
                angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

                err_x = float(cX - self.center_x) / self.center_x
                angle_adjust = -err_x * 0.3

                if dist_to_target <= self.stop_distance_m + 0.05:
                    rospy.loginfo("Reached target zone")
                    open_gripper()
                    self.twist.linear.x = 0
                    self.twist.angular.z = 0
                    approaching = False
                    stopped_at_target = True
                    returning = True
                else:
                    max_speed = 0.15
                    min_speed = 0.02
                    slow_down_radius = 0.3

                    if dist_to_target < slow_down_radius:
                        speed = min_speed + (max_speed - min_speed) * (dist_to_target / slow_down_radius)
                        speed = max(speed, min_speed)
                    else:
                        speed = max_speed

                    if abs(angle_diff) > 0.05:
                        self.twist.linear.x = 0
                        self.twist.angular.z = 0.4 if angle_diff > 0 else -0.4
                    else:
                        self.twist.angular.z = angle_adjust
                        self.twist.linear.x = speed

                self.cmd_vel_pub.publish(self.twist)

            # Visualization (optional)
            cv2.circle(cv_image, (cX, cY), 5, (255, 255, 255), -1)
            cv2.imshow("Camera View - Yellow Zone", cv_image)
            cv2.imshow("Yellow Mask", mask)
            cv2.waitKey(3)

            # Update flags back to self
            self.searching = searching
            self.approaching = approaching
            self.stopped_at_target = stopped_at_target
            self.returning = returning
            self.completed = completed
            self.target_pose = target_pose

        except Exception as e:
            rospy.logerr(f"Error in yellow_zone: {e}")
            self.twist = Twist()
            self.cmd_vel_pub.publish(self.twist)

if __name__ == '__main__':
    try:
        bot = CompleteBot()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
