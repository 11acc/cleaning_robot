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

        self.state = State.YELLOW_ZONE

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

    # Placeholder methods for other states
    def follow_line(self, msg):
        rospy.loginfo("Following line... (not implemented)")
        # TODO: Implement line following

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
        turn(angular_speed=-0.5, duration=4)   # Turn right 90°
        move(speed=0.1, duration=3.5)            # Move forward
        open_gripper()                           # Drop object
        move(speed=-0.1, duration=3.5)           # Move backward
        turn(angular_speed=0.5, duration=4)    # Turn left 90°

        rospy.loginfo("Discard routine complete")

    def yellow_zone(self, msg):
        def open_gripper():
            rospy.loginfo("Opening gripper")
            self.servo_pub.publish(UInt16(data=0))
            self.servo_load_pub.publish(Float64(data=0.5))
            rospy.sleep(1.5)

        def return_to_start():
            if self.current_pose is None or self.start_pose is None:
                rospy.logwarn_throttle(5, "Waiting for odometry data to return to start")
                self.cmd_vel_pub.publish(Twist())
                return

            x_cur, y_cur, yaw_cur = self.current_pose
            x_start, y_start, yaw_start = self.start_pose
            dx = x_start - x_cur
            dy = y_start - y_cur
            dist = math.sqrt(dx**2 + dy**2)
            angle_to_goal = math.atan2(dy, dx)
            angle_diff = (angle_to_goal - yaw_cur + math.pi) % (2 * math.pi) - math.pi

            if dist > self.stop_distance_m:
                if abs(angle_diff) > 0.05:
                    self.twist.linear.x = 0
                    self.twist.angular.z = 0.3 if angle_diff > 0 else -0.3
                else:
                    self.twist.linear.x = 0.1
                    self.twist.angular.z = 0
            else:
                yaw_diff = (yaw_start - yaw_cur + math.pi) % (2 * math.pi) - math.pi
                if abs(yaw_diff) > 0.05:
                    self.twist.linear.x = 0
                    self.twist.angular.z = 0.3 if yaw_diff > 0 else -0.3
                else:
                    rospy.loginfo("Returned to start pose")
                    self.twist = Twist()
                    returning = False
                    completed = True
            self.cmd_vel_pub.publish(self.twist)

        # Load state
        searching = self.searching
        approaching = self.approaching
        returning = self.returning
        completed = self.completed

        try:
            if completed:
                self.cmd_vel_pub.publish(Twist())
                return

            if returning:
                return_to_start()
                self.returning = returning
                self.completed = completed
                return

            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            cX, cY = self.center_x, self.center_y
            mask_area = 0
            if contours:
                largest = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest)
                if M["m00"] > 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                mask_area = cv2.contourArea(largest)

            if searching:
                if not self.recording_360:
                    rospy.loginfo("Starting 360 scan")
                    self.scan_data = []
                    self.start_rotation_time = rospy.Time.now().to_sec()
                    self.recording_360 = True
                else:
                    self.twist.linear.x = 0
                    self.twist.angular.z = self.search_rotation_speed
                    self.cmd_vel_pub.publish(self.twist)

                    if self.current_pose:
                        _, _, yaw = self.current_pose
                        self.scan_data.append((yaw, mask_area))

                    if rospy.Time.now().to_sec() - self.start_rotation_time >= self.rotation_duration:
                        best_yaw = max(self.scan_data, key=lambda x: x[1])[0]
                        rospy.loginfo(f"Scan complete. Best yaw: {best_yaw:.2f}")
                        self.recording_360 = False
                        self.twist = Twist()
                        self.cmd_vel_pub.publish(self.twist)
                        # Now turn to that yaw
                        self.desired_yaw = best_yaw
                        self.aligning = True
                        searching = False
                        approaching = True  # Start visual approach

            elif approaching and not returning:
                err = float(cX - self.center_x) / self.center_x
                area_thresh = 15000  # Tune this value
                twist = Twist()

                if mask_area > area_thresh:
                    rospy.loginfo("Reached yellow zone")
                    twist = Twist()
                    self.cmd_vel_pub.publish(twist)
                    open_gripper()
                    rospy.sleep(1.5)
                    self.twist.linear.x = -0.1
                    self.cmd_vel_pub.publish(self.twist)
                    rospy.sleep(2.5)
                    self.twist.angular.z = 0.4
                    self.twist.linear.x = 0
                    self.cmd_vel_pub.publish(self.twist)
                    rospy.sleep(2.0)
                    self.cmd_vel_pub.publish(Twist())
                    returning = True
                    approaching = False
                else:
                    twist.linear.x = 0.08
                    twist.angular.z = -err * 0.4
                    self.cmd_vel_pub.publish(twist)

            # Visualization
            cv2.circle(cv_image, (cX, cY), 5, (255, 255, 255), -1)
            cv2.imshow("Yellow View", cv_image)
            cv2.imshow("Mask", mask)
            cv2.waitKey(3)

            # Save state back
            self.searching = searching
            self.approaching = approaching
            self.returning = returning
            self.completed = completed

        except Exception as e:
            rospy.logerr(f"yellow_zone error: {e}")
            self.cmd_vel_pub.publish(Twist())

if __name__ == '__main__':
    try:
        bot = CompleteBot()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()