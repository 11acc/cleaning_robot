#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import UInt16, Float64

class DiscardPegRoutine:
    def __init__(self):
        rospy.init_node('discard_peg_routine')
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.servo_pub = rospy.Publisher('/servo', UInt16, queue_size=10)
        self.servo_load_pub = rospy.Publisher('/servoLoad', Float64, queue_size=10)
        rospy.sleep(1)
        rospy.loginfo("DiscardPegRoutine node initialized")

    def stop(self):
        rospy.loginfo("Stopping robot")
        twist = Twist()
        self.cmd_vel_pub.publish(twist)
        rospy.sleep(0.5)

    def turn(self, angular_speed, duration):
        direction = "right" if angular_speed < 0 else "left"
        rospy.loginfo(f"Turning {direction} with speed {angular_speed} for {duration} seconds")

        twist = Twist()
        twist.angular.z = angular_speed
        rate = rospy.Rate(10)  # 10 Hz

        start_time = rospy.Time.now()
        while (rospy.Time.now() - start_time).to_sec() < duration:
            self.cmd_vel_pub.publish(twist)
            rate.sleep()

        self.stop()

    def move(self, speed, duration):
        direction = "forward" if speed > 0 else "backward"
        rospy.loginfo(f"Moving {direction} at speed {speed} for {duration} seconds")

        twist = Twist()
        twist.linear.x = speed
        rate = rospy.Rate(10)  # 10 Hz

        start_time = rospy.Time.now()
        while (rospy.Time.now() - start_time).to_sec() < duration:
            self.cmd_vel_pub.publish(twist)
            rate.sleep()

        self.stop()

    def open_gripper(self):
        rospy.loginfo("Opening gripper")
        servo_angle = UInt16(data=0)
        self.servo_pub.publish(servo_angle)
        self.servo_load_pub.publish(Float64(data=0.5))
        rospy.sleep(1.5)

    def run(self):
        rospy.loginfo("Starting discard peg routine")

        # 1. Turn right 90 degrees
        self.turn(angular_speed=-0.5, duration=10)

        # 2. Move forward 30 cm
        self.move(speed=0.1, duration=5)

        # 3. Open gripper
        self.open_gripper()

        # 4. Move backward 30 cm
        self.move(speed=-0.1, duration=5)

        # 5. Turn left 90 degrees
        self.turn(angular_speed=0.5, duration=10)

        rospy.loginfo("Routine complete")

if __name__ == '__main__':
    try:
        routine = SimplePegRoutine()
        routine.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS Interrupt - shutting down.")
