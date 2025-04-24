#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import UInt16, Float64

class SimplePegRoutine:
    def __init__(self):
        rospy.init_node('simple_peg_routine')
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.servo_pub = rospy.Publisher('/servo', UInt16, queue_size=10)
        self.servo_load_pub = rospy.Publisher('/servoLoad', Float64, queue_size=10)
        rospy.sleep(1)

    def stop(self):
        twist = Twist()
        self.cmd_vel_pub.publish(twist)
        rospy.sleep(0.5)

    def turn(self, angular_speed, duration):
        twist = Twist()
        twist.angular.z = angular_speed
        self.cmd_vel_pub.publish(twist)
        rospy.sleep(duration)
        self.stop()

    def move(self, speed, duration):
        twist = Twist()
        twist.linear.x = speed
        self.cmd_vel_pub.publish(twist)
        rospy.sleep(duration)
        self.stop()

    def open_gripper(self):
        servo_angle = UInt16(data=0)
        self.servo_pub.publish(servo_angle)
        self.servo_load_pub.publish(Float64(data=0.5))
        rospy.sleep(1.5)

    def run(self):
        rospy.loginfo("Starting simple peg routine")

        # 1. Turn right 90 degrees (approx. -0.5 rad/s for 3.1 sec)
        self.turn(angular_speed=-0.5, duration=3.1)

        # 2. Move forward 30 cm (0.1 m/s for 3.0 sec)
        self.move(speed=0.1, duration=3.0)

        # 3. Open gripper
        self.open_gripper()

        # 4. Move backward 30 cm
        self.move(speed=-0.1, duration=3.0)

        # 5. Turn left 90 degrees
        self.turn(angular_speed=0.5, duration=3.1)

        rospy.loginfo("Routine complete")

if __name__ == '__main__':
    try:
        routine = SimplePegRoutine()
        routine.run()
    except rospy.ROSInterruptException:
        pass
