#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float64
import sys, termios, tty

class ArmTeleop:

    def __init__(self):

        rospy.init_node("arm_teleop")

        self.base_pub = rospy.Publisher(
            "/arm_base_joint_controller/command",
            Float64,
            queue_size=10
        )

        self.big_pub = rospy.Publisher(
            "/big_arm_joint_controller/command",
            Float64,
            queue_size=10
        )

        self.small_pub = rospy.Publisher(
            "/small_arm_joint_controller/command",
            Float64,
            queue_size=10
        )

        self.base = 0.0
        self.big = 0.0
        self.small = 0.0

        self.step = 0.1

        rospy.loginfo("机械臂键盘调试")
        rospy.loginfo("q/a : base joint")
        rospy.loginfo("w/s : big arm")
        rospy.loginfo("e/d : small arm")

        self.run()

    def get_key(self):

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)

        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

        return ch

    def publish(self):

        self.base_pub.publish(self.base)
        self.big_pub.publish(self.big)
        self.small_pub.publish(self.small)

    def run(self):

        while not rospy.is_shutdown():

            key = self.get_key()

            if key == 'q':
                self.base += self.step

            elif key == 'a':
                self.base -= self.step

            elif key == 'w':
                self.big += self.step

            elif key == 's':
                self.big -= self.step

            elif key == 'e':
                self.small += self.step

            elif key == 'd':
                self.small -= self.step

            self.publish()

            print("base:", self.base,
                  " big:", self.big,
                  " small:", self.small)

if __name__ == "__main__":

    try:
        ArmTeleop()
    except rospy.ROSInterruptException:
        pass