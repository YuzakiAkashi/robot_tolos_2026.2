#!/usr/bin/env python3
import sys
import termios
import tty
import rospy
from geometry_msgs.msg import Twist

msg = """
Control Your Robot
---------------------------
Moving:
   w
a  s  d

w : forward
s : backward
a : turn left
d : turn right

space : stop
Ctrl-C to quit
"""

move_bindings = {
    'w': (1.0, 0.0),
    's': (-1.0, 0.0),
    'a': (0.0, 1.0),
    'd': (0.0, -1.0),
    ' ': (0.0, 0.0)
}

def get_key():
    tty.setraw(sys.stdin.fileno())
    key = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

if __name__ == "__main__":
    settings = termios.tcgetattr(sys.stdin)

    rospy.init_node('keyboard_teleop')
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

    linear_speed = rospy.get_param("~linear_speed", 0.3)
    angular_speed = rospy.get_param("~angular_speed", 1.0)

    print(msg)

    try:
        while not rospy.is_shutdown():
            key = get_key()
            if key in move_bindings:
                lin, ang = move_bindings[key]
                twist = Twist()
                twist.linear.x = lin * linear_speed
                twist.angular.z = ang * angular_speed
                pub.publish(twist)
            elif key == '\x03':
                break
    finally:
        twist = Twist()
        pub.publish(twist)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)