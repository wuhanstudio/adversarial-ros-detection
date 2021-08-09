#! /usr/bin/env python

import time
import argparse
from typing import Counter
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import Twist

ratio = 1
linear_x = 0.015 * ratio
angular_z = 0

flag_stop = False
counter = 0
stopped = False

def steer_callback(msg):
    global linear_x
    global angular_z
    global flag_stop
    angular_z = msg.angular.z
    print(stopped)
    if stopped:
        angular_z = 0.0

def callback(msg):
    global linear_x
    global angular_z
    global ratio
    global flag_stop
    global counter
    global stopped

    # nothing
    if msg.data == 0:
        print('[0] No object')
        linear_x = 0.015 * ratio
        stopped = False

    # 40
    if msg.data == 1:
        print('[1] 40')
        linear_x = 0.045 * ratio

    # stop 
    if msg.data == 2:
        print('[2] Stop')
        flag_stop = True

    # 20
    if msg.data == 3:
        print('[3] 20')
        linear_x = 0.025 * ratio
        stopped = False

    if flag_stop:
        counter = counter + 1
        print(counter)

    if (flag_stop and counter > 95):
        linear_x = 0.0
        angular_z = 0.0

        counter = 0
        flag_stop = False
        stopped = True

    move = Twist()
    move.angular.z = angular_z
    move.linear.x = linear_x
    pub.publish(move)


parser = argparse.ArgumentParser(description='Object Detection')
parser.add_argument('--env', help='environment', choices=['gazebo', 'turtlebot'], type=str, required=True)
args = parser.parse_args()

if args.env == 'gazebo':
    ratio = 10
if args.env == 'turtlebot':
    ratio = 1

rospy.init_node('control_node')
sub = rospy.Subscriber('/detect', Int32, callback)
sub = rospy.Subscriber('/cmd_vel_steer', Twist, steer_callback)
pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

rospy.spin()
