#! /usr/bin/env python

import argparse
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import Twist

linear_x = 0
angular_z = 0

def steer_callback(msg):
    global linear_x
    global angular_z
    angular_z = msg.angular.z
    move = Twist()
    move.angular.z = angular_z
    move.linear.x = linear_x
    pub.publish(move)

def callback(msg):
    global linear_x
    global angular_z

    # nothing
    if msg.data == 0:
        print('[0] No object')
        linear_x = 0.15

    # stop
    if msg.data == 1:
        print('[1] Stop')
        linear_x = 0.0

    # 30 
    if msg.data == 2:
        print('[2] Deaccelerate')
        linear_x = 0.1

    # 60
    if msg.data == 3:
        print('[3] Accelerate')
        linear_x = 0.2

    move = Twist()
    move.angular.z = angular_z
    move.linear.x = linear_x
    pub.publish(move)

parser = argparse.ArgumentParser(description='Object Detection')
parser.add_argument('--env', help='environment', choices=['gazebo', 'turtlebot'], type=str, required=True)
args = parser.parse_args()

if args.env == 'gazebo':
    linear_x = 0.15
if args.env == 'turtlebot':
    linear_x = 0.015

rospy.init_node('control_node')
sub = rospy.Subscriber('/detect', Int32, callback)
sub = rospy.Subscriber('/cmd_vel_steer', Twist, steer_callback)
pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

rospy.spin()
