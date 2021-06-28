#! /usr/bin/env python

import rospy
from std_msgs.msg import Int32 
from geometry_msgs.msg import Twist

def callback(msg): 
  print(msg.data)

  if msg.data == 1 :
      move.linear.x = 0.01
      #move.angular.z = 0.0

  if msg.data == 2 : 
      move.linear.x = 0.0
      #move.angular.z = 0.0
        
  if msg.data == 3:
      move.linear.x = 0.02
      #move.angular.z = 0.0
      
  pub.publish(move)

rospy.init_node('control_node')
sub = rospy.Subscriber('/detect', Int32, callback)
pub = rospy.Publisher('/cmd_vel', Twist)
move = Twist()

rospy.spin()
