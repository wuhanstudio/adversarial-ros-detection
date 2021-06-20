#!/usr/bin/env python
import argparse

import rospy

import cv2
import numpy as np

from cv_bridge import CvBridge, CvBridgeError

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

bias = 0

class LineFollower(object):

    def __init__(self):
        self.bridge_object = CvBridge()
        self.bias = 0
        self.image_sub = None
        self._pub = None

    def camera_callback(self,data):
        try:
            # We select bgr8 because its the OpneCV encoding by default
            cv_image = self.bridge_object.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)

        # We get image dimensions and crop the parts of the image we don't need
        # Bear in mind that because the first value of the image matrix is start and second value is down limit.
        # Select the limits so that it gets the line not too close and not too far, and the minimum portion possible
        # To make process faster.
        cv_image = cv2.resize(cv_image, (320, 160), interpolation = cv2.INTER_AREA)

        height, width, channels = cv_image.shape
        rows_to_watch = 40
        crop_img = cv_image[int(height/2):int(height/2+rows_to_watch)][1:width]

        
        # Convert from RGB to HSV
        hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array([20,100,100])
        upper_yellow = np.array([50,255,255])
        #lower_green = np.array([55,100,100])
        #upper_green = np.array([85,255,255])
        lower_magenta= np.array([140,100,100])
        upper_magenta= np.array([170,255,255])

        # Threshold the HSV image to get only yellow colors
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Calculate centroid of the blob of binary image using ImageMoments
        m = cv2.moments(mask, False)
        try:
            cx, cy = m['m10']/m['m00'], m['m01']/m['m00']
        except ZeroDivisionError:
            cy, cx = height / 2, width / 2

        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(crop_img,crop_img, mask= mask)
        
        # Draw the centroid in the resultut image
        # cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]]) 
        cv2.circle(res,(int(cx), int(cy)), 10,(0,0,255),-1)

        # cv2.imshow("Original", cv_image)
        # cv2.imshow("HSV", hsv)
        # cv2.imshow("MASK", mask)
        cv2.imshow("RES", res)

        cv2.waitKey(1)

        error_x = cx - width / 2 + self.bias;
        twist_object = Twist();
        twist_object.linear.x = 0.1;
        twist_object.angular.z = -(error_x / 100) * 0.1;
        print(cx, error_x, twist_object.angular.z)
        self._pub.publish(twist_object)

    def clean_up(self):
        cv2.destroyAllWindows()

def main():

    parser = argparse.ArgumentParser(description='Line Following')
    parser.add_argument('--camera', help='camera position', choices=['left', 'center', 'right'], type=str, required=True)
    parser.add_argument('--env', help='environment', choices=['gazebo', 'turtlebot'], type=str, required=True)
    args = parser.parse_args()

    print('Following the {0} of the lane'.format(args.camera))

    rospy.init_node('line_following_node', anonymous=True)
    
    line_follower_object = LineFollower()

    if args.camera == 'left':
        line_follower_object.bias = -40
    if args.camera == 'center':
        line_follower_object.bias = 0
    if args.camera == 'right':
        line_follower_object.bias = 40

    if args.env == 'gazebo':
        image_topic = "/camera/rgb/image_raw"
    if args.env == 'turtlebot':       
        image_topic = "/raspicam_node/image_raw"

    line_follower_object.image_sub = rospy.Subscriber(image_topic, Image,line_follower_object.camera_callback)
    line_follower_object._pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

    rate = rospy.Rate(5)

    def shutdownhook():
        # works better than the rospy.is_shut_down()
        line_follower_object.clean_up()
        rospy.loginfo("shutdown time!")
    
    rospy.on_shutdown(shutdownhook)
    
    # Spin until ctrl + c
    rospy.spin()
    
if __name__ == '__main__':
    main()
