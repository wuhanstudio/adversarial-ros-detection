#Import Relevant Libraries
import argparse

# rospy for the subscriber
import rospy

# ROS Image message
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError

# Image Processing
from PIL import Image as PImage

# ROS Image message -> OpenCV2 image converter
import cv2
from io import BytesIO
import base64
from cv_bridge import CvBridge

import numpy as np
import time

# from tensorflow.keras.models import load_model

class CVDetector():

    def __init__(self, image_topic, net, output_layers, classes, confidence_threashold=0.5):
        self.net = net
        self.output_layers = output_layers
        self.classes = classes
        self.confidence_threshold = confidence_threshold

        # Instantiate CvBridge
        self.bridge = CvBridge()

        rospy.Subscriber(image_topic, Image, self.image_callback)
        self.input_pub = rospy.Publisher('/input_img', String, queue_size=10)
        self.adv_pub = rospy.Publisher('/adv_img', String, queue_size=10)

    def publish_image(self, cv_image, pub_topic):
        # _, buffer = cv2.imencode('.jpg', cv_image)
        img = PImage.fromarray(np.uint8(cv_image))
        b, g, r = img.split()
        img = PImage.merge("RGB", (r, g, b))
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        image_as_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        pub_topic.publish(image_as_str)

    def image_callback(self, msg):
        font = cv2.FONT_HERSHEY_SIMPLEX
        start_time = int(time.time() * 1000)
        try:
            # Convert your ROS Image message to OpenCV2
            cv2_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            img = cv2.resize(cv2_img, (320, 160), interpolation = cv2.INTER_AREA)

            # Publish the model input image
            self.publish_image(img, self.input_pub)

            #get image shape
            height, width, channels = img.shape

            # Detecting objects (YOLO)
            blob = cv2.dnn.blobFromImage(img, 1./255, (320, 160), (0, 0, 0), False, crop=False)
            self.net.setInput(blob)
            outs = self.net.forward(self.output_layers)
            # print(outs)

            # Showing informations on the screen (YOLO)
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    # print(detection.shape)
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > confidence_threshold:
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]]) + "=" + str(round(confidences[i]*100, 2)) + "%"
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 2)
                    cv2.putText(img, label, (x, y), font, 0.5, (255,0,0), 2)
            elapsed_time = int(time.time()*1000) - start_time
            fps = 1000 / elapsed_time
            print ("fps: ", str(round(fps, 2)))
            cv2.imshow("Image", img)
            # Publish the output image
            self.publish_image(img, self.adv_pub)

            cv2.waitKey(1)

        except CvBridgeError as e:
            print(e)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data Collection')
    parser.add_argument('--env', help='environment', choices=['gazebo', 'turtlebot'], type=str, required=True)
    args = parser.parse_args()
    if args.env == 'gazebo':
        image_topic = "/camera/rgb/image_raw"
    if args.env == 'turtlebot':       
        image_topic = "/raspicam_node/image_raw"

    rospy.init_node('cv_detector')

    # Spin until ctrl + c
    '''Load YOLO (YOLOv3 or YOLOv4-Tiny)'''
    #net = cv2.dnn.readNet("yolov3_training_last.weights", "yolov3_training.cfg")
    net = cv2.dnn.readNet("yolov4-tiny-traffic_final.weights", "yolov4-tiny-traffic.cfg")

    classes = []
    with open("obj.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # get last layers names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    confidence_threshold = 0.5

    detector = CVDetector(image_topic, net, output_layers, classes)

    # Spin until ctrl + c
    rospy.spin()
