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

if __name__ == '__main__':
    confidence_threshold = 0.5
    '''Load YOLO (YOLOv3 or YOLOv4-Tiny)'''
    net = cv2.dnn.readNet("yolov3-tiny-traffic-3_final.weights", "yolov3-tiny-traffic-3.cfg")
    # net = cv2.dnn.readNet("yolov4-tiny-traffic-3_final.weights", "yolov4-tiny-traffic-3.cfg")

    classes = []
    with open("obj.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # get last layers names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    font = cv2.FONT_HERSHEY_SIMPLEX
    cam = cv2.VideoCapture(2)
    while True:
        start_time = int(time.time() * 1000)
        try:
            # Convert your ROS Image message to OpenCV2
            ret, frame = cam.read()
            if not ret:
                print("failed to grab frame")
                break
            img = cv2.resize(frame, (320, 160), interpolation = cv2.INTER_AREA)

            #get image shape
            height, width, channels = img.shape

            # Detecting objects (YOLO)
            blob = cv2.dnn.blobFromImage(img, 1./255, (320, 160), (0, 0, 0), False, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)
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

            cv2.waitKey(1)

        except CvBridgeError as e:
            print(e)