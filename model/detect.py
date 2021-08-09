import argparse
import time

# rospy for the subscriber
import rospy

# ROS message
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_msgs.msg import Int32MultiArray
from std_msgs.msg import Int32

# Image Processing
from PIL import Image as PImage

# ROS Image message -> OpenCV2 image converter
import cv2
from io import BytesIO
import base64

# Deep Learning Libraries
import numpy as np
np.set_printoptions(suppress=True)
from keras.models import load_model
from scipy.special import expit, softmax
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import keras.backend as K

from datetime import datetime

class RosTensorFlow():
    def __init__(self, model, attack_type, monochrome, image_topic):
        self.epsilon = 1
        self.graph = tf.compat.v1.get_default_graph()
        self.monochrome = monochrome

        if self.monochrome:
            self.noise = np.zeros((256, 320))
        else:
            self.noise = np.zeros((256, 320, 3))

        self.adv_patch_boxes = []
        self.fixed = False

        self.model = load_model(model)
        self.model.summary()
        self.attack_type = attack_type

        self.delta = None
        for out in self.model.output:
            # Targeted One Box
            if attack_type == "one_targeted":
                loss = K.max(K.sigmoid(K.reshape(out, (-1, 8))[:, 4]) * K.sigmoid(K.reshape(out, (-1, 8))[:, 5]))

            # Targeted Multi boxes
            if attack_type == "multi_targeted":
                loss = K.sigmoid(K.reshape(out, (-1, 8))[:, 4]) * K.sigmoid(K.reshape(out, (-1, 8))[:, 5])

            # Untargeted Multi boxes
            if attack_type == "multi_untargeted":
                loss = K.sigmoid(K.reshape(out, (-1, 8))[:, 4]) * K.sigmoid(K.reshape(out, (-1, 8))[:, 5]) + K.sigmoid(K.reshape(out, (-1, 8))[:, 4]) * K.sigmoid(K.reshape(out, (-1, 8))[:, 6]) + K.sigmoid(K.reshape(out, (-1, 8))[:, 4]) * K.sigmoid(K.reshape(out, (-1, 8))[:, 7])

            grads = K.gradients(loss, self.model.input)
            if self.delta == None:
                self.delta =  K.sign(grads[0])
            else:
                self.delta = self.delta + K.sign(grads[0])

        # Store current patches
        self.patches = []

        # loss = K.sum(K.abs((self.model.input-K.mean(self.model.input))))
        loss = - 0.01 * tf.reduce_sum(tf.image.total_variation(self.model.input))

        # Mirror
        # loss = - 0.01 * tf.reduce_sum(tf.image.total_variation(self.model.input)) - 0.01 * tf.reduce_sum(K.abs(self.model.input - tf.image.flip_left_right(self.model.input)))
        grads = K.gradients(loss, self.model.input)
        self.delta = self.delta + K.sign(grads[0])

        self.iter = 0

        self.sess = tf.compat.v1.keras.backend.get_session()

        # Input Image
        self.input_sub = rospy.Subscriber(image_topic, Image, self.input_callback, queue_size=10)

        # Adversarial Patch box
        self.adv_clear_patch = rospy.Subscriber("/clear_patch", Int32, self.clear_patch_callback, queue_size=10)
        self.adv_fix_patch = rospy.Subscriber("/fix_patch", Int32, self.fix_patch_callback, queue_size=10)
        self.adv_patch_box = rospy.Subscriber("/adv_patch", Int32MultiArray, self.patch_callback, queue_size=10)

        # Publish images to the web UI
        self.input_pub = rospy.Publisher('/input_img', String, queue_size=10)
        self.adv_pub = rospy.Publisher('/adv_img', String, queue_size=10)
        self.patch_pub = rospy.Publisher('/perturb_img', String, queue_size=10)

        # Detection result
        self.detect_pub = rospy.Publisher('/detect', Int32, queue_size=10)

        # This preloads the graph but then it takes more steps to iterate
        #with self.graph.as_default():
        #    _ = self.sess.run(self.delta, feed_dict={self.model.input:np.array([self.noise])})

    def publish_image(self, cv_image, pub_topic):
        # _, buffer = cv2.imencode('.jpg', cv_image)
        img = PImage.fromarray(np.uint8(cv_image))
        b, g, r = img.split()
        img = PImage.merge("RGB", (r, g, b))
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        image_as_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        pub_topic.publish(image_as_str)

    def attack_callback(self, attack_msg):
        self.attack = attack_msg.data
        print('Attack Type:', self.attack)

    def fix_patch_callback(self, clear_msg):
        if(clear_msg.data > 0):
            self.fixed = True
            self.patches = []
            patch_cv_image = np.zeros((256, 320, 3))
            # patch_cv_image = cv2.resize(patch_cv_image, (320, 256), interpolation = cv2.INTER_AREA)
            for box in self.adv_patch_boxes:
                if self.monochrome:
                    patch_cv_image[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2]), 0] = self.noise[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2])]
                    patch_cv_image[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2]), 1] = self.noise[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2])]
                    patch_cv_image[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2]), 2] = self.noise[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2])]
                else:
                    patch_cv_image[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2]), :] = self.noise[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2]), :]
                self.patches.append(self.noise[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2])])
            # Publish the patch image
            self.publish_image(patch_cv_image * 255.0, self.patch_pub)
        else:
            self.fixed = False

    def clear_patch_callback(self, clear_msg):
        if(clear_msg.data > 0):
            self.adv_patch_boxes = []
            self.patches = []
            if self.monochrome:
                self.noise = np.zeros((256, 320))
            else:
                self.noise = np.zeros((256, 320, 3))
            self.iter = 0

    def patch_callback(self, attack_msg):
        box = attack_msg.data[1:]
        if(attack_msg.data[0] < 0):
            self.adv_patch_boxes.append(box)
            self.iter = 0
        else:
            self.adv_patch_boxes[attack_msg.data[0]] = box

    def input_callback(self, input_cv_image):
        classes = ["40", "stop", "20"]
        confidence_threshold = 0.01
        font = cv2.FONT_HERSHEY_SIMPLEX

        start_time = int(time.time() * 1000)

        input_cv_image = np.frombuffer(input_cv_image.data, dtype=np.uint8).reshape(input_cv_image.height, input_cv_image.width, -1)
        input_cv_image = PImage.fromarray(np.uint8(input_cv_image))
        r, g, b = input_cv_image.split()
        input_cv_image = np.array(PImage.merge("RGB", (b, g, r)))

        input_cv_image = cv2.resize(input_cv_image, (320, 256), interpolation = cv2.INTER_AREA)

        # Publish the model input image
        self.publish_image(cv2.resize(input_cv_image, (320, 256), interpolation = cv2.INTER_AREA), self.input_pub)
        
        # get image shape
        height, width, channels = input_cv_image.shape

        input_cv_image = input_cv_image.astype(np.float32) / 255.0

        with self.graph.as_default():
            if not self.fixed:
                for box in self.adv_patch_boxes:
                    if self.monochrome:
                        input_cv_image[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2]), 0] = self.noise[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2])]
                        input_cv_image[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2]), 1] = self.noise[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2])]
                        input_cv_image[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2]), 2] = self.noise[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2])]
                    else:
                        input_cv_image[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2]), :] = self.noise[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2]), :]
            else:
                ib = 0
                for box in self.adv_patch_boxes:
                    if self.monochrome:
                        input_cv_image[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2]), 0] = self.patches[ib]
                        input_cv_image[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2]), 1] = self.patches[ib]
                        input_cv_image[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2]), 2] = self.patches[ib]
                    else:
                        input_cv_image[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2]), :] = self.patches[ib]
                    ib = ib + 1
            if(len(self.adv_patch_boxes) > 0 and (not self.fixed)):
                grads = self.sess.run(self.delta, feed_dict={self.model.input:np.array([input_cv_image])}) / 255.0
                if self.monochrome:
                    self.noise = self.noise + 5 / 3 * (grads[0, :, :, 0] + grads[0, :, :, 1] + grads[0, :, :, 2])
                else:
                    self.noise = self.noise + 5 * grads[0, :, :, :]
                self.iter = self.iter + 1

            outs = self.sess.run(self.model.output, feed_dict={self.model.input:np.array([input_cv_image])})

            # Showing informations on the screen (YOLO)
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                anchors = [[12., 16.], [19., 36.], [40., 28.]]
                # anchors = [[10., 14.],  [23., 27.],  [37., 58.]]
                num_anchors = 3
                grid_size = np.shape(out)[1:3]
                out = out.reshape((-1, 5+len(classes)))
                # generate x_y_offset grid map
                grid_y = np.arange(grid_size[0])
                grid_x = np.arange(grid_size[1])
                x_offset, y_offset = np.meshgrid(grid_x, grid_y)

                x_offset = np.reshape(x_offset, (-1, 1))
                y_offset = np.reshape(y_offset, (-1, 1))

                x_y_offset = np.concatenate((x_offset, y_offset), axis=1)
                x_y_offset = np.tile(x_y_offset, (1, num_anchors))
                x_y_offset = np.reshape(x_y_offset, (-1, 2))

                anchors = np.tile(anchors, (grid_size[0] * grid_size[1], 1))

                box_xy = (expit(out[..., :2]) + x_y_offset) / np.array(grid_size)[::-1]
                box_wh = (np.exp(out[..., 2:4]) * anchors) / np.array((height, width))[::-1]

                scores = expit(out[:, 5:])
                class_id = np.argmax(scores, axis=1)
                confidence = scores[class_id][:, 0] * expit(out[:, 4])

                box_xy = box_xy[confidence > confidence_threshold]
                box_wh = box_wh[confidence > confidence_threshold]
                class_id = class_id[confidence > confidence_threshold]
                confidence = confidence[confidence > confidence_threshold]

                if(len(confidence) > 0):
                    box_tmp = list(np.concatenate((box_xy, box_wh), axis=1))
                    for b in box_tmp:
                        boxes.append(b)
                    for c in confidence:
                        confidences.append(float(c))
                    for c in class_id:
                        class_ids.append(c)
                    # if(len(confidence > 1)):
                        # now = datetime.now()
                        # current_time = now.strftime("%H-%M-%S-%f")
                        # cv2.imwrite(current_time + '.jpg', self.noise * 255.0)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            detected = False
            for i in range(len(boxes)):
                if i in indexes:
                    detected = True
                    self.detect_pub.publish(class_ids[i]+1)

                    x, y, w, h = boxes[i]
                    x = x - w / 2
                    y = y - h / 2
                    x = int(x * width ) 
                    y = int(y * height)
                    w = int(w * width) 
                    h = int(h * height) 
                    label = str(classes[class_ids[i]]) + "=" + str(round(confidences[i]*100, 2)) + "%"
                    print(label)
                    cv2.rectangle(input_cv_image, (x, y), (x + w, y + h), (255,0,0), 2)
                    cv2.putText(input_cv_image, label, (x, y), font, 0.5, (255,0,0), 2)

            if not detected:
                self.detect_pub.publish(0)

        elapsed_time = int(time.time()*1000) - start_time
        fps = 1000 / elapsed_time
        print ("fps: ", str(round(fps, 2)))
        # cv2.imshow("Image", input_cv_image)
        # Publish the output image
        self.publish_image(cv2.resize(input_cv_image, (320, 256), interpolation = cv2.INTER_AREA) * 255.0, self.adv_pub)

        cv2.waitKey(1)

    def main(self):
        rospy.spin()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Object Detection')
    parser.add_argument('--env', help='environment', choices=['camera', 'gazebo', 'turtlebot'], type=str, required=True)
    parser.add_argument('--model', help='deep learning model', type=str, required=True)
    parser.add_argument('--attack', help='adversarial attacks type', choices=['one_targeted', 'multi_targeted', 'multi_untargeted'], type=str, required=False, default="multi_untargeted")
    parser.add_argument('--monochrome', action='store_true', help='monochrome patch')
    args = parser.parse_args()

    rospy.init_node('ros_object_detection')

    # We can also read images from usb_cam
    # rosrun usb_cam usb_cam_node _video_device:=/dev/video0 _image_width:=320 _image_height:=256 _pixel_format:=yuyv
    if args.env == 'camera':
        image_topic = "/usb_cam/image_raw"
    if args.env == 'gazebo':
        image_topic = "/camera/rgb/image_raw"
    if args.env == 'turtlebot':       
        image_topic = "/raspicam_node/image_raw"

    tensor = RosTensorFlow(args.model, args.attack, args.monochrome, image_topic);
    tensor.main()
