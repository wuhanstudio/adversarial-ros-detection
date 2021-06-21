import argparse
import time
from tensorflow.python.ops.control_flow_ops import cond

from tensorflow.python.ops.gen_lookup_ops import hash_table
from tensorflow.python.ops.init_ops import he_normal

# rospy for the subscriber
import rospy

# ROS message
from sensor_msgs.msg import Image
from std_msgs.msg import String

# Image Processing
from PIL import Image as PImage

# ROS Image message -> OpenCV2 image converter
import cv2
from io import BytesIO
import base64
from cv_bridge import CvBridge

# Deep Learning Libraries
import numpy as np
np.set_printoptions(suppress=True)
from keras.models import load_model
from scipy.special import expit, softmax
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import keras.backend as K

class RosTensorFlow():
    def __init__(self, model, image_topic):
        self.epsilon = 1
        self.graph = tf.compat.v1.get_default_graph()

        self.model = load_model(model)
        self.model.summary()

        self.sess = tf.compat.v1.keras.backend.get_session()

        self._cv_bridge = CvBridge()

        # Input Image
        self.input_sub = rospy.Subscriber(image_topic, Image, self.input_callback, queue_size=10)

        # Publish images to the web UI
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

    def attack_callback(self, attack_msg):
        self.attack = attack_msg.data
        print('Attack Type:', self.attack)

    def input_callback(self, input_cv_image):
        classes = ["traffic"]
        confidence_threshold = 0.1
        font = cv2.FONT_HERSHEY_SIMPLEX

        start_time = int(time.time() * 1000)

        input_cv_image = self._cv_bridge.imgmsg_to_cv2(input_cv_image, "bgr8")
        input_cv_image = cv2.resize(input_cv_image, (320, 160), interpolation = cv2.INTER_AREA)

        # Publish the model input image
        self.publish_image(input_cv_image, self.input_pub)
        
        # get image shape
        height, width, channels = input_cv_image.shape

        input_cv_image = input_cv_image.astype(np.float32) / 255.0

        with self.graph.as_default():

            outs = self.sess.run(self.model.output, feed_dict={self.model.input:np.array([input_cv_image])})
            # Showing informations on the screen (YOLO)
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                anchors = [[12., 16.], [19., 36.], [40., 28.]]
                # anchors = [[10., 14.],  [23., 27.],  [37., 58.]]
                num_anchors = int(out.shape[-1] / (5+len(classes)))
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
                box_wh = (np.exp(out[..., 2:4]) * anchors) / np.array((160, 320))[::-1]

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

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    x = x - w / 2
                    y = y - h / 2
                    x = int(x * 320 ) 
                    y = int(y * 160)
                    w = int(w * 320) 
                    h = int(h * 160) 
                    # w = int(w * 320 * 1.5) 
                    # h = int(h * 160 * 1.5) 
                    label = str(classes[class_ids[i]]) + "=" + str(round(confidences[i]*100, 2)) + "%"
                    cv2.rectangle(input_cv_image, (x, y), (x + w, y + h), (255,0,0), 2)
                    cv2.putText(input_cv_image, label, (x, y), font, 0.5, (255,0,0), 2)

        elapsed_time = int(time.time()*1000) - start_time
        fps = 1000 / elapsed_time
        print ("fps: ", str(round(fps, 2)))
        cv2.imshow("Image", input_cv_image)
        # Publish the output image
        self.publish_image(input_cv_image, self.adv_pub)

        cv2.waitKey(1)

    def main(self):
        rospy.spin()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Line Following')
    parser.add_argument('--env', help='environment', choices=['gazebo', 'turtlebot'], type=str, required=True)
    parser.add_argument('--model', help='deep learning model', type=str, required=True)
    args = parser.parse_args()

    rospy.init_node('ros_object_detection')

    if args.env == 'gazebo':
        image_topic = "/camera/rgb/image_raw"
    if args.env == 'turtlebot':       
        image_topic = "/raspicam_node/image_raw"

    tensor = RosTensorFlow(args.model, image_topic);
    tensor.main()
