## Adversarial Detection in ROS

> Attacking End-to-End Object Detection Systems

- [Adversarial Detection in ROS](#adversarial-detection-in-ros)
  * [Overview](#overview)
  * [Quick Start](#quick-start)
    + [Step 0: Prerequisites](#step-0--prerequisites)
    + [Step 1: Setup  the TurtleBot](#step-1--setup--the-turtlebot)
    + [Step 2: Setup the server](#step-2--setup-the-server)
    + [Step 3: Setup the browser](#step-3--setup-the-browser)
  * [Training the model](#training-the-model)
    + [Step 1: Collect the Data](#step-1--collect-the-data)
    + [Step 2: Train the model](#step-2--train-the-model)
      - [Training darknet yolov3-tiny](#training-darknet-yolov3-tiny)
      - [Training darknet yolov4-tiny](#training-darknet-yolov4-tiny)
      - [Train yolov4 mobilenet lite:](#train-yolov4-mobilenet-lite)

### Overview

![](doc/adversarial-ros-driving.png)



### Quick Start

#### Step 0: Prerequisites

```
$ sudo apt install ros-noetic-desktop
$ sudo apt install ros-noetic-rosbridge-suite ros-noetic-turtlebot3-simulations ros-noetic-turtlebot3-gazebo
```

#### Step 1: Setup  the TurtleBot

```
$ cd ros_ws
$ rosdep install --from-paths src --ignore-src -r -y
$ catkin_make
$ source devel/setup.sh
$ export TURTLEBOT3_MODEL=waffle
$ roslaunch turtlebot3_lane turtlebot3_lane_traffic_signs.launch
```

#### Step 2: Setup the server

```
$ roslaunch rosbridge_server rosbridge_websocket.launch
$ cd model

# You may need to put the turtlebot on track first
# rosrun teleop_twist_keyboard teleop_twist_keyboard.py

# For real turtlebot3
$ python3 drive.py --env turtlebot --model yolov4-tiny-traffic.h5
# For Gazebo Simulator
$ python3 drive.py --env gazebo --model yolov4-tiny-traffic.h5
```

#### Step 3: Setup the browser

This is just a website, your can use any web server, just serve all the content under **client/web**.

The client is built as a single executable file.

```
$ ./client
```

For Linux and Mac, or other Unix, the server can be built with:

```
$ go get -u github.com/gobuffalo/packr/packr
$ go get github.com/gobuffalo/packr@v1.30.1
$ packr build
```

The web page will be available at: http://localhost:3333/

That's it!



### Training the model

#### Step 1: Collect the Data

The following script collects image data from the topic **/camera/rgb/image_raw** and corresponding control command in **/cmd_vel**. The log file is saved  in **driving_log.csv**, and images are saved in **IMG/** folder

```
$ cd model/data
$ # Collect center camera data
$ python3 line_follow.py --camera center --env gazebo
$ python3 ros_collect_data.py --camera center --env gazebo
```

#### Step 2: Train the model

Once the data is collected, we need to label images before training the model. This tool [labelimg](https://github.com/tzutalin/labelImg) is used for labelling.

##### Training darknet yolov3-tiny

After labelling, you can use the following script to generate the training and testing set.

```
$ python3 train_test_split.py
```

Now we are ready to train the model using the darknet framework:

```
$ git submodule init
$ git submodule update

$ cd model/utils/darknet
$ # If you have a GPU, you may need to change the Makefile
$ # Compiling the darknet binary
$ make

$ cp -r ../darknet-conf/one-class/* ./
$ ./darknet detector train data/obj.data cfg/yolov3-tiny-traffic.cfg yolov3-tiny.conv.11
```

Finally, we can convert the darknet *.weights file to keras *.h5 file:

```
$ cd model/utils/keras-YOLOv3-model-set
$ python3 tools/model_converter/convert.py cfg/yolov3-tiny-traffic.cfg weights/yolov3-tiny-traffic.weights weights/yolov3-tiny-traffic.h5
```

##### Training darknet yolov4-tiny

```
$ ./darknet detector train data/obj.data cfg/yolov4-tiny-traffic.cfg yolov4-tiny.conv.29
```

Similarly, we can convert the darknet *.weights file to keras *.h5 file:

```
$ cd model/utils/keras-YOLOv3-model-set
$ python3 tools/model_converter/convert.py cfg/yolov4-tiny-traffic.cfg weights/yolov4-tiny-traffic.weights weights/yolov4-tiny-traffic.h5
```

##### Train yolov4 mobilenet lite

Before training the model, we need to convert the format of annotation files:

```
From:
class_id x_center y_center width height
0 0.625000 0.346875 0.081250 0.143750

To:
image_file_path x_min,y_min,x_max,y_max,class_id
path/to/img2.jpg 120,300,250,600,2
```

This can be done using the following script that generates a **trainval.txt** file:

```
$ cd model/utils/keras-YOLOv3-model-set/
$ cp -r ../../data/IMG ../../data/*.txt ./
$ cp ../keras-conf/* ./
$ python3 darknet_to_keras_trainval.py
```

Finally, we can train the model:

```
$ python3 train.py --model_image_size=160x320 --model_type=yolo4_mobilenet_lite --anchors_path=configs/yolo4_anchors.txt --annotation_file=trainval.txt --classes_path=obj.names --eval_online --save_eval_checkpoint
```

Save the model as keras:

```
python3 yolo.py --model_type=yolo4_mobilenet_lite --weights_path=logs/000/REPLACE_WITH_YOUR_CHECK_POINT.h5 --anchors_path=configs/yolo4_anchors.txt --classes_path=obj.names --model_image_size=160x320 --dump_model --output_model_file=yolov4_mobilenet_lite.h5
```

Test the model:

```
python3 yolo.py --model_type=yolo4_mobilenet_lite --weights_path=yolov4_mobilenet_lite.h5 --anchors_path=configs/yolo4_anchors.txt --classes_path=obj.names --model_image_size=160x320 --image
```

