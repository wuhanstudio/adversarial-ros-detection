### Training the model (Optional)

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
$ ./darknet detector train data/obj.data cfg/yolov3-tiny-traffic-3.cfg yolov3-tiny.conv.11
```

Finally, we can convert the darknet *.weights file to keras *.h5 file:

```
$ cd model/utils/keras-YOLOv3-model-set
$ python3 tools/model_converter/convert.py cfg/yolov3-tiny-traffic-3.cfg weights/yolov3-tiny-traffic-3.weights weights/yolov3-tiny-traffic-3.h5
```

##### Training darknet yolov4-tiny

```
$ ./darknet detector train data/obj.data cfg/yolov4-tiny-traffic-3.cfg yolov4-tiny.conv.29
```

Similarly, we can convert the darknet *.weights file to keras *.h5 file:

```
$ cd model/utils/keras-YOLOv3-model-set
$ python3 tools/model_converter/convert.py cfg/yolov4-tiny-traffic-3.cfg weights/yolov4-tiny-traffic-3.weights weights/yolov4-tiny-traffic-3.h5
```

##### Training yolov4 mobilenet lite

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
python3 yolo.py --model_type=yolo4_mobilenet_lite --weights_path=logs/000/trained_final.h5 --anchors_path=configs/yolo4_anchors.txt --classes_path=obj.names --model_image_size=160x320 --dump_model --output_model_file=yolov4_mobilenet_lite.h5
```

Test the model:

```
python3 yolo.py --model_type=yolo4_mobilenet_lite --weights_path=yolov4_mobilenet_lite.h5 --anchors_path=configs/yolo4_anchors.txt --classes_path=obj.names --model_image_size=160x320 --image
```
