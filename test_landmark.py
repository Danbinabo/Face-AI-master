from flask import Flask, request
import cv2
import dlib

from object_detection.utils import ops as utils_ops
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2

import tensorflow as tf
import numpy as np

PATH_TO_FROZEN_GRAPH = "face_detection_model.pb"
PATH_TO_LABELS = "object_detection/face_label_map.pbtxt"
IMAGE_SIZE = (256, 256)
detection_sess = tf.Session()
face_recognition_sess = tf.Session()

with detection_sess.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes', 'detection_masks'
        ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                    tensor_name)
        if 'detection_masks' in tensor_dict:
            # The following processing is only for single image
            detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
            detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
            # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
            real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, IMAGE_SIZE[0], IMAGE_SIZE[1])
            detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
            # Follow the convention by adding back the batch dimension
            tensor_dict['detection_masks'] = tf.expand_dims(
                detection_masks_reframed, 0)
        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')


##################################
###face feature
face_feature_sess = tf.Session()
ff_pb_path = "face_recognition_model.pb"
with face_feature_sess.as_default():
    ff_od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(ff_pb_path, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

        ff_images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        ff_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        ff_embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")


face_landmark_sess = tf.Session()
ff_pb_path = "landmark.pb"
with face_landmark_sess.as_default():
    ff_od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(ff_pb_path, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
        landmark_tensor = tf.get_default_graph().\
            get_tensor_by_name("fully_connected_9/Relu:0")

################

# 加载人脸关键点检测模型

predictor = dlib.shape_predictor('/home/kuan/cc/muke/dataset/'
                                 'shape_predictor_68_face_landmarks.dat')

detector = dlib.get_frontal_face_detector()

def face_landmark():
    #实现图片上传
    im_data = cv2.imread("tmp/tmp_landmark.jpg")

    sp = im_data.shape ##

    im_data_re = cv2.resize(im_data, IMAGE_SIZE)

    output_dict = detection_sess.run(tensor_dict,
                                     feed_dict={image_tensor:
                                         np.expand_dims(
                                             im_data_re, 0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]

    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0

    for i in range(len(output_dict['detection_scores'])):
        if output_dict['detection_scores'][i] > 0.1:
            bbox = output_dict['detection_boxes'][i]
            y1 = bbox[0]
            x1 = bbox[1]
            y2 = (bbox[2])
            x2 = (bbox[3])
            print(output_dict['detection_scores'][i], x1, y1, x2, y2)
            ##提取人脸区域
            y1 = int((y1 + (y2 - y1) * 0.2) * sp[0])
            x1 = int(x1 * sp[1])
            y2 = int(y2 * sp[0])
            x2 = int(x2 * sp[1])

            face_data = im_data[y1:y2, x1:x2]
            cv2.imwrite("face_landmark.jpg", face_data)
            face_data = cv2.resize(face_data, (128, 128))

            pred = face_landmark_sess.run(landmark_tensor, {"Placeholder:0":
                                   np.expand_dims(face_data, 0)})

            pred = pred[0]
            #cv2.imwrite("0_landmark.jpg", face_data)
            res = []
            ##裁剪之后的人脸框中的坐标
            ##
            for i in range(0, 136, 2):
                res.append(str((pred[i] * (x2 - x1) + x1) / sp[1]))
                res.append(str((pred[i + 1] * (y2 - y1) + y1) / sp[0]))

            res = ",".join(res)
            print(res)
            return res

    return "error"



if __name__ == '__main__':
    face_landmark()
