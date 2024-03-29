from flask import Flask, request

from object_detection.utils import ops as utils_ops
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2
from gevent import monkey
monkey.patch_all()
import tensorflow as tf
import numpy as np

app = Flask(__name__)

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

########
face_recognition_sess = tf.Session()
PATH_FR_GRAPH = "/home/kuan/models/facenet/FaceV5/graph.pb"
with face_recognition_sess.as_default():
    fr_od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_FR_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        fr_od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(fr_od_graph_def, name='')

        fr_images_placeholder = tf.get_default_graph().\
            get_tensor_by_name("input:0")
        fr_embeddings = tf.get_default_graph().\
            get_tensor_by_name("embeddings:0")
        fr_phase_train_placeholder = tf.get_default_graph().\
            get_tensor_by_name("phase_train:0")



@app.route("/")
def helloword():
    return '<h1>Hello World!</h1>'

@app.route('/upload', methods=['POST', 'GET'])
def upload():
    f = request.files.get('file')
    print(f)
    upload_path = os.path.join("tmp/tmp." + f.filename.split(".")[-1])
    # secure_filename(f.filename))  #注意：没有的文件夹一定要先创建，不然会提示没有该路径
    print(upload_path)
    f.save(upload_path)
    return upload_path

@app.route("/face_detect")
def inference():

    im_url = request.args.get("url")

    im_data = cv2.imread(im_url)
    sp = im_data.shape
    im_data = cv2.resize(im_data, IMAGE_SIZE)
    output_dict = detection_sess.run(tensor_dict,
                                     feed_dict={image_tensor:
                                                    np.expand_dims(
                                                        im_data, 0)})

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
            y1 =   bbox[0]
            x1 =  bbox[1]
            y2 =   (bbox[2])
            x2 = (bbox[3])
            print(output_dict['detection_scores'][i], x1, y1, x2, y2)

    return str([x1, y1, x2, y2])

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y

@app.route("/face_recognition")
def face_recognition():
    im_data = cv2.imread("/home/kuan/code/mooc_py3_tensorflow/dataset/64_CASIA-FaceV5/image/000/000_2.bmp")


    im_data = np.expand_dims(prewhiten(cv2.resize(im_data, (160, 160))), axis=0)
    print(im_data.shape)
    # Run forward pass to calculate embeddings
    feed_dict = {fr_images_placeholder: im_data,
                 fr_phase_train_placeholder: False}
    emb1 = face_recognition_sess.run(fr_embeddings,
                                    feed_dict=feed_dict)

    im_data = cv2.imread("/home/kuan/code/mooc_py3_tensorflow/dataset/"
                         "64_CASIA-FaceV5/crop_image_160/2/2_0001.jpg")
    im_data = np.expand_dims(prewhiten(cv2.resize(im_data, (160, 160))), axis=0)
    print(im_data.shape)
    # Run forward pass to calculate embeddings
    feed_dict = {fr_images_placeholder: im_data,
                 fr_phase_train_placeholder: False}
    emb2 = face_recognition_sess.run(fr_embeddings,
                                    feed_dict=feed_dict)

    print(emb1)
    dist = np.linalg.norm(emb1 - emb2)
    return str(dist) #str(emb2[0])
if __name__ == '__main__':
    app.run(host="192.168.0.104", port=90, debug=True)

