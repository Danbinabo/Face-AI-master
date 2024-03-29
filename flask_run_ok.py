from flask import Flask, request
import cv2
import dlib
import os
import tensorflow as tf
import numpy as np

app = Flask(__name__)

"""人脸属性识别"""
face_attribute_sess = tf.Session()
ff_pb_path = "face_attribute.pb"
with face_attribute_sess.as_default():
    face_attri_od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(ff_pb_path, 'rb') as fid:
        serialized_graph = fid.read()
        face_attri_od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(face_attri_od_graph_def, name='')

        pred_eyeglasses = tf.get_default_graph().get_tensor_by_name("ArgMax:0")
        pred_young = tf.get_default_graph().get_tensor_by_name("ArgMax_1:0")
        pred_male = tf.get_default_graph().get_tensor_by_name("ArgMax_2:0")
        pred_smiling = tf.get_default_graph().get_tensor_by_name("ArgMax_3:0")

        face_attribute_image_tensor = tf.get_default_graph().get_tensor_by_name("Placeholder:0")

"""人脸检测模型"""
PATH_TO_FROZEN_GRAPH = "face_detection_model.pb"
PATH_TO_LABELS = "face_label_map.pbtxt"
IMAGE_SIZE = (256, 256)
detection_sess = tf.Session()
face_recognition_sess = tf.Session()

with detection_sess.as_default():
    face_detect_od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        face_detect_od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(face_detect_od_graph_def, name='')
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
                      # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
            real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])

        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')



"""face feature"""
face_feature_sess = tf.Session()
ff_pb_path = "face_recognition_model.pb"
with face_feature_sess.as_default():
    face_reco_od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(ff_pb_path, 'rb') as fid:
        serialized_graph = fid.read()
        face_reco_od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(face_reco_od_graph_def, name='')

        ff_images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        ff_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        ff_embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")

"""人脸关键点模型"""
face_landmark_sess = tf.Session()
ff_pb_path = "landmark.pb"
with face_landmark_sess.as_default():
    face_landmark_od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(ff_pb_path, 'rb') as fid:
        serialized_graph = fid.read()
        face_landmark_od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(face_landmark_od_graph_def, name='')
        landmark_tensor = tf.get_default_graph().\
            get_tensor_by_name("fully_connected_9/Relu:0")



# 加载人脸关键点检测模型
# predictor = dlib.shape_predictor('/home/kuan/cc/muke/dataset/'
#                                  'shape_predictor_68_face_landmarks.dat')
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()



@app.route('/face_attribute', methods=['POST', 'GET'])
def face_attribute():
    # 实现图片上传
    f = request.files.get('file')
    print(f)
    upload_path = os.path.join("tmp/tmp_attribute." + f.filename.split(".")[-1])
    # secure_filename(f.filename))  #注意：没有的文件夹一定要先创建，不然会提示没有该路径
    print(upload_path)
    f.save(upload_path)
    ##人脸检测
    im_data = cv2.imread(upload_path)
    rects = detector(im_data, 0)

    if len(rects) == 0:
        return "error"

    x1 = rects[0].left()
    y1 = rects[0].top()
    x2 = rects[0].right()
    y2 = rects[0].bottom()

    y1 = int(max(y1 - 0.3 * (y2 - y1), 0))
    im_data = im_data[y1:y2, x1:x2]

    im_data = cv2.resize(im_data, (128, 128))

    [eye_glass, young, male, smiling] = face_attribute_sess.run(
        [pred_eyeglasses, pred_young, pred_male, pred_smiling],
        feed_dict={face_attribute_image_tensor: np.expand_dims(im_data, 0)})

    return "{},{},{},{}".format(eye_glass[0], young[0], male[0], smiling[0])



@app.route('/face_landmark_tf', methods=['POST', 'GET'])
def face_landmark():
    #实现图片上传
    f = request.files.get('file')
    print(f)
    upload_path = os.path.join("tmp/tmp_landmark." + f.filename.split(".")[-1])
    # secure_filename(f.filename))  #注意：没有的文件夹一定要先创建，不然会提示没有该路径
    print(upload_path)
    f.save(upload_path)
    ##人脸检测
    im_data = cv2.imread(upload_path)

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
    # 检测不到人脸--【0,0,0,0】
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



@app.route('/face_landmark', methods=['POST', 'GET'])
def face_landmark_dlib():
    ### 嘴巴，眼睛，casecade ###1）landmark 关键点定位，眼部关键点，粗位置，
    # 抠取，眼部关键点的回归。2）精细粒度眼部区域的回归
    # 回归模型，人脸区域， ---》 提取眼部区域，
    #
    # sudo pip3 install dlib
    #
    ## http://-----.jpg
    print("begin")
    f = request.files.get("file")
    upload_path = os.path.join("tmp/tmp_landmark." + f.filename.split(".")[-1])
    f.save(upload_path)
    ##
    print(upload_path)

    ##RGB
    im_data = cv2.imread(upload_path)
    im_data = cv2.cvtColor(im_data, cv2.COLOR_BGR2GRAY)
    sp = im_data.shape

    rects = detector(im_data, 0)
    res = []
    for face in rects:
        shape = predictor(im_data, face)
        for pt in shape.parts():
            pt_pos = (pt.x, pt.y)
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()

            ptx = (pt.x - x1) * 1.0 / (x2 - x1)
            pty = (pt.y - y1) *1.0 / (y2 - y1)

            res.append(str(ptx))
            res.append(str(pty))

            res.append(str(pt.x * 1.0 / sp[1]))
            res.append(str(pt.y *1.0 / sp[0]))

        if res.__len__() == 136 * 2:
            res = ",".join(res)
            print(res)
            return res

    return "error"



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
            y1 =   bbox[0]
            x1 =  bbox[1]
            y2 =   (bbox[2])
            x2 = (bbox[3])
            print(output_dict['detection_scores'][i], x1, y1, x2, y2)

    return str([x1, y1, x2, y2])

# 图像数据标准化
def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y

def read_image(path):
    ###
    im_data = cv2.imread(path)
    im_data = prewhiten(im_data)
    im_data = cv2.resize(im_data, (160, 160))
    #1 * h * w * 3
    return im_data

@app.route("/face_feature")
def face_feature():
    im_data1 = read_image("/home/kuan/code/mooc_py3_tensorflow/dataset/64_CASIA-FaceV5/"
               "crop_image_160/0/0_0000.jpg")

    im_data1 = np.expand_dims(im_data1, axis=0)

    emb1 = face_feature_sess.run(ff_embeddings, feed_dict={ff_images_placeholder:im_data1, ff_train_placeholder:False})

    strr = ",".join(str(i) for i in emb1[0])

    return strr

@app.route('/face_register_html', methods=['POST', 'GET'])
def face_register_html():
    ##实现图片上传
    f = request.files.get('imageFile')
    name = request.form.get("username")

    print(f)
    upload_path = os.path.join("tmp/tmp." + f.filename.split(".")[-1])
    # secure_filename(f.filename))  #注意：没有的文件夹一定要先创建，不然会提示没有该路径
    print(upload_path)
    f.save(upload_path)
    ##人脸检测
    im_data = cv2.imread(upload_path)
    sp = im_data.shape
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
            y1 = int(y1 * sp[0])
            x1 = int(x1 * sp[1])
            y2 = int(y2 * sp[0])
            x2 = int(x2 * sp[1])
            face_data = im_data[y1:y2, x1:x2]
            im_data = prewhiten(face_data) #预处理
            im_data = cv2.resize(im_data, (160, 160))
            im_data1 = np.expand_dims(im_data, axis=0)
            ##人脸特征提取
            emb1 = face_feature_sess.run(ff_embeddings,
                                         feed_dict={ff_images_placeholder: im_data1, ff_train_placeholder: False})

            strr = ",".join(str(i) for i in emb1[0])
            ##写去txt
            with open("face/{}_feature.txt".format(name), "w") as f:
                f.writelines(strr)
            f.close()
            mess = "success"
            break
        else:
            mess = "fail"

    return mess


@app.route('/face_register', methods=['POST', 'GET'])
def face_register():
    ##实现图片上传
    f = request.files.get('file')
    print(f)
    upload_path = os.path.join("tmp/tmp." + f.filename.split(".")[-1])
    # secure_filename(f.filename))  #注意：没有的文件夹一定要先创建，不然会提示没有该路径
    print(upload_path)
    f.save(upload_path)
    ##人脸检测
    im_data = cv2.imread(upload_path)
    sp = im_data.shape
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
            y1 = int(y1 * sp[0])
            x1 = int(x1 * sp[1])
            y2 = int(y2 * sp[0])
            x2 = int(x2 * sp[1])
            face_data = im_data[y1:y2, x1:x2]
            im_data = prewhiten(face_data) #预处理
            im_data = cv2.resize(im_data, (160, 160))
            im_data1 = np.expand_dims(im_data, axis=0)
            ##人脸特征提取
            emb1 = face_feature_sess.run(ff_embeddings,
                                         feed_dict={ff_images_placeholder: im_data1, ff_train_placeholder: False})
            strr = ",".join(str(i) for i in emb1[0])
            ##写去txt
            with open("face/feature.txt", "w") as f:
                f.writelines(strr)
            f.close()
            mess = "success"
            break
        else:
            mess = "fail"

    return mess

@app.route('/face_login', methods=['POST', 'GET'])
def face_login():
    #图片上传
    #人脸检测
    #人脸特征提取
    #加载注册人脸（人脸签到，人脸数很多，加载注册人脸放在face_login,
    # 启动服务加载/采用搜索引擎/ES）
    #同注册人脸相似性度量
    #返回度量结果
    f = request.files.get('file')
    print(f)
    upload_path = os.path.join("tmp/login_tmp." + f.filename.split(".")[-1])
    # secure_filename(f.filename))  #注意：没有的文件夹一定要先创建，不然会提示没有该路径
    print(upload_path)
    f.save(upload_path)
    ##人脸检测
    im_data = cv2.imread(upload_path)
    sp = im_data.shape
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
            y1 = int(y1 * sp[0])
            x1 = int(x1 * sp[1])
            y2 = int(y2 * sp[0])
            x2 = int(x2 * sp[1])
            face_data = im_data[y1:y2, x1:x2]
            cv2.imwrite("face.jpg",face_data )
            im_data = prewhiten(face_data)  # 预处理
            im_data = cv2.resize(im_data, (160, 160))
            im_data1 = np.expand_dims(im_data, axis=0)
            ##人脸特征提取
            emb1 = face_feature_sess.run(ff_embeddings,
                                         feed_dict={ff_images_placeholder: im_data1, ff_train_placeholder: False})

            with open("face/feature.txt") as f:
                fea_str = f.readlines()
                f.close()
            emb2_str = fea_str[0].split(",")
            emb2 = []
            for ss in emb2_str:
                emb2.append(float(ss))
            emb2 = np.array(emb2)

            dist = np.linalg.norm(emb1 - emb2)
            print("dist---->", dist)

            if dist < 0.3:
                return "success"
            else:
                return "fail"
    return "fail"

@app.route("/face_dis")
def face_dis():
    im_data1 = read_image("/home/kuan/code/mooc_py3_tensorflow/flask_server_facedetection/tmp/crop_face.jpg")

    im_data1 = np.expand_dims(im_data1, axis=0)

    emb1 = face_feature_sess.run(ff_embeddings, feed_dict={ff_images_placeholder:im_data1, ff_train_placeholder:False})

    im_data1 = read_image("tmp/face.jpg")

    im_data1 = np.expand_dims(im_data1, axis=0)

    emb2 = face_feature_sess.run(ff_embeddings, feed_dict={ff_images_placeholder:im_data1, ff_train_placeholder:False})

    dist = np.linalg.norm(emb1 - emb2)

    return str(dist)

if __name__ == '__main__':
    app.run(host="192.168.1.108", port=90, debug=True)