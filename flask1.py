import h5py
import flask
from keras.models import load_model
import numpy as np
import os
import pandas as pd
from collections import OrderedDict
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
from utils import visualization_utils as vis_utils
from utils import label_map_util
import tensorflow as tf
from PIL import Image
def det_coeff(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

MODEL_NAME = 'weed_detection_inference_graph'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')

NUM_CLASSES = 1
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
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
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict


df= pd.read_csv('data.csv')
#print(df.head())
df.columns=[['date','onion','chickpeas','potato','rice','sugar','wheat']]
#print(df)
index_map = {
    0:'Apple Scab',
    1:'Black Rot, Apple' ,
    2:'Cedar Rust, Apple',
    3:'Healthy Apple',
    4:'Healthy Blueberry',
    5: 'Powdery Mildew, Cherry',
    6:'Healthy Cherry',
    7:'Grey Leaf Spot, Corn',
    8: 'Common Rust of Maize',
    9:'Northern Leaf Blight, Corn',
    10:'Healthy Corn',
    11:'Black Rot, Grape' ,
    12: 'Black Measles, Grape',
    13:'Leaf Spot, Grape',
    14: 'Healthy Grape',
    15:'Citrus Huanglongbing',
    16: 'Bacterial Spot, Peach',
    17:'Healthy Peach',
    18:'Bacterial Spot, Bell Pepper',
    19:'Healthy Bell Pepper',
    20:'Early Blight, Potato',
    21:'Late Blight, Potato',
    22:'Healthy Potato',
    23:'Healthy Raspberry',
    24:'Healthy Soybean',
    25:'Powdery Mildew, Squash',
    26:'Leaf Scorch, Strawberry',
    27:'Healthy Strawberry',
    28:'Bacterial Leaf Spot, Tomato',
    29:'Early Blight, Tomato',
    30:'Late Blight, Tomato',
    31:'Leaf Mold, Tomato',
    32:'Leaf Spot, Tomato',
    33:'Two Spot Spider Mite, Tomato',
    34:'Target Leaf Spot, Tomato',
    35:'Yellow Leaf Curl, Tomato',
    36:'Mosaic, Tomato',
    37:'Healthy Tomato'
}


from flask import Flask,request
app=Flask(__name__)
@app.route("/")
#@app.route("/index")
#@app.route('/predict',methods=['POST'])
def main():
     return flask.render_template("index.html")
@app.route('/predict',methods = ['POST', 'GET'])

def predict():
    if request.method == 'POST' :
        st= request.form['str']
        st= str.lower(str(st))
        model= load_model(str(st+'.h5'),custom_objects={'det_coeff':det_coeff})
        X= np.array(df[st])
        x= X[-30:]
        x.resize((1,30))
        result=model.predict(x)[0]
        print(result)
        r=OrderedDict()
        r['First Month']= result[0]
        r['Second Month'] = result[1]
        r['Third Month '] = result[2]
        r['Fourth Month '] = result[3]
        r['Fifth Month '] = result[4]
        r['Sixth Month '] = result[5]
        return flask.render_template("predict.html",result=r)

@app.route('/disease', methods=['POST', 'GET'])
def disease():
    if request.method == 'POST':
        temp = request.files['image']
        print("got image")
        temp=image.load_img(temp,target_size=(224,224,3))
        temp= image.img_to_array(temp)
        temp= preprocess_input(temp)
        temp=temp.reshape(1,224,224,3)
        model =load_model('disease.h5')
        result = model.predict(temp)
        result = np.argmax(result)
        print(index_map[result])
        result= index_map[result]
        #return '<html> <body> hai </body></html>'
        return flask.render_template("disease.html",result=result)



@app.route('/weed', methods=['POST', 'GET'])
def weed():
    if request.method == 'POST':
        temp= request.files['image']
        img= Image.open(temp)
        #w,h=img.size
        #img.load()
        #array= np.asarray(img,dtype=np.uint8)
        #image_np= array.reshape(w,h,3)
        image_np = load_image_into_numpy_array(img)
        #image_np_expanded = np.expand_dims(image_np, axis=0)
        output_dict = run_inference_for_single_image(image_np, detection_graph)
        # Visualization of the results of a detection.
        vis_utils.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=8)
        w,h= 600,400
        plt.imshow(image_np)
        print('hai')
        img= Image.fromarray(image_np,'RGB')
        img.save('img.jpg')
        image = Image.open('img.jpg')
        #image.show()
        image.save('static/img.jpg')
        #return '<html> <body> hai </body></html>'
        return flask.render_template("weed.html")
if __name__ == '__main__':
    app.run(debug=True)
