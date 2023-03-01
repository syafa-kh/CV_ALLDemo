import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg19 import preprocess_input

import os
import urllib.request
@st.cache_resource
def load_model():
    if not os.path.isfile('model.h5'):
        urllib.request.urlretrieve('https://github.com/syafa-kh/CV_ALLDemo/raw/main/all_fixed-vgg19.h5', 'model.h5')
    return tf.keras.models.load_model('model.h5')

def pred_class(img):
    # load model
    model = load_model()

    # prepare image
    img_prep = np.expand_dims(img_to_array(img), axis=0)
    # preprocess_input
    img_fin = preprocess_input(img_prep)
    # predict
    labels = ['Benign','Benign','Pre','Pro']
    pred_arr = model.predict(img_fin)
    class_label = labels[pred_arr.argmax(1)[0]]
    return model, pred_arr, class_label