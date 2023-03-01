import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg19 import preprocess_input

import os
import subprocess

def pred_class(img):
    # load model
    if not os.path.isfile('all_fixed-vgg19.h5'):
        subprocess.run(['curl --output model.h5 "https://github.com/syafa-kh/CV_ALLDemo/blob/main/all_fixed-vgg19.h5"'], shell=True)
    model = load_model('model.h5', compile=False)

    # prepare image
    img_prep = np.expand_dims(img_to_array(img), axis=0)
    # preprocess_input
    img_fin = preprocess_input(img_prep)
    # predict
    labels = ['Benign','Benign','Pre','Pro']
    pred_arr = model.predict(img_fin)
    class_label = labels[pred_arr.argmax(1)[0]]
    return model, pred_arr, class_label