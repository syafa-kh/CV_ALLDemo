import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg19 import preprocess_input

def pred_class(img):
    # load model
    model = load_model('all_fixed-vgg19.h5', compile=False)

    # prepare image
    img_prep = np.expand_dims(img_to_array(img), axis=0)
    # preprocess_input
    img_fin = preprocess_input(img_prep)
    # predict
    labels = ['Benign','Benign','Pre','Pro']
    pred_arr = model.predict(img_fin)
    class_label = labels[pred_arr.argmax(1)[0]]
    return model, pred_arr, class_label