import streamlit as st
import numpy as np
from tensorflow.keras.utils import load_img
import pred
from explain import gradCAM

import os
import subprocess
if not os.path.isfile('all_fixed-vgg19.h5'):
    subprocess.run(['curl --output all_fixed-vgg19.h5 "https://github.com/syafa-kh/CV_ALLDemo/blob/main/all_fixed-vgg19.h5"'], shell=True)

st.markdown('<h1 style="color:white;">🩸 VGG 19 Model for ALL Detection</h1>', unsafe_allow_html=True)
st.markdown('<h3 style="color:#d8d8d8;">Find out whether a <span style="color: #f77c7c">Peripheral Blood Smear (PBS)</span> image classifies into <span style="color: #f77c7c">Benign, Early, Pre, or Pro</span</h2>', unsafe_allow_html=True)

upload = st.file_uploader('Insert .JPG image for classification', type=['jpg'])
col_in, col_out = st.columns(2)
if upload is not None:
    img = load_img(upload)
    # show input image
    col_in.header('Input Image')
    col_in.image(img)

    # print prediction
    col_out.header('Prediction & GradCAM')
    # show gradcam
    # predict
    model, pred_arr, class_lbl = pred.pred_class(img)
    img_gradcam = gradCAM(img_path=upload,size=(224,224),model=model).display_gradcam()
    col_out.image(img_gradcam)
    # print prediction
    col_out.markdown(f'Probability for each class: **{pred_arr}**')
    col_out.markdown(f'Most likely class: **{class_lbl}**')

st.markdown("""---""") 
st.markdown('<h2 style="color:white;">How Did <span style="color: #f77c7c">This Model</span> Came To Be?</h2>', unsafe_allow_html=True)
st.markdown('Find out more about my development process here: [GitHub Repository](https://github.com/syafa-kh/CV_ALLDetection)')