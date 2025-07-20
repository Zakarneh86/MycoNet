import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tempfile
from tensorflow.keras.models import load_model

@st.cache_resource
def loading_model():
    try:
        model = load_model('./model/model4.h5')
        return model
    except Exception as e:
        print (f'Fail to Load Model. Error: {e}')

model = loading_model()

st.set_page_config(
    page_title="Microscopic Fungus Colony Classifier",
    page_icon="🧫",
    layout="wide"
)

# ========== TITLE ==========
st.markdown(
    "<h1 style='text-align: center; color: #4B8BBE;'>Microscopic Fungus Colony Classification with CNN Resedual Network </h1>",
    unsafe_allow_html=True
)
st.markdown("<hr>", unsafe_allow_html=True)
