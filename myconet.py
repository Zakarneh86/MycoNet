import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tempfile
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Microscopic Fungus Colony Classifier",
    page_icon="🧫",
    layout="wide"
)

@st.cache_resource
def loading_model():
    try:
        model = load_model('./model/model4.h5')
        return model
    except Exception as e:
        st.write(f'Fail to Load Model. Error: {e}')

model = loading_model()
class_ids = ['H1', 'H2', 'H3', 'H5', 'H6']
# ========== TITLE ==========
st.markdown(
    "<h1 style='text-align: center; color: #4B8BBE;'>🧫 Microscopic Fungus Colony Classification with CNN Resedual Network </h1>",
    unsafe_allow_html=True
)
st.markdown("<hr>", unsafe_allow_html=True)

# ========== SIDEBAR ==========
st.sidebar.markdown("## 📤 Upload Microscopic Image")
uploaded_file = st.sidebar.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    original_img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    height, width, channels = original_img_rgb.shape
    if height !=224 or width !=224:
        original_img_rgb = cv2.resize(original_img_rgb, (224,224), interpolation=cv2.INTER_LANCZOS4)
    
    file_type = uploaded_file.type
    file_type = file_type.split('/')[-1]

    img_input = original_img_rgb.astype(np.float32) / 255.0
    img_input = np.expand_dims(img_input, axis=0)
    
    results = model.predict(img_input)
    predicted_class = class_ids[results.argmax()]
    conf = results.max()

    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🧪 Prediction Result")
    st.sidebar.success(f'''Predicted Class: {predicted_class}\n
                       Confidence: {conf:.2f}''')