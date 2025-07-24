import streamlit as st

st.set_page_config(page_title=" Plant Disease Detection", layout="centered")

import os
import tensorflow as tf
import numpy as np
from PIL import Image

# ── CONFIG ─────────────────────────────
MODEL_PATH = r"models/CustomCNN.h5"
IMG_SIZE = (128, 128)

# ── Load model once and cache ──────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Define class names (replace with your real class names if needed)
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___healthy',
    'Corn_(maize)___Northern_Leaf_Blight', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___healthy',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___healthy', 'Potato___Late_blight', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___healthy', 'Strawberry___Leaf_scorch', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
    'Tomato___healthy', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
]

# ── UI ────────────────────────────────
st.title(" Plant Disease Detector")
st.write("Upload a leaf image and let the model predict the disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width =True)

    # Preprocess
    img_resized = image.resize(IMG_SIZE)
    img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

    # Predict
    preds = model.predict(img_array)
    pred_idx = np.argmax(preds[0])
    pred_class = class_names[pred_idx]
    confidence = round(float(np.max(preds[0])) * 100, 2)

    st.subheader(" Prediction")
    st.markdown(f"**Class:** {pred_class}")
    st.markdown(f"**Confidence:** {confidence} %")
else:
    st.info("Please upload an image to get a prediction.")
