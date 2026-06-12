import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import os
from tensorflow import keras
import tensorflow.keras.backend as K

# CONFIGURATION
MODEL_URL = "https://github.com/jtw011/skin-lesion-classifier/releases/download/binaries/best_model.keras"
MODEL_PATH = "/tmp/best_model.keras"

# Define the focal loss here so the model can load it
@tf.keras.utils.register_keras_serializable()
def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
    pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)
    loss_1 = -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))
    loss_0 = -K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return loss_1 + loss_0

@st.cache_resource
def get_model():
    # 1. Download
    if not os.path.exists(MODEL_PATH):
        st.write("Downloading model from drive...")
        response = requests.get(MODEL_URL, allow_redirects=True)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
            
    # 2. Check for success (Google Drive often returns HTML instead of the model)
    file_size = os.path.getsize(MODEL_PATH)
    if file_size < 5000000: # Less than 5MB? It's probably an error page.
        os.remove(MODEL_PATH) # Clear the bad file
        raise ValueError(f"Download failed! The file is too small ({file_size} bytes). Your Google Drive link is likely not a 'Direct Download' link.")
    
    # 3. Load
    return tf.keras.models.load_model(MODEL_PATH, custom_objects={"focal_loss": focal_loss})

# --- UI ---
st.title("Skin Lesion Diagnostic Prototype")
st.warning("⚠️ DISCLAIMER: This tool is for educational purposes only and is NOT a medical device.")

# Load model (this will trigger the download if needed)
try:
    model = get_model()
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

uploaded_file = st.file_uploader("Upload a dermoscopy image...", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Force normalization to [0, 1] range manually
    img_array = img_array.astype('float32') / 255.0

    # Add this in app.py before model.predict()
    st.write(f"Image Min: {img_array.min()}, Image Max: {img_array.max()}")
    
    # Predict
    prob = model.predict(img_array)[0][0]
    
# --- UI Display Result ---
    
    # Define your thresholds
    HIGH_RISK_THRESHOLD = 0.5
    ELEVATED_RISK_THRESHOLD = 0.168
    
    # Logic to interpret probability
    if prob >= HIGH_RISK_THRESHOLD:
        st.error(f"Prediction: High Risk | Probability: {prob:.2%}")
        st.write("Immediate expert review required.")
    elif prob >= ELEVATED_RISK_THRESHOLD:
        st.warning(f"Prediction: Elevated Risk | Probability: {prob:.2%}")
        st.write("Consult a dermatologist for a professional evaluation.")
    else:
        st.success(f"Prediction: Low Risk | Probability: {prob:.2%}")
        st.write("Likely benign, but consult a professional if you have concerns.")