import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow import keras
import tensorflow.keras.backend as K

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

# Cache the model so it doesn't reload every time you click something
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model.keras", custom_objects={"focal_loss": focal_loss})

model = load_model()

# --- UI ---
st.title("Skin Lesion Diagnostic Prototype")
st.warning("⚠️ DISCLAIMER: This tool is for educational purposes only and is NOT a medical device. Consult a dermatologist for medical advice.")

uploaded_file = st.file_uploader("Upload a dermoscopy image...", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    prob = model.predict(img_array)[0][0]
    
    # Display Result
    # Replace 0.5 with the 'optimal_threshold' you found in your script
    threshold = 0.5 
    
    if prob >= threshold:
        st.error(f"Prediction: Malignant | Probability: {prob:.2%}")
    else:
        st.success(f"Prediction: Benign | Probability: {prob:.2%}")