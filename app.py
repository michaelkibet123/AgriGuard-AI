import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import keras

# --- 1. UI SETUP ---
st.set_page_config(page_title="AgriGuard Pro", page_icon="🌿")
st.title("🌿 AgriGuard AI: Smart Crop Doctor")
st.markdown("### High-Precision Plant Disease Detection")

# --- 2. THE DICTIONARY ---
categories = [
    'Apple Scab', 'Apple Black Rot', 'Apple Cedar Rust', 'Apple Healthy',
    'Blueberry Healthy', 'Cherry Powdery Mildew', 'Cherry Healthy',
    'Corn Gray Leaf Spot', 'Corn Common Rust', 'Corn Northern Blight', 'Corn Healthy', 
    'Grape Black Rot', 'Grape Black Measles', 'Grape Leaf Blight', 'Grape Healthy', 
    'Orange Citrus Greening', 'Peach Bacterial Spot', 'Peach Healthy', 
    'Pepper Bell Bacterial Spot', 'Pepper Bell Healthy', 'Potato Early Blight', 
    'Potato Late Blight', 'Potato Healthy', 'Raspberry Healthy', 'Soybean Healthy', 
    'Squash Powdery Mildew', 'Strawberry Leaf Scorch', 'Strawberry Healthy', 
    'Tomato Bacterial Spot', 'Tomato Early Blight', 'Tomato Late Blight', 
    'Tomato Leaf Mold', 'Tomato Septoria Leaf Spot', 'Tomato Spider Mites', 
    'Tomato Target Spot', 'Tomato Yellow Leaf Curl Virus', 'Tomato Mosaic Virus', 
    'Tomato Healthy'
]

# --- 3. THE "BYPASS" LOADER ---
@st.cache_resource
def load_model():
    # This specific line bypasses the 'batch_shape' error by using the Legacy loader
    return tf.keras.models.load_model('agri_guard_brain.h5', compile=False, safe_mode=False)

with st.spinner('Waking up the AI Intelligence...'):
    model = load_model()

# --- 4. THE INTERFACE ---
uploaded_file = st.file_uploader("📸 Scan a leaf photo...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Target Leaf", use_container_width=True)
    
    with st.spinner('Analyzing cellular patterns...'):
        img = image.resize((224, 224))
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Final double-input safety check
        try:
            predictions = model.predict([img_array, img_array])
        except Exception:
            predictions = model.predict(img_array)

        idx = np.argmax(predictions)
        conf = np.max(predictions) * 100

    # --- 5. THE REVEAL ---
    st.success(f"**Diagnosis:** {categories[idx]}")
    st.info(f"**AI Confidence:** {conf:.1f}%")
