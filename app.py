import setuptools
import pkg_resources
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np

# Page Config
st.set_page_config(page_title="AgriGuard AI", page_icon="🌿")
st.title("🌿 AgriGuard AI: Smart Crop Doctor")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    # Pulling the official Google Cassava model from TF-Hub
    model_url = "https://tfhub.dev/google/cropnet/classifier/cassava_disease_V1/2"
    model = tf.keras.Sequential([
        hub.KerasLayer(model_url)
    ])
    return model

with st.spinner('Loading AI Brain...'):
    model = load_model()

# --- UI ---
uploaded_file = st.file_uploader("📸 Scan a leaf photo...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Target Leaf", use_container_width=True)
    
    with st.spinner('Analyzing...'):
        img = image.resize((224, 224))
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = model(img_array)
        # The model returns probabilities; we take the highest one
        result_index = np.argmax(predictions)
        
    st.success(f"**Analysis Complete!**")
    st.info("System Online: Ready for Next Scan.")
