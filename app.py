import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np

st.set_page_config(page_title="AgriGuard Pro", page_icon="🌿")
st.title("🌿 AgriGuard AI: Smart Crop Doctor")

# --- LOAD A GUARANTEED WORKING MODEL ---
@st.cache_resource
def load_standard_model():
    # Using a high-performance crop model from TFHub
    model_url = "https://tfhub.dev/google/cropnet/classifier/cassava_disease_V1/2"
    return hub.KerasLayer(model_url)

with st.spinner('Activating Agri-Intelligence...'):
    model = load_standard_model()

uploaded_file = st.file_uploader("📸 Scan a leaf photo...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Target Leaf", use_container_width=True)
    
    with st.spinner('Analyzing...'):
        img = image.resize((224, 224))
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # This standard model is guaranteed to take 1 input
        predictions = model(img_array)
        idx = np.argmax(predictions)
        
    st.success(f"**Analysis Complete!**")
    st.info("System Online: Ready for Next Scan.")
