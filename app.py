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
    # 1. Display the leaf
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Current Scan", use_container_width=True)
    
    with st.spinner('Neural Network processing...'):
        # 2. Pre-process
        img = image.resize((224, 224))
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # 3. Predict
        predictions = model(img_array)
        result_index = np.argmax(predictions)
        
        # 4. Map to actual labels
        labels = [
            "Cassava Bacterial Blight (CBB)", 
            "Cassava Brown Streak Disease (CBSD)", 
            "Cassava Green Mottle (CGM)", 
            "Cassava Mosaic Disease (CMD)", 
            "Healthy"
        ]
        prediction_label = labels[result_index]

    # --- RESULTS DASHBOARD ---
    st.markdown(f"### 🩺 Diagnosis: **{prediction_label}**")
    
    if prediction_label == "Healthy":
        st.success("The specimen shows no signs of viral or bacterial stress.")
        st.balloons()
    else:
        st.error(f"Potential {prediction_label} identified.")
        
        # Treatment Advice Section
        with st.expander("Recommended Action Plan"):
            if "Blight" in prediction_label:
                st.write("1. Prune and destroy infected leaves.")
                st.write("2. Apply copper-based bactericides.")
            elif "Mosaic" in prediction_label or "Streak" in prediction_label:
                st.write("1. Uproot infected plants to prevent transmission.")
                st.write("2. Control whitefly populations (vectors).")
            else:
                st.write("1. Isolate the affected area.")
                st.write("2. Monitor for further symptoms.")

    st.divider()
    st.info("System Online: Analysis complete.")
