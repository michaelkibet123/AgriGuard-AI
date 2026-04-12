import streamlit as st
import tensorflow as tf




st.title('AgriGuard AI')
st.write('App is running successfully')




@st.cache_resource
def load_cassava_model():
    path = "agri_guard_brain_v2.keras"
    return tf.keras.models.load_model(path, compile=False, custom_objects={})

cassava_model = load_cassava_model()
