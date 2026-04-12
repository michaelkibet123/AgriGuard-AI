import tensorflow as tf



def load_cassava_model():
    path = "agri_guard_brain_v2.keras"
    return tf.keras.models.load_model(path, compile=False)

st.title('AgriGuard AI')
st.write('App is running successfully')
