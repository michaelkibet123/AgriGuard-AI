import tensorflow as tf
        return tf.keras.models.load_model(path, compile=False)
    return None



def load_cassava_model():
    path = "agri_guard_brain_v2.keras"
    return tf.keras.models.load_model(path, compile=False)
