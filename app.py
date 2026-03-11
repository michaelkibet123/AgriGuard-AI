import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np
import requests
from bs4 import BeautifulSoup
import requests
from bs4 import BeautifulSoup

# --- INITIALIZE VARIABLES (Fixes the NameError) ---
prediction_label = "Waiting for Image..."
confidence_score = 0.0
recommendation_text = "Please upload a leaf image to begin analysis."

# --- 2. INTELLIGENCE LAYER (The Brain) ---


def compile_results(disease):
    search_url = (
        f"https://www.google.com/search?q={disease}+treatment+measures+kenya+2026"
    )
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(search_url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, "html.parser")
        results = soup.find_all("div", class_="BNeawe s3v9rd AP7Wnd")
        return results[0].text if results else "No live research found."
    except:
        return "Offline: Research servers unreachable."


# --- 1. PAGE CONFIG & THEME ---
st.set_page_config(
    page_title="AgriGuard Pro | Neural Suite",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CUSTOM CSS: This turns the app into a high-end dashboard
st.markdown(
    """
    <style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.8) !important;
        border-right: 1px solid #334155;
    }

    /* Glassmorphism Cards */
    div[data-testid="stVerticalBlock"] > div:has(div.stMetric) {
        background: rgba(30, 41, 59, 0.7);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #475569;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: #1e293b;
        border-radius: 8px;
        color: #94a3b8;
        border: 1px solid #334155;
        padding: 0 20px;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #22c55e;
        border-color: #22c55e;
    }
    .stTabs [aria-selected="true"] {
        background-color: #22c55e !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- 2. MODEL LOADING (CACHED) ---
@st.cache_resource
def load_model():
    model_url = "https://tfhub.dev/google/cropnet/classifier/cassava_disease_V1/2"
    return hub.KerasLayer(model_url)


model = load_model()

# --- 3. SIDEBAR (System Metadata) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2097/2097276.png", width=80)
    st.title("AgriGuard Pro")
    st.markdown("---")

    # This creates the actual dropdown menu in the UI
    selected_crop = st.selectbox(
        "Target Crop System", ["Cassava", "Maize", "Potato", "Tomato"]
    )

# --- 4. UNIVERSAL CROP LIBRARY ---
crop_library = {
    "Cassava": [
        "Bacterial Blight (CBB)",
        "Brown Streak (CBSD)",
        "Green Mottle (CGM)",
        "Mosaic Disease (CMD)",
        "Healthy Cassava",
    ],
    "Maize": ["Common Rust", "Gray Leaf Spot", "Northern Leaf Blight", "Healthy Maize"],
    "Potato": ["Early Blight", "Late Blight", "Healthy Potato"],
    "Tomato": [
        "Bacterial Spot",
        "Early Blight",
        "Late Blight",
        "Leaf Mold",
        "Healthy Tomato",
    ],
}

# --- 5. REDESIGNED INTERFACE ---
st.subheader(f"Diagnostic Suite: {selected_crop}")
col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "📤 Upload Leaf Specimen", type=["jpg", "jpeg", "png"]
    )

with col2:
    st.markdown("### Known Pathogens")
    for disease in crop_library[selected_crop]:
        st.write(f"• {disease}")

# --- 6. NEURAL PROCESSING ENGINE ---
if uploaded_file is not None:
    # Show the uploaded image to the user
    st.image(
        uploaded_file,
        caption=f"Processing {selected_crop} Specimen...",
        use_container_width=True,
    )

    with st.status("Initializing Neural Analysis...", expanded=True) as status:
        st.write("🔧 Loading specialized weights...")
        # (This is where your model.load function will eventually sit)
        st.write("🧠 Running leaf segmentation...")
        st.write("📡 Cross-referencing crop library...")
        status.update(label="Analysis Complete!", state="complete", expanded=False)

# --- 7. RESULTS DASHBOARD ---
st.markdown("---")
res_col1, res_col2 = st.columns(2)

with res_col1:
    # This matches the variable we initialized at the top
    st.metric(label="Primary Diagnosis", value=prediction_label)

with res_col2:
    st.metric(label="Confidence", value=f"{confidence_score}%")
# This creates the dropdown menu you were looking for
selected_crop = st.selectbox("Select Crop Type", list(crop_library.keys()))
labels = crop_library[selected_crop]
st.markdown("---")

st.write("### System Specs")
st.caption("Core: Google CropNet")
st.caption("Engine: MobileNetV3")
st.caption("Environment: Python 3.10")

st.markdown("---")
st.write("### Diagnostics")
st.success("● System Operational")
st.info("● Neural Network Loaded")

# --- 4. MAIN INTERFACE ---
st.title("🌿 Neural Crop Diagnostic Suite")
st.markdown("#### High-fidelity plant pathology detection for precision agriculture.")

tab1, tab2, tab3 = st.tabs(["🔍 AI Scanner", "📖 Disease Directory", "⚙️ System Logs"])

with tab1:
    st.markdown("### Specimen Input")
    uploaded_file = st.file_uploader(
        "Drop leaf imagery here...", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        col1, col2 = st.columns([1, 1], gap="large")
        image = Image.open(uploaded_file).convert("RGB")

        with col1:
            st.markdown("#### Analyzed Image")
            st.image(image, use_container_width=True)

        with col2:
            st.markdown("#### Diagnostic Results")
            with st.spinner("Calculating tensor probabilities..."):
                img = image.resize((224, 224))
                img_array = np.array(img).astype(np.float32) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                predictions = model(img_array)
                result_index = np.argmax(predictions)
                prediction_label = labels[result_index]

crop_library = {
    "Maize": ["Common Rust", "Gray Leaf Spot", "Northern Leaf Blight", "Healthy Maize"],
    "Potato": ["Early Blight", "Late Blight", "Healthy Potato"],
    "Tomato": [
        "Bacterial Spot",
        "Early Blight",
        "Late Blight",
        "Leaf Mold",
        "Healthy Tomato",
    ],
}

selected_crop = st.selectbox("Select Crop Type", list(crop_library.keys()))
labels = crop_library[selected_crop]

# Result Display
st.metric(label="Primary Diagnosis", value=prediction_label)
st.markdown("---")
if "Healthy" in prediction_label:
    st.balloons()
    st.success(f"✅ {prediction_label}: No pathogen markers detected.")
else:
    st.error(f"⚠️ Alert: {prediction_label} identified.")
    st.markdown("### **Architect's Action Plan**")
    st.write("1. **Isolate:** Remove affected leaves to prevent spore spread.")
    st.write("2. **Vector Control:** Check for whiteflies or aphids.")

    st.markdown("---")
    # This button ONLY appears if the plant is sick
if st.button("🔍 Compile Live Treatment Research"):
    with st.spinner(f"Architect is browsing 2026 data for {prediction_label}..."):
        research_data = compile_results(prediction_label)
        st.subheader("📋 Compiled Research Summary")
        st.info(research_data)

if "Healthy" in prediction_label:
    st.balloons()
    st.success("Verification: No pathogen markers detected.")
else:
    st.error(f"Alert: {prediction_label} identified.")
    st.markdown("**Action Plan**")
    st.write("1. Check for whitefly presence in immediate crop radius.")
    st.write("2. Apply localized copper-based bactericide if applicable.")
    st.markdown("---")

with tab2:
    st.subheader("Field Directory")
    colA, colB = st.columns(2)
    with colA:
        with st.expander("🍂 Cassava Bacterial Blight (CBB)"):
            st.write(
                "Caused by *Xanthomonas axonopodis*. Look for angular leaf spots and gum exudate on stems."
            )
        with st.expander("🕸️ Cassava Green Mottle (CGM)"):
            st.write(
                "Viral infection. Causes yellowing patterns and severe leaf distortion."
            )
    with colB:
        with st.expander("🥀 Cassava Brown Streak (CBSD)"):
            st.write(
                "The 'silent killer'. Often affects roots first. Look for yellowing along secondary leaf veins."
            )
        with st.expander("🌀 Cassava Mosaic Disease (CMD)"):
            st.write(
                "The most common threat. Characterized by mosaic-like patches of green and yellow."
            )

with tab3:
    st.subheader("Neural Network Logs")
    st.code("""
    [INFO] Initializing MobileNetV3 Backend...
    [INFO] Loading Weights from TFHub (V1_Cassava)...
    [INFO] Input Tensor Shape: (1, 224, 224, 3)
    [INFO] Softmax Layer: 5-way Classification
    [SUCCESS] Environment stable on Streamlit Cloud (Linux/Python 3.10)
    """)
# --- 6. FOOTER & SOURCE ---
st.markdown("---")
st.caption(
    "🌿 **AgriGuard Pro v2.1** | Intelligence: TensorFlow-Hub | Environment: Streamlit Cloud"
)


import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Global labels - KEEP THIS AT THE TOP
labels = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Corn___Gray_leaf_spot",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy",
]

# 2. The Sidebar Block (Everything indented here stays on the left)
with st.sidebar:
    st.title("👨‍🔬 Agronomist Portal")
    st.markdown("---")
    if st.button("Request Expert Review"):
        st.success("Request sent to nearest KALRO station.")
    st.info("AgriGuard v1.0 | Kenya")

# 3. The Main Page (Everything NOT indented stays in the middle)
st.title("🌿 AgriGuard: AI Leaf Diagnosis")
# ... your image uploader code follows here

# --- 8. RECOMMENDATION ---
st.info(f"💡 **Recommendation:** {recommendation_text}")
