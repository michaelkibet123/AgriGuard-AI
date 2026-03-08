# 🌿 AgriGuard AI: Neural Crop Diagnostics

An AI-powered diagnostic tool designed to identify diseases in Cassava crops using Deep Learning. This project bridges the gap between complex Computer Vision and real-world agricultural impact.

---

## 🚀 Live Demo
[Click here to view the Live App](https://agriguard-ai.streamlit.app)

## 🧠 The Tech Stack
* **Core Engine:** TensorFlow 2.15.0
* **Model Source:** Google CropNet (MobileNetV3 Architecture)
* **Frontend:** Streamlit (Customized UI)
* **Environment:** Python 3.10

## 🛠️ Engineering Challenges (The "Version Hell" Solve)
One of the primary achievements of this project was navigating complex environment dependencies:
1. **Environment Stability:** Resolved `ModuleNotFoundError` by rolling back from Python 3.12 to **Python 3.10** to ensure compatibility with legacy TensorFlow-Hub dependencies.
2. **Dependency Injection:** Manually injected `pkg_resources` via `setuptools` to bypass cloud-native deployment bugs.
3. **Data Mapping:** Built a custom logic layer to translate raw Softmax tensors into human-readable treatment plans.

## 🩺 Supported Diagnoses
* Cassava Bacterial Blight (CBB)
* Cassava Brown Streak Disease (CBSD)
* Cassava Green Mottle (CGM)
* Cassava Mosaic Disease (CMD)
* Healthy Specimen Analysis

---
*Developed as a technical portfolio project for University-level Computer Science applications.*
