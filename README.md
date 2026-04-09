🌿 AgriGuard Pro
AI-Powered Plant Disease Detection for Kenyan Smallholder Farmers
�
�
�
�
�
�
Load image
Load image
Load image
Load image
🚀 Live Demo: agriguard-ai.streamlit.app
🌍 Overview
AgriGuard Pro is a full-stack AI agricultural diagnostic tool built to help Kenyan smallholder farmers identify plant diseases in real time — directly from a photo of a leaf.
The project was built entirely on a Samsung tablet using Termux, with model training conducted on Kaggle. It demonstrates that impactful AI systems can be developed with minimal hardware resources, making it a proof-of-concept for accessible AI in Sub-Saharan Africa.
"Built from a tablet. Designed for a continent."
🎯 Problem Statement
Kenya loses an estimated 30–50% of crop yields annually to plant diseases, most of which go undetected until it is too late. Smallholder farmers — who represent over 75% of Kenya's agricultural workforce — lack access to fast, affordable diagnostic tools. The closest agronomist may be hours away. AgriGuard Pro puts expert-level disease detection in the farmer's pocket.
🧠 How It Works
Farmer uploads leaf photo
        ↓
Leaf vs. non-leaf detection (green-channel heuristic)
        ↓
Custom-trained TensorFlow model classifies disease
        ↓
Confidence score + probability breakdown generated
        ↓
Disease hotspot overlay highlights affected areas
        ↓
Farmer-friendly recommendations displayed
        ↓
Live web scraping fetches latest treatment data (2025)
        ↓
Scan saved to personal history in SQLite database
✨ Features
Feature
Description
🔬 AI Diagnosis
Custom-trained MobileNet model identifies plant diseases from leaf images
🛡️ Leaf Detection
Intelligently rejects non-leaf images with a friendly message
🗺️ Hotspot Mapping
Highlights disease spread zones on the uploaded image
📊 Confidence Scores
Visual probability breakdown for all possible diagnoses
🌐 Live Research
Scrapes latest 2025 treatment data from agricultural databases
👤 User Accounts
Full registration and login system with SQLite database
📋 Scan History
Every scan saved and accessible per user with summary metrics
👨‍⚕️ Vet Directory
Database of Kenyan agronomists filterable by region with call links
📖 Disease Directory
Farmer-friendly field guide for all supported crops and diseases
📷 Camera Input
Upload from gallery or capture live using device camera
🌱 Supported Crops & Diseases
Crop
Diseases Detected
🌿 Cassava
Bacterial Blight (CBB), Brown Streak (CBSD), Green Mottle (CGM), Mosaic Disease (CMD), Healthy
🌽 Maize
Common Rust, Gray Leaf Spot, Northern Leaf Blight, Healthy
🍅 Tomato
Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Healthy
🥔 Potato
Early Blight, Late Blight, Healthy
🤖 The AI Model
The core model (agri_guard_brain.h5) is a fine-tuned MobileNetV3 convolutional neural network trained on the PlantVillage dataset via Kaggle. Key technical details:
Architecture: MobileNetV3 (lightweight, mobile-optimised)
Input shape: 224 × 224 × 3 (RGB)
Output: Softmax over disease classes
Training platform: Kaggle (GPU-accelerated)
Framework: TensorFlow 2.x / Keras
Fallback: Google CropNet (TFHub) if local model unavailable
The model prioritises inference speed suitable for low-bandwidth mobile environments.
🏗️ Architecture
AgriGuard Pro
├── app.py                  # Main Streamlit application
├── agri_guard_brain.h5     # Trained TensorFlow model
├── agriguard.db            # SQLite database (auto-created on first run)
├── requirements.txt        # Python dependencies
└── README.md

Database Schema
├── users                   # Accounts (username, password hash, location, phone)
├── scans                   # Scan history (diagnosis, confidence, timestamp)
└── vets                    # Agronomist contacts directory
🚀 Getting Started
Prerequisites
Python 3.10+
pip
Installation
# Clone the repository
git clone https://github.com/michaelkibet123/AgriGuard-AI.git
cd AgriGuard-AI

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
The app will open in your browser at http://localhost:8501
📱 Running on Mobile (Termux)
This project was developed and runs on a Samsung tablet using Termux — no laptop required.
# Install Termux from F-Droid, then:
pkg install python
pip install streamlit tensorflow pillow numpy requests beautifulsoup4
git clone https://github.com/michaelkibet123/AgriGuard-AI.git
cd AgriGuard-AI
streamlit run app.py
Access the app from your tablet browser at http://localhost:8501
📊 User Research & Impact
AgriGuard Pro is currently being tested with smallholder farmers across Kenya. User feedback is being collected to inform a peer-reviewed publication on AI-assisted disease detection in East African agriculture.
Target user study metrics:
Diagnosis accuracy vs. expert agronomist assessment
Time to diagnosis (AI vs. traditional methods)
Farmer usability score (SUS questionnaire)
Geographic distribution of detected diseases
📄 Publication
A research paper is in preparation:
"AgriGuard: Real-Time AI-Powered Plant Disease Detection for Smallholder Farmers in Kenya"
Michael Kibet, 2025
Target journal: MDPI Agriculture / arXiv preprint
🏆 Competitions
This project is being submitted to:
Zindi Africa — Plant Disease Detection Challenge
Google AI for Social Good
🤝 Contributing
Contributions are welcome, especially:
Adding more crop models (beans, wheat, coffee)
Expanding the vet contacts database
Translations to Swahili and other Kenyan languages
Improving the leaf detection algorithm
Please open an issue or pull request.
📬 Contact
Michael Kibet
🔗 GitHub: @michaelkibet123
📧 Email: michaelkibet123@gmail.com
🌍 Built in Kenya 🇰🇪
📜 License
This project is licensed under the MIT License — see LICENSE for details.
�
Built with 🌿 to feed Africa · Powered by TensorFlow · Deployed on Streamlit 


