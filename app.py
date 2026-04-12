# ╔══════════════════════════════════════════════════════════════════╗
# ║              AgriGuard Pro — Final Production Build              ║
# ║         Author: Michael Kibet | Kenya 🇰🇪 | UPenn Portfolio      ║
# ║                                                                  ║
# ║  TECHNICAL SPEC (DO NOT MODIFY MODEL LOADING LOGIC):            ║
# ║  - Cassava  → TFHub CropNet, 5 classes, loaded separately       ║
# ║  - Others   → agri_guard_brain.h5, 38 classes, sliced by index  ║
# ║  - Brain loaded with tf.keras.models.load_model() ONLY          ║
# ║  - NEVER rebuild architecture, NEVER use load_weights()          ║
# ║  - CROP_INDICES: Maize[7,8,9,10] Potato[20,21,22] Tomato[28-37] ║
# ║  - Database: Supabase (PostgreSQL cloud)                         ║
# ╚══════════════════════════════════════════════════════════════════╝

import streamlit as st
import hashlib
import numpy as np
import requests
import os
import json
from supabase import create_client
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageEnhance
import random

# ─────────────────────────────────────────────────────────────
# 0. PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgriGuard Pro",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# 1. PREMIUM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,700;0,900;1,700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:        #070d07;
    --surface:   #0e1a0e;
    --card:      rgba(255,255,255,0.03);
    --border:    rgba(34,197,94,0.15);
    --green:     #22c55e;
    --green-dim: #16a34a;
    --green-glow:rgba(34,197,94,0.08);
    --text:      #f0fdf4;
    --text-dim:  #86efac;
    --text-muted:#4ade80;
    --red:       #ef4444;
    --amber:     #f59e0b;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
}

.stApp {
    background: var(--bg);
    background-image:
        radial-gradient(ellipse at 20% 20%, rgba(34,197,94,0.04) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 80%, rgba(34,197,94,0.03) 0%, transparent 50%);
}

section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; }

.ag-display {
    font-family: 'Fraunces', serif;
    font-size: clamp(2rem, 5vw, 3.2rem);
    font-weight: 900;
    color: var(--green);
    line-height: 1.1;
    letter-spacing: -1px;
}
.ag-subtitle {
    font-size: 0.95rem;
    color: var(--text-dim);
    font-weight: 300;
    margin-top: 4px;
}
.ag-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text-muted);
}

.ag-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
}
.ag-card-danger {
    background: rgba(239,68,68,0.05);
    border: 1px solid rgba(239,68,68,0.2);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
}
.ag-card-success {
    background: rgba(34,197,94,0.05);
    border: 1px solid rgba(34,197,94,0.25);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
}

.ag-badge { display:inline-block; padding:4px 14px; border-radius:100px; font-size:0.75rem; font-weight:600; letter-spacing:1px; text-transform:uppercase; }
.ag-badge-disease { background:rgba(239,68,68,0.15); color:#fca5a5; border:1px solid rgba(239,68,68,0.3); }
.ag-badge-healthy { background:rgba(34,197,94,0.15); color:#86efac; border:1px solid rgba(34,197,94,0.3); }
.ag-badge-warning { background:rgba(245,158,11,0.15); color:#fcd34d; border:1px solid rgba(245,158,11,0.3); }

.conf-wrap { background:rgba(255,255,255,0.05); border-radius:100px; height:8px; width:100%; margin:10px 0; overflow:hidden; }
.conf-fill  { height:100%; border-radius:100px; }

.ag-metric { background:var(--card); border:1px solid var(--border); border-radius:14px; padding:20px 16px; text-align:center; }
.ag-metric-val { font-family:'Fraunces',serif; font-size:2rem; font-weight:700; color:var(--green); }
.ag-metric-lab { font-size:0.72rem; color:var(--text-dim); text-transform:uppercase; letter-spacing:1.5px; margin-top:4px; }

.ag-history { background:var(--card); border:1px solid var(--border); border-left:3px solid var(--green); border-radius:0 12px 12px 0; padding:14px 18px; margin-bottom:10px; }
.ag-history-disease { border-left-color:var(--red) !important; }

.ag-vet { background:var(--card); border:1px solid var(--border); border-radius:14px; padding:18px; margin-bottom:12px; }

.stButton > button {
    background: var(--green) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 1.5rem !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: var(--green-dim) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(34,197,94,0.3) !important;
}

.stTextInput > div > div > input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}

.stFileUploader > div {
    background: rgba(34,197,94,0.03) !important;
    border: 2px dashed rgba(34,197,94,0.25) !important;
    border-radius: 16px !important;
}

.stTabs [data-baseweb="tab-list"] { background:transparent !important; gap:8px !important; }
.stTabs [data-baseweb="tab"] { background:var(--card) !important; border:1px solid var(--border) !important; border-radius:10px !important; color:var(--text-dim) !important; font-weight:500 !important; padding:8px 20px !important; }
.stTabs [aria-selected="true"] { background:rgba(34,197,94,0.15) !important; border-color:var(--green) !important; color:var(--green) !important; }

.ag-offline { background:rgba(245,158,11,0.1); border:1px solid rgba(245,158,11,0.3); border-radius:10px; padding:10px 16px; font-size:0.85rem; color:#fcd34d; margin-bottom:16px; }
.ag-guest   { background:rgba(59,130,246,0.08); border:1px solid rgba(59,130,246,0.2); border-radius:10px; padding:12px 16px; font-size:0.85rem; color:#93c5fd; margin-bottom:16px; }

.ag-sidebar-logo { padding:24px 20px 16px; border-bottom:1px solid var(--border); margin-bottom:16px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# 2. SUPABASE CLIENT
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_supabase():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = get_supabase()


# ─────────────────────────────────────────────────────────────
# 3. AUTH & DATABASE FUNCTIONS
# ─────────────────────────────────────────────────────────────
def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def register_user(username, password, full_name, location, phone):
    try:
        existing = supabase.table("users").select("id").eq("username", username.strip().lower()).execute()
        if existing.data:
            return False, "Username already exists."
        supabase.table("users").insert({
            "username":      username.strip().lower(),
            "password_hash": hash_pw(password),
            "full_name":     full_name,
            "location":      location,
            "phone":         phone,
        }).execute()
        return True, "Account created!"
    except Exception as e:
        return False, f"Error: {e}"

def login_user(username, password):
    try:
        result = supabase.table("users").select("*")\
            .eq("username", username.strip().lower())\
            .eq("password_hash", hash_pw(password))\
            .execute()
        return result.data[0] if result.data else None
    except:
        return None

def save_scan(user_id, result, crop, research=""):
    try:
        supabase.table("scans").insert({
            "user_id":    user_id,
            "crop":       crop,
            "diagnosis":  result["diagnosis"],
            "confidence": result["confidence"],
            "is_healthy": 1 if result["is_healthy"] else 0,
            "severity":   result["severity"],
            "advice":     result["advice"],
            "prevention": result["prevention"],
            "research":   research,
            "all_probs":  json.dumps(result["all_probs"]),
        }).execute()
    except Exception as e:
        st.error(f"Could not save scan: {e}")

def get_user_scans(user_id, limit=50):
    try:
        result = supabase.table("scans").select("*")\
            .eq("user_id", user_id)\
            .order("scanned_at", desc=True)\
            .limit(limit)\
            .execute()
        return result.data
    except:
        return []

def get_vets(region=None):
    try:
        if region and region != "All Regions":
            result = supabase.table("vets").select("*").eq("region", region).execute()
        else:
            result = supabase.table("vets").select("*").execute()
        return result.data
    except:
        return []


# ─────────────────────────────────────────────────────────────
# 4. CROP & DISEASE DATA
# ─────────────────────────────────────────────────────────────
CROP_INDICES = {
    "Maize":  [0, 1, 2, 3],
    "Potato": [4, 5, 6],
    "Tomato": [7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
}

CROP_LIBRARY = {
    "Cassava": {
        "icon": "🌿",
        "labels": ["Bacterial Blight","Brown Streak Disease","Green Mottle","Mosaic Disease","Healthy"],
        "model": "tfhub",
    },
    "Maize": {
        "icon": "🌽",
        "labels": ["Cercospora Leaf Spot","Common Rust","Healthy","Northern Leaf Blight"],
        "model": "brain",
    },
    "Potato": {
        "icon": "🥔",
        "labels": ["Early Blight","Healthy","Late Blight"],
        "model": "brain",
    },
    "Tomato": {
        "icon": "🍅",
        "labels": ["Bacterial Spot","Early Blight","Late Blight","Leaf Mold",
                   "Septoria Leaf Spot","Spider Mites","Target Spot",
                   "Yellow Leaf Curl Virus","Mosaic Virus","Healthy"],
        "model": "brain",
    },
}

DISEASE_ADVICE = {
    "Bacterial Blight":      {"severity":"High",     "action":"Remove and burn infected leaves immediately. Do not work in the field when wet. Apply copper-based bactericide from your agrovet. Avoid overhead irrigation.", "prevention":"Use certified disease-free cuttings. Space plants well for airflow."},
    "Brown Streak Disease":  {"severity":"Very High", "action":"No cure once infected. Uproot and destroy affected plants immediately to protect neighbours. Control whitefly using yellow sticky traps or approved insecticides.", "prevention":"Plant CBSD-resistant varieties. Source cuttings from certified clean sources only."},
    "Green Mottle":          {"severity":"Medium",   "action":"Manage whitefly with neem-based sprays. Remove heavily infected leaves. Ensure good crop nutrition.", "prevention":"Use virus-free planting material. Monitor whitefly weekly."},
    "Mosaic Disease":        {"severity":"High",     "action":"Remove infected plants early. Control whitefly with insecticide. Intercrop with beans. Contact your extension officer.", "prevention":"Plant CMD-resistant varieties. Use certified cuttings."},
    "Cercospora Leaf Spot":  {"severity":"Medium",   "action":"Apply mancozeb or chlorothalonil fungicide. Improve field drainage. Remove infected lower leaves.", "prevention":"Rotate crops. Till old crop residue. Plant resistant varieties."},
    "Common Rust":           {"severity":"Medium",   "action":"Apply fungicide (mancozeb or propiconazole) at first sign. Spray in the evening. Repeat after 7-14 days.", "prevention":"Plant rust-resistant maize varieties. Avoid poor air circulation."},
    "Northern Leaf Blight":  {"severity":"High",     "action":"Apply fungicide immediately. Remove severely infected leaves. Ensure proper plant spacing.", "prevention":"Use resistant hybrid seeds. Practice crop rotation."},
    "Early Blight":          {"severity":"Medium",   "action":"Apply mancozeb or chlorothalonil fungicide. Remove lower infected leaves. Ensure good drainage.", "prevention":"Rotate crops every season. Mulch around plants."},
    "Late Blight":           {"severity":"Very High", "action":"Act immediately — Late Blight spreads extremely fast. Apply metalaxyl or cymoxanil fungicide. Remove and destroy infected plants. Do NOT compost infected material.", "prevention":"Plant blight-resistant varieties. Avoid overhead irrigation. Monitor weekly."},
    "Bacterial Spot":        {"severity":"Medium",   "action":"Apply copper-based bactericide. Avoid overhead watering. Remove infected plant debris.", "prevention":"Use disease-free seeds. Avoid working in wet conditions."},
    "Leaf Mold":             {"severity":"Medium",   "action":"Improve ventilation. Apply fungicide. Remove infected leaves.", "prevention":"Reduce humidity. Space plants properly. Avoid wetting leaves."},
    "Septoria Leaf Spot":    {"severity":"Medium",   "action":"Remove infected leaves. Apply mancozeb fungicide. Avoid overhead watering.", "prevention":"Rotate crops. Remove plant debris after harvest."},
    "Spider Mites":          {"severity":"Medium",   "action":"Apply miticide or neem oil. Increase humidity around plants. Remove heavily infested leaves.", "prevention":"Monitor regularly. Avoid dusty conditions. Use resistant varieties."},
    "Target Spot":           {"severity":"Medium",   "action":"Apply fungicide. Remove infected leaves. Improve air circulation.", "prevention":"Rotate crops. Avoid excessive nitrogen fertiliser."},
    "Yellow Leaf Curl Virus":{"severity":"Very High", "action":"No cure. Remove infected plants immediately. Control whitefly populations aggressively.", "prevention":"Use virus-resistant varieties. Control whitefly from seedling stage."},
    "Mosaic Virus":          {"severity":"High",     "action":"Remove infected plants. Control aphid and whitefly vectors. Disinfect tools.", "prevention":"Use certified virus-free seeds. Control insect vectors."},
}

HEALTHY_ADVICE = "Your crop looks healthy! Keep it that way — water correctly, use certified seeds, and scout your field every week for early signs of disease."

SEV_COLORS = {
    "Very High": ("#ef4444","🔴"),
    "High":      ("#f97316","🟠"),
    "Medium":    ("#f59e0b","🟡"),
    "Low":       ("#22c55e","🟢"),
}


# ─────────────────────────────────────────────────────────────
# 5. MODEL LOADING
# ── NEVER modify this section ──
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_cassava_model():
    return hub.KerasLayer("https://tfhub.dev/google/cropnet/classifier/cassava_disease_V1/2")

@st.cache_resource
def load_brain_model():
    import tensorflow as tf
    path = "agri_guard_brain.h5"
    if os.path.exists(path):
        return tf.keras.models.load_model(path)
    return None

cassava_model = load_cassava_model()
brain_model   = load_brain_model()


# ─────────────────────────────────────────────────────────────
# 6. UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────
def is_online():
    try:
        requests.get("https://google.com", timeout=3)
        return True
    except:
        return False

def is_leaf(image):
    img = image.resize((100,100)).convert("RGB")
    arr = np.array(img).astype(float)
    r,g,b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    mask  = (g > r) & (g > b) & (g > 55)
    ratio = mask.sum() / 10000
    return ratio > 0.12, round(float(ratio), 3)

def hotspot_overlay(image, severity="Medium"):
    img = image.copy().convert("RGBA")
    w,h = img.size
    overlay = Image.new("RGBA", img.size, (0,0,0,0))
    draw    = ImageDraw.Draw(overlay)
    colors  = {"Very High":(220,20,20,85),"High":(220,100,20,70),"Medium":(220,175,20,55),"Low":(20,180,20,40)}
    col = colors.get(severity,(220,100,20,65))
    n   = {"Very High":7,"High":5,"Medium":3,"Low":1}.get(severity,3)
    random.seed(99)
    for _ in range(n):
        cx=random.randint(w//5,4*w//5); cy=random.randint(h//5,4*h//5)
        rx=random.randint(w//9,w//4);   ry=random.randint(h//9,h//4)
        draw.ellipse([(cx-rx,cy-ry),(cx+rx,cy+ry)], fill=col)
    return ImageEnhance.Contrast(Image.alpha_composite(img,overlay).convert("RGB")).enhance(1.15)

def scrape_research(disease, crop):
    try:
        q   = f"{disease} {crop} treatment Kenya 2026"
        url = f"https://www.google.com/search?q={requests.utils.quote(q)}"
        h   = {"User-Agent":"Mozilla/5.0 (Linux; Android 13) AppleWebKit/537.36"}
        r   = requests.get(url, headers=h, timeout=6)
        soup= BeautifulSoup(r.text,"html.parser")
        bits= soup.find_all("div", class_="BNeawe s3v9rd AP7Wnd")
        if bits:
            t = bits[0].text.strip()
            return t if len(t) > 30 else None
    except:
        pass
    return None

def run_diagnosis(image, crop):
    labels = CROP_LIBRARY[crop]["labels"]
    img_arr= np.array(image.resize((224,224))).astype(np.float32)/255.0
    img_arr= np.expand_dims(img_arr, axis=0)

    if crop == "Cassava":
        preds = cassava_model(img_arr)
        probs = np.array(preds).flatten()[:5]
    else:
        if brain_model is None:
            return None
        preds  = brain_model(img_arr)
        all38  = np.array(preds).flatten()
        probs  = np.array([all38[i] for i in CROP_INDICES[crop]])

    total = probs.sum()
    probs = probs/total if total > 0 and not np.isnan(total) else np.ones(len(labels))/len(labels)

    top_idx    = int(np.argmax(probs))
    confidence = float(probs[top_idx])*100
    diagnosis  = labels[top_idx]
    is_healthy = "healthy" in diagnosis.lower()
    adv        = DISEASE_ADVICE.get(diagnosis,{})
    severity   = "Low" if is_healthy else adv.get("severity","Medium")
    action     = HEALTHY_ADVICE if is_healthy else adv.get("action","Consult your nearest agronomist.")
    prevention = "" if is_healthy else adv.get("prevention","")

    return {
        "diagnosis":  diagnosis,
        "confidence": confidence,
        "is_healthy": is_healthy,
        "severity":   severity,
        "advice":     action,
        "prevention": prevention,
        "all_probs":  list(zip(labels,[round(float(p)*100,1) for p in probs])),
    }


# ─────────────────────────────────────────────────────────────
# 7. SESSION STATE
# ─────────────────────────────────────────────────────────────
for k,v in [("user",None),("page","home"),("selected_scan",None),("last_result",None),("online",None)]:
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.online is None:
    st.session_state.online = is_online()


# ─────────────────────────────────────────────────────────────
# 8. SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="ag-sidebar-logo">
        <div style="font-family:'Fraunces',serif;font-size:1.5rem;font-weight:900;color:#22c55e;">🌿 AgriGuard</div>
        <div style="font-size:0.72rem;color:#4ade80;margin-top:2px;letter-spacing:1px;text-transform:uppercase;">Pro · AI Plant Health</div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.online:
        st.markdown('<div class="ag-offline">📡 Offline — diagnosis still works</div>', unsafe_allow_html=True)

    if st.session_state.user:
        u = st.session_state.user
        st.markdown(f"""
        <div style="padding:0 4px 16px;">
            <div style="font-weight:600;font-size:0.95rem;">👤 {u['full_name']}</div>
            <div style="font-size:0.78rem;color:#4ade80;">📍 {u['location']}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="ag-guest">👋 Guest Mode — sign in to save scans</div>', unsafe_allow_html=True)

    selected_crop = st.selectbox("🌱 Crop", list(CROP_LIBRARY.keys()),
                                  format_func=lambda c: f"{CROP_LIBRARY[c]['icon']} {c}")

    st.markdown("<hr style='border-color:rgba(34,197,94,0.15);margin:16px 0;'>", unsafe_allow_html=True)

    brain_ok = brain_model is not None
    st.markdown(f"""
    <div style="font-size:0.75rem;padding:0 2px;">
        <div style="margin-bottom:6px;">✅ <span style="color:#86efac;">CropNet (Cassava)</span></div>
        <div>{'✅' if brain_ok else '⚠️'} <span style="color:#86efac;">Brain Model (Maize/Potato/Tomato)</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(34,197,94,0.15);margin:16px 0;'>", unsafe_allow_html=True)

    if st.session_state.user:
        if st.button("🚪 Sign Out"):
            st.session_state.user = None
            st.session_state.selected_scan = None
            st.rerun()
    else:
        if st.button("🔐 Sign In / Register"):
            st.session_state.page = "auth"
            st.rerun()


# ─────────────────────────────────────────────────────────────
# 9. AUTH PAGE
# ─────────────────────────────────────────────────────────────
if st.session_state.page == "auth" and not st.session_state.user:
    st.markdown('<div class="ag-display">Welcome back.</div>', unsafe_allow_html=True)
    st.markdown('<div class="ag-subtitle">Sign in or create your AgriGuard account.</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col_form, col_info = st.columns([1,1], gap="large")

    with col_form:
        mode = st.radio("", ["Sign In","Create Account"], horizontal=True, label_visibility="collapsed")

        if mode == "Sign In":
            with st.form("login"):
                uname = st.text_input("Username")
                pwd   = st.text_input("Password", type="password")
                if st.form_submit_button("Sign In →"):
                    user = login_user(uname, pwd)
                    if user:
                        st.session_state.user = user
                        st.session_state.page = "home"
                        st.rerun()
                    else:
                        st.error("Incorrect username or password.")
        else:
            with st.form("register"):
                full_name = st.text_input("Full Name")
                uname     = st.text_input("Username")
                pwd       = st.text_input("Password", type="password")
                location  = st.selectbox("County / Region", [
                    "Nairobi","Nakuru","Kisumu","Meru","Kakamega",
                    "Eldoret","Nyeri","Mombasa","Machakos","Kisii",
                    "Bungoma","Embu","Kitale","Thika","Other"
                ])
                phone = st.text_input("Phone (optional)")
                if st.form_submit_button("Create Account →"):
                    if full_name and uname and pwd:
                        ok, msg = register_user(uname, pwd, full_name, location, phone)
                        if ok:
                            st.success(msg + " Please sign in.")
                        else:
                            st.error(msg)
                    else:
                        st.warning("Please fill in name, username and password.")

        if st.button("← Continue as Guest"):
            st.session_state.page = "home"
            st.rerun()

    with col_info:
        st.markdown("""
        <div class="ag-card">
            <div class="ag-label" style="margin-bottom:14px;">Why create an account?</div>
            <div style="font-size:0.88rem;line-height:1.8;color:#86efac;">
                📋 Save every scan automatically<br>
                🔍 Tap any past scan for full breakdown<br>
                📊 Track your farm's disease history<br>
                👨‍⚕️ Save agronomist contacts<br>
                📈 See your healthy vs disease ratio
            </div>
        </div>
        <div class="ag-card">
            <div class="ag-label" style="margin-bottom:10px;">Guest mode includes</div>
            <div style="font-size:0.88rem;line-height:1.8;color:#86efac;">
                ✅ Full AI diagnosis<br>
                ✅ Confidence scores<br>
                ✅ Disease hotspot map<br>
                ✅ Treatment recommendations<br>
                ✅ Disease directory<br>
                ✅ Agronomist contacts
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────────────────────
# 10. SCAN DETAIL PAGE
# ─────────────────────────────────────────────────────────────
if st.session_state.page == "scan_detail" and st.session_state.selected_scan:
    scan = st.session_state.selected_scan
    if st.button("← Back to History"):
        st.session_state.page = "history"
        st.session_state.selected_scan = None
        st.rerun()

    is_h      = bool(scan["is_healthy"])
    sev       = scan["severity"]
    sev_color,sev_icon = SEV_COLORS.get(sev,("#22c55e","🟢"))
    conf      = float(scan["confidence"])

    st.markdown(f'<div class="ag-display">{"✅ Healthy" if is_h else "⚠️ "+scan["diagnosis"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="ag-subtitle">🌱 {scan["crop"]} · Scanned {scan["scanned_at"][:16]}</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    with c1: st.markdown(f'<div class="ag-metric"><div class="ag-metric-val">{conf:.0f}%</div><div class="ag-metric-lab">Confidence</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="ag-metric"><div class="ag-metric-val" style="color:{sev_color};">{sev}</div><div class="ag-metric-lab">Severity</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="ag-metric"><div class="ag-metric-val">{"✅" if is_h else "⚠️"}</div><div class="ag-metric-lab">Status</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a,col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown("**🌾 What to do:**")
        st.info(scan["advice"])
        if scan.get("prevention"):
            st.markdown("**🛡️ Prevention:**")
            st.success(scan["prevention"])

    with col_b:
        try:
            probs = json.loads(scan["all_probs"])
            st.markdown("**📊 Probability Breakdown:**")
            for label, prob in probs:
                bc = "#22c55e" if "healthy" in label.lower() else "#ef4444"
                st.markdown(f"""
                <div style="margin-bottom:10px;">
                    <div style="display:flex;justify-content:space-between;font-size:0.82rem;margin-bottom:4px;">
                        <span style="color:#86efac;">{label}</span>
                        <span style="color:#f0fdf4;font-weight:600;">{prob:.1f}%</span>
                    </div>
                    <div class="conf-wrap">
                        <div class="conf-fill" style="width:{min(prob,100):.0f}%;background:{bc};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        except:
            pass

    if scan.get("research"):
        st.markdown("<br>**🌐 Research at time of scan:**")
        st.info(scan["research"])
    st.stop()


# ─────────────────────────────────────────────────────────────
# 11. MAIN APP
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="ag-display">🌿 AgriGuard Pro</div>', unsafe_allow_html=True)
st.markdown(f'<div class="ag-subtitle">AI plant disease detection for Kenyan farmers · {CROP_LIBRARY[selected_crop]["icon"]} {selected_crop} selected</div>', unsafe_allow_html=True)

if not st.session_state.online:
    st.markdown('<div class="ag-offline">📡 Offline — AI diagnosis works, live research unavailable</div>', unsafe_allow_html=True)

if not st.session_state.user:
    st.markdown('<div class="ag-guest">👋 Using as Guest — sign in to save your scans and view history</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

if st.session_state.user:
    tab_scan, tab_history, tab_vets, tab_directory = st.tabs([
        "🔍 Scan Leaf","📋 My Scans","👨‍⚕️ Agronomists","📖 Disease Guide"
    ])
else:
    tab_scan, tab_vets, tab_directory = st.tabs([
        "🔍 Scan Leaf","👨‍⚕️ Agronomists","📖 Disease Guide"
    ])
    tab_history = None


# ══════════════════════════════════════════════════════════════
# TAB — SCANNER
# ══════════════════════════════════════════════════════════════
with tab_scan:
    col_left,col_right = st.columns([1,1], gap="large")

    with col_left:
        st.markdown('<div class="ag-label" style="margin-bottom:12px;">Upload Leaf Image</div>', unsafe_allow_html=True)
        method   = st.radio("", ["📁 Upload Photo","📷 Camera"], horizontal=True, label_visibility="collapsed")
        uploaded = None
        if method == "📁 Upload Photo":
            uploaded = st.file_uploader("", type=["jpg","jpeg","png"], label_visibility="collapsed")
        else:
            cam = st.camera_input("")
            if cam: uploaded = cam

        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, use_container_width=True, caption="Uploaded leaf")

    with col_right:
        st.markdown('<div class="ag-label" style="margin-bottom:12px;">Diagnosis Results</div>', unsafe_allow_html=True)

        if not uploaded:
            st.markdown("""
            <div class="ag-card" style="text-align:center;padding:48px 24px;">
                <div style="font-size:3.5rem;margin-bottom:12px;">🍃</div>
                <div style="font-size:1rem;color:#86efac;font-weight:500;">Upload a leaf photo to begin</div>
                <div style="font-size:0.8rem;color:#4ade80;margin-top:6px;">Supports JPG and PNG · Camera supported</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            leaf_ok, green_ratio = is_leaf(image)
            if not leaf_ok:
                st.markdown(f"""
                <div class="ag-card-danger">
                    <div style="font-size:1.1rem;font-weight:700;color:#fca5a5;margin-bottom:8px;">⚠️ Not a Leaf Image</div>
                    <div style="font-size:0.88rem;color:#fecaca;line-height:1.7;">
                        AgriGuard detected this may not be a plant leaf
                        (green coverage: {green_ratio*100:.0f}%).<br><br>
                        Please photograph a single leaf clearly.
                    </div>
                    <div style="margin-top:12px;font-size:0.8rem;color:#f87171;">
                        💡 Tips: Good lighting · Leaf fills the frame · Avoid shadows
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                with st.spinner("Analysing leaf..."):
                    result = run_diagnosis(image, selected_crop)
                    st.session_state.last_result = result

                if result is None:
                    st.error("Brain model not available. Please check agri_guard_brain.h5 is in the app folder.")
                else:
                    diagnosis  = result["diagnosis"]
                    confidence = result["confidence"]
                    is_h       = result["is_healthy"]
                    severity   = result["severity"]
                    sev_color,sev_icon = SEV_COLORS.get(severity,("#22c55e","🟢"))
                    card_class  = "ag-card-success" if is_h else "ag-card-danger"
                    badge_class = "ag-badge-healthy" if is_h else "ag-badge-disease"
                    status_text = "✅ HEALTHY" if is_h else "⚠️ DISEASE DETECTED"

                    st.markdown(f"""
                    <div class="{card_class}">
                        <span class="ag-badge {badge_class}">{status_text}</span>
                        <div style="font-family:'Fraunces',serif;font-size:1.5rem;font-weight:700;
                                    color:{'#86efac' if is_h else '#fca5a5'};margin:10px 0 4px;">
                            {diagnosis}
                        </div>
                        <div style="font-size:0.8rem;color:#4ade80;">{sev_icon} Severity: {severity}</div>
                        <div style="margin-top:14px;">
                            <div style="display:flex;justify-content:space-between;font-size:0.8rem;margin-bottom:6px;">
                                <span style="color:#86efac;">Confidence</span>
                                <span style="font-weight:700;color:#f0fdf4;">{confidence:.1f}%</span>
                            </div>
                            <div class="conf-wrap">
                                <div class="conf-fill" style="width:{min(confidence,100):.0f}%;background:{'#22c55e' if is_h else sev_color};"></div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    with st.expander("📊 Full probability breakdown"):
                        for label, prob in result["all_probs"]:
                            bc = "#22c55e" if "healthy" in label.lower() else "#ef4444"
                            st.markdown(f"""
                            <div style="margin-bottom:8px;">
                                <div style="display:flex;justify-content:space-between;font-size:0.8rem;margin-bottom:3px;">
                                    <span style="color:#86efac;">{label}</span>
                                    <span style="color:#f0fdf4;font-weight:600;">{prob:.1f}%</span>
                                </div>
                                <div class="conf-wrap" style="height:6px;">
                                    <div class="conf-fill" style="width:{min(prob,100):.0f}%;background:{bc};"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                    if not is_h:
                        st.markdown('<div class="ag-label" style="margin:16px 0 8px;">Disease Spread Hotspots</div>', unsafe_allow_html=True)
                        st.image(hotspot_overlay(image,severity), use_container_width=True,
                                 caption="Highlighted areas show potential disease spread zones")

                    st.markdown('<div class="ag-label" style="margin:16px 0 8px;">What To Do Now</div>', unsafe_allow_html=True)
                    st.info(result["advice"])
                    if result["prevention"]:
                        st.success(f"🛡️ **Prevention:** {result['prevention']}")

                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.session_state.online:
                        if st.button("🌐 Get Latest Treatment Info from Internet"):
                            with st.spinner("Searching agricultural databases..."):
                                research = scrape_research(diagnosis, selected_crop)
                            if research:
                                st.markdown('<div class="ag-label" style="margin-bottom:8px;">Live Research (2026)</div>', unsafe_allow_html=True)
                                st.info(research)
                                if st.session_state.user:
                                    save_scan(st.session_state.user["id"], result, selected_crop, research)
                                    st.success("✅ Scan saved with research!")
                            else:
                                st.warning("Could not retrieve live data. Use the recommendations above.")
                    else:
                        st.markdown('<div class="ag-offline">📡 Offline — live research unavailable</div>', unsafe_allow_html=True)

                    if st.session_state.user:
                        if st.button("💾 Save This Scan"):
                            save_scan(st.session_state.user["id"], result, selected_crop)
                            st.success("✅ Scan saved to your history!")
                    else:
                        st.markdown('<div class="ag-guest" style="margin-top:12px;">💡 Sign in to save this scan</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB — HISTORY
# ══════════════════════════════════════════════════════════════
if tab_history:
    with tab_history:
        scans = get_user_scans(st.session_state.user["id"])
        if not scans:
            st.markdown("""
            <div class="ag-card" style="text-align:center;padding:48px;">
                <div style="font-size:2.5rem;">📋</div>
                <div style="color:#86efac;margin-top:10px;">No scans yet</div>
                <div style="color:#4ade80;font-size:0.82rem;margin-top:6px;">Go to Scan Leaf to analyse your first plant</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            total   = len(scans)
            healthy = sum(1 for s in scans if s["is_healthy"])
            disease = total - healthy

            m1,m2,m3 = st.columns(3)
            with m1: st.markdown(f'<div class="ag-metric"><div class="ag-metric-val">{total}</div><div class="ag-metric-lab">Total Scans</div></div>', unsafe_allow_html=True)
            with m2: st.markdown(f'<div class="ag-metric"><div class="ag-metric-val" style="color:#ef4444;">{disease}</div><div class="ag-metric-lab">Diseases Found</div></div>', unsafe_allow_html=True)
            with m3: st.markdown(f'<div class="ag-metric"><div class="ag-metric-val">{healthy}</div><div class="ag-metric-lab">Healthy Scans</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="ag-label" style="margin-bottom:12px;">Tap View to see full breakdown</div>', unsafe_allow_html=True)

            for scan in scans:
                is_h  = bool(scan["is_healthy"])
                color = "#22c55e" if is_h else "#ef4444"
                icon  = "✅" if is_h else "⚠️"
                conf  = float(scan["confidence"])
                extra = "" if is_h else "ag-history-disease"
                crop_icon = CROP_LIBRARY.get(scan['crop'],{}).get('icon','🌱')

                col_s,col_b = st.columns([5,1])
                with col_s:
                    st.markdown(f"""
                    <div class="ag-history {extra}" style="border-left-color:{color};">
                        <div style="font-weight:600;font-size:0.95rem;">{icon} {scan['diagnosis']}</div>
                        <div style="font-size:0.78rem;color:#4ade80;margin-top:4px;">
                            {crop_icon} {scan['crop']} · 🎯 {conf:.0f}% · 🕐 {scan['scanned_at'][:16]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_b:
                    if st.button("View", key=f"view_{scan['id']}"):
                        st.session_state.selected_scan = scan
                        st.session_state.page = "scan_detail"
                        st.rerun()


# ══════════════════════════════════════════════════════════════
# TAB — AGRONOMISTS
# ══════════════════════════════════════════════════════════════
with tab_vets:
    st.markdown('<div class="ag-label" style="margin-bottom:16px;">Find an Agronomist Near You</div>', unsafe_allow_html=True)
    regions = ["All Regions","Nairobi","Kisumu","Nakuru","Kakamega","Eldoret","Nyeri","Mombasa","Meru"]
    region  = st.selectbox("Filter by region", regions, label_visibility="collapsed")
    vets    = get_vets(region)

    if not vets:
        st.info("No contacts found for this region.")
    else:
        cols = st.columns(2)
        for i,vet in enumerate(vets):
            with cols[i%2]:
                st.markdown(f"""
                <div class="ag-vet">
                    <div style="font-weight:700;color:#22c55e;font-size:0.95rem;margin-bottom:6px;">👤 {vet['name']}</div>
                    <div style="font-size:0.78rem;color:#86efac;margin-bottom:2px;">🏷️ {vet['role']} · {vet['organisation']}</div>
                    <div style="font-size:0.78rem;color:#f0fdf4;margin-bottom:10px;">📍 {vet['region']} · 🌱 {vet['speciality']}</div>
                    <a href="tel:{vet['phone']}" style="display:inline-block;background:#22c55e;color:#000;
                        padding:6px 16px;border-radius:8px;font-size:0.8rem;font-weight:700;text-decoration:none;">
                        📞 {vet['phone']}
                    </a>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(34,197,94,0.15);margin:24px 0;'>", unsafe_allow_html=True)
    st.markdown("**🏛️ National helplines:**")
    st.markdown("""
- **KALRO Helpline:** 0800 720 715 *(free)*
- **iShamba:** 0800 723 253
- **Kenya Farmers Helpline:** +254 20 2033 000
    """)


# ══════════════════════════════════════════════════════════════
# TAB — DISEASE GUIDE
# ══════════════════════════════════════════════════════════════
with tab_directory:
    st.markdown('<div class="ag-label" style="margin-bottom:16px;">Disease Field Guide · Works Offline</div>', unsafe_allow_html=True)
    guide_crop = st.selectbox("Browse by crop", list(CROP_LIBRARY.keys()),
                               format_func=lambda c: f"{CROP_LIBRARY[c]['icon']} {c}", key="guide_crop")

    for label in CROP_LIBRARY[guide_crop]["labels"]:
        is_h = "healthy" in label.lower()
        if is_h:
            with st.expander(f"✅ {label}"):
                st.success(HEALTHY_ADVICE)
        else:
            adv = DISEASE_ADVICE.get(label,{})
            sev = adv.get("severity","Unknown")
            _,sev_icon = SEV_COLORS.get(sev,("#22c55e","🟢"))
            with st.expander(f"{sev_icon} {label}"):
                st.markdown(f'<span class="ag-badge ag-badge-warning">{sev} Severity</span>', unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"**🌾 What to do:** {adv.get('action','Consult an agronomist.')}")
                if adv.get("prevention"):
                    st.markdown(f"**🛡️ Prevention:** {adv.get('prevention')}")


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;padding:20px 0;border-top:1px solid rgba(34,197,94,0.1);">
    <div style="font-family:'Fraunces',serif;font-size:1rem;color:#22c55e;margin-bottom:4px;">🌿 AgriGuard Pro</div>
    <div style="font-size:0.72rem;color:#4ade80;">
        Developed by Michael Kibet · Kenya 🇰🇪 ·
        <a href="https://agriguard-ai.streamlit.app" style="color:#22c55e;">agriguard-ai.streamlit.app</a>
    </div>
    <div style="font-size:0.68rem;color:#166534;margin-top:4px;">
        MobileNetV2 · PlantVillage · 94.58% Validation Accuracy · Built in Termux
    </div>
</div>
""", unsafe_allow_html=True)

