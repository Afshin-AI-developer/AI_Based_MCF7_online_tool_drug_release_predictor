# -*- coding: utf-8 -*-
"""
Professional Streamlit app for predicting drug release amount (%)
Designed for a scientific, premium, publication-style interface.
"""

import os
import json
import joblib
import pandas as pd
import streamlit as st

# =========================================================
# PATHS
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
METADATA_PATH = os.path.join(BASE_DIR, "best_model_metadata.json")
RANGES_PATH = os.path.join(BASE_DIR, "input_ranges.json")

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="MCF-7 Nanoparticle Drug Release Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    model_name = metadata["best_model_name"]
    model_path = os.path.join(BASE_DIR, f"Best_{model_name}_model.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)
    return model, metadata

# =========================================================
# LOAD RANGES IF AVAILABLE
# =========================================================
@st.cache_data
def load_ranges():
    if os.path.exists(RANGES_PATH):
        with open(RANGES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

model, metadata = load_model()
input_ranges = load_ranges()

feature_cols = metadata["feature_columns"]
best_model_name = metadata["best_model_name"]

# =========================================================
# LABELS
# =========================================================
labels = {
    "size (DLS) of nanoparticle (nm)-mean": "Nanoparticle Size (DLS, nm)",
    "Polydispersity Index (PDI) of nanoparticle-mean": "Polydispersity Index (PDI)",
    "Zeta potential of nanoparticle (mV)-mean": "Zeta Potential (mV)",
    "Drug loading capacity (%)-mean": "Drug Loading Capacity (%)",
    "Entrapment efficiency of nanoparticle (%)-mean": "Entrapment Efficiency (%)",
    "Temperature °C": "Temperature (°C)",
    "PH": "pH",
    "Time of Drug release (h)": "Drug Release Time (h)"
}

short_names = {
    "size (DLS) of nanoparticle (nm)-mean": "Size",
    "Polydispersity Index (PDI) of nanoparticle-mean": "PDI",
    "Zeta potential of nanoparticle (mV)-mean": "Zeta",
    "Drug loading capacity (%)-mean": "DLC",
    "Entrapment efficiency of nanoparticle (%)-mean": "EE",
    "Temperature °C": "Temp",
    "PH": "pH",
    "Time of Drug release (h)": "Time"
}

default_values = {
    "size (DLS) of nanoparticle (nm)-mean": 100.000,
    "Polydispersity Index (PDI) of nanoparticle-mean": 0.200,
    "Zeta potential of nanoparticle (mV)-mean": -20.000,
    "Drug loading capacity (%)-mean": 10.000,
    "Entrapment efficiency of nanoparticle (%)-mean": 70.000,
    "Temperature °C": 37.000,
    "PH": 7.400,
    "Time of Drug release (h)": 24.000
}

# =========================================================
# CSS (FIXED STRUCTURE + DOWNLOAD BUTTON ADDED)
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #F1F5F9 !important;
}

.stApp {
    background:
        linear-gradient(rgba(8, 20, 40, 0.88), rgba(8, 20, 40, 0.88)),
        url("https://images.unsplash.com/photo-1532187863486-abf9dbad1b69?auto=format&fit=crop&w=1600&q=80");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Sidebar text fix */
[data-testid="stSidebar"] * {
    color: #102033 !important;
}

/* Table text fix */
[data-testid="stDataFrame"] * {
    color: #F1F5F9 !important;
}

/* DOWNLOAD BUTTON FIX (YOUR REQUEST) */
.stDownloadButton > button {
    width: 100%;
    border-radius: 16px;
    padding: 15px 22px;
    font-size: 18px;
    font-weight: 800;
    color: #FFFFFF !important;
    background: linear-gradient(90deg, #174ea6, #0f766e);
    border: none;
    box-shadow: 0 12px 24px rgba(23, 78, 166, 0.20);
}

.stDownloadButton > button:hover {
    background: linear-gradient(90deg, #1d4ed8, #047857);
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# INPUT SECTION
# =========================================================
st.title("🧬 Nanoparticle Drug Release Predictor")

inputs = {}
col1, col2 = st.columns(2)

for i, var in enumerate(feature_cols):
    with (col1 if i < 4 else col2):
        if input_ranges and var in input_ranges:
            r = input_ranges[var]
            value = st.number_input(
                labels[var],
                min_value=float(r["min"]),
                max_value=float(r["max"]),
                value=float(r["median"]),
                step=0.01
            )
        else:
            value = st.number_input(labels[var], value=default_values[var])
        inputs[var] = value

# =========================================================
# PREDICTION
# =========================================================
if st.button("Predict Drug Release (%)"):
    X = [[inputs[col] for col in feature_cols]]
    pred = model.predict(X)[0]
    st.success(f"Predicted Drug Release: {pred:.3f}%")

# =========================================================
# CURRENT INPUT TABLE
# =========================================================
current_df = pd.DataFrame({
    "Variable": [short_names[col] for col in feature_cols],
    "Scientific Label": [labels[col] for col in feature_cols],
    "Value": [f"{inputs[col]:.3f}" for col in feature_cols]
})

st.dataframe(current_df, use_container_width=True)

# =========================================================
# DOWNLOAD BUTTON
# =========================================================
csv_data = current_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Current Input Profile (CSV)",
    data=csv_data,
    file_name="current_input_profile.csv",
    mime="text/csv",
    use_container_width=True
)
