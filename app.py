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
# CSS
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
    color: #F1F5F9 !important;
}

.main .block-container {
    max-width: 1450px;
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}

.hero-box {
    background: linear-gradient(135deg, rgba(12, 27, 52, 0.96), rgba(25, 62, 118, 0.92));
    border-radius: 24px;
    padding: 34px 34px 28px 34px;
    box-shadow: 0 18px 38px rgba(12, 27, 52, 0.18);
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 18px;
}

.hero-title {
    color: #ffffff;
    font-size: 40px;
    font-weight: 800;
    line-height: 1.15;
    margin-bottom: 8px;
    letter-spacing: -0.02em;
}

.hero-subtitle {
    color: #d9e8ff;
    font-size: 17px;
    line-height: 1.6;
    max-width: 980px;
}

.metric-card {
    background: rgba(255,255,255,0.14);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 20px;
    padding: 18px 20px;
    box-shadow: 0 10px 24px rgba(16, 32, 51, 0.08);
    min-height: 120px;
}

.metric-label {
    color: #CBD5E1;
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 8px;
}

.metric-value {
    color: #FFFFFF;
    font-size: 26px;
    font-weight: 800;
}

.metric-subvalue {
    color: #BFD7FF;
    font-size: 14px;
    margin-top: 8px;
    font-weight: 600;
}

.section-box {
    background: rgba(255,255,255,0.10);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border-radius: 22px;
    padding: 22px 22px 18px 22px;
    box-shadow: 0 10px 24px rgba(16, 32, 51, 0.10);
    border: 1px solid rgba(255,255,255,0.18);
    margin-top: 14px;
    margin-bottom: 16px;
}

.section-title {
    color: #FFFFFF;
    font-size: 22px;
    font-weight: 800;
    margin-bottom: 6px;
}

.section-subtitle {
    color: #D6E4F5;
    font-size: 14px;
    margin-bottom: 16px;
}

.info-panel {
    background: linear-gradient(135deg, rgba(15, 118, 110, 0.18), rgba(23, 78, 166, 0.16));
    border-left: 6px solid #60A5FA;
    border-radius: 18px;
    padding: 18px 20px;
    color: #E2E8F0;
    line-height: 1.8;
    font-size: 15px;
}

.guide-text {
    font-size: 12.5px;
    color: #D6E4F5;
    margin-top: -6px;
    margin-bottom: 12px;
    line-height: 1.5;
}

.pred-box {
    background: linear-gradient(135deg, rgba(23, 78, 166, 0.20), rgba(15, 118, 110, 0.18));
    border: 1px solid rgba(96, 165, 250, 0.20);
    border-radius: 22px;
    padding: 28px;
    text-align: center;
    box-shadow: 0 12px 26px rgba(16, 32, 51, 0.08);
}

.pred-title {
    color: #D6E4F5;
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 10px;
}

.pred-value {
    color: #FFFFFF;
    font-size: 56px;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 10px;
}

.pred-meta {
    color: #E2E8F0;
    font-size: 15px;
    line-height: 1.7;
}

div[data-baseweb="input"] > div {
    background-color: rgba(255,255,255,0.96) !important;
    border: 1px solid #d6e0ea !important;
    border-radius: 14px !important;
    box-shadow: 0 4px 12px rgba(16, 32, 51, 0.04);
}

div[data-baseweb="input"] input {
    color: #102033 !important;
    font-size: 17px !important;
    font-weight: 600 !important;
}

label, .stNumberInput label {
    color: #E2E8F0 !important;
    font-weight: 700 !important;
    font-size: 15px !important;
}

.stButton > button {
    width: 100%;
    border-radius: 16px;
    padding: 15px 22px;
    font-size: 18px;
    font-weight: 800;
    color: white !important;
    background: linear-gradient(90deg, #174ea6, #0f766e);
    border: none;
    box-shadow: 0 12px 24px rgba(23, 78, 166, 0.20);
    transition: all 0.2s ease-in-out;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 16px 28px rgba(23, 78, 166, 0.24);
}

[data-testid="stDataFrame"] {
    background: rgba(255,255,255,0.10) !important;
    border-radius: 16px !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
    overflow: hidden !important;
}

[data-testid="stDataFrame"] * {
    color: #F1F5F9 !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f7fbff 0%, #eef5fc 100%);
    border-right: 1px solid #dbe6f0;
}

/* Sidebar text fix */
[data-testid="stSidebar"] * {
    color: #102033 !important;
}

.sidebar-box {
    background: rgba(255,255,255,0.95);
    border: 1px solid rgba(16, 32, 51, 0.10);
    border-radius: 18px;
    padding: 16px;
    margin-bottom: 14px;
    box-shadow: 0 8px 20px rgba(16, 32, 51, 0.06);
}

.footer-note {
    color: #D6E4F5;
    font-size: 13px;
    text-align: center;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)
/* Download button fix */
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
# =========================================================
# FUNCTIONS
# =========================================================
def get_status(value, var_name):
    if input_ranges is None or var_name not in input_ranges:
        return "Range unavailable", "⚪", "Unknown"

    r = input_ranges[var_name]
    if value < r["min"] or value > r["max"]:
        return "Out of observed range", "🔴", "Low"
    elif value < r["p5"] or value > r["p95"]:
        return "Outside recommended range", "🟡", "Moderate"
    else:
        return "Within recommended range", "🟢", "High"

def get_reliability_label(input_dict):
    if input_ranges is None:
        return "Unknown"

    statuses = [get_status(v, k)[0] for k, v in input_dict.items()]
    if "Out of observed range" in statuses:
        return "Low"
    elif "Outside recommended range" in statuses:
        return "Moderate"
    return "High"

def make_input(var_name):
    if input_ranges is not None and var_name in input_ranges:
        r = input_ranges[var_name]
        value = st.number_input(
            labels.get(var_name, var_name),
            min_value=float(r["min"]),
            max_value=float(r["max"]),
            value=float(r["median"]),
            step=0.01,
            format="%.3f",
            key=var_name
        )
        st.markdown(
            f'<div class="guide-text">Recommended: <b>{r["p5"]:.3f} to {r["p95"]:.3f}</b> '
            f'| Typical: {r["p25"]:.3f} to {r["p75"]:.3f} '
            f'| Observed: {r["min"]:.3f} to {r["max"]:.3f}</div>',
            unsafe_allow_html=True
        )
    else:
        value = st.number_input(
            labels.get(var_name, var_name),
            value=float(default_values.get(var_name, 0.0)),
            step=0.01,
            format="%.3f",
            key=var_name
        )
        st.markdown(
            '<div class="guide-text">Range information unavailable because input_ranges.json is missing.</div>',
            unsafe_allow_html=True
        )
    return value

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown('<div class="sidebar-box">', unsafe_allow_html=True)
    st.markdown("### Model Overview")
    st.write(f"**Model:** {best_model_name}")
    if "test_r2_mean" in metadata:
        st.write(f'**Test R² Mean:** {metadata["test_r2_mean"]:.3f}')
    if "test_rmse_mean" in metadata:
        st.write(f'**Test RMSE Mean:** {metadata["test_rmse_mean"]:.3f}')
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-box">', unsafe_allow_html=True)
    st.markdown("### Scientific Scope")
    st.write("- MCF-7 environment")
    st.write("- Synthetic polymeric nanoparticles")
    st.write("- Post-drug-loading properties")
    st.write("- Research-use-only application")
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# HERO SECTION
# =========================================================
st.markdown(
    """
    <div class="hero-box">
        <div class="hero-title">MCF-7 Synthetic Polymeric Nanoparticle Drug Release Predictor</div>
        <div class="hero-subtitle">
            A machine learning–based scientific platform for estimating drug release amount (%) from synthetic polymeric nanoparticles under MCF-7-related experimental conditions. This interface is designed for professional research presentation, decision support, and rapid exploratory analysis.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# TOP METRIC CARDS
# =========================================================
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">Prediction Target</div>
            <div class="metric-value">Drug Release Amount (%)</div>
            <div class="metric-subvalue">Regression-based output</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Input Variables</div>
            <div class="metric-value">{len(feature_cols)} Scientific Features</div>
            <div class="metric-subvalue">Nanoparticle + release conditions</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c3:
    test_r2_text = f'{metadata["test_r2_mean"]:.3f}' if "test_r2_mean" in metadata else "Available"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Model Performance</div>
            <div class="metric-value">Test R² = {test_r2_text}</div>
            <div class="metric-subvalue">Loaded from metadata</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================================================
# NOTES
# =========================================================
st.markdown('<div class="section-box">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Scientific Notes</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="info-panel">
        <b>1.</b> This application is intended specifically for the <b>MCF-7 environment</b>.<br><br>
        <b>2.</b> The model is appropriate for <b>synthetic polymeric nanoparticle systems</b> and is not intended for natural or semi-synthetic polymer systems.<br><br>
        <b>3.</b> All nanoparticle-related inputs should be entered using <b>post-drug-loading values</b>.<br><br>
        <b>4.</b> Prediction reliability is highest when the entered values remain within the recommended range derived from the training dataset.<br><br>
        <b>5.</b> This tool is intended for <b>research support only</b> and should not replace experimental validation.
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True)

if input_ranges is None:
    st.warning("input_ranges.json was not found. The app can still predict, but range-based reliability guidance will be limited.")

# =========================================================
# INPUT SECTION
# =========================================================
st.markdown('<div class="section-box">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Enter Experimental Parameters</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">Provide nanoparticle characteristics and release conditions to generate a prediction.</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
inputs = {}

for i, var in enumerate(feature_cols):
    with (col1 if i < 4 else col2):
        inputs[var] = make_input(var)

st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# VALIDATION TABLE
# =========================================================
rows = []
for var, value in inputs.items():
    status_text, icon, reliability = get_status(value, var)
    rows.append({
        "Variable": short_names.get(var, var),
        "Scientific Label": labels.get(var, var),
        "Value": f"{value:.3f}",
        "Status": f"{icon} {status_text}",
        "Reliability": reliability
    })

summary_df = pd.DataFrame(rows)
overall_reliability = get_reliability_label(inputs)

st.markdown('<div class="section-box">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Input Validation Summary</div>', unsafe_allow_html=True)
st.markdown(f'<div class="section-subtitle">Overall input reliability category: <b>{overall_reliability}</b></div>', unsafe_allow_html=True)
st.dataframe(summary_df, use_container_width=True, hide_index=True)

if overall_reliability == "Low":
    st.error("One or more inputs are outside the observed training-data range. Prediction reliability is low.")
elif overall_reliability == "Moderate":
    st.warning("One or more inputs are outside the recommended range. Prediction can still be generated, but reliability is lower.")
elif overall_reliability == "High":
    st.success("All inputs are within the recommended training-data range.")
else:
    st.info("Range guidance is unavailable because input_ranges.json is missing.")
st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# PREDICTION
# =========================================================
st.markdown('<div class="section-box">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Prediction Panel</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">Click the button below to estimate drug release amount (%).</div>', unsafe_allow_html=True)

if st.button("Predict Drug Release Amount (%)", use_container_width=True):
    X = [[inputs[col] for col in feature_cols]]
    pred = float(model.predict(X)[0])

    st.markdown(
        f"""
        <div class="pred-box">
            <div class="pred-title">Predicted Drug Release Amount</div>
            <div class="pred-value">{pred:.3f}%</div>
            <div class="pred-meta">
                Model used: <b>{best_model_name}</b><br>
                Input reliability category: <b>{overall_reliability}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    pred_df = pd.DataFrame({
        "Item": [
            "Predicted Drug Release Amount (%)",
            "Model Used",
            "Input Reliability"
        ],
        "Value": [
            f"{pred:.3f}",
            best_model_name,
            overall_reliability
        ]
    })

    st.markdown("")
    st.dataframe(pred_df, use_container_width=True, hide_index=True)

st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# CURRENT INPUTS
# =========================================================
st.markdown('<div class="section-box">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Current Input Profile</div>', unsafe_allow_html=True)
current_df = pd.DataFrame({
    "Variable": [short_names.get(col, col) for col in feature_cols],
    "Scientific Label": [labels.get(col, col) for col in feature_cols],
    "Value": [f"{inputs[col]:.3f}" for col in feature_cols]
})
st.dataframe(current_df, use_container_width=True, hide_index=True)

csv_data = current_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Current Input Profile (CSV)",
    data=csv_data,
    file_name="current_input_profile.csv",
    mime="text/csv",
    use_container_width=True
)
st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================
st.markdown(
    '<div class="footer-note">Research-use-only application. Interpret predictions together with the underlying experimental design and dataset limitations.</div>',
    unsafe_allow_html=True
)
