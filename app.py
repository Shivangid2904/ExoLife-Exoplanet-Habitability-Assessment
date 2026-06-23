import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import io
import matplotlib.pyplot as plt

from src.predict import predict, predict_proba, load_model
from src.explain import explain_prediction, export_html_explanation
from src.preprocess import BASELINE_FEATURES, PROXY_FEATURES

# Resolve base directories path-independently
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_absolute_path(relative_path):
    return os.path.join(BASE_DIR, relative_path)

# Page configuration
st.set_page_config(
    page_title="ExoLife Assessment Platform",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom branding CSS styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        color: #1E90FF;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #B0C4DE;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1e293b;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1E90FF;
        margin-bottom: 1rem;
    }
    .leakage-warning {
        background-color: #3b1e1e;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ff4b4b;
        color: #ffcccc;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# App Title & Branding Header
st.markdown("<div class='main-title'>🪐 ExoLife Exoplanet Habitability Platform</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>AI-Powered Exoplanet Habitability Assessment Platform • Dual-Model Framework</div>", unsafe_allow_html=True)

# ----------------- SIDEBAR: MODEL SELECTOR & PRESETS -----------------
st.sidebar.markdown("### 🛠️ Configuration")

# Model Mode Selector
model_mode_display = st.sidebar.radio(
    "Select Model Mode",
    [
        "Enhanced Habitability Model (All Features)",
        "Physics-Informed Proxy Model"
    ]
)
model_mode = "baseline" if "All Features" in model_mode_display else "proxy"

st.sidebar.markdown("---")

# Planetary Presets definition
presets = {
    "Custom (Manual Input)": None,
    "Earth (Habitable Baseline)": {
        "pl_rade": 1.0, "pl_bmasse": 1.0, "pl_orbper": 365.25, "pl_eqt": 288.0, "pl_insol": 1.0,
        "pl_orbeccen": 0.0167, "st_teff": 5778.0, "st_rad": 1.0, "st_mass": 1.0, "st_met": 0.0, "sy_dist": 0.0
    },
    "Kepler-22b (Super-Earth candidate)": {
        "pl_rade": 2.4, "pl_bmasse": 8.3, "pl_orbper": 289.86, "pl_eqt": 262.0, "pl_insol": 1.1,
        "pl_orbeccen": 0.0, "st_teff": 5518.0, "st_rad": 0.989, "st_mass": 0.97, "st_met": -0.03, "sy_dist": 195.0
    },
    "Kepler-186f (Earth-sized HZ planet)": {
        "pl_rade": 1.17, "pl_bmasse": 1.4, "pl_orbper": 129.9, "pl_eqt": 188.0, "pl_insol": 0.32,
        "pl_orbeccen": 0.04, "st_teff": 3755.0, "st_rad": 0.52, "st_mass": 0.48, "st_met": -0.28, "sy_dist": 179.0
    },
    "Mars (Cold Rocky Desert)": {
        "pl_rade": 0.53, "pl_bmasse": 0.11, "pl_orbper": 687.0, "pl_eqt": 210.0, "pl_insol": 0.43,
        "pl_orbeccen": 0.0934, "st_teff": 5778.0, "st_rad": 1.0, "st_mass": 1.0, "st_met": 0.0, "sy_dist": 0.0
    },
    "Kepler-10b (Lava World)": {
        "pl_rade": 1.47, "pl_bmasse": 4.6, "pl_orbper": 0.84, "pl_eqt": 2130.0, "pl_insol": 3560.0,
        "pl_orbeccen": 0.0, "st_teff": 5627.0, "st_rad": 1.06, "st_mass": 0.91, "st_met": -0.15, "sy_dist": 186.0
    }
}

selected_preset = st.sidebar.selectbox("🚀 Load Exoplanet Preset", list(presets.keys()))
preset_data = presets[selected_preset]

def get_val(key, default):
    if preset_data is not None and key in preset_data:
        return float(preset_data[key])
    return float(default)

# ----------------- SESSION STATE SYSTEM -----------------
if 'inputs' not in st.session_state:
    st.session_state.inputs = None
if 'model_mode' not in st.session_state:
    st.session_state.model_mode = "baseline"

# ----------------- APP TAB NAVIGATION -----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🌌 Habitability Assessment",
    "📊 Model Comparison",
    "🧠 Explainability (SHAP)",
    "📖 About Project"
])

# =====================================================================
# TAB 1: HABITABILITY ASSESSMENT
# =====================================================================
with tab1:
    st.markdown("### 🪐 Exoplanet Parameter Input")
    st.write("Configure the planetary and stellar properties below to assess habitability potential.")
    
    col1, col2 = st.columns(2)
    
    # Check if we should disable direct-rule features in Proxy Mode
    proxy_disabled = (model_mode == "proxy")
    
    if proxy_disabled:
        st.info("💡 **Proxy Model Active**: Direct-rule boundary features (`pl_rade`, `pl_eqt`, and `pl_insol`) are automatically disabled and ignored. Predictions will be generated using only secondary physical proxies.")
        
    with col1:
        st.markdown("#### 🌍 Planetary Properties")
        # Direct rule features (disabled in Proxy mode)
        pl_rade = st.number_input("Planet Radius (Earth radii)", 0.0, 20.0, get_val("pl_rade", 1.0), step=0.1, disabled=proxy_disabled, key="val_pl_rade")
        pl_eqt = st.number_input("Equilibrium Temperature (Kelvin)", 0.0, 3000.0, get_val("pl_eqt", 288.0), step=5.0, disabled=proxy_disabled, key="val_pl_eqt")
        pl_insol = st.number_input("Insolation Flux (Earth flux units)", 0.0, 5000.0, get_val("pl_insol", 1.0), step=0.1, disabled=proxy_disabled, key="val_pl_insol")
        # Allowed proxy features
        pl_bmasse = st.number_input("Planet Mass (Earth masses)", 0.0, 5000.0, get_val("pl_bmasse", 1.0), step=0.1, key="val_pl_bmasse")
        pl_orbper = st.number_input("Orbital Period (days)", 0.0, 20000.0, get_val("pl_orbper", 365.25), step=1.0, key="val_pl_orbper")
        pl_orbeccen = st.number_input("Orbital Eccentricity", 0.0, 1.0, get_val("pl_orbeccen", 0.0), step=0.01, key="val_pl_orbeccen")
        
    with col2:
        st.markdown("#### ☀️ Stellar & System Properties")
        st_teff = st.number_input("Stellar Effective Temperature (Kelvin)", 0.0, 25000.0, get_val("st_teff", 5778.0), step=50.0, key="val_st_teff")
        st_rad = st.number_input("Stellar Radius (Solar radii)", 0.0, 50.0, get_val("st_rad", 1.0), step=0.05, key="val_st_rad")
        st_mass = st.number_input("Stellar Mass (Solar masses)", 0.0, 20.0, get_val("st_mass", 1.0), step=0.05, key="val_st_mass")
        st_met = st.number_input("Stellar Metallicity (dex)", -3.0, 2.0, get_val("st_met", 0.0), step=0.05, key="val_st_met")
        sy_dist = st.number_input("System Distance (parsecs)", 0.0, 15000.0, get_val("sy_dist", 0.0), step=1.0, key="val_sy_dist")
        
    # Gather inputs
    inputs = {
        "pl_rade": pl_rade,
        "pl_bmasse": pl_bmasse,
        "pl_orbper": pl_orbper,
        "pl_eqt": pl_eqt,
        "pl_insol": pl_insol,
        "pl_orbeccen": pl_orbeccen,
        "st_teff": st_teff,
        "st_rad": st_rad,
        "st_mass": st_mass,
        "st_met": st_met,
        "sy_dist": sy_dist
    }
    
    st.markdown("---")
    
    if st.button("🔮 Run Habitability Assessment", type="primary"):
        # Save inputs and mode to session state for SHAP tab access
        st.session_state.inputs = inputs
        st.session_state.model_mode = model_mode
        
        # Run inference
        pred = predict(inputs, model_mode)
        score = predict_proba(inputs, model_mode)
        
        # Classification category based on confidence threshold
        if score < 0.30:
            classification = "Low Potential"
            color = "#ff4b4b"
        elif score < 0.70:
            classification = "Moderate Potential"
            color = "#ffa500"
        else:
            classification = "High Potential"
            color = "#00e676"
            
        # Display Results
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.write("### 🧾 Assessment Results")
            if pred == 1:
                st.success("✅ **HABITABLE CANDIDATE 🌍**")
            else:
                st.error("❌ **NOT HABITABLE CANDIDATE**")
                
            st.markdown(f"""
            <div class='metric-card'>
                <h4>Predictive Assessment Statistics</h4>
                <p><b>Model Selected:</b> {model_mode_display}</p>
                <p><b>Habitability Score:</b> <span style='font-size: 1.5rem; color:#1E90FF; font-weight:bold;'>{score*100:.1f}%</span></p>
                <p><b>Confidence Category:</b> <span style='font-size: 1.2rem; color:{color}; font-weight:bold;'>{classification}</span></p>
            </div>
            """, unsafe_allow_html=True)
            
        with res_col2:
            st.write("### 📏 Parameter Summary")
            features_to_show = PROXY_FEATURES if model_mode == "proxy" else BASELINE_FEATURES
            summary_df = pd.DataFrame({
                "Feature Description": [f for f in features_to_show],
                "Input Value": [inputs[f] for f in features_to_show]
            })
            st.dataframe(summary_df, use_container_width=True)

# =====================================================================
# TAB 2: MODEL COMPARISON
# =====================================================================
with tab2:
    st.markdown("### 📊 Model Architecture Comparison")
    st.write("ExoLife compares a rule-leaky baseline model with a physics-informed proxy model to evaluate performance under realistic scientific constraints.")
    
    # Load comparison metrics JSON
    metrics_path = get_absolute_path("models/metrics_comparison.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
            
        # Structure side-by-side comparative dataframe
        metrics_df = pd.DataFrame({
            "Evaluation Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "Mean CV F1 Score"],
            "Enhanced Habitability Model (All Features)": [
                f"{metrics['baseline']['holdout_accuracy']*100:.2f}%",
                f"{metrics['baseline']['holdout_precision']*100:.2f}%",
                f"{metrics['baseline']['holdout_recall']*100:.2f}%",
                f"{metrics['baseline']['holdout_f1']*100:.2f}%",
                f"{metrics['baseline']['holdout_roc_auc']*100:.2f}%",
                f"{metrics['baseline']['cv_mean_f1']*100:.2f}%"
            ],
            "Physics-Informed Proxy Model": [
                f"{metrics['proxy']['holdout_accuracy']*100:.2f}%",
                f"{metrics['proxy']['holdout_precision']*100:.2f}%",
                f"{metrics['proxy']['holdout_recall']*100:.2f}%",
                f"{metrics['proxy']['holdout_f1']*100:.2f}%",
                f"{metrics['proxy']['holdout_roc_auc']*100:.2f}%",
                f"{metrics['proxy']['cv_mean_f1']*100:.2f}%"
            ]
        })
        
        col_m1, col_m2 = st.columns([2, 1])
        with col_m1:
            st.write("#### 📈 Shared holdout Evaluation metrics")
            st.table(metrics_df)
        with col_m2:
            st.write("#### ℹ️ Target Leakage Warning")
            st.markdown("""
            <div class='leakage-warning'>
                <h5>⚠️ Target Leakage Alert</h5>
                <p>The <b>Enhanced Habitability Model</b> achieves a nearly perfect F1 score because it has direct access to the 3 features that mathematically define the label. It merely memorizes the rectangular cuts of the definition.</p>
                <p>The <b>Physics-Informed Proxy Model</b> is forced to predict habitability using only secondary features. Its performance represents an authentic machine learning challenge, demonstrating genuine physical mapping.</p>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        
        # Display side-by-side confusion matrix images
        st.write("#### 🗺️ Confusion Matrices Comparison")
        col_cm1, col_cm2 = st.columns(2)
        
        cm_path_base = get_absolute_path("outputs/evaluation/confusion_matrix_baseline.png")
        cm_path_proxy = get_absolute_path("outputs/evaluation/confusion_matrix_proxy.png")
        
        with col_cm1:
            st.write("**Model A: Baseline (Leaky)**")
            if os.path.exists(cm_path_base):
                st.image(cm_path_base, use_container_width=True)
            else:
                st.info("Baseline confusion matrix plot not found.")
                
        with col_cm2:
            st.write("**Model B: Proxy (Physics-Based)**")
            if os.path.exists(cm_path_proxy):
                st.image(cm_path_proxy, use_container_width=True)
            else:
                st.info("Proxy confusion matrix plot not found.")
    else:
        st.warning("Comparative metrics file `models/metrics_comparison.json` not found. Please run model training and evaluation scripts first.")

# =====================================================================
# TAB 3: EXPLAINABILITY (SHAP)
# =====================================================================
with tab3:
    st.markdown("### 🧠 Explainable AI (SHAP) Analysis")
    st.write("SHAP (SHapley Additive exPlanations) values show how each input parameter pushes the model to predict habitability or non-habitability.")
    
    # waterfall prediction explanation
    st.write("#### 🔮 Active Prediction Explanation")
    if st.session_state.inputs is not None:
        try:
            active_inputs = st.session_state.inputs
            active_mode = st.session_state.model_mode
            active_mode_display = "Enhanced Habitability Model" if active_mode == "baseline" else "Physics-Informed Proxy Model"
            
            st.info(f"Showing SHAP waterfall explanation for the last run exoplanet using the **{active_mode_display}**.")
            
            # Load selected pipeline model
            pipeline = load_model(active_mode)
            explain_prediction(active_inputs, pipeline)
            
            # Downloadable explanation
            html_path = export_html_explanation(active_inputs, pipeline)
            if os.path.exists(html_path):
                with open(html_path, 'rb') as f:
                    st.download_button("📄 Download Local SHAP Report (HTML)", f, file_name=f"shap_explanation_{active_mode}.html")
        except Exception as e:
            st.error(f"Error computing SHAP values: {e}")
    else:
        st.info("No active prediction found. Please run a 'Habitability Assessment' in Tab 1 first to view its waterfall explanation.")
        
    st.markdown("---")
    
    # Global beeswarm plots comparison
    st.write("#### 🐝 Global Model Beeswarm Plots")
    st.write("These plots show the overall importance of features across 100 random exoplanets in the catalog. Red dots represent high feature values; blue represents low values.")
    
    col_shap1, col_shap2 = st.columns(2)
    
    summary_path_base = get_absolute_path("outputs/shap/shap_summary_baseline.png")
    summary_path_proxy = get_absolute_path("outputs/shap/shap_summary_proxy.png")
    
    with col_shap1:
        st.write("**Model A: Baseline beeswarm**")
        if os.path.exists(summary_path_base):
            st.image(summary_path_base, use_container_width=True)
            st.write("Notice that `pl_insol` and `pl_eqt` dominate the baseline model's decision splits, capturing almost all of the SHAP impact.")
        else:
            st.info("Baseline beeswarm plot not found.")
            
    with col_shap2:
        st.write("**Model B: Proxy beeswarm**")
        if os.path.exists(summary_path_proxy):
            st.image(summary_path_proxy, use_container_width=True)
            st.write("In the proxy model, importance is distributed physically: high stellar mass/radius and orbital period are required to map to the HZ.")
        else:
            st.info("Proxy beeswarm plot not found.")

# =====================================================================
# TAB 4: ABOUT PROJECT
# =====================================================================
with tab4:
    st.markdown("### 📖 About the ExoLife Project")
    
    st.markdown("""
    #### 🎯 Problem Statement
    In exoplanetary research, determining which distant worlds could support surface liquid water is a major bottleneck. The NASA Exoplanet Catalog contains thousands of entries, but labels are highly imbalanced, and direct indicators of energy and temperature are frequently noisy or unobserved in early detection sweeps. 
    
    This project constructs a predictive model to identify habitable candidates and highlights a key Data Science challenge: **Target Leakage**.
    
    #### 🪐 Habitability Criteria
    Habitability labels are defined deterministically based on three conservative boundaries:
    1.  **Planet Radius (`pl_rade`)**: $0.5 - 2.5$ Earth Radii (rocky worlds).
    2.  **Equilibrium Temperature (`pl_eqt`)**: $160 - 330$ Kelvin (permits surface temperatures suitable for liquid water).
    3.  **Insolation Flux (`pl_insol`)**: $0.3 - 1.8$ times the energy flux Earth receives from the Sun.
    
    #### ⚠️ The Target Leakage Dilemma & Proxy Solution
    *   **The Trivial Model**: Standard modeling approaches train on all features. However, since the label is generated *from* Radius, Temperature, and Insolation, passing these three variables to a Random Forest causes **circular learning**. The model reaches ~100% accuracy by simply reconstructing the logic boundaries.
    *   **The Physics-Informed Proxy**: To create an authentic predictive challenge, we exclude the defining variables. The model must predict habitability using only secondary features like stellar mass/temp, orbital period, and metallicity. This forces it to learn the underlying physics:
        *   **Luminosity** is reconstructed via the Stefan-Boltzmann relation: $L \propto R_{\star}^2 \cdot T_{\star}^4$.
        *   **Orbital distance** is reconstructed via Kepler's Third Law: $a \propto (P^2 M_{\star})^{1/3}$.
        *   This showcases robust, mathematically validated machine learning.
    """)
    
    st.markdown("""
    #### 🗺️ Project Workflow Diagram
    """)
    
    st.code("""
    [ NASA Exoplanet Catalog ] ──> [ Preprocess: load_and_clean_data ]
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
        [ Baseline Model: All Features ]               [ Proxy Model: 8 Features ]
        - Trained on 11 parameters                      - Excludes: pl_rade, pl_eqt, pl_insol
        - Learns simple threshold splits                - Learns HZ physics mapping
        - Saved: model_baseline.pkl                     - Saved: model_proxy.pkl
                      │                                               │
                      └───────────────────────┬───────────────────────┘
                                              ▼
                             [ Shared Holdout Train/Test Split ]
                                              │
                       ┌──────────────────────┴──────────────────────┐
                       ▼                                             ▼
          [ confusion_matrix_baseline ]                 [ confusion_matrix_proxy ]
                       │                                             │
                       └──────────────────────┬──────────────────────┘
                                              ▼
                                 [ Streamlit UI Assessment ]
                                 - Interactive exoplanet presets
                                 - Live SHAP waterfall prediction explains
                                 - Global beeswarm model summaries
    """, language="text")
    
    st.markdown("---")
    st.markdown("🚀 Built as a premium Machine Learning portfolio project to demonstrate clean data engineering, target leakage mitigation, and Explainable AI (XAI).")
