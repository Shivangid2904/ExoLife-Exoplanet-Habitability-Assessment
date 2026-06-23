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
    .science-note {
        background-color: #1e2d3b;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #38bdf8;
        color: #cbd5e1;
        margin-bottom: 1rem;
    }
    .accuracy-note {
        background-color: #1e293b;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #fbbf24;
        color: #fef3c7;
        margin-bottom: 1rem;
    }
    .dataset-stat {
        background-color: #0f172a;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        border: 1px solid #1e3a5f;
        text-align: center;
    }
    .stat-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #38bdf8;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 0.2rem;
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
        "Baseline Model (All Features)",
        "Physics-Informed Proxy Model"
    ]
)
model_mode = "baseline" if "All Features" in model_mode_display else "proxy"

st.sidebar.markdown("---")

# Planetary Presets definition
presets = {
    "Custom (Manual Input)": None,
    # --- Habitable Candidates ---
    "🌍 Earth (Habitable Baseline)": {
        "pl_rade": 1.0, "pl_bmasse": 1.0, "pl_orbper": 365.25, "pl_eqt": 288.0, "pl_insol": 1.0,
        "pl_orbeccen": 0.0167, "st_teff": 5778.0, "st_rad": 1.0, "st_mass": 1.0, "st_met": 0.0, "sy_dist": 0.0
    },
    "🌍 Kepler-22b (Super-Earth Candidate)": {
        "pl_rade": 2.4, "pl_bmasse": 8.3, "pl_orbper": 289.86, "pl_eqt": 262.0, "pl_insol": 1.1,
        "pl_orbeccen": 0.0, "st_teff": 5518.0, "st_rad": 0.989, "st_mass": 0.97, "st_met": -0.03, "sy_dist": 195.0
    },
    "🌍 Kepler-186f (Earth-Sized HZ Planet)": {
        "pl_rade": 1.17, "pl_bmasse": 1.4, "pl_orbper": 129.9, "pl_eqt": 188.0, "pl_insol": 0.32,
        "pl_orbeccen": 0.04, "st_teff": 3755.0, "st_rad": 0.52, "st_mass": 0.48, "st_met": -0.28, "sy_dist": 179.0
    },
    # --- Non-Habitable Examples ---
    "🔥 Mars (Cold Rocky Desert)": {
        "pl_rade": 0.53, "pl_bmasse": 0.11, "pl_orbper": 687.0, "pl_eqt": 210.0, "pl_insol": 0.43,
        "pl_orbeccen": 0.0934, "st_teff": 5778.0, "st_rad": 1.0, "st_mass": 1.0, "st_met": 0.0, "sy_dist": 0.0
    },
    "🔥 Kepler-10b (Lava World)": {
        "pl_rade": 1.47, "pl_bmasse": 4.6, "pl_orbper": 0.84, "pl_eqt": 2130.0, "pl_insol": 3560.0,
        "pl_orbeccen": 0.0, "st_teff": 5627.0, "st_rad": 1.06, "st_mass": 0.91, "st_met": -0.15, "sy_dist": 186.0
    }
}

st.sidebar.markdown("**🌍 Habitable Candidates:** Earth, Kepler-22b, Kepler-186f")
st.sidebar.markdown("**🔥 Non-Habitable Examples:** Mars, Kepler-10b")
selected_preset = st.sidebar.selectbox("Load Exoplanet Preset", list(presets.keys()))
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
        st.info("💡 **Physics-Informed Proxy Model Active**: Planet Radius, Equilibrium Temperature, and Insolation Flux are not used by this model. Predictions are generated exclusively from secondary orbital and stellar properties.")
        
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
            # Human-readable labels for all model features
            FEATURE_LABELS = {
                "pl_rade":     "Planet Radius (Earth Radii)",
                "pl_bmasse":   "Planet Mass (Earth Masses)",
                "pl_orbper":   "Orbital Period (Days)",
                "pl_eqt":      "Equilibrium Temperature (K)",
                "pl_insol":    "Insolation Flux (Earth Units)",
                "pl_orbeccen": "Orbital Eccentricity",
                "st_teff":     "Stellar Temperature (K)",
                "st_rad":      "Stellar Radius (Solar Radii)",
                "st_mass":     "Stellar Mass (Solar Masses)",
                "st_met":      "Stellar Metallicity (dex)",
                "sy_dist":     "System Distance (Parsecs)"
            }
            features_to_show = PROXY_FEATURES if model_mode == "proxy" else BASELINE_FEATURES
            summary_df = pd.DataFrame({
                "Parameter": [FEATURE_LABELS.get(f, f) for f in features_to_show],
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
            "Baseline Model (All Features)": [
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

        # Why accuracy is misleading card
        st.markdown("""
        <div class='accuracy-note'>
            <b>⚠️ Why Accuracy Is Misleading Here</b><br>
            The dataset is <b>98.7% non-habitable</b>. A model that predicts every planet as uninhabitable would still achieve ~98.7% accuracy — without identifying a single habitable candidate.
            <b>F1 Score</b> (harmonic mean of Precision and Recall) and <b>Recall</b> are the primary metrics because they directly measure how well each model finds the rare habitable planets.
        </div>
        """, unsafe_allow_html=True)

        col_m1, col_m2 = st.columns([2, 1])
        with col_m1:
            st.write("#### 📈 Holdout Evaluation Metrics")
            st.table(metrics_df)
        with col_m2:
            st.write("#### 🔬 Scientific Modeling Note")
            st.markdown("""
            <div class='science-note'>
                <h5>🔬 Scientific Modeling Note</h5>
                <p>The <b>Baseline Model</b> includes variables that directly participate in label generation. As a result, it achieves extremely high performance by reconstructing the habitability rules.</p>
                <p>The <b>Physics-Informed Proxy Model</b> excludes these variables and therefore represents a more realistic predictive task, requiring the model to infer the Habitable Zone from secondary physical properties.</p>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        
        # Display side-by-side confusion matrix images
        st.write("#### 🗺️ Confusion Matrices Comparison")
        col_cm1, col_cm2 = st.columns(2)
        
        cm_path_base = get_absolute_path("outputs/evaluation/confusion_matrix_baseline.png")
        cm_path_proxy = get_absolute_path("outputs/evaluation/confusion_matrix_proxy.png")
        
        with col_cm1:
            st.write("**Baseline Model**")
            if os.path.exists(cm_path_base):
                st.image(cm_path_base, use_container_width=True)
            else:
                st.info("Baseline confusion matrix plot not found.")

        with col_cm2:
            st.write("**Physics-Informed Proxy Model**")
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
    st.write("SHAP (SHapley Additive exPlanations) values reveal how much each input parameter contributes to the final habitability prediction.")

    # SHAP interpretation guide
    st.markdown("""
    <div class='science-note'>
        <b>How to Read SHAP Values</b><br>
        &nbsp;• <b>Positive SHAP values</b> push the prediction <b>toward habitability</b> (increase the probability).<br>
        &nbsp;• <b>Negative SHAP values</b> push the prediction <b>away from habitability</b> (decrease the probability).<br>
        &nbsp;• The longer the bar, the greater the feature's influence on this specific prediction.
    </div>
    """, unsafe_allow_html=True)

    # Waterfall prediction explanation
    st.write("#### 🔮 Active Prediction Explanation")
    if st.session_state.inputs is not None:
        try:
            active_inputs = st.session_state.inputs
            active_mode = st.session_state.model_mode
            active_mode_display = "Baseline Model" if active_mode == "baseline" else "Physics-Informed Proxy Model"

            st.info(f"Showing SHAP waterfall explanation for the last assessed exoplanet using the **{active_mode_display}**.")

            # Load selected pipeline model
            pipeline = load_model(active_mode)
            explain_prediction(active_inputs, pipeline)

            # Downloadable explanation
            html_path = export_html_explanation(active_inputs, pipeline)
            if os.path.exists(html_path):
                with open(html_path, 'rb') as f:
                    st.download_button("📄 Download SHAP Report (HTML)", f, file_name=f"shap_explanation_{active_mode}.html")
        except Exception as e:
            st.error(f"Error computing SHAP values: {e}")
    else:
        st.info("No prediction found yet. Run a Habitability Assessment in Tab 1 first to view its SHAP waterfall explanation.")
        
    st.markdown("---")
    
    # Global beeswarm plots comparison
    st.write("#### 🐝 Global Model Beeswarm Plots")
    st.write("These plots show the overall importance of features across 100 random exoplanets in the catalog. Red dots represent high feature values; blue represents low values.")
    
    col_shap1, col_shap2 = st.columns(2)
    
    summary_path_base = get_absolute_path("outputs/shap/shap_summary_baseline.png")
    summary_path_proxy = get_absolute_path("outputs/shap/shap_summary_proxy.png")
    
    with col_shap1:
        st.write("**Baseline Model — Global Feature Impact**")
        if os.path.exists(summary_path_base):
            st.image(summary_path_base, use_container_width=True)
            st.caption("Insolation Flux and Equilibrium Temperature dominate almost all decision weight — a signature of circular rule reconstruction.")
        else:
            st.info("Baseline beeswarm plot not found. Run `python -m src.explain` to generate.")

    with col_shap2:
        st.write("**Physics-Informed Proxy Model — Global Feature Impact**")
        if os.path.exists(summary_path_proxy):
            st.image(summary_path_proxy, use_container_width=True)
            st.caption("Importance is distributed across Orbital Period, Stellar Radius, and Stellar Mass — reflecting genuine physical Habitable Zone reconstruction.")
        else:
            st.info("Proxy beeswarm plot not found. Run `python -m src.explain` to generate.")

# =====================================================================
# TAB 4: ABOUT PROJECT
# =====================================================================
with tab4:
    st.markdown("### 📖 About the ExoLife Project")

    # Dataset Stats Cards
    st.write("#### 🗄️ Dataset Overview")
    ds1, ds2, ds3, ds4, ds5 = st.columns(5)
    with ds1:
        st.markdown("<div class='dataset-stat'><div class='stat-value'>NASA</div><div class='stat-label'>Dataset Source — Exoplanet Archive</div></div>", unsafe_allow_html=True)
    with ds2:
        st.markdown("<div class='dataset-stat'><div class='stat-value'>3,757</div><div class='stat-label'>Total Exoplanets (cleaned)</div></div>", unsafe_allow_html=True)
    with ds3:
        st.markdown("<div class='dataset-stat'><div class='stat-value'>49</div><div class='stat-label'>Potentially Habitable</div></div>", unsafe_allow_html=True)
    with ds4:
        st.markdown("<div class='dataset-stat'><div class='stat-value'>3,708</div><div class='stat-label'>Non-Habitable</div></div>", unsafe_allow_html=True)
    with ds5:
        st.markdown("<div class='dataset-stat'><div class='stat-value'>75.7 : 1</div><div class='stat-label'>Class Imbalance Ratio</div></div>", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    #### 🎯 Problem Statement
    In exoplanetary research, determining which distant worlds could support surface liquid water is a major bottleneck. The NASA Exoplanet Catalog contains thousands of entries, but labels are highly imbalanced and direct indicators of energy flux and surface temperature are frequently noisy or unobserved in early detection surveys.

    This project builds a predictive model to identify habitable candidates and addresses a critical Data Science modeling challenge: **Target Leakage**.

    #### 🪐 Habitability Criteria
    Habitability labels are defined deterministically based on three conservative astrophysical boundaries:
    1. **Planet Radius**: 0.5 – 2.5 Earth Radii (selects rocky, terrestrial worlds).
    2. **Equilibrium Temperature**: 160 – 330 Kelvin (permits surface conditions for liquid water).
    3. **Insolation Flux**: 0.3 – 1.8 times the solar flux Earth receives.

    #### 🔬 The Leakage Discovery and Proxy Solution
    - **Baseline Model**: Trained on all 11 features, including the three that directly define the habitability label. The model reconstructs simple threshold rules, reaching near-perfect accuracy without learning any real physical relationships.
    - **Physics-Informed Proxy Model**: Excludes the three defining variables entirely. The model must predict habitability using only secondary orbital and stellar parameters, forcing it to approximate the underlying astrophysics:
        - Stellar Luminosity reconstructed from Radius and Temperature via the Stefan-Boltzmann Law.
        - Orbital distance reconstructed from Orbital Period and Stellar Mass via Kepler's Third Law.
    """)

    st.write("#### 🗺️ Project Workflow")
    st.code("""
 [ NASA Exoplanet Archive ] --> [ preprocess.py: load_and_clean_data() ]
                                            |
                    +-----------------------+------------------------+
                    v                                               v
    [ Baseline Model (All 11 Features) ]       [ Proxy Model (8 Secondary Features) ]
    - Learns threshold rule boundaries          - Learns HZ physics reconstruction
    - model_baseline.pkl                        - model_proxy.pkl
                    |                                               |
                    +-----------------------+------------------------+
                                           v
                          [ Shared 80/20 Holdout Evaluation ]
                          [ metrics_comparison.json ]
                                           v
                             [ Streamlit Multi-Tab Dashboard ]
                             - Preset planet loader
                             - Dual-model inference & probability score
                             - SHAP waterfall + beeswarm analysis
    """, language="text")

    st.markdown("---")

    # Footer
    st.markdown("""
    <div style='text-align:center; padding: 1.5rem 0 0.5rem 0;'>
        <p style='color:#64748b; font-size:0.85rem;'>
            <b style='color:#94a3b8;'>Built with:</b> Python &nbsp;|&nbsp; Scikit-Learn &nbsp;|&nbsp; SHAP &nbsp;|&nbsp; Streamlit &nbsp;|&nbsp; imbalanced-learn
        </p>
        <p style='color:#64748b; font-size:0.85rem;'>
            <b style='color:#94a3b8;'>Dataset:</b> NASA Exoplanet Archive &nbsp;|&nbsp;
            <b style='color:#94a3b8;'>Project Focus:</b> Explainable AI &amp; Physics-Informed Habitability Assessment
        </p>
        <p style='color:#475569; font-size:0.8rem; margin-top:0.5rem;'>
            Made with by <a href='https://www.linkedin.com/in/shivangi-dubey-1783511a6/' target='_blank' style='color:#38bdf8;'>Shivangi Dubey</a>
            &nbsp;&bull;&nbsp;
            <a href='https://github.com/Shivangid2904/ExoLife-Exoplanet-Habitability-Assessment' target='_blank' style='color:#38bdf8;'>GitHub Repository</a>
            &nbsp;&bull;&nbsp;
            <a href='https://exoplanetarchive.ipac.caltech.edu/' target='_blank' style='color:#38bdf8;'>NASA Exoplanet Archive</a>
        </p>
    </div>
    """, unsafe_allow_html=True)
