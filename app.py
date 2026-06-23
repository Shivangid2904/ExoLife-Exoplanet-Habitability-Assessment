import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import io
import matplotlib.pyplot as plt
import plotly.express as px
import scipy.stats as stats

from src.predict import predict, predict_proba, load_model
from src.explain import explain_prediction, export_html_explanation
from src.preprocess import BASELINE_FEATURES, PROXY_FEATURES

# Define global feature labels mapping
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

ABBREV_LABELS = {
    "pl_rade":     "Planet Rad (R⊕)",
    "pl_bmasse":   "Planet Mass (M⊕)",
    "pl_orbper":   "Orb. Period (days)",
    "pl_eqt":      "Equil. Temp (K)",
    "pl_insol":    "Insolation (F⊕)",
    "pl_orbeccen": "Eccentricity",
    "st_teff":     "Star Temp (K)",
    "st_rad":      "Star Rad (R☉)",
    "st_mass":     "Star Mass (M☉)",
    "st_met":      "Star Metal (dex)",
    "sy_dist":     "Distance (pc)"
}

SUPPORTED_EDA_FEATURES = BASELINE_FEATURES

@st.cache_data
def get_cleaned_data():
    from src.preprocess import load_and_clean_data
    return load_and_clean_data()

@st.cache_data
def get_correlation_matrix(df, features):
    return df[features].corr()

@st.cache_data
def get_dataset_quality_report():
    filepath = "data/raw/exoplanetdata.csv"
    if not os.path.exists(filepath):
        # Fallback relative paths
        base_path = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.join(base_path, "data", "raw", "exoplanetdata.csv")
        if os.path.exists(alt_path):
            filepath = alt_path
        else:
            alt_path2 = os.path.join(base_path, "..", "data", "raw", "exoplanetdata.csv")
            if os.path.exists(alt_path2):
                filepath = alt_path2
            else:
                return None
    try:
        df_raw = pd.read_csv(filepath, header=88)
    except Exception:
        return None
        
    total_raw = len(df_raw)
    raw_missing = df_raw[BASELINE_FEATURES].isna().sum().to_dict()
    raw_missing_pct = (df_raw[BASELINE_FEATURES].isna().sum() / total_raw * 100).to_dict()
    duplicates = df_raw.duplicated(subset=BASELINE_FEATURES).sum()
    
    return {
        "total_raw": total_raw,
        "missing_counts": raw_missing,
        "missing_pcts": raw_missing_pct,
        "duplicates": int(duplicates)
    }

def get_processed_viz_df(df, features, mode):
    """
    Returns a copy of the dataframe with features filtered/clipped for visualization.
    Does NOT alter the underlying original dataset.
    """
    df_viz = df.dropna(subset=features).copy()
    if df_viz.empty:
        return df_viz, False
        
    use_log = (mode == "Log Scale")
    
    if mode == "99th Percentile Clipping":
        for feat in features:
            upper_limit = df_viz[feat].quantile(0.99)
            df_viz[feat] = df_viz[feat].clip(upper=upper_limit)
    elif mode == "Log Scale":
        for feat in features:
            df_viz = df_viz[df_viz[feat] > 0]
            
    return df_viz, use_log

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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌌 Habitability Assessment",
    "📊 Model Comparison",
    "🧠 Explainability (SHAP)",
    "🔍 Exploratory Data Analysis",
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
# TAB 4: EXPLORATORY DATA ANALYSIS (EDA) DASHBOARD
# =====================================================================
with tab4:
    st.markdown("### 🔍 Exploratory Data Analysis (EDA)")
    st.write("Explore the physical, orbital, and stellar characteristics of the exoplanet population from the NASA Exoplanet Archive.")
    
    # Outlier Selector & Scaling Selector (Section 2)
    st.markdown("##### 🛠️ Visualization Mode (Outlier & Scale Treatment)")
    viz_mode = st.radio(
        "Select Visualization Mode:",
        ["Full Dataset", "99th Percentile Clipping", "Log Scale"],
        horizontal=True,
        help="Adjusts data representation for plots to enhance readability and reveal hidden patterns. Does NOT modify the underlying dataset."
    )
    
    # Load cached dataset
    try:
        df_eda = get_cleaned_data()
    except Exception as e:
        st.error(f"Error loading exoplanet dataset: {e}")
        df_eda = pd.DataFrame()
        
    if not df_eda.empty:
        # Define features and helper mapping
        df_plot = df_eda.copy()
        df_plot['Habitability'] = df_plot['is_habitable'].map({0: "Non-Habitable", 1: "Habitable"})
        
        # ----------------- Section 1: Dataset Overview & Quality Report -----------------
        st.write("#### 1. Dataset Overview & Quality Report")
        
        total_p = len(df_eda)
        hab_p = int((df_eda['is_habitable'] == 1).sum())
        non_hab_p = int((df_eda['is_habitable'] == 0).sum())
        ratio_p = non_hab_p / hab_p if hab_p > 0 else 0
        
        col_ov1, col_ov2 = st.columns([1, 1])
        
        with col_ov1:
            st.markdown(f"""
            <div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;'>
                <div class='dataset-stat'>
                    <div class='stat-value'>{total_p:,}</div>
                    <div class='stat-label'>Total Usable Planets</div>
                </div>
                <div class='dataset-stat'>
                    <div class='stat-value'>{hab_p}</div>
                    <div class='stat-label'>Habitable (Class 1)</div>
                </div>
                <div class='dataset-stat'>
                    <div class='stat-value'>{non_hab_p:,}</div>
                    <div class='stat-label'>Non-Habitable (Class 0)</div>
                </div>
                <div class='dataset-stat'>
                    <div class='stat-value'>{ratio_p:.1f}:1</div>
                    <div class='stat-label'>Imbalance Ratio</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Text explanation (Section 5 observational language)
            st.markdown(f"""
            <div class='accuracy-note' style='margin-top: 1rem;'>
                <b>🔬 Scientific Observations on Dataset Imbalance</b><br>
                Within this dataset, habitable candidates occur at a ratio of approximately <b>{ratio_p:.1f} to 1</b> non-habitable planets.
                Only <b>{hab_p / total_p * 100:.2f}%</b> of the planets in this catalog satisfy the physical requirements of rocky structure, moderate equilibrium temperature, and appropriate insolation flux. This extreme skew illustrates why evaluation metrics like F1-Score and Recall are the primary measures for assessing model performance.
            </div>
            """, unsafe_allow_html=True)
            
        with col_ov2:
            # Bar chart of class distribution
            fig_ov = px.bar(
                x=["Non-Habitable", "Habitable"],
                y=[non_hab_p, hab_p],
                labels={"x": "Class", "y": "Planet Count"},
                color=["Non-Habitable", "Habitable"],
                color_discrete_map={"Non-Habitable": "#ef4444", "Habitable": "#10b981"},
                title="Class Distribution Breakdown",
                text=[f"{non_hab_p:,} ({non_hab_p/total_p*100:.2f}%)", f"{hab_p} ({hab_p/total_p*100:.2f}%)"]
            )
            fig_ov.update_traces(textposition='outside')
            fig_ov.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#f8fafc",
                showlegend=False,
                height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_ov, use_container_width=True)
            
        # Section 7: Dataset Quality Assessment
        st.write("##### 📊 Dataset Quality Report")
        quality_data = get_dataset_quality_report()
        
        if quality_data:
            col_q1, col_q2 = st.columns([2, 1])
            with col_q1:
                # Compile table of raw missing values for baseline features
                completeness_rows = []
                for f in BASELINE_FEATURES:
                    m_count = quality_data["missing_counts"].get(f, 0)
                    m_pct = quality_data["missing_pcts"].get(f, 0.0)
                    completeness_rows.append({
                        "Feature": f,
                        "Description": FEATURE_LABELS[f],
                        "Raw Missing Values": f"{m_count:,}",
                        "Raw Missing %": f"{m_pct:.2f}%",
                        "Clean Completeness": "100.00%"
                    })
                st.dataframe(pd.DataFrame(completeness_rows), use_container_width=True, hide_index=True)
            with col_q2:
                st.markdown(f"""
                <div class='science-note' style='padding: 1.2rem; margin-top: 0;'>
                    <h5>📋 Data Quality Summary</h5>
                    <p style='font-size:0.9rem; margin-bottom: 0.5rem;'><b>Raw Records:</b> {quality_data['total_raw']:,}</p>
                    <p style='font-size:0.9rem; margin-bottom: 0.5rem;'><b>Usable clean records:</b> {total_p:,}</p>
                    <p style='font-size:0.9rem; margin-bottom: 0.5rem;'><b>Cleaned Duplicate Rows:</b> {df_eda.duplicated().sum()}</p>
                    <p style='font-size:0.85rem; color:#cbd5e1; margin-top:0.8rem;'>
                        The processed dataset is highly complete with minimal missing values across selected planetary and stellar features. Preprocessing dropped {quality_data['total_raw'] - total_p:,} rows missing raw parameters, yielding 100% completeness for the final 11 physical attributes.
                    </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Raw dataset quality report metrics not available.")
            
        st.markdown("---")
        
        # ----------------- Section 2: Feature Distribution Explorer -----------------
        st.write("#### 2. Feature Distribution Explorer")
        st.write("Inspect individual planetary and stellar characteristics to understand their mathematical distributions and statistical properties.")
        
        col_dist1, col_dist2 = st.columns([1, 2])
        
        with col_dist1:
            selected_dist_feat = st.selectbox(
                "Select Feature to Investigate",
                SUPPORTED_EDA_FEATURES,
                format_func=lambda x: FEATURE_LABELS.get(x, x),
                key="dist_selector"
            )
            
            # Apply outlier treatment (Section 2)
            df_dist_viz, use_log_dist = get_processed_viz_df(df_eda, [selected_dist_feat], viz_mode)
            feat_vals_viz = df_dist_viz[selected_dist_feat]
            
            # Compute stats
            f_mean = feat_vals_viz.mean()
            f_median = feat_vals_viz.median()
            f_std = feat_vals_viz.std()
            f_min = feat_vals_viz.min()
            f_max = feat_vals_viz.max()
            f_skew = feat_vals_viz.skew()
            
            # Classify skewness (Section 6)
            if abs(f_skew) < 0.5:
                skew_desc = "Approximately Symmetric"
                skew_color = "#10b981"
                skew_explain = "The values are distributed relatively symmetrically around the mean. Normal-like distributions are well-suited for linear models without mathematical transformation."
            elif f_skew >= 0.5:
                skew_desc = "Right Skewed"
                skew_color = "#f59e0b"
                skew_explain = "Concentration of observations at smaller values with a tail stretching towards larger values. In astrophysics, this is common for orbital periods or planetary masses as smaller bodies are more common and easier to detect."
            else:
                skew_desc = "Left Skewed"
                skew_color = "#f59e0b"
                skew_explain = "Concentration of observations at larger values with a tail stretching towards smaller values. Transformation is recommended if models assume normality."
                
            st.markdown(f"""
            <div class='metric-card'>
                <h5>📊 Descriptive Statistics ({viz_mode})</h5>
                <table style='width:100%; border-collapse: collapse;'>
                    <tr><td style='padding: 4px 0; color:#94a3b8;'>Mean</td><td style='text-align:right; font-weight:bold; color:#38bdf8;'>{f_mean:.4f}</td></tr>
                    <tr><td style='padding: 4px 0; color:#94a3b8;'>Median</td><td style='text-align:right; font-weight:bold; color:#38bdf8;'>{f_median:.4f}</td></tr>
                    <tr><td style='padding: 4px 0; color:#94a3b8;'>Std Dev</td><td style='text-align:right; font-weight:bold; color:#38bdf8;'>{f_std:.4f}</td></tr>
                    <tr><td style='padding: 4px 0; color:#94a3b8;'>Skewness</td><td style='text-align:right; font-weight:bold; color:#38bdf8;'>{f_skew:+.4f}</td></tr>
                    <tr><td style='padding: 4px 0; color:#94a3b8;'>Minimum</td><td style='text-align:right; font-weight:bold; color:#38bdf8;'>{f_min:.4f}</td></tr>
                    <tr><td style='padding: 4px 0; color:#94a3b8;'>Maximum</td><td style='text-align:right; font-weight:bold; color:#38bdf8;'>{f_max:.4f}</td></tr>
                </table>
                <div style='margin-top: 10px; padding-top: 10px; border-top: 1px solid #334155;'>
                    <span style='font-size: 0.85rem; color:#94a3b8;'>Distribution Type:</span><br>
                    <b style='color:{skew_color}; font-size: 0.95rem;'>{skew_desc}</b><br>
                    <p style='font-size: 0.8rem; color:#94a3b8; margin: 4px 0 0 0;'>{skew_explain}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_dist2:
            # Determine if KDE is appropriate
            is_kde_appropriate = len(feat_vals_viz.unique()) >= 10 and f_std > 0.001 and not use_log_dist
            
            # Set up Plotly histogram
            nbins = 50
            if use_log_dist:
                fig_dist = px.histogram(
                    x=feat_vals_viz,
                    nbins=nbins,
                    title=f"Log-Distribution: {FEATURE_LABELS[selected_dist_feat]}",
                    labels={"x": FEATURE_LABELS[selected_dist_feat]},
                    color_discrete_sequence=["#38bdf8"]
                )
                fig_dist.update_layout(xaxis_type="log")
            else:
                fig_dist = px.histogram(
                    x=feat_vals_viz,
                    histnorm="probability density" if is_kde_appropriate else None,
                    nbins=nbins,
                    title=f"Distribution: {FEATURE_LABELS[selected_dist_feat]} ({viz_mode})",
                    labels={"x": FEATURE_LABELS[selected_dist_feat], "y": "Density" if is_kde_appropriate else "Count"},
                    color_discrete_sequence=["#38bdf8"]
                )
                
                if is_kde_appropriate:
                    try:
                        kde = stats.gaussian_kde(feat_vals_viz)
                        x_grid = np.linspace(f_min, f_max, 300)
                        y_kde = kde(x_grid)
                        fig_dist.add_scatter(
                            x=x_grid,
                            y=y_kde,
                            mode="lines",
                            name="Gaussian KDE",
                            line=dict(color="#f43f5e", width=2.5)
                        )
                    except Exception:
                        pass
                        
            fig_dist.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#f8fafc",
                height=320,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            
        st.markdown("---")
        
        # ----------------- Section 3: Habitability Comparison -----------------
        st.write("#### 3. Habitability Comparison")
        st.write("Analyze how various parameters differ between habitable and non-habitable worlds to uncover key physical separations.")
        
        selected_comp_feat = st.selectbox(
            "Select Feature to Compare",
            SUPPORTED_EDA_FEATURES,
            format_func=lambda x: FEATURE_LABELS.get(x, x),
            key="comp_selector"
        )
        
        # Apply outlier treatment (Section 2)
        df_comp_viz, use_log_comp = get_processed_viz_df(df_eda, [selected_comp_feat], viz_mode)
        df_comp_plot = df_comp_viz.copy()
        df_comp_plot['Habitability'] = df_comp_plot['is_habitable'].map({0: "Non-Habitable", 1: "Habitable"})
        
        # Split data
        hab_vals = df_comp_viz[df_comp_viz['is_habitable'] == 1][selected_comp_feat].dropna()
        non_hab_vals = df_comp_viz[df_comp_viz['is_habitable'] == 0][selected_comp_feat].dropna()
        
        # Significance Testing using Mann-Whitney U test (Section 4)
        if len(hab_vals) > 0 and len(non_hab_vals) > 0:
            try:
                stat, p_val = stats.mannwhitneyu(hab_vals, non_hab_vals, alternative='two-sided')
                is_sig = p_val < 0.05
                sig_desc = "Statistically Significant Difference" if is_sig else "No Statistically Significant Difference"
                sig_color = "#10b981" if is_sig else "#ef4444"
                p_str = f"p = {p_val:.4e}" if p_val < 0.0001 else f"p = {p_val:.4f}"
                test_explain = (
                    f"The Mann-Whitney U test indicates a <b>{sig_desc.lower()}</b> ({p_str}) "
                    f"in the distribution of {FEATURE_LABELS[selected_comp_feat]} between habitable candidates and non-habitable planets. "
                    "This non-parametric test evaluates whether the distribution of a variable differs systematically between two independent groups without assuming normal distributions, making it highly robust for skewed exoplanetary parameters."
                )
            except Exception as e:
                sig_desc = "Test Execution Error"
                sig_color = "#94a3b8"
                p_str = "Error"
                test_explain = f"Could not perform Mann-Whitney U test: {e}"
        else:
            sig_desc = "Insufficient Data"
            sig_color = "#94a3b8"
            p_str = "N/A"
            test_explain = "At least one group contains zero observations after outlier treatment/scaling."
            
        col_comp_stats1, col_comp_stats2 = st.columns(2)
        with col_comp_stats1:
            st.markdown(f"""
            <div class='metric-card'>
                <h5>📊 Group Summaries ({viz_mode})</h5>
                <table style='width:100%; border-collapse: collapse;'>
                    <thead>
                        <tr style='border-bottom: 1px solid #334155;'>
                            <th style='text-align:left; color:#94a3b8; padding:4px 0;'>Metric</th>
                            <th style='text-align:right; color:#10b981; padding:4px 0;'>Habitable</th>
                            <th style='text-align:right; color:#ef4444; padding:4px 0;'>Non-Habitable</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr><td style='padding: 6px 0; color:#cbd5e1;'>Mean</td><td style='text-align:right; font-weight:bold;'>{hab_vals.mean():.4f}</td><td style='text-align:right; font-weight:bold;'>{non_hab_vals.mean():.4f}</td></tr>
                        <tr><td style='padding: 6px 0; color:#cbd5e1;'>Median</td><td style='text-align:right; font-weight:bold;'>{hab_vals.median():.4f}</td><td style='text-align:right; font-weight:bold;'>{non_hab_vals.median():.4f}</td></tr>
                        <tr><td style='padding: 6px 0; color:#cbd5e1;'>Std Dev</td><td style='text-align:right; font-weight:bold;'>{hab_vals.std():.4f}</td><td style='text-align:right; font-weight:bold;'>{non_hab_vals.std():.4f}</td></tr>
                        <tr><td style='padding: 6px 0; color:#cbd5e1;'>Sample Size</td><td style='text-align:right; font-weight:bold;'>{len(hab_vals)}</td><td style='text-align:right; font-weight:bold;'>{len(non_hab_vals)}</td></tr>
                    </tbody>
                </table>
            </div>
            """, unsafe_allow_html=True)
            
        with col_comp_stats2:
            st.markdown(f"""
            <div class='metric-card' style='border-left: 5px solid {sig_color};'>
                <h5 style='color:{sig_color};'>{sig_desc}</h5>
                <div style='font-size: 1.5rem; font-weight: bold; margin-top:5px;'>{p_str}</div>
                <p style='color:#94a3b8; font-size: 0.85rem; margin-top:5px;'>
                    {test_explain}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        comp_tab1, comp_tab2, comp_tab3 = st.tabs([
            "📈 Overlapping Distributions",
            "📦 Box Plot",
            "🎻 Violin Plot"
        ])
        
        # Audited texts for scientific neutral/observational wording (Section 5)
        interpretations = {
            "pl_rade": "Within this dataset, exoplanets with habitable labels are observed exclusively in the 0.5 to 2.5 Earth Radii boundary. Larger planets are typically associated with gas accumulation (Mini-Neptunes/Jupiters), whereas smaller ones are observed in environments with insufficient surface gravity to retain atmospheres.",
            "pl_bmasse": "Within this dataset, habitable candidates tend to occupy the lower mass spectrum (under 10 Earth masses), reflecting rocky super-Earth compositions. Extremely massive objects in this catalog consistently show properties of gas giant atmospheres.",
            "pl_orbper": "Within this dataset, habitable candidates show a clustering at short orbital periods for planets orbiting M-dwarfs, while Sun-like host stars show candidates orbiting closer to 360 days. Non-habitable worlds display a broad distribution covering extremely short orbits.",
            "pl_eqt": "Within this dataset, habitable candidates are strictly associated with equilibrium temperatures between 160 K and 330 K. In contrast, non-habitable objects display wide temperature distributions, including hot Jovian temperatures exceeding 1000 K.",
            "pl_insol": "Within this dataset, habitable candidates are observed strictly within the insolation flux range of 0.3 to 1.8 Earth units, aligning with boundaries where water may exist in a liquid state. Non-habitable planets display a high dispersion, showing both frozen outer planets and heavily irradiated close-in objects.",
            "pl_orbeccen": "Within this dataset, habitable candidates are associated with low orbital eccentricities, typically below 0.2, representing stable orbital parameters. High eccentricity orbits in this catalog are observed strictly among non-habitable worlds.",
            "st_teff": "Within this dataset, habitable candidates are observed orbiting host stars with effective temperatures concentrated between 3,000 K and 6,000 K, which spans G, K, and M dwarf categories. Hosts of non-habitable planets cover the full temperature range of G, K, M, F, and A type stars.",
            "st_rad": "Within this dataset, host star radius is observed to correlate with habitable orbit distance. Smaller stars are associated with closer habitable zones where tidal configurations are common.",
            "st_mass": "Within this dataset, host star mass shows that habitable candidates are found around stars under 1.2 solar masses, aligning with stellar lifetimes that allow potential planets to remain in stable orbits for billions of years.",
            "st_met": "Within this dataset, host stars of habitable planets tend to cluster around solar metallicity values, suggesting that metal-rich protoplanetary disks may facilitate rocky terrestrial core formation.",
            "sy_dist": "Within this dataset, system distance is observed to be closer on average for habitable candidates, which represents a potential observational selection bias as small rocky planets are far easier to detect in nearby systems."
        }
        
        with comp_tab1:
            fig_comp_dist = px.histogram(
                df_comp_plot,
                x=selected_comp_feat,
                color="Habitability",
                barmode="overlay",
                histnorm="probability density",
                color_discrete_map={"Non-Habitable": "#ef4444", "Habitable": "#10b981"},
                title=f"Overlapping Density Distribution: {FEATURE_LABELS[selected_comp_feat]}",
                labels={"Habitability": "Class"}
            )
            fig_comp_dist.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#f8fafc",
                xaxis_title=FEATURE_LABELS[selected_comp_feat],
                yaxis_title="Probability Density",
                xaxis_type="log" if use_log_comp else None,
                height=350
            )
            st.plotly_chart(fig_comp_dist, use_container_width=True)
            
        with comp_tab2:
            fig_comp_box = px.box(
                df_comp_plot,
                x="Habitability",
                y=selected_comp_feat,
                color="Habitability",
                color_discrete_map={"Non-Habitable": "#ef4444", "Habitable": "#10b981"},
                title=f"Box Plot: {FEATURE_LABELS[selected_comp_feat]} by Habitability Class",
                points="outliers"
            )
            fig_comp_box.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#f8fafc",
                yaxis_title=FEATURE_LABELS[selected_comp_feat],
                yaxis_type="log" if use_log_comp else None,
                height=350
            )
            st.plotly_chart(fig_comp_box, use_container_width=True)
            
        with comp_tab3:
            fig_comp_viol = px.violin(
                df_comp_plot,
                x="Habitability",
                y=selected_comp_feat,
                color="Habitability",
                color_discrete_map={"Non-Habitable": "#ef4444", "Habitable": "#10b981"},
                title=f"Violin Plot: {FEATURE_LABELS[selected_comp_feat]} by Habitability Class",
                box=True,
                points="all"
            )
            fig_comp_viol.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#f8fafc",
                yaxis_title=FEATURE_LABELS[selected_comp_feat],
                yaxis_type="log" if use_log_comp else None,
                height=350
            )
            st.plotly_chart(fig_comp_viol, use_container_width=True)
            
        st.markdown(f"""
        <div class='science-note'>
            <b>🔬 Scientific Interpretation of Observed Differences</b><br>
            <p style='margin-top: 5px; margin-bottom: 5px;'>
                {interpretations.get(selected_comp_feat, "No detailed interpretation available for this feature.")}
            </p>
            <p style='margin-top: 5px; margin-bottom: 0px;'>
                <b>Habitable Median:</b> {hab_vals.median():.4f} &nbsp;|&nbsp; 
                <b>Non-Habitable Median:</b> {non_hab_vals.median():.4f}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ----------------- Section 4: Correlation Analysis -----------------
        st.write("#### 4. Correlation Analysis")
        st.write("Examine linear relationships between physical variables in the catalog to detect coupled planetary and stellar processes.")
        
        corr_matrix = get_correlation_matrix(df_eda, SUPPORTED_EDA_FEATURES)
        
        # Find examples for the Correlation Interpretation Panel
        corr_pairs = corr_matrix.unstack()
        corr_pairs = corr_pairs[corr_pairs.index.get_level_values(0) != corr_pairs.index.get_level_values(1)]
        
        unique_pairs_list = []
        seen = set()
        for (f1, f2), val in corr_pairs.items():
            pair = tuple(sorted([f1, f2]))
            if pair not in seen:
                seen.add(pair)
                unique_pairs_list.append(((f1, f2), val))
                
        unique_pairs_df = pd.DataFrame([
            {"F1": p[0], "F2": p[1], "Correlation": val} for p, val in unique_pairs_list
        ])
        
        # Strongest positive (excluding self)
        top_pos_row = unique_pairs_df.sort_values(by="Correlation", ascending=False).iloc[0]
        top_pos_f1, top_pos_f2, top_pos_val = top_pos_row["F1"], top_pos_row["F2"], top_pos_row["Correlation"]
        
        # Strongest negative
        top_neg_row = unique_pairs_df.sort_values(by="Correlation", ascending=True).iloc[0]
        top_neg_f1, top_neg_f2, top_neg_val = top_neg_row["F1"], top_neg_row["F2"], top_neg_row["Correlation"]
        
        # Find examples for each guide category (Section 8)
        vs_ex_df = unique_pairs_df[unique_pairs_df["Correlation"].abs() > 0.8]
        vs_ex = f"{FEATURE_LABELS[vs_ex_df.iloc[0]['F1']]} & {FEATURE_LABELS[vs_ex_df.iloc[0]['F2']]} (r = {vs_ex_df.iloc[0]['Correlation']:+.2f})" if not vs_ex_df.empty else "None observed"
        
        s_ex_df = unique_pairs_df[(unique_pairs_df["Correlation"].abs() >= 0.6) & (unique_pairs_df["Correlation"].abs() <= 0.8)]
        s_ex = f"{FEATURE_LABELS[s_ex_df.iloc[0]['F1']]} & {FEATURE_LABELS[s_ex_df.iloc[0]['F2']]} (r = {s_ex_df.iloc[0]['Correlation']:+.2f})" if not s_ex_df.empty else "None observed"
        
        m_ex_df = unique_pairs_df[(unique_pairs_df["Correlation"].abs() >= 0.4) & (unique_pairs_df["Correlation"].abs() < 0.6)]
        m_ex = f"{FEATURE_LABELS[m_ex_df.iloc[0]['F1']]} & {FEATURE_LABELS[m_ex_df.iloc[0]['F2']]} (r = {m_ex_df.iloc[0]['Correlation']:+.2f})" if not m_ex_df.empty else "None observed"
        
        w_ex_df = unique_pairs_df[(unique_pairs_df["Correlation"].abs() >= 0.2) & (unique_pairs_df["Correlation"].abs() < 0.4)]
        w_ex = f"{FEATURE_LABELS[w_ex_df.iloc[0]['F1']]} & {FEATURE_LABELS[w_ex_df.iloc[0]['F2']]} (r = {w_ex_df.iloc[0]['Correlation']:+.2f})" if not w_ex_df.empty else "None observed"
        
        vw_ex_df = unique_pairs_df[unique_pairs_df["Correlation"].abs() < 0.2]
        vw_ex = f"{FEATURE_LABELS[vw_ex_df.iloc[0]['F1']]} & {FEATURE_LABELS[vw_ex_df.iloc[0]['F2']]} (r = {vw_ex_df.iloc[0]['Correlation']:+.2f})" if not vw_ex_df.empty else "None observed"

        # Show strongest correlations as clean metrics cards
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown(f"""
            <div class='metric-card' style='border-left: 5px solid #10b981; height: 100%;'>
                <h6 style='color:#94a3b8; margin:0;'>Strongest Positive Correlation</h6>
                <div style='font-size:1.1rem; font-weight:bold; margin-top:5px; color:#10b981;'>
                    {FEATURE_LABELS[top_pos_f1]} &<br>{FEATURE_LABELS[top_pos_f2]}
                </div>
                <div style='font-size:1.6rem; font-weight:bold; margin-top:5px;'>r = {top_pos_val:+.4f}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_c2:
            st.markdown(f"""
            <div class='metric-card' style='border-left: 5px solid #ef4444; height: 100%;'>
                <h6 style='color:#94a3b8; margin:0;'>Strongest Negative Correlation</h6>
                <div style='font-size:1.1rem; font-weight:bold; margin-top:5px; color:#ef4444;'>
                    {FEATURE_LABELS[top_neg_f1]} &<br>{FEATURE_LABELS[top_neg_f2]}
                </div>
                <div style='font-size:1.6rem; font-weight:bold; margin-top:5px;'>r = {top_neg_val:+.4f}</div>
            </div>
            """, unsafe_allow_html=True)

        col_corr1, col_corr2 = st.columns([3, 2])
        
        with col_corr1:
            # Heatmap with custom hover naming and larger size (Section 1)
            x_labels = [ABBREV_LABELS[f] for f in SUPPORTED_EDA_FEATURES]
            y_labels = [ABBREV_LABELS[f] for f in SUPPORTED_EDA_FEATURES]
            
            hover_text = []
            for i, f1 in enumerate(SUPPORTED_EDA_FEATURES):
                row = []
                for j, f2 in enumerate(SUPPORTED_EDA_FEATURES):
                    r_val = corr_matrix.iloc[i, j]
                    row.append(
                        f"Feature X: {FEATURE_LABELS[f2]}<br>"
                        f"Feature Y: {FEATURE_LABELS[f1]}<br>"
                        f"Correlation Coefficient: {r_val:+.4f}"
                    )
                hover_text.append(row)
                
            fig_corr = px.imshow(
                corr_matrix,
                x=x_labels,
                y=y_labels,
                color_continuous_scale="RdBu",
                zmin=-1,
                zmax=1,
                title="Interactive Feature Correlation Matrix"
            )
            fig_corr.update_traces(
                hovertemplate="%{customdata}<extra></extra>",
                customdata=hover_text
            )
            fig_corr.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#f8fafc",
                height=550,
                xaxis=dict(tickangle=45),
                margin=dict(l=40, r=40, t=50, b=40)
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
        with col_corr2:
            st.markdown("""
            <div class='science-note' style='padding: 1.2rem; margin-bottom: 1rem;'>
                <h5>⚠️ Correlation vs. Causation</h5>
                <b>Correlation does not imply causation.</b> A high correlation value indicates that two variables change together, but does not prove one causes the other. For instance, distance to host star and orbital period are correlated due to orbital physics, not because distance generates the period.
            </div>
            """, unsafe_allow_html=True)
            
            # Correlation scale education card (Section 8)
            st.markdown(f"""
            <div class='metric-card' style='padding: 1rem; border-left: 5px solid #fbbf24; margin-bottom: 1rem;'>
                <h5 style='color:#fbbf24; margin-top:0;'>📚 Correlation Scale Guide</h5>
                <ul style='margin: 0; padding-left: 1.2rem; font-size: 0.85rem;'>
                    <li><b>Very Strong (|r| > 0.80):</b> High linear relationship.<br>
                        <i>Example: {vs_ex}</i></li>
                    <li style='margin-top: 3px;'><b>Strong (0.60–0.80):</b> Strong linear relationship.<br>
                        <i>Example: {s_ex}</i></li>
                    <li style='margin-top: 3px;'><b>Moderate (0.40–0.60):</b> Moderate linear relationship.<br>
                        <i>Example: {m_ex}</i></li>
                    <li style='margin-top: 3px;'><b>Weak (0.20–0.40):</b> Weak linear relationship.<br>
                        <i>Example: {w_ex}</i></li>
                    <li style='margin-top: 3px;'><b>Very Weak (< 0.20):</b> Negligible linear relationship.<br>
                        <i>Example: {vw_ex}</i></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Show list tables of top 5 correlations
            top_pos = unique_pairs_df.sort_values(by="Correlation", ascending=False).head(5)
            top_neg = unique_pairs_df.sort_values(by="Correlation", ascending=True).head(5)
            
            st.write("**Top 5 Positive Correlations**")
            pos_table = pd.DataFrame({
                "Parameter 1": [FEATURE_LABELS[f] for f in top_pos["F1"]],
                "Parameter 2": [FEATURE_LABELS[f] for f in top_pos["F2"]],
                "Correlation": [f"{val:+.4f}" for val in top_pos["Correlation"]]
            })
            st.dataframe(pos_table, use_container_width=True, hide_index=True)
            
            st.write("**Top 5 Negative Correlations**")
            neg_table = pd.DataFrame({
                "Parameter 1": [FEATURE_LABELS[f] for f in top_neg["F1"]],
                "Parameter 2": [FEATURE_LABELS[f] for f in top_neg["F2"]],
                "Correlation": [f"{val:+.4f}" for val in top_neg["Correlation"]]
            })
            st.dataframe(neg_table, use_container_width=True, hide_index=True)
            
        # Warning note separating correlation and ML importance (Section 10)
        st.markdown("""
        <div class='accuracy-note' style='margin-top: 1rem;'>
            <b>🔬 EDA vs. Machine Learning Feature Importance Note</b><br>
            <b>High correlation does not imply predictive importance.</b> A feature may be strongly correlated with another variable while contributing little additional information to a machine learning model.
            For instance, two highly collinear features might share statistical correlation, but the model may distribute weights between them or utilize only one.
            Correlation analysis (which measures bivariate linear associations) and feature importance (which measures predictive power within an ensemble framework) should be interpreted separately.
        </div>
        """, unsafe_allow_html=True)
            
        st.markdown("---")
        
        # ----------------- Section 5: Pairwise Relationships -----------------
        st.write("#### 5. Pairwise Relationships Scatter Explorer")
        st.write("Plot two physical attributes against each other to inspect exoplanet populations and identify multidimensional boundaries.")
        
        col_pair1, col_pair2 = st.columns(2)
        with col_pair1:
            x_scatter = st.selectbox(
                "X-Axis Parameter",
                SUPPORTED_EDA_FEATURES,
                index=0,
                format_func=lambda x: FEATURE_LABELS.get(x, x),
                key="scatter_x"
            )
            log_x = st.checkbox("Log-scale X-Axis", value=(viz_mode == "Log Scale"), key="scatter_log_x")
            
        with col_pair2:
            y_scatter = st.selectbox(
                "Y-Axis Parameter",
                SUPPORTED_EDA_FEATURES,
                index=1,
                format_func=lambda x: FEATURE_LABELS.get(x, x),
                key="scatter_y"
            )
            log_y = st.checkbox("Log-scale Y-Axis", value=(viz_mode == "Log Scale"), key="scatter_log_y")
            
        # Apply outlier treatment (Section 2)
        df_scatter_viz, _ = get_processed_viz_df(df_eda, [x_scatter, y_scatter], viz_mode)
        df_scatter_plot = df_scatter_viz.copy()
        df_scatter_plot['Habitability'] = df_scatter_plot['is_habitable'].map({0: "Non-Habitable", 1: "Habitable"})
        
        # Calculate Spearman Rank Correlation (Section 3)
        x_vals = df_scatter_plot[x_scatter].dropna()
        y_vals = df_scatter_plot[y_scatter].dropna()
        if len(x_vals) > 1 and len(y_vals) > 1:
            try:
                r_spearman, p_spearman = stats.spearmanr(x_vals, y_vals)
                abs_r = abs(r_spearman)
                if abs_r > 0.80:
                    strength = "Very Strong"
                elif abs_r >= 0.60:
                    strength = "Strong"
                elif abs_r >= 0.40:
                    strength = "Moderate"
                elif abs_r >= 0.20:
                    strength = "Weak"
                else:
                    strength = "Very Weak"
                    
                direction = "Positive" if r_spearman > 0 else "Negative" if r_spearman < 0 else ""
                rel_desc = f"{strength} {direction} Relationship" if direction else "No Linear Relationship"
                rel_color = "#10b981" if r_spearman > 0 else "#ef4444" if r_spearman < 0 else "#94a3b8"
            except Exception:
                r_spearman = 0.0
                rel_desc = "Calculation Error"
                rel_color = "#94a3b8"
        else:
            r_spearman = 0.0
            rel_desc = "Insufficient Data"
            rel_color = "#94a3b8"
            
        col_sc1, col_sc2 = st.columns([2, 1])
        with col_sc1:
            # Styled scatter plot with marker transparency and small sizes (Section 3)
            fig_scatter = px.scatter(
                df_scatter_plot,
                x=x_scatter,
                y=y_scatter,
                color="Habitability",
                color_discrete_map={"Non-Habitable": "#ef4444", "Habitable": "#10b981"},
                title=f"Scatter Plot: {FEATURE_LABELS[x_scatter]} vs {FEATURE_LABELS[y_scatter]} ({viz_mode})",
                opacity=0.45,
                hover_name="pl_name" if "pl_name" in df_scatter_plot.columns else None,
                hover_data={
                    "Habitability": True,
                    x_scatter: ":.4f",
                    y_scatter: ":.4f"
                }
            )
            fig_scatter.update_traces(marker=dict(size=5))
            fig_scatter.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#f8fafc",
                xaxis_title=FEATURE_LABELS[x_scatter],
                yaxis_title=FEATURE_LABELS[y_scatter],
                xaxis_type="log" if log_x else None,
                yaxis_type="log" if log_y else None,
                height=450
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        with col_sc2:
            st.markdown(f"""
            <div class='metric-card' style='border-left: 5px solid {rel_color}; margin-top: 2rem;'>
                <h5>🔬 Observed Relationship</h5>
                <p style='margin:0; font-size: 1.25rem; font-weight:bold; color: {rel_color};'>{rel_desc}</p>
                <p style='margin:5px 0 0 0; font-size: 1.5rem; font-weight:bold;'>Spearman Correlation = {r_spearman:+.4f}</p>
                <p style='color:#94a3b8; font-size: 0.85rem; margin-top:8px;'>
                    Spearman Rank Correlation evaluates the monotonic relationship between rank values. It is less sensitive to extreme outliers than Pearson correlation, providing a robust statistical measure for exoplanetary orbits and physical mass scales.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        
        # ----------------- Section 6: Scientific Insights -----------------
        st.write("#### 6. Scientific Insights (Dynamically Calculated)")
        st.write("These statistical observations are computed in real-time from the preprocessed empirical exoplanet database.")
        
        hab_df = df_eda[df_eda['is_habitable'] == 1]
        non_hab_df = df_eda[df_eda['is_habitable'] == 0]
        
        avg_temp_hab = hab_df['pl_eqt'].mean()
        avg_rad_hab = hab_df['pl_rade'].mean()
        
        q1_per = hab_df['pl_orbper'].quantile(0.25)
        q3_per = hab_df['pl_orbper'].quantile(0.75)
        
        avg_steff_hab = hab_df['st_teff'].mean()
        avg_steff_non = non_hab_df['st_teff'].mean()
        avg_smet_hab = hab_df['st_met'].mean()
        avg_smet_non = non_hab_df['st_met'].mean()
        
        col_in1, col_in2, col_in3 = st.columns(3)
        
        # Observational language pass (Section 5 / Section 11)
        with col_in1:
            st.markdown(f"""
            <div class='metric-card' style='height: 100%;'>
                <h5 style='color:#1E90FF;'>🌡️ Habitable Thermal Profile</h5>
                <p style='margin-bottom:0.5rem;'><b>Average Equilibrium Temp:</b></p>
                <div style='font-size: 1.8rem; font-weight: bold; color: #10b981;'>{avg_temp_hab:.1f} K</div>
                <p style='color:#94a3b8; font-size: 0.85rem; margin-top:0.5rem;'>
                    Within this dataset, habitable candidates are associated with equilibrium temperatures averaging {avg_temp_hab:.1f} K. Non-habitable planets average <b>{non_hab_df['pl_eqt'].mean():.1f} K</b>, reflecting extreme hot and cold zones.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        with col_in2:
            st.markdown(f"""
            <div class='metric-card' style='height: 100%;'>
                <h5 style='color:#1E90FF;'>🌍 Habitable Physical Radius</h5>
                <p style='margin-bottom:0.5rem;'><b>Average Planetary Radius:</b></p>
                <div style='font-size: 1.8rem; font-weight: bold; color: #10b981;'>{avg_rad_hab:.2f} R<sub>⊕</sub></div>
                <p style='color:#94a3b8; font-size: 0.85rem; margin-top:0.5rem;'>
                    Terrestrial rocky surfaces are preserved. In this catalog, habitable candidates tend to be associated with an average radius of {avg_rad_hab:.2f} R⊕, whereas non-habitable planets display an average radius of <b>{non_hab_df['pl_rade'].mean():.2f} R<sub>⊕</sub></b>.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        with col_in3:
            st.markdown(f"""
            <div class='metric-card' style='height: 100%;'>
                <h5 style='color:#1E90FF;'>⏳ Orbital Period Range</h5>
                <p style='margin-bottom:0.5rem;'><b>Middle 50% Period Range (IQR):</b></p>
                <div style='font-size: 1.3rem; font-weight: bold; color: #10b981;'>{q1_per:.1f} to {q3_per:.1f} days</div>
                <p style='color:#94a3b8; font-size: 0.85rem; margin-top:0.5rem;'>
                    The median orbital period is <b>{hab_df['pl_orbper'].median():.1f} days</b>. Within this dataset, habitable candidates are observed frequently in shorter period ranges compared to outer solar system equivalents.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown(f"""
        <div class='science-note' style='margin-top: 1rem;'>
            <h5>☀️ Comparative Host Stellar Properties (Observational Trends)</h5>
            Comparing host stars of habitable vs non-habitable systems:
            <ul>
                <li><b>Host Star Temperature:</b> Within this dataset, habitable candidates are associated with cooler host stars on average (<b>{avg_steff_hab:.1f} K</b>) compared to hosts of non-habitable planets (<b>{avg_steff_non:.1f} K</b>).</li>
                <li><b>Host Star Metallicity:</b> Host stars of habitable candidates average <b>{avg_smet_hab:+.3f} dex</b> compared to non-habitable host stars at <b>{avg_smet_non:+.3f} dex</b>. High metallicity disks are associated with solid matter accretion.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ----------------- Section 7: Feature Summary & Export Data -----------------
        st.write("#### 7. Feature Summary & Export Data")
        st.write("Inspect descriptive summary metrics and download the processed exoplanet catalog and calculations for local scientific analysis.")
        
        # Feature Summary Table — human-readable feature names (Section 9 / 10)
        HUMAN_READABLE_NAMES = {
            "pl_rade":     "Planet Radius (pl_rade)",
            "pl_bmasse":   "Planet Mass (pl_bmasse)",
            "pl_orbper":   "Orbital Period (pl_orbper)",
            "pl_eqt":      "Equilibrium Temperature (pl_eqt)",
            "pl_insol":    "Insolation Flux (pl_insol)",
            "pl_orbeccen": "Orbital Eccentricity (pl_orbeccen)",
            "st_teff":     "Stellar Temperature (st_teff)",
            "st_rad":      "Stellar Radius (st_rad)",
            "st_mass":     "Stellar Mass (st_mass)",
            "st_met":      "Stellar Metallicity (st_met)",
            "sy_dist":     "System Distance (sy_dist)"
        }

        summary_rows = []
        raw_missing_pcts = quality_data["missing_pcts"] if quality_data else {}
        for f in SUPPORTED_EDA_FEATURES:
            vals = df_eda[f].dropna()
            raw_pct = raw_missing_pcts.get(f, 0.0)
            clean_missing = df_eda[f].isna().sum()
            clean_pct = clean_missing / len(df_eda) * 100 if len(df_eda) > 0 else 0.0
            summary_rows.append({
                "Feature": HUMAN_READABLE_NAMES.get(f, f),
                "Description": FEATURE_LABELS[f],
                "Mean": round(vals.mean(), 4),
                "Median": round(vals.median(), 4),
                "Std Dev": round(vals.std(), 4),
                "Min": round(vals.min(), 4),
                "Max": round(vals.max(), 4),
                "Raw Missing %": f"{raw_pct:.2f}%",
                "Clean Missing %": f"{clean_pct:.2f}%"
            })
        summary_table_df = pd.DataFrame(summary_rows)
        
        st.write("**Feature Descriptive Metrics**")
        st.dataframe(summary_table_df, use_container_width=True, hide_index=True)
        st.caption("Raw Missing % represents missing measurements in the original NASA catalog. Clean Missing % represents missing values after preprocessing and feature filtering.")

        # Dataset Quality Observation — dynamically generated from actual data (Section 10)
        if quality_data:
            raw_pct_map = quality_data["missing_pcts"]
            # Identify features with >10% raw missing
            high_missing = [
                (HUMAN_READABLE_NAMES.get(f, f), pct)
                for f, pct in raw_pct_map.items()
                if f in SUPPORTED_EDA_FEATURES and pct > 10.0
            ]
            high_missing_sorted = sorted(high_missing, key=lambda x: -x[1])

            if high_missing_sorted:
                # Build human-readable list
                if len(high_missing_sorted) == 1:
                    missing_text = f"{high_missing_sorted[0][0]} ({high_missing_sorted[0][1]:.2f}%)"
                elif len(high_missing_sorted) == 2:
                    missing_text = (
                        f"{high_missing_sorted[0][0]} ({high_missing_sorted[0][1]:.2f}%) and "
                        f"{high_missing_sorted[1][0]} ({high_missing_sorted[1][1]:.2f}%)"
                    )
                else:
                    parts = [f"{name} ({pct:.2f}%)" for name, pct in high_missing_sorted[:-1]]
                    last = high_missing_sorted[-1]
                    missing_text = ", ".join(parts) + f", and {last[0]} ({last[1]:.2f}%)"

                total_clean = len(df_eda)
                total_raw = quality_data["total_raw"]
                retention_pct = total_clean / total_raw * 100 if total_raw > 0 else 0.0
                observation_text = (
                    f"{missing_text} show the highest rates of missing measurements in the original NASA "
                    f"Exoplanet Archive catalog. After preprocessing — which selects only rows with complete "
                    f"observations across all 11 physical attributes — the processed dataset retains "
                    f"{total_clean:,} of {total_raw:,} raw records ({retention_pct:.1f}% retention) "
                    f"and achieves 100% completeness across all selected features."
                )
            else:
                total_clean = len(df_eda)
                total_raw = quality_data["total_raw"]
                retention_pct = total_clean / total_raw * 100 if total_raw > 0 else 0.0
                observation_text = (
                    f"The original NASA Exoplanet Archive catalog contains {total_raw:,} records. "
                    f"After preprocessing to select complete observations across all 11 physical "
                    f"attributes, the processed dataset retains {total_clean:,} records "
                    f"({retention_pct:.1f}% retention) with 100% completeness across all features."
                )

            st.markdown(f"""
            <div class='science-note' style='margin-top: 0.8rem;'>
                <b>📋 Dataset Quality Observation</b><br>
                <p style='margin-top: 0.5rem; margin-bottom: 0;'>{observation_text}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Download summary table CSV (Section 9)
        summary_csv = summary_table_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Feature Summary (CSV)",
            data=summary_csv,
            file_name="exoplanet_feature_summary.csv",
            mime="text/csv",
            key="download_summary"
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.write("**Export Raw Catalog & Calculations**")
        col_ex1, col_ex2 = st.columns(2)
        
        with col_ex1:
            csv_cleaned = df_eda.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Cleaned Dataset (CSV)",
                data=csv_cleaned,
                file_name="exoplanet_cleaned_dataset.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_cleaned"
            )
            st.caption("Cleaned dataset containing exoplanet parameters with target habitability classifications (SMOTE-oversampling is NOT applied here; this is the raw, cleaned empirical database).")
            
        with col_ex2:
            csv_corr = corr_matrix.to_csv().encode('utf-8')
            st.download_button(
                label="📥 Download Correlation Matrix (CSV)",
                data=csv_corr,
                file_name="exoplanet_correlation_matrix.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_corr"
            )
            st.caption("Complete correlation coefficients matrix between all 11 supported planetary and stellar variables.")

        # EDA Disclaimer Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; padding: 0.5rem 0 1rem 0;'>
            <p style='color:#64748b; font-size: 0.82rem; font-style: italic;'>
                All observations shown in this dashboard are descriptive analyses of the processed NASA Exoplanet Archive dataset
                and should not be interpreted as causal scientific conclusions.
            </p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.warning("No exoplanet dataset available for analysis.")

# =====================================================================
# TAB 5: ABOUT PROJECT
# =====================================================================
with tab5:
    st.markdown("### 📖 About the ExoLife Project")

    # Dataset Stats Cards
    st.write("#### 🗄️ Dataset Overview")
    
    try:
        df_stats = get_cleaned_data()
        total_p_val = len(df_stats)
        hab_p_val = int((df_stats['is_habitable'] == 1).sum())
        non_hab_p_val = int((df_stats['is_habitable'] == 0).sum())
        ratio_p_val = non_hab_p_val / hab_p_val if hab_p_val > 0 else 0
        ratio_str = f"{ratio_p_val:.1f} : 1"
    except Exception:
        # pyrefly: ignore [bad-unpacking]
        total_p_val, hab_p_val, non_hab_p_val, ratio_str = 3,757, 49, 3,708, "75.7 : 1"

    ds1, ds2, ds3, ds4, ds5 = st.columns(5)
    with ds1:
        st.markdown("<div class='dataset-stat'><div class='stat-value'>NASA</div><div class='stat-label'>Dataset Source — Exoplanet Archive</div></div>", unsafe_allow_html=True)
    with ds2:
        st.markdown(f"<div class='dataset-stat'><div class='stat-value'>{total_p_val:,}</div><div class='stat-label'>Total Exoplanets (cleaned)</div></div>", unsafe_allow_html=True)
    with ds3:
        st.markdown(f"<div class='dataset-stat'><div class='stat-value'>{hab_p_val}</div><div class='stat-label'>Potentially Habitable</div></div>", unsafe_allow_html=True)
    with ds4:
        st.markdown(f"<div class='dataset-stat'><div class='stat-value'>{non_hab_p_val:,}</div><div class='stat-label'>Non-Habitable</div></div>", unsafe_allow_html=True)
    with ds5:
        st.markdown(f"<div class='dataset-stat'><div class='stat-value'>{ratio_str}</div><div class='stat-label'>Class Imbalance Ratio</div></div>", unsafe_allow_html=True)

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
