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

SUPPORTED_EDA_FEATURES = BASELINE_FEATURES

@st.cache_data
def get_cleaned_data():
    from src.preprocess import load_and_clean_data
    return load_and_clean_data()

@st.cache_data
def get_correlation_matrix(df, features):
    return df[features].corr()

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
        
        # ----------------- Section 1: Dataset Overview -----------------
        st.write("#### 1. Dataset Overview & Imbalance")
        
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
                    <div class='stat-label'>Total Planets</div>
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
            
            # Text explanation
            st.markdown(f"""
            <div class='accuracy-note' style='margin-top: 1rem;'>
                <b>🔬 How imbalanced is the dataset?</b><br>
                The dataset is extremely imbalanced with a ratio of <b>{ratio_p:.1f} to 1</b> non-habitable planets for every habitable one. 
                Only <b>{hab_p / total_p * 100:.2f}%</b> of the planets in this catalog satisfy the physical requirements of rocky structure, moderate equilibrium temperature, and appropriate insolation flux. This extreme skew illustrates why evaluation metrics like F1-Score and Recall are critical.
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
                title="Exoplanet Distribution Breakdown",
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
            
            # Log scale toggle
            log_scale_dist = st.checkbox("Log-scale X-Axis", value=False, key="dist_log_toggle")
            
            # Compute stats
            feat_vals = df_eda[selected_dist_feat].dropna()
            f_mean = feat_vals.mean()
            f_median = feat_vals.median()
            f_std = feat_vals.std()
            f_min = feat_vals.min()
            f_max = feat_vals.max()
            
            st.markdown(f"""
            <div class='metric-card'>
                <h5>📊 Descriptive Statistics</h5>
                <table style='width:100%; border-collapse: collapse;'>
                    <tr><td style='padding: 4px 0; color:#94a3b8;'>Mean</td><td style='text-align:right; font-weight:bold; color:#38bdf8;'>{f_mean:.4f}</td></tr>
                    <tr><td style='padding: 4px 0; color:#94a3b8;'>Median</td><td style='text-align:right; font-weight:bold; color:#38bdf8;'>{f_median:.4f}</td></tr>
                    <tr><td style='padding: 4px 0; color:#94a3b8;'>Std Dev</td><td style='text-align:right; font-weight:bold; color:#38bdf8;'>{f_std:.4f}</td></tr>
                    <tr><td style='padding: 4px 0; color:#94a3b8;'>Minimum</td><td style='text-align:right; font-weight:bold; color:#38bdf8;'>{f_min:.4f}</td></tr>
                    <tr><td style='padding: 4px 0; color:#94a3b8;'>Maximum</td><td style='text-align:right; font-weight:bold; color:#38bdf8;'>{f_max:.4f}</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
            
        with col_dist2:
            # Determine if KDE is appropriate (avoid on sparse or extremely low variance)
            is_kde_appropriate = len(feat_vals.unique()) >= 10 and f_std > 0.001
            
            # Set up Plotly histogram
            nbins = 50
            if log_scale_dist:
                hist_vals = feat_vals[feat_vals > 0]
                fig_dist = px.histogram(
                    x=hist_vals,
                    nbins=nbins,
                    title=f"Log-Distribution of {FEATURE_LABELS[selected_dist_feat]}",
                    labels={"x": FEATURE_LABELS[selected_dist_feat]},
                    color_discrete_sequence=["#38bdf8"]
                )
                fig_dist.update_layout(xaxis_type="log")
            else:
                fig_dist = px.histogram(
                    x=feat_vals,
                    histnorm="probability density" if is_kde_appropriate else None,
                    nbins=nbins,
                    title=f"Distribution of {FEATURE_LABELS[selected_dist_feat]}",
                    labels={"x": FEATURE_LABELS[selected_dist_feat], "y": "Density" if is_kde_appropriate else "Count"},
                    color_discrete_sequence=["#38bdf8"]
                )
                
                if is_kde_appropriate:
                    try:
                        kde = stats.gaussian_kde(feat_vals)
                        x_grid = np.linspace(f_min, f_max, 300)
                        y_kde = kde(x_grid)
                        fig_dist.add_scatter(
                            x=x_grid,
                            y=y_kde,
                            mode="lines",
                            name="KDE Curve",
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
        
        # Split data
        hab_vals = df_eda[df_eda['is_habitable'] == 1][selected_comp_feat].dropna()
        non_hab_vals = df_eda[df_eda['is_habitable'] == 0][selected_comp_feat].dropna()
        
        comp_tab1, comp_tab2, comp_tab3 = st.tabs([
            "📈 Overlapping Distributions",
            "📦 Box Plot",
            "🎻 Violin Plot"
        ])
        
        interpretations = {
            "pl_rade": "<b>Planet Radius</b> is a core defining metric of rocky planets. Potentially habitable candidates are tightly constrained between 0.5 and 2.5 Earth Radii. Planets exceeding 2.5 Earth Radii accumulate heavy hydrogen/helium envelopes, turning into volatile-rich Mini-Neptunes or Gas Giants without a solid surface, while planets smaller than 0.5 Earth Radii have insufficient gravity to retain habitable atmospheres.",
            "pl_bmasse": "Although <b>Planet Mass</b> is not directly locked in the habitability label, it displays a strong separation. Habitable candidates reside in the lower mass spectrum (typically under 10 Earth masses), which corresponds to rocky super-Earths and Earth-analogs. Extremely massive planets become gas giants, retaining thick envelopes that lead to immense pressure and temperatures hostile to life.",
            "pl_orbper": "<b>Orbital Period</b> shows that habitable planets around cool M-dwarf host stars reside in short orbits (under 100 days), whereas habitable candidates orbiting Sun-like stars have periods closer to 360 days. Non-habitable planets cover the entire span, showing a dense population of hot Jupiters and hot super-Earths with extremely short periods (under 10 days) that lie well inside the inner boundary of the Habitable Zone.",
            "pl_eqt": "<b>Equilibrium Temperature</b> represents the stellar heating balance. Habitable planets are strictly confined to the 160 – 330 K range. Non-habitable worlds exhibit wide dispersion, including scorching close-in planets exceeding 1,000 K and frozen outer worlds. The 160–330 K range allows carbon-dioxide-water vapor feedback cycles to support liquid water on a solid surface.",
            "pl_insol": "<b>Insolation Flux</b> represents the stellar energy received. Habitable planets require 0.3 to 1.8 times Earth's insolation. Exceeding 1.8 causes a runaway greenhouse effect (evaporating oceans, like Venus), while falling below 0.3 leads to global glaciation (like Mars, without strong greenhouse gases).",
            "pl_orbeccen": "<b>Orbital Eccentricity</b> determines orbit circularity. Habitable candidates consistently have low eccentricity (<0.2), ensuring stable, continuous stellar flux year-round. Planets with highly eccentric orbits experience extreme seasonal swings, causing surface sterilization when passing close to the star and deep freezing when furthest away.",
            "st_teff": "<b>Stellar Effective Temperature</b> shows habitable planets can orbit a wide range of stars, but are frequently discovered around cooler M-dwarfs (3,000-4,000 K) and Sun-like G-dwarfs (5,000-6,000 K). M-dwarf systems have narrow, close-in habitable zones, whereas hotter stars have wider, distant habitable zones.",
            "st_rad": "<b>Stellar Radius</b> correlates directly with star luminosity. Smaller stars have lower energy output, requiring habitable planets to orbit very close to their host. This increases the likelihood of tidal locking, where one side of the planet permanently faces the star.",
            "st_mass": "<b>Stellar Mass</b> dictates the star's evolutionary lifespan. Habitable candidates are typically found around low to intermediate mass stars (0.1 to 1.2 solar masses) because massive stars burn their fuel too rapidly (in millions rather than billions of years), exhausting their fuel before life can develop.",
            "st_met": "<b>Stellar Metallicity</b> represents the fraction of heavy elements. Higher metallicity host stars indicate dust-rich protoplanetary disks, which facilitate rocky core accretion. Habitable planets occur across a range of metallicities, but tend to center near solar values.",
            "sy_dist": "<b>System Distance</b> is an observational bias rather than a physical requirement. Habitable planets tend to be clustered closer to Earth (<500 parsecs) because small, terrestrial worlds are extremely difficult to detect at great distances with current transit or radial velocity instruments."
        }
        
        with comp_tab1:
            fig_comp_dist = px.histogram(
                df_plot,
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
                height=350
            )
            st.plotly_chart(fig_comp_dist, use_container_width=True)
            
        with comp_tab2:
            fig_comp_box = px.box(
                df_plot,
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
                height=350
            )
            st.plotly_chart(fig_comp_box, use_container_width=True)
            
        with comp_tab3:
            fig_comp_viol = px.violin(
                df_plot,
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
                height=350
            )
            st.plotly_chart(fig_comp_viol, use_container_width=True)
            
        st.markdown(f"""
        <div class='science-note'>
            <b>🔬 How do habitable planets differ from non-habitable planets?</b><br>
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
        
        col_corr1, col_corr2 = st.columns([3, 2])
        
        with col_corr1:
            readable_labels = [FEATURE_LABELS[f] for f in SUPPORTED_EDA_FEATURES]
            fig_corr = px.imshow(
                corr_matrix,
                labels=dict(color="Correlation"),
                x=readable_labels,
                y=readable_labels,
                color_continuous_scale="RdBu",
                zmin=-1,
                zmax=1,
                title="Interactive Feature Correlation Matrix"
            )
            fig_corr.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#f8fafc",
                height=450,
                xaxis=dict(tickangle=45),
                margin=dict(l=40, r=40, t=50, b=40)
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
        with col_corr2:
            st.markdown("""
            <div class='science-note' style='padding: 1rem; margin-bottom: 1rem;'>
                <h5>⚠️ Core Correlation Concept</h5>
                <b>Correlation does not imply causation.</b> A high correlation value indicates that two variables change together, but does not prove one causes the other. For instance, distance to host star and orbital period are correlated due to orbital physics, not because distance generates the period.
            </div>
            """, unsafe_allow_html=True)
            
            # Extract top positive and negative correlations
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
            log_x = st.checkbox("Log-scale X-Axis", value=False, key="scatter_log_x")
            
        with col_pair2:
            y_scatter = st.selectbox(
                "Y-Axis Parameter",
                SUPPORTED_EDA_FEATURES,
                index=1,
                format_func=lambda x: FEATURE_LABELS.get(x, x),
                key="scatter_y"
            )
            log_y = st.checkbox("Log-scale Y-Axis", value=False, key="scatter_log_y")
            
        fig_scatter = px.scatter(
            df_plot,
            x=x_scatter,
            y=y_scatter,
            color="Habitability",
            color_discrete_map={"Non-Habitable": "#ef4444", "Habitable": "#10b981"},
            title=f"Scatter Plot: {FEATURE_LABELS[x_scatter]} vs {FEATURE_LABELS[y_scatter]}",
            hover_name="pl_name" if "pl_name" in df_plot.columns else None,
            hover_data={
                "Habitability": True,
                x_scatter: ":.4f",
                y_scatter: ":.4f"
            }
        )
        
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
        
        st.markdown("---")
        
        # ----------------- Section 6: Scientific Insights -----------------
        st.write("#### 6. Scientific Insights (Dynamically Calculated)")
        st.write("These insights are computed in real-time from the active dataset to characterize properties of habitable candidates.")
        
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
        
        with col_in1:
            st.markdown(f"""
            <div class='metric-card' style='height: 100%;'>
                <h5 style='color:#1E90FF;'>🌡️ Habitable Thermal Profile</h5>
                <p style='margin-bottom:0.5rem;'><b>Average Equilibrium Temp:</b></p>
                <div style='font-size: 1.8rem; font-weight: bold; color: #10b981;'>{avg_temp_hab:.1f} K</div>
                <p style='color:#94a3b8; font-size: 0.85rem; margin-top:0.5rem;'>
                    This thermal profile aligns with liquid water surface limits. By comparison, non-habitable worlds average <b>{non_hab_df['pl_eqt'].mean():.1f} K</b>, reflecting extreme hot and cold zones.
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
                    Terrestrial rocky surfaces are preserved. Non-habitable planets have an average radius of <b>{non_hab_df['pl_rade'].mean():.2f} R<sub>⊕</sub></b>, which is dominated by massive gaseous Jovian configurations.
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
                    The median orbital period is <b>{hab_df['pl_orbper'].median():.1f} days</b>. This demonstrates that many habitable candidates reside in close orbits around cool M-dwarf host stars.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown(f"""
        <div class='science-note' style='margin-top: 1rem;'>
            <h5>☀️ Comparative Host Stellar Properties</h5>
            Comparing host stars of habitable vs non-habitable systems:
            <ul>
                <li><b>Host Star Temperature:</b> Habitable hosts average <b>{avg_steff_hab:.1f} K</b> (typically late K and M dwarfs or solar G dwarfs) vs. non-habitable hosts which average <b>{avg_steff_non:.1f} K</b>.</li>
                <li><b>Host Star Metallicity:</b> Habitable hosts average <b>{avg_smet_hab:+.3f} dex</b> compared to non-habitable hosts at <b>{avg_smet_non:+.3f} dex</b>. High metallicity increases solid accretion matter, facilitating rocky terrestrial cores.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ----------------- Section 7: Export -----------------
        st.write("#### 7. Export Data & Matrices")
        st.write("Download the processed exoplanet catalog and correlation calculations for local scientific analysis.")
        
        col_ex1, col_ex2 = st.columns(2)
        
        with col_ex1:
            csv_cleaned = df_eda.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Cleaned Dataset (CSV)",
                data=csv_cleaned,
                file_name="exoplanet_cleaned_dataset.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.caption("Cleaned dataset containing exoplanet parameters with target habitability classifications (SMOTE-oversampling is NOT applied here; this is the raw, cleaned empirical database).")
            
        with col_ex2:
            csv_corr = corr_matrix.to_csv().encode('utf-8')
            st.download_button(
                label="📥 Download Correlation Matrix (CSV)",
                data=csv_corr,
                file_name="exoplanet_correlation_matrix.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.caption("Complete correlation coefficients matrix between all 11 supported planetary and stellar variables.")
            
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
