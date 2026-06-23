import shap
import matplotlib.pyplot as plt
import streamlit as st
import io
import pandas as pd
import numpy as np
import os

def get_shap_explanation(input_data, pipeline):
    """
    Consolidated helper to compute SHAP values and return a shap.Explanation object.
    Supports list, 2D numpy, and 3D numpy shap values formats returned by TreeExplainer.
    Automatically detects and handles BASELINE_FEATURES vs PROXY_FEATURES based on 
    pipeline model dimension to prevent shape mismatch.
    """
    from src.preprocess import BASELINE_FEATURES, PROXY_FEATURES

    if not isinstance(input_data, pd.DataFrame):
        input_df = pd.DataFrame([input_data])
    else:
        input_df = input_data.copy()

    # Extract classifier from pipeline (handles imblearn pipeline wrapper)
    model = pipeline.named_steps['clf']

    # Auto-detect correct feature set
    n_features = getattr(model, "n_features_in_", len(BASELINE_FEATURES))
    if n_features == len(PROXY_FEATURES):
        features = PROXY_FEATURES
    else:
        features = BASELINE_FEATURES

    # Keep only the required features in the correct order to match model input shape
    input_sliced = input_df[features]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_sliced)
    predicted_class = model.predict(input_sliced)[0]

    # Handle various return shapes/types of SHAP values
    if isinstance(shap_values, list):
        # Multi-class output representation in some tree explainers
        shap_for_pred_class = shap_values[predicted_class][0]
        base_value = explainer.expected_value[predicted_class]
    elif isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3:
            # Shape (samples, features, classes)
            shap_for_pred_class = shap_values[0, :, predicted_class]
            base_value = explainer.expected_value[predicted_class]
        elif shap_values.ndim == 2:
            # Shape (samples, features) for binary output
            shap_for_pred_class = shap_values[0]
            base_value = explainer.expected_value
        else:
            raise ValueError("Unexpected shape of shap_values: " + str(shap_values.shape))
    else:
        raise ValueError("shap_values type not supported: " + str(type(shap_values)))

    explanation = shap.Explanation(
        values=shap_for_pred_class,
        base_values=base_value,
        data=input_sliced.iloc[0],
        feature_names=input_sliced.columns
    )
    return explanation, input_sliced

# --- Enhanced SHAP Explanation Function ---
def explain_prediction(input_data, pipeline):
    """
    Computes SHAP waterfall plot and displays it in Streamlit.
    Also renders the contributions table.
    """
    explanation, input_df = get_shap_explanation(input_data, pipeline)

    # --- SHAP Waterfall Plot ---
    # Draw waterfall plot on current active figure
    shap.plots.waterfall(explanation, show=False)
    fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    st.image(buf)
    plt.close(fig)

    # --- Display SHAP Value Table ---
    df_expl = pd.DataFrame({
        "Feature": input_df.columns,
        "Value": input_df.iloc[0].values,
        "SHAP Contribution": explanation.values
    })
    df_expl["|Impact|"] = df_expl["SHAP Contribution"].abs()
    df_expl = df_expl.sort_values("|Impact|", ascending=False)
    st.write("#### 🔍 Feature Contributions")
    st.dataframe(df_expl.drop(columns=["|Impact|"]))

# --- Save HTML Explanation ---
def export_html_explanation(input_data, pipeline, filename="outputs/shap/shap_explanation.html"):
    """
    Generates a SHAP waterfall plot as an SVG and saves it inside an HTML file.
    """
    explanation, _ = get_shap_explanation(input_data, pipeline)

    # Ensure output directory exists if filename includes paths
    dir_name = os.path.dirname(filename)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    fig = plt.figure()
    shap.plots.waterfall(explanation, show=False)
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", dpi=100)
    buf.seek(0)
    svg_data = buf.getvalue().decode("utf-8")

    with open(filename, "w", encoding="utf-8") as f:
        f.write("<html><body>")
        f.write("<h2>SHAP Waterfall Explanation</h2>")
        f.write(svg_data)
        f.write("</body></html>")

    plt.close(fig)
    return filename

# --- Save SHAP Summary Beeswarm Plot ---
def generate_shap_summary_plot(pipeline, background_data, filepath="outputs/shap/shap_summary.png"):
    """
    Generates a SHAP summary (beeswarm) plot for a set of data and saves it.
    Automatically aligns features and processes multiclass or array SHAP values.
    """
    from src.preprocess import BASELINE_FEATURES, PROXY_FEATURES
    
    dir_name = os.path.dirname(filepath)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    
    model = pipeline.named_steps['clf']
    
    # Auto-detect correct feature set
    n_features = getattr(model, "n_features_in_", len(BASELINE_FEATURES))
    if n_features == len(PROXY_FEATURES):
        features = PROXY_FEATURES
    else:
        features = BASELINE_FEATURES
        
    df_sliced = background_data[features]
    
    explainer = shap.TreeExplainer(model)
    
    # Sample a representative subset for efficiency and readability
    sample_df = df_sliced.sample(min(100, len(df_sliced)), random_state=42)
    shap_values = explainer.shap_values(sample_df)
    
    # Handle class indexing for SHAP summary plots
    if isinstance(shap_values, list):
        # Index 1 is the habitable class
        shap_values_class = shap_values[1]
    elif isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3:
            shap_values_class = shap_values[:, :, 1]
        else:
            shap_values_class = shap_values
    else:
        shap_values_class = shap_values

    fig = plt.figure()
    shap.summary_plot(shap_values_class, sample_df, show=False)
    plt.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return filepath

