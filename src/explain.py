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
    """
    if not isinstance(input_data, pd.DataFrame):
        input_df = pd.DataFrame([input_data])
    else:
        input_df = input_data

    # Extract classifier from pipeline (handles imblearn pipeline wrapper)
    model = pipeline.named_steps['clf']

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    predicted_class = model.predict(input_df)[0]

    # Handle various return shapes/types of SHAP values
    if isinstance(shap_values, list):
        shap_for_pred_class = shap_values[predicted_class][0]
        base_value = explainer.expected_value[predicted_class]
    elif isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3:
            shap_for_pred_class = shap_values[0, :, predicted_class]
            base_value = explainer.expected_value[predicted_class]
        elif shap_values.ndim == 2:
            shap_for_pred_class = shap_values[0]
            base_value = explainer.expected_value
        else:
            raise ValueError("Unexpected shape of shap_values: " + str(shap_values.shape))
    else:
        raise ValueError("shap_values type not supported: " + str(type(shap_values)))

    explanation = shap.Explanation(
        values=shap_for_pred_class,
        base_values=base_value,
        data=input_df.iloc[0],
        feature_names=input_df.columns
    )
    return explanation, input_df

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
