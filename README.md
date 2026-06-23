# ExoLife: Exoplanet Habitability Assessment Platform

An AI-powered web application that predicts the potential habitability of exoplanets using machine learning, explainable AI, and scientific benchmarking techniques.

Built using NASA Exoplanet Archive data, ExoLife combines predictive modeling, SHAP explainability, exploratory data analysis, and multi-model benchmarking into a single interactive platform.

---

## Live Demo

https://exolife-exoplanet-habitability-assessment.streamlit.app/

---

## Features

### Habitability Assessment
- Predicts whether an exoplanet is potentially habitable
- Supports manual parameter input
- Includes Earth, Mars, Kepler-22b, and Kepler-186f presets
- Dual-model framework:
  - Baseline Model (All Features)
  - Physics-Informed Proxy Model

### Explainable AI (SHAP)
- Local prediction explanations using SHAP
- Waterfall visualizations showing feature contributions
- Scientific interpretation of prediction drivers

### Exploratory Data Analysis
- Dataset quality assessment
- Class imbalance analysis
- Feature distributions
- Correlation analysis
- Habitability trend exploration

### Model Benchmarking & Validation
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- Physics-Informed Proxy Random Forest

Includes:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC
- Confusion Matrix Analysis
- Stratified 5-Fold Cross Validation
- Feature Importance Comparison

---

## Dataset

Source: NASA Exoplanet Archive

After preprocessing:

| Metric | Value |
|----------|----------|
| Total Planets | 3,757 |
| Habitable | 49 |
| Non-Habitable | 3,708 |
| Imbalance Ratio | 75.7 : 1 |

---

## Machine Learning Pipeline

### Baseline Model
Uses all available planetary and stellar parameters.

### Physics-Informed Proxy Model
Uses indirect physical indicators to avoid target leakage:

```python
PROXY_FEATURES = [
    "pl_bmasse",
    "pl_orbper",
    "pl_orbeccen",
    "st_teff",
    "st_rad",
    "st_mass",
    "st_met",
    "sy_dist"
]
```

Purpose:

* Prevent target leakage
* Encourage learning of underlying astrophysical relationships
* Improve scientific validity

---

## Project Structure

```text
ExoLife/

├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── cleanup_report.md

├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── explain.py
│   ├── predict.py
│   └── benchmark.py

├── data/
│   ├── raw/
│   │   └── exoplanetdata.csv
│   └── processed/

├── models/
│   ├── model_baseline.pkl
│   ├── model_proxy.pkl
│   ├── metrics_comparison.json
│   ├── feature_importance_baseline.csv
│   └── feature_importance_proxy.csv

├── outputs/
│   ├── eda/
│   ├── evaluation/
│   │   ├── confusion_matrix_baseline.png
│   │   └── confusion_matrix_proxy.png
│   ├── shap/
│   │   ├── shap_summary_baseline.png
│   │   ├── shap_summary_proxy.png
│   │   └── shap_explanation.html
│   └── benchmark/
│       ├── benchmark_results.csv
│       ├── cross_validation_results.csv
│       └── feature_importance_comparison.csv

└── assets/
    └── screenshots/
```

---

## Benchmark Results

| Model               | Accuracy | Precision | Recall | F1 Score | PR-AUC |
| ------------------- | -------- | --------- | ------ | -------- | ------ |
| Proxy RF            | 0.989    | 0.583     | 0.700  | 0.636    | 0.695  |
| Logistic Regression | 0.918    | 0.139     | 1.000  | 0.244    | 0.279  |
| Decision Tree       | 0.992    | 0.700     | 0.700  | 0.700    | 0.702  |
| Random Forest       | 0.989    | 0.583     | 0.700  | 0.636    | 0.695  |
| XGBoost             | 0.992    | 0.750     | 0.600  | 0.667    | 0.773  |

---

## Recommended Model

XGBoost demonstrated the strongest overall performance based on:

* Highest PR-AUC (0.773)
* Best cross-validation F1 score
* Strong generalization performance
* Lowest validation variance

Cross-Validation Results:

| Metric       | Value           |
| ------------ | --------------- |
| CV Accuracy  | 0.9862 ± 0.0032 |
| CV Precision | 0.5107 ± 0.1223 |
| CV Recall    | 0.7356 ± 0.0997 |
| CV F1        | 0.5858 ± 0.0435 |

---

## Technologies Used

### Machine Learning

* Scikit-Learn
* XGBoost
* Imbalanced-Learn (SMOTE)

### Explainable AI

* SHAP

### Data Analysis

* Pandas
* NumPy
* SciPy

### Visualization

* Plotly
* Matplotlib
* Seaborn

### Deployment

* Streamlit

---

## Screenshots

### Habitability Assessment

[Insert Screenshot]

### Explainable AI (SHAP)

[Insert Screenshot]

### Exploratory Data Analysis

[Insert Screenshot]

### Model Benchmarking

[Insert Screenshot]

---

## Scientific Considerations

The dataset is highly imbalanced, with only 1.3% of planets classified as habitable.

A naïve classifier predicting every planet as non-habitable would achieve approximately 98.7% accuracy while failing to identify any habitable candidates.

For this reason, ExoLife emphasizes:

* Recall
* F1 Score
* PR-AUC
* Cross-Validation Stability

as primary evaluation metrics.

---

## Future Work

* Deep learning approaches for habitability prediction
* Bayesian uncertainty estimation
* Automated feature engineering
* Real-time NASA archive integration
* Multi-class habitability scoring
* Planet similarity search

---

## Author

**Shivangi Dubey**  
B.Tech Computer Science & Engineering (AI & ML)  
SRM University AP  
