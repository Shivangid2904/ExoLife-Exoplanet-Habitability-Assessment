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
## Why Two Models?

### Baseline Model

The Baseline Model uses all available planetary and stellar parameters, including planet radius, equilibrium temperature, and insolation flux. Since these features are directly related to the rule-based habitability criteria used to generate labels, the model achieves very high predictive performance.

### Physics-Informed Proxy Model

The Physics-Informed Proxy Model excludes the three direct labeling features:

- `pl_rade` (Planet Radius)
- `pl_eqt` (Equilibrium Temperature)
- `pl_insol` (Insolation Flux)

This design reduces target leakage and forces the model to infer habitability using indirect astrophysical indicators such as orbital dynamics and stellar properties.

The objective is to validate whether machine learning models can learn meaningful physical relationships rather than simply reproducing the labeling rules used to create the target variable.

Despite this constraint, the proxy model achieves strong performance (F1 Score: 0.636), demonstrating that underlying astrophysical patterns remain informative for habitability assessment.
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

├── models/
│   ├── model_baseline.pkl
│   ├── model_proxy.pkl
│   ├── metrics_comparison.json
│   ├── feature_importance_baseline.csv
│   └── feature_importance_proxy.csv

├── outputs/
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

<img width="957" height="479" alt="Screenshot 2026-06-23 154601" src="https://github.com/user-attachments/assets/a70de6f9-e96d-4d85-80f0-34966e5c9cc5" />


### Explainable AI (SHAP)

<img width="957" height="472" alt="Screenshot 2026-06-23 154634" src="https://github.com/user-attachments/assets/7b7f8b13-a29a-4416-a011-0c3a126b8321" />



### Exploratory Data Analysis

<img width="957" height="467" alt="Screenshot 2026-06-23 154745" src="https://github.com/user-attachments/assets/3f4e7eed-5b00-44b8-b798-664dd682c3e0" />



### Model Benchmarking

<img width="959" height="460" alt="Screenshot 2026-06-23 154814" src="https://github.com/user-attachments/assets/0d3a3253-c33f-475c-b36b-b34d4c9b8005" />



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

## Limitations

- Habitability labels are generated using astrophysical threshold-based criteria and do not represent confirmed evidence of extraterrestrial life.
- The platform should be interpreted as an automated habitability assessment framework rather than a definitive predictor of planetary habitability.
- Atmospheric composition, magnetic field strength, geological activity, cloud dynamics, and other complex planetary factors are not incorporated into the current model.
- Model performance depends on the quality, completeness, and availability of measurements within the NASA Exoplanet Archive dataset.
- The dataset contains a severe class imbalance (49 habitable planets out of 3,757 total planets), which introduces uncertainty when generalizing to newly discovered exoplanets.
- SMOTE-generated synthetic samples are used during model benchmarking to mitigate class imbalance and may not perfectly represent real planetary distributions.
- Results should be interpreted as probabilistic assessments of habitability potential rather than direct indicators of life.
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
