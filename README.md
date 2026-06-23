# ExoLife – AI-Powered Exoplanet Habitability Assessment Platform

## Overview

ExoLife is an end-to-end Data Science and Machine Learning platform designed to assess the potential habitability of exoplanets using data from the NASA Exoplanet Archive. The project combines data preprocessing, exploratory data analysis, machine learning, explainable AI, and interactive deployment to provide interpretable habitability assessments.

Rather than manually analyzing thousands of discovered exoplanets, ExoLife automates the evaluation process by leveraging planetary and stellar characteristics to identify planets that may support conditions favorable for life.

---

## Problem Statement

Thousands of exoplanets have been discovered by modern telescopes, making manual habitability assessment increasingly difficult.

The objective of ExoLife is to build an explainable machine learning framework that evaluates exoplanet habitability using scientific criteria and provides transparent predictions through interpretable AI techniques.

---

## Objectives

* Analyze exoplanet and stellar data from the NASA Exoplanet Archive.
* Perform comprehensive Exploratory Data Analysis (EDA).
* Build and compare multiple machine learning models.
* Handle class imbalance using advanced resampling techniques.
* Explain model predictions using SHAP.
* Generate interpretable habitability scores.
* Deploy an interactive web application for real-time assessment.

---

## Dataset

**Source:** NASA Exoplanet Archive

### Features Used

| Feature     | Description                   |
| ----------- | ----------------------------- |
| pl_rade     | Planet Radius                 |
| pl_bmasse   | Planet Mass                   |
| pl_orbper   | Orbital Period                |
| pl_eqt      | Equilibrium Temperature       |
| pl_insol    | Insolation Flux               |
| pl_orbeccen | Orbital Eccentricity          |
| st_teff     | Stellar Effective Temperature |
| st_rad      | Stellar Radius                |
| st_mass     | Stellar Mass                  |
| st_met      | Stellar Metallicity           |
| sy_dist     | Distance from Earth           |

---

## Project Workflow

NASA Exoplanet Dataset
↓
Data Cleaning & Preprocessing
↓
Exploratory Data Analysis (EDA)
↓
Feature Engineering
↓
Habitability Label Generation
↓
Class Imbalance Analysis (SMOTE)
↓
Model Training & Comparison
↓
Model Evaluation
↓
SHAP Explainability
↓
Habitability Score Generation
↓
Streamlit Deployment

---

## Exploratory Data Analysis (EDA)

### Analysis Performed

* Planet Radius Distribution
* Planet Mass Distribution
* Orbital Period Distribution
* Equilibrium Temperature Distribution
* Insolation Flux Distribution
* Correlation Heatmaps
* Statistical Summary Analysis
* Habitability Class Distribution

### Key Questions

* What characteristics are common among potentially habitable planets?
* Which features are strongly correlated?
* How severe is class imbalance?
* Which planetary properties influence habitability the most?

---

## Data Preprocessing

### Steps Performed

* Loaded NASA dataset from the correct header row.
* Selected scientifically relevant planetary and stellar features.
* Removed records containing missing values.
* Applied feature scaling using StandardScaler.
* Generated habitability labels using astrophysical criteria.

---

## Habitability Criteria

A planet is labeled as potentially habitable if:

* Planet Radius between 0.5 and 2.5 Earth radii
* Equilibrium Temperature between 180 K and 310 K
* Insolation Flux between 0.3 and 1.8 Earth flux units

Classification:

* 1 → Potentially Habitable
* 0 → Non-Habitable

---

## Class Imbalance Handling

The dataset contains significantly fewer habitable planets than non-habitable planets.

To address this issue:

* Analyzed class distributions.
* Applied SMOTE (Synthetic Minority Over-sampling Technique).
* Compared model performance before and after balancing.
* Evaluated final performance on the original test set.

This helps reduce bias toward the majority class and improves model generalization.

---

## Machine Learning Models

Three models are trained and compared.

### Logistic Regression

Used as a baseline model.

### Decision Tree

Provides interpretability and nonlinear decision boundaries.

### Random Forest

Selected as the primary model because it:

* Handles nonlinear relationships effectively.
* Reduces overfitting through ensemble learning.
* Performs well on structured tabular data.
* Supports feature importance analysis.

---

## Model Evaluation

Models are evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score

### Evaluation Visualizations

* Confusion Matrix
* ROC Curves
* Model Comparison Table
* Feature Importance Analysis

The best-performing model is selected based on overall predictive performance rather than accuracy alone.

---

## Explainable AI (XAI)

ExoLife integrates SHAP (SHapley Additive Explanations) to improve transparency and interpretability.

### Global Explanations

Used to identify:

* Most influential features
* Overall model behavior
* Feature importance rankings

### Local Explanations

Used to explain:

* Individual planet predictions
* Why a planet was classified as habitable or non-habitable

### SHAP Visualizations

* Summary Plot
* Beeswarm Plot
* Waterfall Plot
* Force Plot

---

## Habitability Score System

Instead of providing only a binary prediction, ExoLife generates a Habitability Score.

### Score Categories

| Score Range | Classification     |
| ----------- | ------------------ |
| 0–39%       | Low Potential      |
| 40–69%      | Moderate Potential |
| 70–100%     | High Potential     |

This provides a more intuitive interpretation of model predictions.

---

## Streamlit Application

### Features

* Interactive user interface
* Real-time habitability assessment
* Habitability score generation
* SHAP-based explanations
* Data exploration dashboard
* Model performance overview

---

## Technology Stack

### Programming Language

* Python

### Data Analysis

* Pandas
* NumPy

### Machine Learning

* Scikit-Learn
* Imbalanced-Learn

### Explainable AI

* SHAP

### Visualization

* Matplotlib
* Seaborn
* Plotly

### Deployment

* Streamlit

### Development Tools

* VS Code
* Git
* GitHub

---

## Project Structure

```text
ExoLife/

├── app.py
├── requirements.txt
├── README.md

├── src/
│   ├── eda.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── explain.py
│   └── predict.py

├── data/
│   ├── raw/
│   └── processed/

├── models/
│   ├── rf_model.pkl
│   ├── scaler.pkl
│   └── explainer.pkl

├── outputs/
│   ├── eda/
│   ├── evaluation/
│   └── shap/

└── assets/
    └── screenshots/
```

---

## Key Learning Outcomes

* Data Cleaning and Preprocessing
* Exploratory Data Analysis
* Feature Engineering
* Handling Imbalanced Datasets
* Model Comparison and Selection
* Explainable AI using SHAP
* Machine Learning Evaluation Metrics
* Streamlit Application Development
* End-to-End ML Pipeline Design
* Project Structuring and Deployment

---

## Limitations

* Habitability labels are generated using established astrophysical thresholds rather than direct evidence of life.
* Atmospheric composition and planetary geology are not considered.
* The model should be interpreted as a habitability assessment framework rather than a definitive predictor of extraterrestrial life.
* Results depend on currently available exoplanet observations and may evolve as new discoveries are made.

---

## Future Enhancements

* Experiment with XGBoost and Gradient Boosting models.
* Add hyperparameter optimization.
* Integrate real-time NASA API data.
* Incorporate atmospheric composition features.
* Expand explainability using feature interaction analysis.
* Deploy using cloud infrastructure for scalability.

---

## Conclusion

ExoLife demonstrates a complete Data Science workflow from raw NASA exoplanet data to explainable machine learning predictions and interactive deployment. By combining exploratory data analysis, class imbalance handling, model comparison, SHAP explainability, and Streamlit deployment, the platform provides a transparent and scientifically grounded framework for exoplanet habitability assessment.
