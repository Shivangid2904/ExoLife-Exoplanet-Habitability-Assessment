import pandas as pd
import joblib
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from src.preprocess import BASELINE_FEATURES, PROXY_FEATURES, load_and_clean_data

def train_pipeline(data_path="data/raw/exoplanetdata.csv", model_dir="models"):
    """
    Trains the Dual-Model Framework:
    1. Baseline Random Forest using BASELINE_FEATURES.
    2. Proxy models (Random Forest and Logistic Regression) using PROXY_FEATURES.
    Selects the best Proxy model based on CV F1 score, and saves all outputs.
    """
    os.makedirs(model_dir, exist_ok=True)

    # 1. Load and preprocess data
    df = load_and_clean_data(data_path)
    y = df["is_habitable"]

    print("Original Class Distribution:\n", y.value_counts())

    # --- MODEL A: BASELINE MODEL ---
    print("\n--- Training Model A: Baseline Random Forest ---")
    X_baseline = df[BASELINE_FEATURES]
    
    baseline_pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_baseline_f1 = np.mean(cross_val_score(baseline_pipeline, X_baseline, y, cv=skf, scoring='f1'))
    print(f"Model A (Baseline RF) Mean CV F1 Score: {cv_baseline_f1:.4f}")

    baseline_pipeline.fit(X_baseline, y)
    baseline_model_path = os.path.join(model_dir, "model_baseline.pkl")
    joblib.dump(baseline_pipeline, baseline_model_path)
    print(f"Model A saved to {baseline_model_path}")

    # Save Baseline Feature Importance
    rf_clf = baseline_pipeline.named_steps['clf']
    importance_df_baseline = pd.DataFrame({
        "Feature": BASELINE_FEATURES,
        "Importance": rf_clf.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    importance_path_baseline = os.path.join(model_dir, "feature_importance_baseline.csv")
    importance_df_baseline.to_csv(importance_path_baseline, index=False)
    print(f"Model A feature importances saved to {importance_path_baseline}")


    # --- MODEL B: PROXY ML MODEL ---
    print("\n--- Training Model B: Proxy ML Models ---")
    X_proxy = df[PROXY_FEATURES]

    # Proxy RF Pipeline (No scaling needed for RF, but keeping SMOTE)
    proxy_rf_pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Proxy LR Pipeline (Requires Scaling first)
    proxy_lr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])

    print("Evaluating Proxy Random Forest...")
    cv_proxy_rf_f1 = np.mean(cross_val_score(proxy_rf_pipeline, X_proxy, y, cv=skf, scoring='f1'))
    print(f"Proxy Random Forest Mean CV F1 Score: {cv_proxy_rf_f1:.4f}")

    print("Evaluating Proxy Logistic Regression...")
    cv_proxy_lr_f1 = np.mean(cross_val_score(proxy_lr_pipeline, X_proxy, y, cv=skf, scoring='f1'))
    print(f"Proxy Logistic Regression Mean CV F1 Score: {cv_proxy_lr_f1:.4f}")

    # Selection
    if cv_proxy_rf_f1 >= cv_proxy_lr_f1:
        print("\nSelected Proxy Model: Random Forest")
        best_proxy_pipeline = proxy_rf_pipeline
        proxy_name = "Random Forest"
        proxy_cv_f1 = cv_proxy_rf_f1
    else:
        print("\nSelected Proxy Model: Logistic Regression")
        best_proxy_pipeline = proxy_lr_pipeline
        proxy_name = "Logistic Regression"
        proxy_cv_f1 = cv_proxy_lr_f1

    # Fit and save selected Proxy model
    print("Fitting selected Proxy model on entire dataset...")
    best_proxy_pipeline.fit(X_proxy, y)
    proxy_model_path = os.path.join(model_dir, "model_proxy.pkl")
    joblib.dump(best_proxy_pipeline, proxy_model_path)
    print(f"Model B saved to {proxy_model_path}")

    # Extract & Save Feature Importance / Weights for Proxy Model
    clf_proxy = best_proxy_pipeline.named_steps['clf']
    if hasattr(clf_proxy, 'feature_importances_'):
        importances_proxy = clf_proxy.feature_importances_
    elif hasattr(clf_proxy, 'coef_'):
        # For Logistic Regression, absolute coefficients represent importance magnitude
        importances_proxy = np.abs(clf_proxy.coef_[0])
    else:
        importances_proxy = np.zeros(len(PROXY_FEATURES))

    importance_df_proxy = pd.DataFrame({
        "Feature": PROXY_FEATURES,
        "Importance": importances_proxy
    }).sort_values(by="Importance", ascending=False)
    importance_path_proxy = os.path.join(model_dir, "feature_importance_proxy.csv")
    importance_df_proxy.to_csv(importance_path_proxy, index=False)
    print(f"Model B feature importances saved to {importance_path_proxy}")

    return {
        "baseline_cv_f1": cv_baseline_f1,
        "proxy_rf_cv_f1": cv_proxy_rf_f1,
        "proxy_lr_cv_f1": cv_proxy_lr_f1,
        "selected_proxy": proxy_name,
        "selected_proxy_cv_f1": proxy_cv_f1
    }

if __name__ == "__main__":
    train_pipeline()

