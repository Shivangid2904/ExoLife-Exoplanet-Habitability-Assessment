import os
import json
import numpy as np
import pandas as pd
import matplotlib
# Use non-interactive backend to prevent GUI blocking
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

from src.preprocess import BASELINE_FEATURES, PROXY_FEATURES, load_and_clean_data

def plot_and_save_cm(y_true, y_pred, title, file_path):
    """
    Plots confusion matrix and saves it as a PNG file.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.savefig(file_path, bbox_inches='tight', dpi=150)
    plt.close()

def evaluate_models(data_path="data/raw/exoplanetdata.csv", model_dir="models", output_dir="outputs/evaluation"):
    """
    Evaluates both Baseline and Proxy models on a shared holdout test set (80/20 split)
    and performs cross-validation. Saves metrics to models/metrics_comparison.json
    and confusion matrices to outputs/evaluation/.
    """
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load data
    df = load_and_clean_data(data_path)
    y = df["is_habitable"]

    # Class distribution analysis
    class_counts = y.value_counts()
    class_ratio = class_counts / len(y)
    print("\n--- Dataset Class Distribution ---")
    print(f"Total exoplanets in cleaned set: {len(y)}")
    for val, count in class_counts.items():
        print(f"Class {val} ({'Habitable' if val == 1 else 'Not Habitable'}): {count} planets ({class_ratio[val]*100:.2f}%)")

    # 2. Shared holdout train/test split (80/20)
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=42, stratify=y
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 3. Model A: Baseline Random Forest
    print("\n--- Evaluating Model A: Baseline Random Forest ---")
    X_train_base = X_train_df[BASELINE_FEATURES]
    X_test_base = X_test_df[BASELINE_FEATURES]

    baseline_pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Run CV F1 Score
    cv_baseline_f1 = np.mean(cross_val_score(baseline_pipeline, df[BASELINE_FEATURES], y, cv=skf, scoring='f1'))

    # Fit on training split and predict
    baseline_pipeline.fit(X_train_base, y_train)
    y_pred_base = baseline_pipeline.predict(X_test_base)
    y_proba_base = baseline_pipeline.predict_proba(X_test_base)[:, 1]

    # Calculate metrics
    acc_base = accuracy_score(y_test, y_pred_base)
    prec_base = precision_score(y_test, y_pred_base, zero_division=0)
    rec_base = recall_score(y_test, y_pred_base, zero_division=0)
    f1_base = f1_score(y_test, y_pred_base, zero_division=0)
    roc_auc_base = roc_auc_score(y_test, y_proba_base)

    print(f"Accuracy: {acc_base:.4f}")
    print(f"Precision: {prec_base:.4f}")
    print(f"Recall: {rec_base:.4f}")
    print(f"F1 Score: {f1_base:.4f}")
    print(f"ROC-AUC: {roc_auc_base:.4f}")
    print(f"Mean CV F1 Score: {cv_baseline_f1:.4f}")

    # Plot and save confusion matrix
    cm_path_base = os.path.join(output_dir, "confusion_matrix_baseline.png")
    plot_and_save_cm(y_test, y_pred_base, "Confusion Matrix - Baseline (Leaky)", cm_path_base)
    print(f"Confusion matrix saved to {cm_path_base}")

    # 4. Model B: Proxy Random Forest (the selected proxy)
    print("\n--- Evaluating Model B: Proxy Random Forest ---")
    X_train_proxy = X_train_df[PROXY_FEATURES]
    X_test_proxy = X_test_df[PROXY_FEATURES]

    proxy_pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Run CV F1 Score
    cv_proxy_f1 = np.mean(cross_val_score(proxy_pipeline, df[PROXY_FEATURES], y, cv=skf, scoring='f1'))

    # Fit on training split and predict
    proxy_pipeline.fit(X_train_proxy, y_train)
    y_pred_proxy = proxy_pipeline.predict(X_test_proxy)
    y_proba_proxy = proxy_pipeline.predict_proba(X_test_proxy)[:, 1]

    # Calculate metrics
    acc_proxy = accuracy_score(y_test, y_pred_proxy)
    prec_proxy = precision_score(y_test, y_pred_proxy, zero_division=0)
    rec_proxy = recall_score(y_test, y_pred_proxy, zero_division=0)
    f1_proxy = f1_score(y_test, y_pred_proxy, zero_division=0)
    roc_auc_proxy = roc_auc_score(y_test, y_proba_proxy)

    print(f"Accuracy: {acc_proxy:.4f}")
    print(f"Precision: {prec_proxy:.4f}")
    print(f"Recall: {rec_proxy:.4f}")
    print(f"F1 Score: {f1_proxy:.4f}")
    print(f"ROC-AUC: {roc_auc_proxy:.4f}")
    print(f"Mean CV F1 Score: {cv_proxy_f1:.4f}")

    # Plot and save confusion matrix
    cm_path_proxy = os.path.join(output_dir, "confusion_matrix_proxy.png")
    plot_and_save_cm(y_test, y_pred_proxy, "Confusion Matrix - Proxy (Physics-Based)", cm_path_proxy)
    print(f"Confusion matrix saved to {cm_path_proxy}")

    # 5. Consolidate and Save Metrics to JSON
    metrics = {
        "baseline": {
            "holdout_accuracy": float(acc_base),
            "holdout_precision": float(prec_base),
            "holdout_recall": float(rec_base),
            "holdout_f1": float(f1_base),
            "holdout_roc_auc": float(roc_auc_base),
            "cv_mean_f1": float(cv_baseline_f1)
        },
        "proxy": {
            "holdout_accuracy": float(acc_proxy),
            "holdout_precision": float(prec_proxy),
            "holdout_recall": float(rec_proxy),
            "holdout_f1": float(f1_proxy),
            "holdout_roc_auc": float(roc_auc_proxy),
            "cv_mean_f1": float(cv_proxy_f1)
        }
    }

    metrics_path = os.path.join(model_dir, "metrics_comparison.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nComparative evaluation metrics saved to {metrics_path}")

    return metrics

if __name__ == "__main__":
    evaluate_models()

