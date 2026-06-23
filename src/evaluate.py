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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

from src.preprocess import load_and_clean_data

def evaluate_model(data_path="data/raw/exoplanetdata.csv", model_dir="models", output_dir="outputs/evaluation"):
    """
    Evaluates model performance using cross-validation and a holdout test set (80/20 split).
    Saves performance metrics to models/metrics.json and confusion matrix plot to outputs/evaluation/.
    """
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load data
    df = load_and_clean_data(data_path)

    # 2. Extract features and target
    X = df[[ 
        "pl_rade", "pl_bmasse", "pl_orbper", "pl_eqt", "pl_insol",
        "pl_orbeccen", "st_teff", "st_rad", "st_mass", "st_met", "sy_dist"
    ]]
    y = df["is_habitable"]

    # 3. Create evaluation pipeline
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # 4. Stratified 5-Fold Cross-Validation F1-score
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores = cross_val_score(pipeline, X, y, cv=skf, scoring='f1')
    mean_cv_f1 = np.mean(f1_scores)

    # 5. Holdout Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fit pipeline on the training split only
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # 6. Compute Metrics
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

    print("\n--- Holdout Evaluation Metrics ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (Habitable Class): {precision:.4f}")
    print(f"Recall (Habitable Class): {recall:.4f}")
    print(f"F1 Score (Habitable Class): {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 7. Save Metrics to JSON
    metrics = {
        "holdout_accuracy": float(acc),
        "holdout_precision": float(precision),
        "holdout_recall": float(recall),
        "holdout_f1": float(f1),
        "cv_mean_f1": float(mean_cv_f1)
    }
    
    metrics_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Evaluation metrics saved to {metrics_path}")

    # 8. Save Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Holdout Test Set)')
    
    plot_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Confusion matrix plot saved to {plot_path}")

    return metrics

if __name__ == "__main__":
    evaluate_model()
