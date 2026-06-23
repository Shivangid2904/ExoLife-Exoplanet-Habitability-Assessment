"""
src/benchmark.py
----------------
Phase 5 — Model Benchmarking & Validation

Trains and evaluates five classifiers on the Physics-Informed Proxy feature set
(PROXY_FEATURES) using an identical 80/20 stratified holdout split and
Stratified 5-Fold Cross-Validation.

Models evaluated:
  1. Proxy Random Forest  (matches existing proxy model)
  2. Logistic Regression
  3. Decision Tree
  4. Random Forest
  5. XGBoost              (skipped gracefully if not installed)

Outputs (all written to outputs/benchmark/):
  benchmark_results.csv          — holdout metrics for every model
  cross_validation_results.csv   — 5-fold CV mean ± std per model
  feature_importance_comparison.csv — feature importances / LR coefficients
"""

import os
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, precision_recall_curve, auc
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import make_pipeline

from src.preprocess import PROXY_FEATURES, load_and_clean_data

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
OUTPUT_DIR   = "outputs/benchmark"
N_FOLDS      = 5

# ── Feature labels for display ──────────────────────────────────────────────
FEATURE_LABELS = {
    "pl_bmasse":   "Planet Mass (M-Earth)",
    "pl_orbper":   "Orbital Period (days)",
    "pl_orbeccen": "Eccentricity",
    "st_teff":     "Stellar Temp (K)",
    "st_rad":      "Stellar Radius (R-Sun)",
    "st_mass":     "Stellar Mass (M-Sun)",
    "st_met":      "Stellar Metallicity (dex)",
    "sy_dist":     "System Distance (pc)",
}


def _build_pipelines():
    """
    Returns an ordered dict of (name → imblearn/sklearn Pipeline).
    All pipelines apply SMOTE inside the pipeline so CV folds see
    only the training portion.
    """
    pipelines = {}

    # 1. Proxy Random Forest  (mirrors existing model_proxy.pkl logic)
    pipelines["Proxy Random Forest"] = Pipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf",   RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
    ])

    # 2. Logistic Regression  (needs scaling; scale before SMOTE)
    pipelines["Logistic Regression"] = Pipeline([
        ("scaler", StandardScaler()),
        ("smote",  SMOTE(random_state=RANDOM_STATE)),
        ("clf",    LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])

    # 3. Decision Tree
    pipelines["Decision Tree"] = Pipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf",   DecisionTreeClassifier(random_state=RANDOM_STATE)),
    ])

    # 4. Random Forest  (no explicit scaler needed)
    pipelines["Random Forest"] = Pipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf",   RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
    ])

    # 5. XGBoost  (optional — skip gracefully if not installed)
    try:
        from xgboost import XGBClassifier
        pipelines["XGBoost"] = Pipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf",   XGBClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                use_label_encoder=False,
                verbosity=0,
            )),
        ])
        print("XGBoost detected and included in benchmark.")
    except ImportError:
        print("XGBoost not installed — skipping. Run `pip install xgboost` to include it.")

    return pipelines


def _pr_auc(y_true, y_score):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return auc(recall, precision)


def _extract_importance(pipeline, features):
    """
    Extracts feature importances from the classifier step.
    Returns a numpy array of length == len(features), or None.
    """
    clf = pipeline.named_steps.get("clf")
    if clf is None:
        return None
    if hasattr(clf, "feature_importances_"):
        return clf.feature_importances_
    if hasattr(clf, "coef_"):
        # Logistic Regression: use absolute coefficients
        return np.abs(clf.coef_[0])
    return None


def run_benchmark(
    data_path="data/raw/exoplanetdata.csv",
    output_dir=OUTPUT_DIR,
):
    """
    Main benchmark entry point.  Returns three DataFrames:
      (benchmark_df, cv_df, importance_df)
    and writes them as CSV files to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────────
    print("Loading and preprocessing exoplanet data…")
    df = load_and_clean_data(data_path)
    X  = df[PROXY_FEATURES]
    y  = df["is_habitable"]

    print(f"  Dataset: {len(y)} rows — {int(y.sum())} habitable / {int((y==0).sum())} non-habitable")

    # ── Shared 80/20 holdout split ───────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train: {len(y_train)} rows  |  Test: {len(y_test)} rows\n")

    # ── Build pipelines ──────────────────────────────────────────────────────
    pipelines = _build_pipelines()

    # ── Shared Stratified K-Fold ─────────────────────────────────────────────
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    holdout_rows   = []
    cv_rows        = []
    importance_rows = {}

    for model_name, pipeline in pipelines.items():
        print(f"  Training & evaluating: {model_name} …")

        # ── Holdout evaluation ───────────────────────────────────────────────
        pipeline.fit(X_train, y_train)
        y_pred  = pipeline.predict(X_test)
        y_score = pipeline.predict_proba(X_test)[:, 1]

        tn, fp, fn, tp = _confusion_parts(y_test.values, y_pred)

        holdout_rows.append({
            "Model":      model_name,
            "Accuracy":   accuracy_score(y_test, y_pred),
            "Precision":  precision_score(y_test, y_pred, zero_division=0),
            "Recall":     recall_score(y_test, y_pred, zero_division=0),
            "F1":         f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC":    roc_auc_score(y_test, y_score),
            "PR-AUC":     _pr_auc(y_test, y_score),
            "TP":         int(tp),
            "TN":         int(tn),
            "FP":         int(fp),
            "FN":         int(fn),
        })

        # ── Cross-validation ─────────────────────────────────────────────────
        cv_scoring = {
            "accuracy":  "accuracy",
            "precision": "precision",
            "recall":    "recall",
            "f1":        "f1",
        }
        # Re-train a fresh copy for CV (avoids data leakage from the holdout fit)
        cv_pipeline = _rebuild_pipeline(model_name, pipelines)
        cv_results  = cross_validate(
            cv_pipeline, X, y, cv=skf, scoring=cv_scoring,
            return_train_score=False, error_score="raise"
        )

        cv_rows.append({
            "Model":              model_name,
            "CV Accuracy Mean":   cv_results["test_accuracy"].mean(),
            "CV Accuracy Std":    cv_results["test_accuracy"].std(),
            "CV Precision Mean":  cv_results["test_precision"].mean(),
            "CV Precision Std":   cv_results["test_precision"].std(),
            "CV Recall Mean":     cv_results["test_recall"].mean(),
            "CV Recall Std":      cv_results["test_recall"].std(),
            "CV F1 Mean":         cv_results["test_f1"].mean(),
            "CV F1 Std":          cv_results["test_f1"].std(),
        })

        # ── Feature importance ───────────────────────────────────────────────
        imp = _extract_importance(pipeline, PROXY_FEATURES)
        if imp is not None:
            importance_rows[model_name] = imp

        print(f"    Accuracy={holdout_rows[-1]['Accuracy']:.4f}  "
              f"Recall={holdout_rows[-1]['Recall']:.4f}  "
              f"F1={holdout_rows[-1]['F1']:.4f}  "
              f"PR-AUC={holdout_rows[-1]['PR-AUC']:.4f}")

    # ── Assemble DataFrames ──────────────────────────────────────────────────
    benchmark_df = pd.DataFrame(holdout_rows)
    cv_df        = pd.DataFrame(cv_rows)

    # Feature importance: wide-format (Feature | ModelA | ModelB | …)
    imp_df = pd.DataFrame({
        "Feature":     PROXY_FEATURES,
        "Feature Label": [FEATURE_LABELS.get(f, f) for f in PROXY_FEATURES],
    })
    for model_name, imp in importance_rows.items():
        imp_df[model_name] = imp

    # ── Save to CSV ──────────────────────────────────────────────────────────
    bench_path = os.path.join(output_dir, "benchmark_results.csv")
    cv_path    = os.path.join(output_dir, "cross_validation_results.csv")
    imp_path   = os.path.join(output_dir, "feature_importance_comparison.csv")

    benchmark_df.to_csv(bench_path, index=False)
    cv_df.to_csv(cv_path,           index=False)
    imp_df.to_csv(imp_path,         index=False)

    print(f"\n  Benchmark results   -> {bench_path}")
    print(f"  Cross-validation    -> {cv_path}")
    print(f"  Feature importances -> {imp_path}")

    # ── Print leaderboard ────────────────────────────────────────────────────
    print("\n=========================== LEADERBOARD ===========================")
    display_cols = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    print(benchmark_df[display_cols].to_string(index=False, float_format="{:.4f}".format))
    print("==================================================================\n")

    # Best-model summary
    best = {
        "Best Accuracy": benchmark_df.loc[benchmark_df["Accuracy"].idxmax(), "Model"],
        "Best Recall":   benchmark_df.loc[benchmark_df["Recall"].idxmax(),   "Model"],
        "Best F1":       benchmark_df.loc[benchmark_df["F1"].idxmax(),       "Model"],
        "Best PR-AUC":   benchmark_df.loc[benchmark_df["PR-AUC"].idxmax(),   "Model"],
    }
    print("Champion models:")
    for metric, model in best.items():
        print(f"  {metric:20s} → {model}")

    return benchmark_df, cv_df, imp_df


def _confusion_parts(y_true, y_pred):
    """Returns (TN, FP, FN, TP) from binary arrays."""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Edge case: only one class in test split
        tn = fp = fn = tp = 0
    return tn, fp, fn, tp


def _rebuild_pipeline(model_name, original_pipelines):
    """
    Returns a fresh (unfitted) clone of the pipeline for cross-validation.
    We use sklearn clone to avoid sharing state.
    """
    from sklearn.base import clone
    return clone(original_pipelines[model_name])


# ── CLI entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_benchmark()
