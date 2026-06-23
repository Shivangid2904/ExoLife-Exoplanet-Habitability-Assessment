import pandas as pd
import joblib
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from src.preprocess import load_and_clean_data

def train_pipeline(data_path="data/raw/exoplanetdata.csv", model_dir="models"):
    """
    Trains the SMOTE + Random Forest pipeline on the full dataset.
    Performs stratified 5-fold cross-validation and saves the trained pipeline
    along with feature importances.
    """
    os.makedirs(model_dir, exist_ok=True)

    # 1. Load and preprocess data
    df = load_and_clean_data(data_path)

    # 2. Split features and target
    X = df[[ 
        "pl_rade", "pl_bmasse", "pl_orbper", "pl_eqt", "pl_insol",
        "pl_orbeccen", "st_teff", "st_rad", "st_mass", "st_met", "sy_dist"
    ]]
    y = df["is_habitable"]

    print("Original Class Distribution:\n", y.value_counts())

    # 3. Define the pipeline (SMOTE + Random Forest Classifier)
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # 4. Cross-Validation
    print("\n--- Running 5-Fold Cross-Validation ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores = cross_val_score(pipeline, X, y, cv=skf, scoring='f1')

    mean_f1 = np.mean(f1_scores)
    print("Cross-Validated F1 Scores (Positive Class):", f1_scores)
    print(f"Mean F1 Score: {mean_f1:.4f}")

    # 5. Fit on the entire dataset and save
    print("\n--- Fitting Final Model on Entire Dataset ---")
    pipeline.fit(X, y)
    model_path = os.path.join(model_dir, "model.pkl")
    joblib.dump(pipeline, model_path)
    print(f"Final model pipeline saved to {model_path}")

    # 6. Extract and Save Feature Importance
    rf_classifier = pipeline.named_steps['clf']
    importances = rf_classifier.feature_importances_
    features = X.columns
    
    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)
    
    importance_path = os.path.join(model_dir, "feature_importance.csv")
    importance_df.to_csv(importance_path, index=False)
    print(f"Feature importances saved to {importance_path}")

    return pipeline, mean_f1

if __name__ == "__main__":
    train_pipeline()
