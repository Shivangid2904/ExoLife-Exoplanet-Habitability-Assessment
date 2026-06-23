import joblib
import pandas as pd
import os

_model = None

def load_model(model_path="models/model.pkl"):
    """
    Lazy-loads and caches the model pipeline to prevent duplicate memory allocation.
    """
    global _model
    if _model is None:
        # Resolve paths for both project root and nested execution
        if not os.path.exists(model_path):
            alternative_path = os.path.join("..", model_path)
            if os.path.exists(alternative_path):
                model_path = alternative_path
            else:
                # If neither works, try absolute path fallback relative to this file
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                fallback_path = os.path.join(base_dir, "models", "model.pkl")
                if os.path.exists(fallback_path):
                    model_path = fallback_path
                else:
                    raise FileNotFoundError(f"Trained model not found. Checked: {model_path}, {alternative_path}, {fallback_path}")
        
        _model = joblib.load(model_path)
    return _model

def predict(input_data):
    """
    Runs model inference on input features.
    input_data: dictionary or DataFrame of features matching model requirements.
    """
    model = load_model()
    if not isinstance(input_data, pd.DataFrame):
        input_df = pd.DataFrame([input_data])
    else:
        input_df = input_data
    
    # Run prediction through the pipeline (which handles SMOTE internally)
    prediction = model.predict(input_df)[0]
    return prediction
