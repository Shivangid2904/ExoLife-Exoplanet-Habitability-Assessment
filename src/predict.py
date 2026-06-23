import joblib
import pandas as pd
import os

_models = {"baseline": None, "proxy": None}

def load_model(model_mode="baseline"):
    """
    Lazy-loads and caches the model pipeline to prevent duplicate memory allocation.
    Supports either model_mode ('baseline' or 'proxy') or a direct path to a pickle file.
    """
    global _models
    
    # Check if a direct file path was passed for backward compatibility
    if isinstance(model_mode, str) and (model_mode.endswith(".pkl") or "/" in model_mode or "\\" in model_mode):
        path = model_mode
        if not os.path.exists(path):
            alternative_path = os.path.join("..", path)
            if os.path.exists(alternative_path):
                path = alternative_path
            else:
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                fallback_path = os.path.join(base_dir, path)
                if os.path.exists(fallback_path):
                    path = fallback_path
                else:
                    raise FileNotFoundError(f"Custom model file not found at: {model_mode}")
        return joblib.load(path)

    # Standard model mode handling
    if model_mode not in _models:
        raise ValueError(f"Invalid model_mode: {model_mode}. Must be 'baseline' or 'proxy'.")

    if _models[model_mode] is None:
        filename = f"model_{model_mode}.pkl"
        
        # Check standard path
        model_path = os.path.join("models", filename)
        
        # Resolve paths for both project root and nested execution
        if not os.path.exists(model_path):
            alternative_path = os.path.join("..", model_path)
            if os.path.exists(alternative_path):
                model_path = alternative_path
            else:
                # Try absolute path fallback relative to this file
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                fallback_path = os.path.join(base_dir, "models", filename)
                if os.path.exists(fallback_path):
                    model_path = fallback_path
                else:
                    # Backward compatibility fallback for baseline (try original model.pkl)
                    if model_mode == "baseline":
                        fallback_original = os.path.join(base_dir, "models", "model.pkl")
                        if os.path.exists(fallback_original):
                            model_path = fallback_original
                        else:
                            raise FileNotFoundError(f"Trained baseline model not found. Checked: {model_path}, {alternative_path}, {fallback_path}, {fallback_original}")
                    else:
                        raise FileNotFoundError(f"Trained proxy model not found. Checked: {model_path}, {alternative_path}, {fallback_path}")
        
        _models[model_mode] = joblib.load(model_path)
    
    return _models[model_mode]

def predict(input_data, model_mode="baseline"):
    """
    Runs model inference on input features.
    input_data: dictionary or DataFrame of features.
    model_mode: 'baseline' or 'proxy'.
    """
    from src.preprocess import BASELINE_FEATURES, PROXY_FEATURES

    model = load_model(model_mode)
    
    # Determine correct feature set
    if model_mode == "proxy":
        features = PROXY_FEATURES
    else:
        features = BASELINE_FEATURES

    # Convert input to DataFrame
    if not isinstance(input_data, pd.DataFrame):
        input_df = pd.DataFrame([input_data])
    else:
        input_df = input_data
    
    # Check for missing features
    missing_features = [f for f in features if f not in input_df.columns]
    if missing_features:
        raise ValueError(f"Input data is missing features required for '{model_mode}' mode: {missing_features}")
        
    input_sliced = input_df[features]
    
    # Run prediction through the pipeline (which handles SMOTE internally)
    prediction = model.predict(input_sliced)[0]
    return prediction

