import numpy as np
import pandas as pd
import joblib
import streamlit as st
PATH_PIPE = "models\\xgboost_model.joblib"

# 1. Define your exact training pipeline feature output order
PIPELINE_OUTPUT_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]

def preprocess_data(X):
    """Preprocesses user input features before model inference."""
    X_clean = X.copy()
    
    if 'SeniorCitizen' in X_clean.columns:
        X_clean['SeniorCitizen'] = pd.to_numeric(X_clean['SeniorCitizen'], errors='raise')
        
    if 'gender' in X_clean.columns:
        X_clean['gender'] = X_clean['gender'].map({'Male': 1, 'Female': 0})
        
    return X_clean

def preprocess_target(df, target_col="Churn"):
    """
    Safely extracts and maps the target column from raw text ("Yes"/"No")
    to binary integers (1/0) for model evaluation.
    """
    if target_col not in df.columns:
        return None
        
    return df[target_col].map({'Yes': 1, 'No': 0}).reset_index(drop=True)

@st.cache_resource
def load_prediction_pipeline(pipeline_path=PATH_PIPE):
    """Loads and caches the trained machine learning pipeline."""
    return joblib.load(pipeline_path)

@st.cache_data
def generate_predictions(_pipeline, X_processed):
    """Generates and caches predictions to prevent redundant computations."""
    preds = _pipeline.predict(X_processed)
    # If your model outputs probabilities, you can also use:
    # probs = _pipeline.predict_proba(X_processed)[:, 1]
    return preds

def transform_full_features(pipeline_obj, X_raw):
    """
    Passes raw data through transformers, tracking and fixing column 
    names even if custom FunctionTransformers break the scikit-learn chain.
    """
    X_clean = preprocess_data(X_raw)
    
    if not hasattr(pipeline_obj, 'steps'):
        return X_clean

    transformers = pipeline_obj[:-1]
    X_transformed = transformers.transform(X_clean)
    
    feature_names = None
    
    # 1. Standard native extraction attempt
    if hasattr(transformers, "get_feature_names_out"):
        try:
            feature_names = transformers.get_feature_names_out()
        except Exception:
            pass
            
    # 2. 🚀 THE FIX FOR THE CULPRIT: 
    # If scikit-learn returns None or errors out because of FunctionTransformer, 
    # but the shape hasn't changed from our cleaned input layout, recover the original names!
    if (feature_names is None or "Unknown" in str(feature_names[0])) and X_transformed.shape[1] == X_clean.shape[1]:
        feature_names = X_clean.columns

    # 3. Clean up formatting and strip transformer prefixes
    if feature_names is not None and len(feature_names) == X_transformed.shape[1]:
        clean_names = [str(col).split("__")[-1] for col in feature_names]
        col_names = clean_names
    else:
        col_names = [f"Unknown_Feature_{i}" for i in range(X_transformed.shape[1])]
        
    return pd.DataFrame(X_transformed, columns=col_names, index=X_clean.index)



def extract_feature_importance(pipeline_obj, feature_names):
    """
    Safely extracts global feature importance matrices from the XGBoost step
    at the end of a scikit-learn Pipeline.
    """
    if not hasattr(pipeline_obj, "steps"):
        return None
        
    model = pipeline_obj.steps[-1][1] # Safely unpack step tuple
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        
        # Ensure names align perfectly with feature length
        if len(feature_names) == len(importances):
            return pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False)
            
    return None

def generate_prediction_probabilities(pipeline_obj, X_encoded):
    """
    Safely extracts raw probability values for decision-boundary tracking.
    Falls back gracefully to discrete steps if predict_proba is restricted.
    """
    if hasattr(pipeline_obj, "predict_proba"):
        try:
            return pipeline_obj.predict_proba(X_encoded)[:, 1]
        except Exception:
            pass
    if hasattr(pipeline_obj, "predict"):
        return pipeline_obj.predict(X_encoded).astype(float)
    return np.zeros(len(X_encoded))
