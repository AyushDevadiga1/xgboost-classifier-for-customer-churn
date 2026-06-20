import numpy as np
import pandas as pd
import joblib
import streamlit as st

PATH_PIPE = "models\\xgboost_model.joblib"

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
