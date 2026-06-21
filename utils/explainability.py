import numpy as np
import pandas as pd
import shap
import streamlit as st

@st.cache_resource
def get_shap_importance_df(_pipeline, X_pipeline_fully_encoded):
    """
    Computes SHAP values and aggregates them into a clean pandas DataFrame.
    Dynamically fetches the final estimator to avoid KeyErrors.
    """
    # FIX: Extract the very last step in the pipeline by index, ignoring step name strings
    model = _pipeline[-1]
    
    # Compute SHAP values using the optimized TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_pipeline_fully_encoded)
    
    # Calculate mean absolute SHAP values for global importance
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    
    # Build and sort the structured dataframe
    importance_df = pd.DataFrame({
        "Feature": X_pipeline_fully_encoded.columns,
        "SHAP Importance": mean_abs_shap
    })
    
    importance_df = importance_df.sort_values(by="SHAP Importance", ascending=False).reset_index(drop=True)
    
    return importance_df
