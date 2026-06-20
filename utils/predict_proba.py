import numpy as np
import pandas as pd

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