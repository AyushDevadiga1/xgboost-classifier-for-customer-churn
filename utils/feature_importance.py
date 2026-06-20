import pandas as pd
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