import numpy as np
import pandas as pd

from src.preprocessor import preprocess_data

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


