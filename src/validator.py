import pandas as pd
import numpy as np

def validate_dataset(uploaded_df: pd.DataFrame) -> tuple[bool, str, str]:
    """
    Validates the dataset layout, data types, checks for null/empty values, 
    and automatically detects if it contains X or X and y.
    
    Returns:
        is_valid (bool)
        status (str): "X_ONLY", "X_AND_Y", or "INVALID"
        message (str): Detailed success or error message
    """
    
    expected_features = {
        'customerID': 'str',
        'gender': 'str',
        'SeniorCitizen': 'int64',
        'Partner': 'str',
        'Dependents': 'str',
        'tenure': 'int64',
        'PhoneService': 'str',
        'MultipleLines': 'str',
        'InternetService': 'str',
        'OnlineSecurity': 'str',
        'OnlineBackup': 'str',
        'DeviceProtection': 'str',
        'TechSupport': 'str',
        'StreamingTV': 'str',
        'StreamingMovies': 'str',
        'Contract': 'str',
        'PaperlessBilling': 'str',
        'PaymentMethod': 'str',
        'MonthlyCharges': 'float64',
        'TotalCharges': 'float64'   
    }
    
    target_col = "Churn"
    expected_target_dtype = "str"
    
    uploaded_cols = list(uploaded_df.columns)
    mandatory_cols = list(expected_features.keys())
    
    # ----------------------------------------------------
    # STEP 1: Verify Columns & Ordering
    # ----------------------------------------------------
    if len(uploaded_cols) < len(mandatory_cols):
        missing = set(mandatory_cols) - set(uploaded_cols)
        return False, "INVALID", f"Dataset contains too few columns. Missing columns: {list(missing)}"

    uploaded_base_cols = uploaded_cols[:len(mandatory_cols)]
    
    if uploaded_base_cols != mandatory_cols:
        missing = set(mandatory_cols) - set(uploaded_cols)
        msg = "Missing or misaligned feature columns. "
        if missing:
            msg += f"Missing mandatory columns: {list(missing)}. "
        msg += f"The first {len(mandatory_cols)} columns must exactly match the template features layout."
        return False, "INVALID", msg

    # ----------------------------------------------------
    # STEP 2: Check & Coerce Data Types Flexibly
    # ----------------------------------------------------
    for col, expected_dtype in expected_features.items():
        actual_dtype = str(uploaded_df[col].dtype)
        
        # Standardize Text Types
        if expected_dtype in ["str", "object", "string"] and actual_dtype in ["object", "string" , "str"]:
            continue
            
        # Standardize Integer Options (Handles int32 vs int64 platform differences)
        if expected_dtype == "int64" and ("int" in actual_dtype):
            continue
            
        # Handle TotalCharges object parsing due to whitespace string presence
        if col == 'TotalCharges' and actual_dtype == 'object':
            try:
                # Dynamically try converting it to numeric to assist validation
                coerced_series = pd.to_numeric(uploaded_df[col].replace(r'^\s*$', np.nan, regex=True))
                actual_dtype = str(coerced_series.dtype)
            except Exception:
                return False, "INVALID", f"Column '{col}' contains unparseable non-numeric values."

        # Standardize Float Options
        if expected_dtype == "float64" and ("float" in actual_dtype):
            continue
            
        if expected_dtype != actual_dtype:
            return False, "INVALID", f"Type mismatch in column '{col}': Expected '{expected_dtype}', found '{actual_dtype}'."

    # ----------------------------------------------------
    # STEP 3: Check for Null Values & Empty Spaces
    # ----------------------------------------------------
    for col in mandatory_cols:
        # Check for standard NaNs
        null_count = uploaded_df[col].isnull().sum()
        
        # Check for hidden empty strings if parsed as string/object
        actual_dtype = str(uploaded_df[col].dtype)
        if actual_dtype in ["object", "string", "str"]:
            empty_space_count = (uploaded_df[col].astype(str).str.strip() == '').sum()
            null_count += empty_space_count

        # TotalCharges explicit rule for tenure == 0
        if col == 'TotalCharges':
            # Isolate blank spaces or NaNs
            is_null_or_blank = uploaded_df[col].isnull() | (uploaded_df[col].astype(str).str.strip() == '')
            invalid_blanks = uploaded_df[is_null_or_blank & (uploaded_df['tenure'] != 0)]
            
            if not invalid_blanks.empty:
                return False, "INVALID", f"Column '{col}' contains {len(invalid_blanks)} missing values where customer tenure is greater than 0."
            continue # Clean pass: values where tenure == 0 are allowed blanks

        if null_count > 0:
            return False, "INVALID", f"Column '{col}' contains {null_count} missing (NaN or empty) values. Please clean your dataset."

    # ----------------------------------------------------
    # STEP 4: Automated Target (y) Detection
    # ----------------------------------------------------
    if len(uploaded_cols) == len(mandatory_cols):
        return True, "X_ONLY", "Validated successfully! Dataset contains only features (X) for inference."
        
    # FIX: Corrected Python list lookup logic (replaced dictionary syntax with -1 index matching)
    elif len(uploaded_cols) == len(mandatory_cols) + 1 and uploaded_cols[-1] == target_col:
        actual_target_dtype = str(uploaded_df[target_col].dtype)
        
        target_nulls = uploaded_df[target_col].isnull().sum()
        if actual_target_dtype in ["object", "string", "str"]:
            target_nulls += (uploaded_df[target_col].astype(str).str.strip() == '').sum()
            
        if target_nulls > 0:
            return False, "INVALID", f"Target column '{target_col}' contains {target_nulls} missing values."
            
        valid_text_types = ["object", "string", "str"]
        if expected_target_dtype in valid_text_types and actual_target_dtype in valid_text_types:
            return True, "X_AND_Y", "Validated successfully! Dataset contains features and labels (X and y) for evaluation."
            
        return False, "INVALID", f"Type mismatch in target column '{target_col}': Expected '{expected_target_dtype}', found '{actual_target_dtype}'."

    else:
        extra_cols = uploaded_cols[len(mandatory_cols):]
        return False, "INVALID", f"Unexpected columns found: {extra_cols}. Dataset must contain either just X features, or X features plus exactly one target column named '{target_col}'."
