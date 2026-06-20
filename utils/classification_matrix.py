import numpy as np
import pandas as pd

# Helper function to generate structural metric statistics manually without sklearn dependency
def compute_metrics(y_true, y_pred):
    # Ensure standard numpy array operations
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Calculate confusion matrix quadrants
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    
    # Class 0 calculations
    p0 = tn / (tn + fn) if (tn + fn) > 0 else 0
    r0 = tn / (tn + fp) if (tn + fp) > 0 else 0
    f0 = 2 * p0 * r0 / (p0 + r0) if (p0 + r0) > 0 else 0
    
    # Class 1 calculations
    p1 = tp / (tp + fp) if (tp + fp) > 0 else 0
    r1 = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p1 * r1 / (p1 + r1) if (p1 + r1) > 0 else 0
    
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
    
    report_dict = {
        "0 (No Churn)": {"Precision": p0, "Recall": r0, "F1-Score": f0, "Count": int(tn + fp)},
        "1 (Churn)": {"Precision": p1, "Recall": r1, "F1-Score": f1, "Count": int(tp + fn)}
    }
    
    return report_dict, [[tn, fp], [fn, tp]], accuracy