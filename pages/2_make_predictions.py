import streamlit as st
import pandas as pd

# Core structural logic and data patches
from src.validator import validate_dataset
from src.preprocessor import load_prediction_pipeline

# Isolated view rendering engines
from frontend.page_2.views.dashboard_view import render_dashboard_page
from frontend.page_2.views.landing_view import render_landing_page


# Ensure foundational application session state keys are available globally
if "dataset_approved" not in st.session_state:
    st.session_state.dataset_approved = False
    st.session_state.processed_data = None
    st.session_state.data_status = None

def main():
    st.set_page_config(
        page_title=" Customer Churn Prediction via XGBOOST",
        page_icon="🔮", 
        layout="wide", 
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': None,
            'Report a bug': None,
            'About': None
        }
    )
    
    # -------------------------------------------------------------------------
    # ROUTE A: Dataset Not Uploaded/Approved -> Keep user on onboarding panel
    # -------------------------------------------------------------------------
    if not st.session_state.dataset_approved:
        render_landing_page()

    # -------------------------------------------------------------------------
    # ROUTE B: Dataset Processed Successfully -> Forward pipeline to dashboard
    # -------------------------------------------------------------------------
    else:
        # Load the model resource securely
        try:
            pipeline = load_prediction_pipeline()
        except Exception as e:
            st.error(f"Failed to load ML Pipeline: {e}")
            st.stop() # Halts app loop execution safely if model paths are corrupted
            
        # Pass the model down to let the decoupled layout handle tabs and inference
        render_dashboard_page(pipeline)

if __name__ == '__main__':
    main()
