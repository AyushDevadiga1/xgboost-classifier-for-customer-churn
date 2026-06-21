import streamlit as st
import pandas as pd
from src.validator import validate_dataset

# Keep paths relative to the project root directory
PATH_INTRO = "frontend\\page_2\\introduction.md"
PATH_SAMPLE = "frontend\\page_2\\data-sample.csv"

def render_landing_page():
    """Handles home page asset delivery, user uploads, and template verification."""
    try:
        with open(PATH_INTRO, 'r', encoding='utf-8') as file:
            intro_markdown = file.read() 
        st.markdown(intro_markdown)
    except Exception as e: 
        st.error(f'The instruction asset could not be loaded: {e}')

    try:
        with open(PATH_SAMPLE, 'r', encoding='utf-8') as file:
            sample_data = file.read() 
    except Exception as e: 
        st.error(f'The sample template data could not be loaded: {e}')
        sample_data = ""

    # Double columns layout for template handling
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Get Template")
        st.download_button(
            label="Download Sample CSV",
            data=sample_data,
            file_name="telco_dataset_sample.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col2:
        st.subheader(" Submit Data ")
        uploaded_file = st.file_uploader(
            label="Upload Workspace Data",
            type=["csv"],
            accept_multiple_files=False,
            label_visibility="collapsed"
        )

    if uploaded_file is not None:
        try:
            user_df = pd.read_csv(uploaded_file)

            # Sanitize explicit spaces inside total charges columns before verification runs
            if 'TotalCharges' in user_df.columns:
                user_df = user_df[user_df['TotalCharges'] != ' ']
                user_df['TotalCharges'] = pd.to_numeric(user_df['TotalCharges'], errors='coerce')
            
            is_valid, status, message = validate_dataset(user_df)
            
            if is_valid:
                # Update global state flags
                st.session_state.dataset_approved = True
                st.session_state.processed_data = user_df
                st.session_state.data_status = status
                st.rerun()
            else:
                st.warning(f"Rejected: {message}")
                
        except Exception as e:
            st.error(f"Critical file reading error: {e}")
