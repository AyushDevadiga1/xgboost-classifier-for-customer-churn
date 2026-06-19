import streamlit as st
import pandas as pd
from pathlib import Path
from src.validator import validate_dataset

# Maintain persistence across workspace view swaps
if "dataset_approved" not in st.session_state:
    st.session_state.dataset_approved = False
    st.session_state.processed_data = None
    st.session_state.data_status = None

# File paths 
PATH_INTRO = "frontend\\page-2\\introduction.md"
PATH_SAMPLE = "frontend\\page-2\\data-sample.csv"
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
    
    if not st.session_state.dataset_approved:

        try :
            with open(PATH_INTRO , 'r' , encoding = 'utf-8') as file:
                intro_markdown = file.read() 

            st.markdown(intro_markdown)

        except Exception as e : 
            print(f'The instruction-page was not found .\n Cause : {e}')

        try :
            with open(PATH_SAMPLE , 'r' , encoding = 'utf-8') as file:
                sample_data = file.read() 

        except Exception as e : 
            print(f'The sample path was not found .\n Cause : {e}')




        # 3. Create two side-by-side columns for Download and Upload
        col1, col2 = st.columns(2)

        with col1:
            st.subheader(" Get Template")
            st.download_button(
                label="📥 Download Sample CSV",
                data=sample_data,
                file_name="telco_dataset_sample.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col2:
            st.subheader(" Submit Data ")
            uploaded_file = st.file_uploader(
                label="Upload your completed Telco dataset",
                type=["csv"],
                accept_multiple_files=False,
                label_visibility="collapsed"
            )


        if uploaded_file is not None:

            try:
                user_df = pd.read_csv(uploaded_file)

                # FIX: Force TotalCharges to float, turning any remaining empty spaces into NaN
                if 'TotalCharges' in user_df.columns:
                    user_df = user_df[user_df['TotalCharges'] != ' ']
                    user_df['TotalCharges'] = pd.to_numeric(user_df['TotalCharges'], errors='coerce')
                
                # Execute automated validation & structure detection
                is_valid, status, message = validate_dataset(user_df)
                
                if is_valid:
                    # Store structural state variables globally
                    st.session_state.dataset_approved = True
                    st.session_state.processed_data = user_df
                    st.session_state.data_status = status
                    
                    # Instantly refresh to switch to Layout 2
                    st.rerun()
                else:
                    st.warning(f"❌ Rejected: {message}")
                    
            except Exception as e:
                st.error(f"❌ Critical file reading error: {e}")

    else:
        st.title("📊 Telco Analytics Workspace")
        
        if st.sidebar.button("🔄 Upload Different Dataset"):
            st.session_state.dataset_approved = False
            st.session_state.processed_data = None
            st.session_state.data_status = None
            st.rerun()

        df = st.session_state.processed_data
        status = st.session_state.data_status
        
        st.sidebar.success("📋 Active Session: Validated Schema")
        st.sidebar.metric(label="Total Records Loaded", value=len(df))
        
        if status == "CN_AND_Y" or status == "X_AND_Y":
            st.sidebar.info("Detected Mode: **Training/Evaluation**")
            st.header("⚡ Evaluation Dashboard ($X$ and $y$)")
            
            # Explicit backend variables for model computations
            X = df.drop(columns=["churn"])
            y = df["churn"]
            
            st.dataframe(df)
            
        elif status == "X_ONLY":
            st.sidebar.info("Detected Mode: **Inference Pipeline**")
            st.header("🔮 Inference Pipeline ($X$ only)")
            
            X = df
            
            st.dataframe(X)
if __name__ == '__main__':
    main()