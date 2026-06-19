import streamlit as st
import pandas as pd
from pathlib import Path
from src.validator import validate_dataset
from src.preprocessor import preprocess_data

# Flags to change the page interface depending on differnent session states 

if "dataset_approved" not in st.session_state:
    st.session_state.dataset_approved = False
    st.session_state.processed_data = None
    st.session_state.data_status = None

# File paths for different assests
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
    
    # First condition to check if the dataset is not approved
    # If the dataset is not approved we want the user to be still on home page 
 
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




        # Two side-by-side columns for Download and Upload
        col1, col2 = st.columns(2)

        with col1:
            st.subheader(" Get Template")
            st.download_button(
                label="⬇ Download Sample CSV",
                data=sample_data,
                file_name="telco_dataset_sample.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col2:
            st.subheader(" Submit Data ")
            uploaded_file = st.file_uploader(
                label="⬆ Upload Your Sample CSV",
                type=["csv"],
                accept_multiple_files=False,
                label_visibility="collapsed"
            )

        # Check whether the file was actually uploaded or not

        if uploaded_file is not None:

            try:
                user_df = pd.read_csv(uploaded_file)

                # FIX: Force TotalCharges to float, turning any remaining empty spaces into NaN
                if 'TotalCharges' in user_df.columns:
                    user_df = user_df[user_df['TotalCharges'] != ' ']
                    user_df['TotalCharges'] = pd.to_numeric(user_df['TotalCharges'], errors='coerce')
                
                # Execute automated validation & structure detection
                # Validatior is responsible for passing the status and if all flags are set then only
                # can user continue or else the rejectede message will be displayed 

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

    # Else block for the first conditional
    # This means will run if condition becomes false - dataset_aprroved is Not None

    else:
        # The default title
        st.title("Telco Analytics Workspace")
        
        # A sidebar to upload your own data
        if st.sidebar.button(" Upload Different Dataset"):
            st.session_state.dataset_approved = False
            st.session_state.processed_data = None
            st.session_state.data_status = None
            st.rerun()

        # Fetch the data and statuses
        df = st.session_state.processed_data
        status = st.session_state.data_status
        
        st.sidebar.success(" Active Session: Validated Schema")
        st.sidebar.metric(label="Total Records Loaded", value=len(df))
        
        # Runs when the dataset has both X and y Columns and their flag was set 
        if status == "CN_AND_Y" or status == "X_AND_Y":
            st.sidebar.info("Detected Mode: **Training/Evaluation**")
            st.header("⚡ Evaluation Dashboard ($X$ and $y$)")
            
            # Explicit backend variables for model computations
            X = df.drop(columns=["Churn"])
            y = df["Churn"]
            
            # Split the screen into two columns: 4 parts for X, 1 part for y
            col_x, col_y = st.columns([4, 1])
            
            with col_x:
                st.markdown("### Features ($X$)")
                st.dataframe(X, use_container_width=True)
                
            with col_y:
                st.markdown("### Label ($y$)")
                st.dataframe(y, use_container_width=True)

        # Runs only if X column conditions are met    
        elif status == "X_ONLY":
            st.sidebar.info("Detected Mode: **Inference Pipeline**")
            st.header("🔮 Inference Pipeline ($X$ only)")
            
            X = df  
            
            st.dataframe(X)
if __name__ == '__main__':
    main()