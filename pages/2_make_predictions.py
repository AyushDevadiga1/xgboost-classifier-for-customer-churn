import streamlit as st

import numpy as np
import pandas as pd

import sys
from pathlib import Path

from src.validator import validate_dataset
from src.preprocessor import (
    preprocess_data, 
    preprocess_target, 
    load_prediction_pipeline, 
    generate_predictions,
    transform_full_features,
    extract_feature_importance,        # Added backend helper
    generate_prediction_probabilities  # Added backend helper
)

import plotly.express as px  # Recommended for dynamic graphs

from sklearn.metrics import classification_report, confusion_matrix

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

# 1. Re-create the exact function your pipeline is looking for
def to_binary_map(X_sliced):
    col_map = {
        'Yes': 1,
        'No': 0,
        'No internet service': 0,
        'No phone service': 0
    }
    return X_sliced.apply(lambda x: x.map(col_map).fillna(0))

# 2. Inject it into the main module before joblib executes
import __main__
__main__.to_binary_map = to_binary_map


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

         # 1. Load the pipeline safely using try-except
        try:
            pipeline = load_prediction_pipeline()
        except Exception as e:
            st.error(f"❌ Failed to load ML Pipeline: {e}")
            st.stop() # Halts app execution safely

        
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
        
        if status in ["CN_AND_Y", "X_AND_Y"]:
            st.sidebar.info("Detected Mode: **Training/Evaluation**")
            st.header("⚡ Evaluation Dashboard")
            
            # ---------------------------------------------------------
            # DATA LAYER SETUP
            # ---------------------------------------------------------
            X_raw = df.drop(columns=["Churn"])
            y = preprocess_target(df, target_col="Churn")
            
            X_encoded_manual = preprocess_data(X_raw)
            predictions = generate_predictions(pipeline, X_encoded_manual)
            
            # Advanced Diagnostics Fields
            probabilities = generate_prediction_probabilities(pipeline, X_encoded_manual)
            X_pipeline_fully_encoded = transform_full_features(pipeline, X_raw)
            
            # =========================================================
            # ROW 1: MODEL PERFORMANCE INDICATORS (3 TABS)
            # =========================================================
            st.markdown("### 📈 Model Performance Indicators")
            
            tab1, tab2, tab3 = st.tabs([
                "📋 Classification Report", 
                "🗺️ Confusion Matrix Heatmap", 
                "🎯 Prediction Confidence Distribution"
            ])
            
            with tab1:
                report_dict = classification_report(y, predictions, output_dict=True)
                rep_df = pd.DataFrame(report_dict).T
                st.dataframe(rep_df.style.format(precision=2), use_container_width=True)
                
            with tab2:
                matrix_data = confusion_matrix(y, predictions)
                fig_hm = px.imshow(
                    matrix_data, x=['Predicted No', 'Predicted Yes'], y=['Actual No', 'Actual Yes'],
                    text_auto=True, color_continuous_scale='Blues',
                    labels=dict(x="Model Predictions", y="Ground Truth Labels"), aspect="auto"
                )
                fig_hm.update_layout(coloraxis_showscale=False, margin=dict(l=40, r=40, t=30, b=40), height=450)
                st.plotly_chart(fig_hm, use_container_width=True, key="tab_heatmap")
                
            with tab3:
                # Confidence distribution analysis map
                prob_df = pd.DataFrame({
                    "Probability": probabilities,
                    "Actual Churn State": df["Churn"].to_numpy() # Keeps "Yes"/"No" strings cleanly for chart legends
                })
                fig_prob = px.histogram(
                    prob_df, x="Probability", color="Actual Churn State",
                    nbins=30, barmode="overlay", marginal="box",
                    color_discrete_sequence=["#1f77b4", "#ff7f0e"]
                )
                fig_prob.update_layout(margin=dict(l=40, r=40, t=40, b=40), height=450,
                                    xaxis_title="Predicted Churn Probability Score", yaxis_title="Customer Segment Base Count")
                st.plotly_chart(fig_prob, use_container_width=True, key="tab_confidence_dist")

            # =========================================================
            # ROW 2: EXPLAINABLE AI & ERROR ANALYSIS (3 TABS)
            # =========================================================
            st.write("")
            st.markdown("### 🔍Explainable AI & Risk Factor Analysis")
            
            tab4, tab5, tab6 = st.tabs([
                "🔑 Global Feature Importance",
                "📉 Churn Risk vs Tenure",
                "💲 Churn Risk vs Monthly Charges"
            ])
            
            with tab4:
                feat_df = extract_feature_importance(pipeline, X_pipeline_fully_encoded.columns)
                if feat_df is not None:
                    # Displays the top 10 core real feature names beautifully sorted
                    fig_feat = px.bar(
                        feat_df.head(10), 
                        x="Importance", 
                        y="Feature", 
                        orientation="h",
                        color="Importance", 
                        color_continuous_scale="Blues", 
                        text_auto=".2f"
                    )
                    fig_feat.update_layout(
                        height=450, 
                        yaxis={'categoryorder':'total ascending'}, # Sorts highest importance to the top
                        coloraxis_showscale=False, 
                        margin=dict(l=120, r=40, t=30, b=40) # Extra left margin so long names aren't cut off
                    )
                    st.plotly_chart(fig_feat, use_container_width=True, key="tab_feat_importance")
                else:
                    st.info("Feature importance data is currently loading or unavailable.")

                    
            with tab5:
                # Grouped line graph mapping actual trends against system estimations across Account Age
                trend_tn = pd.DataFrame({"Tenure": X_raw["tenure"], "Actual": y, "Predicted": predictions})
                grouped_tn = trend_tn.groupby("Tenure").mean().reset_index()
                
                fig_tn = px.line(grouped_tn, x="Tenure", y=["Actual", "Predicted"],
                                labels={"value": "Segment Churn Density", "variable": "Data Stream"},
                                color_discrete_sequence=["#1f77b4", "#ff7f0e"])
                fig_tn.update_layout(height=450, yaxis_tickformat=".0%", margin=dict(l=40, r=40, t=40, b=40))
                st.plotly_chart(fig_tn, use_container_width=True, key="tab_tenure_line")
                
            with tab6:
                # Bin continuous charges to check operational performance gaps safely
                trend_mc = pd.DataFrame({"MonthlyCharges": X_raw["MonthlyCharges"], "Actual": y, "Predicted": predictions})
                trend_mc["Charges Bin"] = pd.qcut(trend_mc["MonthlyCharges"], q=10, duplicates="drop").astype(str)
                grouped_mc = trend_mc.groupby("Charges Bin")[["Actual", "Predicted"]].mean().reset_index()
                
                fig_mc = px.bar(grouped_mc, x="Charges Bin", y=["Actual", "Predicted"], barmode="group",
                                labels={"value": "Segment Churn Density", "variable": "Data Stream"},
                                color_discrete_sequence=["#1f77b4", "#ff7f0e"])
                fig_mc.update_layout(height=450, yaxis_tickformat=".0%", margin=dict(l=40, r=40, t=40, b=40))
                st.plotly_chart(fig_mc, use_container_width=True, key="tab_charges_bar")


            # ==========================================
            # REVISED SECTION 2: ISOLATED COMPARISON & DOWNLOAD
            # ==========================================
            st.write("---")
            d_col1, d_col2 = st.columns([2, 1])
            
            with d_col1:
                st.markdown("#### 🔍 Actual vs Predicted Targets (Ground Truth Only)")
                
                # Create a tiny 2-column comparison slice containing ONLY target trends
                comparison_view = pd.DataFrame({
                    "Actual Churn (y)": y,
                    "Predicted Churn (ŷ)": predictions
                }, index=df.index)
                
                st.dataframe(comparison_view, use_container_width=True)
                
            with d_col2:
                st.markdown("#### 📥 Export Metrics")
                st.markdown("Download the unified dataset containing original parameters matched with model inference tracking values.")
                
                # Compile full sheet for download tracking backgrounds
                results_df = X_raw.copy()
                results_df["Actual Churn Flag"] = y
                results_df["Predicted Churn Flag"] = predictions
                csv_output = results_df.to_csv(index=False).encode('utf-8')
                
                st.write("") # Clean vertical spacing block
                st.download_button(
                    label="📥 Download Completed Evaluation CSV",
                    data=csv_output,
                    file_name="telco_evaluation_compiled.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
            # ==========================================
            # REVISED SECTION 3: TRUE BEFORE & AFTER STATE CHANGES
            # ==========================================
            st.write("---")
            st.markdown("### ⛓️ Pipeline Transform Architecture (State Changes)")
            
            state_col1, state_col2 = st.columns(2)
            
            with state_col1:
                st.markdown("#### 🟥 Input Dataset Space")
                st.dataframe(X_raw.head(8), use_container_width=True)
                st.caption("Raw features before mappings are deployed.")
                
            with state_col2:
                st.markdown("#### 🟩 Encoded Pipeline Space")
                st.dataframe(X_pipeline_fully_encoded.head(8), use_container_width=True)
                st.caption("Clean numeric vectors extracted directly from your scikit-learn transformers.")

            # --- MODE 2: INFERENCE PIPELINE ---
        elif status == "X_ONLY":

            st.sidebar.info("Detected Mode: **Inference Pipeline**")
            st.header("🔮 Inference Pipeline ($X$ only)")
            
            # Preprocess and Predict
            X_encoded = preprocess_data(df)
            predictions = generate_predictions(pipeline, X_encoded)
            
            # Create a downloadable dataframe
            inference_results = df.copy()
            inference_results["Predicted Churn"] = predictions
            
            st.markdown("### 📋 Generated Predictions")
            st.dataframe(inference_results, use_container_width=True)
            
            # Download Button
            csv_output = inference_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Predictions CSV",
                data=csv_output,
                file_name="telco_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )
if __name__ == '__main__':
    main()