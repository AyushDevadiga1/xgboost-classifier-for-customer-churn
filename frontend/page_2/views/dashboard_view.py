import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
from sklearn.metrics import classification_report, confusion_matrix

# Shared preprocessing and parsing helper utilities
from src.preprocessor import preprocess_data, preprocess_target, generate_predictions ,load_prediction_pipeline
from utils.feature_importance import extract_feature_importance
from utils.feature_transformer import transform_full_features
from utils.predict_proba import generate_prediction_probabilities
from utils.explainability import get_shap_importance_df

def render_dashboard_page(pipeline):
    """
    Renders a fully interactive workspace that seamlessly switches depending on y-label presence.
    The function is flexible to automatically detect whether user specified the y-column or not.
    """
    df = st.session_state.processed_data
    status = st.session_state.data_status
    
    # Sidebar resetting infrastructure
    if st.sidebar.button("Upload Different Dataset"):
        st.session_state.dataset_approved = False
        st.session_state.processed_data = None
        st.session_state.data_status = None
        st.rerun()

    st.sidebar.success(" Active Session: Validated Schema")
    st.sidebar.metric(label="Total Records Loaded", value=len(df))

    with st.sidebar.expander("Data Processing Notice", expanded=False):
            st.markdown(
                "**Why did my row count decrease?**\n\n"
                "The pipeline automatically strips out records where `TotalCharges` contains "
                "blank spaces (commonly found in brand-new customers with 0 months of tenure) "
                "to prevent model matrix calculation crashes."
            )
    
    # --- AUTOMATED MODE DETECTION LAYER ---
    has_target = status in ["CN_AND_Y", "X_AND_Y"]
    
    if has_target:
        st.sidebar.info("Detected Mode: **Training/Evaluation**")
        st.header("Evaluation & Inference Workspace")
        X_raw = df.drop(columns=["Churn"])
        y = preprocess_target(df, target_col="Churn")
    else:
        st.sidebar.info("Detected Mode: **Bulk Inference Pipeline**")
        st.header("Automated Inference Pipeline")
        X_raw = df.copy()
        y = None

    # Consolidated multi-step vector operations
    X_encoded_manual = preprocess_data(X_raw)
    predictions = generate_predictions(pipeline, X_encoded_manual)
    probabilities = generate_prediction_probabilities(pipeline, X_encoded_manual)
    X_pipeline_fully_encoded = transform_full_features(pipeline, X_raw)

    # =========================================================
    # ROW 1: PERFORMANCE INDICATORS (DYNAMIC FALLBACK)
    # =========================================================
    if has_target:
        st.markdown("### Model Performance Indicators")
        tab1, tab2, tab3 = st.tabs([
            "Classification Report", 
            "Confusion Matrix Heatmap", 
            "Prediction Confidence Distribution"
        ])
        
        with tab1:
            report_dict = classification_report(y, predictions, output_dict=True)
            rep_df = pd.DataFrame(report_dict).T
            st.dataframe(rep_df.style.format(precision=2), width="stretch", height=212)
            
        with tab2:
            matrix_data = confusion_matrix(y, predictions)
            fig_hm = px.imshow(
                matrix_data, x=['Predicted No', 'Predicted Yes'], y=['Actual No', 'Actual Yes'],
                text_auto=True, color_continuous_scale='Blues', aspect="auto"
            )
            fig_hm.update_layout(coloraxis_showscale=False, margin=dict(l=40, r=40, t=30, b=40), height=450)
            st.plotly_chart(fig_hm, width="stretch", key="eval_heatmap")
            
        with tab3:
            prob_df = pd.DataFrame({"Probability": probabilities, "Actual Churn State": df["Churn"].to_numpy()})
            fig_prob = px.histogram(
                prob_df, x="Probability", color="Actual Churn State", nbins=30, 
                barmode="overlay", marginal="box", color_discrete_sequence=["#197bc1", "#f18322"]
            )
            fig_prob.update_layout(margin=dict(l=40, r=40, t=40, b=40), height=450, xaxis_title="Predicted Probability Score", yaxis_title="Customer Segment Count")
            st.plotly_chart(fig_prob, width="stretch", key="eval_confidence")
    else:
        st.markdown("### Pipeline Prediction Distribution")
        prob_df = pd.DataFrame({"Probability": probabilities})
        fig_prob = px.histogram(prob_df, x="Probability", nbins=30, color_discrete_sequence=["#2ca02c"])
        fig_prob.update_layout(height=350, xaxis_title="Predicted Churn Probability Score", yaxis_title="Total Inference Count")
        st.plotly_chart(fig_prob, width="stretch", key="inference_distribution")

    # =========================================================
    # ROW 2: EXPLAINABLE AI & RISK DRIVER ANALYSIS
    # =========================================================
    st.write("")
    st.markdown("### Explainable AI & Risk Factor Analysis")
    
    # Conditional tab unpacking strips out line charts if ground-truth targets do not exist
    if has_target:
        tab4, tab5, tab6 = st.tabs(["Global Feature Importance", "Churn Risk vs Tenure", "SHAP Values"])
    else:
        tab4, = st.tabs(["Global Feature Importance"])

    with tab4:
        feat_df = extract_feature_importance(pipeline, X_pipeline_fully_encoded.columns)
        if feat_df is not None:
            fig_feat = px.bar(
                feat_df.head(10), x="Importance", y="Feature", orientation="h",
                color="Importance", color_continuous_scale="blugrn", text_auto=".2f"
            )
            fig_feat.update_layout(height=450, yaxis={'categoryorder':'total ascending'}, coloraxis_showscale=False, margin=dict(l=150, r=40, t=30, b=40))
            st.plotly_chart(fig_feat, width="stretch", key="feature_importance_plot")
        else:
            st.info("Feature importance metrics are unavailable for this model pipeline configuration.")
            
    if has_target:

        with tab5:
            trend_tn = pd.DataFrame({"Tenure": X_raw["tenure"], "Actual": y, "Predicted": predictions}).groupby("Tenure").mean().reset_index()
            fig_tn = px.line(trend_tn, x="Tenure", y=["Actual", "Predicted"], color_discrete_sequence=["#1f77b4", "#ff7f0e"])
            fig_tn.update_layout(height=450, yaxis_tickformat=".0%", margin=dict(l=40, r=40, t=40, b=40))
            st.plotly_chart(fig_tn, width="stretch", key="tenure_trend")
            
        with tab6:
            st.subheader("Global Feature Importance (SHAP)")
            
            # 1. Fetch your cached machine learning pipeline
            pipeline = load_prediction_pipeline()
            
            # 2. Get the structured importance data cleanly using your renamed variable
            with st.spinner("Analyzing feature contributions..."):
                importance_df = get_shap_importance_df(pipeline, X_pipeline_fully_encoded)
            
            if importance_df is not None and not importance_df.empty:
                # 3. Generate a clean interactive Plotly horizontal bar chart
                # Used 'Viridis' color scale to match your original Seaborn palette choice
                fig_shap = px.bar(
                    importance_df.head(15), 
                    x="SHAP Importance", 
                    y="Feature", 
                    orientation="h",
                    color="SHAP Importance", 
                    color_continuous_scale="agsunset", 
                    text_auto=".3f",
                    title="Top 15 Features Driving Model Decisions"
                )
                
                # 4. Match layout dimensions and margins exactly with Tab 4 for strict visual consistency
                fig_shap.update_layout(
                    height=450, 
                    yaxis={'categoryorder': 'total ascending'}, 
                    coloraxis_showscale=False, 
                    margin=dict(l=150, r=40, t=50, b=40) # Slightly larger top margin (t=50) to accommodate the title
                )
                
                # 5. Render the Plotly chart inside your Streamlit container
                st.plotly_chart(fig_shap, width="stretch", key="shap_importance_plot")
            else:
                st.info("SHAP importance metrics are unavailable for this model pipeline configuration.")


    # =========================================================
    # ROW 3: TARGET TRACKING & FILE COMPILATION LOGS
    # =========================================================
    st.write("---")
    d_col1, d_col2 = st.columns(2)
    
    with d_col1:
        if has_target:
            st.markdown("#### Comparison of Ground Truth and Predictions")
            comparison_view = pd.DataFrame({"Actual Churn (y)": y, "Predicted Churn (ŷ)": predictions}, index=df.index)
            st.dataframe(comparison_view, width="stretch")
        else:
            st.markdown("#### Generated Output Head-Slice")
            comparison_view = pd.DataFrame({"Predicted Churn (ŷ)": predictions, "Churn Probability": probabilities}, index=df.index)
            st.dataframe(comparison_view, width="stretch")
            
    with d_col2:
        st.markdown("#### Export Workspace Outputs")
        
        # 2. Wrap the download workflow inside a clean, padded visual container card
        with st.container(border=True):
            st.markdown("##### **Export Pipeline Audit Logs**")
            st.markdown(
                f"Generates a complete dataset with all original feature parameters "
                f"matched row-by-row with system classifications."
            )
            
            # Add micro-metrics to make the download block look data-rich
            m1, m2 = st.columns(2)
            m1.metric("Total Records", f"{len(df)}")
            m2.metric("Flagged Churn", f"{int(predictions.sum())}")
            
            # Setup data payload
            results_df = X_raw.copy()
            if has_target:
                results_df["Actual Churn Flag"] = y
            results_df["Predicted Churn Flag"] = predictions
            results_df["Churn Probability Score"] = probabilities
            csv_data = results_df.to_csv(index=False).encode('utf-8')
            
            st.write("") # Clean structural separation line
            
            # 3. Deploy a high-utility primary download button
            st.download_button(
                label="Download Completed Results CSV", 
                data=csv_data, 
                file_name="telco_predictions_compiled.csv", 
                mime="text/csv", 
                width="stretch",
                type="primary" # Uses your theme's bold primary action color!
            )
        
    # =========================================================
    # ROW 4: DATASTATE STATE ARCHITECTURE
    # =========================================================
    st.write("---")
    st.markdown("### Pipeline Transformation on Features")
    state_col1, state_col2 = st.columns(2)
    with state_col1:
        st.markdown("#### $X$ : Raw Strings")
        st.dataframe(X_raw.head(8), width="stretch")
    with state_col2:
        st.markdown("#### $X$ : Numeric Encodings")
        st.dataframe(X_pipeline_fully_encoded.head(8), width="stretch")
