import streamlit as st
import pandas as pd
from pathlib import Path

# File paths 
BASE_PATH = "frontend/page_1"
PATH_INTRO = "frontend/page_1/introduction.md"
PATH_CONF_ROC = "frontend/page_1/conf-matrix-and-roc-curve.png"
PATH_CORR_HEAT = "frontend/page_1/corr-matrix.png"
PATH_FEAT_IMPORTANCE = "frontend/page_1/xgbc-feature-importance.png"

def main():

    st.set_page_config(
                        page_title="Model Metadata",
                        page_icon="🏗", 
                        layout="wide", 
                        initial_sidebar_state="expanded",
                        menu_items={
                                        'Get Help': None,
                                        'Report a bug': None,
                                        'About': None
                                    }
                    )

    st.sidebar.markdown(
        "**Go Back To:**\n\n"
        "• :gray[**app**] — Introduction to churning and XGBoost Classifier.\n\n"
        "• :green[**model info**] — Information about how the model was trained and it's performance.\n\n"
        "• :orange[**make predictions**] — Make predictions with your custom dataset in this tab."
    )
    try :
        with open(PATH_INTRO , 'r' , encoding = 'utf-8') as file:
            intro_markdown = file.read() 

        st.markdown(intro_markdown)

    except Exception as e : 
        print(f'The introduction-page was not found .\n Cause : {e}')

    st.markdown("## Model Evaluation Dashboard")
    st.write(
        "This section outlines the evaluation metrics for the trained XGBoost pipeline, "
        "assessing its predictive confidence, feature associations, and underlying feature impact thresholds."
    )


    tab1, tab2, tab3 = st.tabs([" Classification Metrics ", " Data Correlations ", " Feature Significance "])

    with tab1:
        st.write("### Prediction Metrics & Threshold Performance")
        
        # Row 1: Large Performance Plots
        if Path(PATH_CONF_ROC).exists():
            try:
                st.image(
                    PATH_CONF_ROC, 
                    caption="Confusion Matrix & Receiver Operating Characteristic (ROC) Layout", 
                   width="stretch"
                )
            except Exception as e:
                st.error(f"Error loading Confusion Matrix plot: {e}")
        else:
            st.warning(f"File not found at target directory path: `{PATH_CONF_ROC}`")
        
        # Row 2: Graph Descriptions
        st.write("#### Visual Performance ")
        st.write(
            "The **Confusion Matrix** on the left displays the breakdown of correct and incorrect predictions made by the model. "
            "It highlights how many true churners were correctly identified versus how many were missed (false negatives). "
            "The **ROC-AUC Curve** on the right illustrates the model's diagnostic ability to separate the two classes across "
            "different threshold variations. A higher area under the curve (AUC) indicates superior performance at maximizing true positives "
            "while minimizing false alarms."
        )
        
        st.write("---")
        
        # Row 3: Classification Report Table
        st.write("#### Classification Report ")
        st.write(
            "The table below shows the core statistical metrics evaluated across both classes. "
            "For customer retention, **Recall (0.79)** on Class 1 (Churned) is highly prioritized because it demonstrates that the model "
            "successfully captures nearly 80% of all actual churning customers, giving the business an opportunity to step in with active retention offers."
        )
        
        # Exact classification metrics payload matching your training logs
        report_data = {
            "Metric Class": ["0 (Retained)", "1 (Churned)", "Accuracy Score", "Macro Average", "Weighted Average"],
            "Precision": [0.91, 0.51, "", 0.71, 0.80],
            "Recall": [0.73, 0.79, "", 0.76, 0.75],
            "F1-Score": [0.81, 0.62, 0.75, 0.72, 0.76],
            "Support": [1033, 374, 1407, 1407, 1407]
        }
        
        df_report = pd.DataFrame(report_data)
        st.dataframe(df_report, hide_index=True, use_container_width=True)
        st.caption("*Note: Class 1 (Churned) properties directly drive the focus of targeted marketing outreach.*")


    with tab2:
        st.write("### Feature Correlation Analysis")
        st.write(
            "The heat matrix illustrates the co-dependency between feature matrices. "
            "This was the status of the model when all the features were encoded , causing interdependence."
            "To reduce this the xgboost_pipeline uses several methods like Target encoding ,  OneHot Encoding with dropping" \
            " , dropping redundant columns and smart feature engineering ."
        )
        
        if Path(PATH_CORR_HEAT).exists():
            try:
                st.image(
                    PATH_CORR_HEAT, 
                    caption="Feature Correlation Heatmap Matrix", 
                   width="stretch"
                )
            except Exception as e:
                st.error(f"Error loading Correlation Matrix plot: {e}")
        else:
            st.warning(f"File not found at target directory path: `{PATH_CORR_HEAT}`")

    with tab3:
        st.write("### XGBoost Weight Gain Rankings")
        st.write(
            "The calculation maps the structural contribution value of features across internal "
            "decision splits, highlighting what dictates active customer departure."
        )
        
        if Path(PATH_FEAT_IMPORTANCE).exists():
            try:
                st.image(
                    PATH_FEAT_IMPORTANCE, 
                    caption="XGBoost Feature Significance Ranking Spectrum", 
                   width="stretch"
                )
            except Exception as e:
                st.error(f"Error loading Feature Importance plot: {e}")
        else:
            st.warning(f"File not found at target directory path: `{PATH_FEAT_IMPORTANCE}`")


if __name__ == '__main__':
    main()