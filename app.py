import streamlit as st

# File paths 
PATH_INTRO = "frontend\\main-page\\introduction-page.md"

def main():
    st.set_page_config(
                        page_title=" Customer Churn Prediction via XGBOOST",
                        page_icon="🏢", 
                        layout="wide", 
                        initial_sidebar_state="expanded",
                        menu_items={
                                        'Get Help': None,
                                        'Report a bug': None,
                                        'About': None
                                    }
                    )
    try :
        with open(PATH_INTRO , 'r' , encoding = 'utf-8') as file:
            intro_markdown = file.read() 

        st.markdown(intro_markdown)

    except Exception as e : 
        print(f'The introduction-page was not found .\n Cause : {e}')



    st.sidebar.markdown("## Quick Guide ")

    st.sidebar.markdown(
        "🚦 **Where to navigate:**\n\n"
        "• :green[**app**] — Introduction to churning and XGBoost Classifier.\n\n"
        "• :orange[**model info**] — Model training and its performance evaluation."
    )

    st.sidebar.markdown("---") # Clean divider line

if __name__ == '__main__':
    main()