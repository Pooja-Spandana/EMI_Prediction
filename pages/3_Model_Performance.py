"""
EMI Prediction Application - Model Performance Dashboard
View detailed model metrics, visualizations, and performance analysis.
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Page configuration
st.set_page_config(
    page_title="Model Performance - EMI Prediction Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for consistent styling
st.markdown("""
    <style>
    .performance-header {
        font-size: 3rem;
        font-weight: bold;
        color: #3399FF;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        color: #3399FF;
        text-align: center;
    }
    h1, h2, h3 {
        color: #3399FF !important;
        text-align: center !important;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="performance-header">📈 Model Performance Dashboard</div>', unsafe_allow_html=True)

# Introduction
st.markdown("""<p style="text-align: center;">
This dashboard provides comprehensive insights into the performance of our EMI prediction models.</p>
""")
st.markdown("""<p style="text-align: center;">Monitor key metrics, visualizations, and model behavior to ensure optimal performance.</p>
""")

st.markdown("---")

# Model Information
st.header("ℹ️ Model Information")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Model Dataset")
    st.markdown("""
    - **Total Samples:** ~4,00,000
    - **Training Set:** 70%
    - **Validation Set:** 15%
    - **Test Set:** 15%
    - **Features:** 42 input features (after going through preprocessor)
    - **Target Variables:** 2 (Emi_Eligibility, Max_Monthly_Emi)
    """)

with col2:
    st.subheader("Model Algorithms")
    st.markdown("""
    **Classification:**
    - Feature engineering and scaling
    - SMOTE for handling class imbalance
    - LightGBM with hyperparameter tuning with RandomSearchCV
    
    **Regression:**
    - Feature engineering and scaling
    - XGBoost with hyperparameter tuning with RandomSearchCV
    """)

st.markdown("---")

# Model Overview
st.header("🎯 Model Overview")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Classification Model (LightGBM)")
    st.markdown("""
    **Purpose:** Predict EMI eligibility (Not_Eligible, Eligible, High_Risk)
    
    **Key Metrics:**
    - Accuracy: 97.16%
    - Precision: 0.9712
    - Recall: 0.9703
    - F1-Score: 0.9707
    - ROC-AUC: 0.9945
    """)

with col2:
    st.subheader("Regression Model (XGBoost)")
    st.markdown("""
    **Purpose:** Predict maximum EMI amount per month
    
    **Key Metrics:**
    - R² Score: 0.9817
    - RMSE: 996.55
    - MAE: 612.89
    - MAPE: 8.23%
    """)

st.markdown("---")

# MLflow Integration
st.header("🔬 MLflow Experiment Tracking")

st.markdown("""
Our models are tracked using MLflow for comprehensive experiment management:

- **Experiment Tracking:** All training runs are logged with parameters, metrics, and artifacts
- **Model Registry:** Production models are registered and versioned
- **DagsHub Integration:** Remote tracking server for collaboration
- **Artifact Storage:** Models, plots, and data artifacts are stored for reproducibility

To access the MLflow UI, please refer to the project documentation for the tracking server URL.
""")

st.markdown("---")

# Footer
st.info("💡 **Tip:** For detailed model artifacts, confusion matrices, and plots, check the `Reports/` directory or access the MLflow tracking server.")
