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

# Header
st.title("📈 Model Performance Dashboard")
st.markdown("---")

# Introduction
st.markdown("""
This dashboard provides comprehensive insights into the performance of our EMI prediction models.
Monitor key metrics, visualizations, and model behavior to ensure optimal performance.
""")

st.markdown("---")

# Model Overview
st.header("🎯 Model Overview")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Classification Model (LightGBM)")
    st.markdown("""
    **Purpose:** Predict EMI eligibility (Approved/Rejected)
    
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
    **Purpose:** Predict maximum EMI amount
    
    **Key Metrics:**
    - R² Score: 0.9817
    - RMSE: 996.55
    - MAE: 612.89
    - MAPE: 8.23%
    """)

st.markdown("---")

# Performance Metrics
st.header("📊 Detailed Performance Metrics")

tab1, tab2 = st.tabs(["Classification Model", "Regression Model"])

with tab1:
    st.subheader("Classification Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", "97.16%", delta="2.3%")
    with col2:
        st.metric("Precision", "0.9712", delta="0.015")
    with col3:
        st.metric("Recall", "0.9703", delta="0.012")
    with col4:
        st.metric("F1-Score", "0.9707", delta="0.014")
    
    st.info("📌 **Note:** To view confusion matrices, ROC curves, and feature importance plots, please check the MLflow tracking server or the Reports directory.")

with tab2:
    st.subheader("Regression Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("R² Score", "0.9817", delta="0.023")
    with col2:
        st.metric("RMSE", "996.55", delta="-45.2")
    with col3:
        st.metric("MAE", "612.89", delta="-28.5")
    with col4:
        st.metric("MAPE", "8.23%", delta="-1.2%")
    
    st.info("📌 **Note:** To view residual plots, prediction vs actual plots, and feature importance, please check the MLflow tracking server or the Reports directory.")

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

# Model Information
st.header("ℹ️ Model Information")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Training Dataset")
    st.markdown("""
    - **Total Samples:** 252,000
    - **Training Set:** 70%
    - **Validation Set:** 15%
    - **Test Set:** 15%
    - **Features:** 11 input features
    - **Target Variables:** 2 (Eligibility, EMI Amount)
    """)

with col2:
    st.subheader("Model Algorithms")
    st.markdown("""
    **Classification:**
    - LightGBM with hyperparameter tuning
    - SMOTE for handling class imbalance
    
    **Regression:**
    - XGBoost with hyperparameter tuning
    - Feature engineering and scaling
    """)

st.markdown("---")

# Footer
st.info("💡 **Tip:** For detailed model artifacts, confusion matrices, and plots, check the `Reports/` directory or access the MLflow tracking server.")
