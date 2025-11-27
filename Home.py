"""
EMI Prediction Application - Home Page
A comprehensive financial risk assessment platform for EMI prediction.
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

# Page configuration
st.set_page_config(
    page_title="EMI Prediction Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #3399FF;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #B0B0B0;
        margin-bottom: 2rem;
    }
    .feature-card {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #262730;
        margin: 1rem 0;
    }
    .metric-container {
        text-align: center;
        padding: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🏦 EMI Prediction Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Intelligent Financial Risk Assessment with Machine Learning</div>', unsafe_allow_html=True)

# Introduction
st.markdown("""
<div style="text-align: center;">
This application leverages advanced machine learning models to provide accurate EMI eligibility predictions and maximum EMI amount calculations.</div>
""", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center;">Built with state-of-the-art algorithms and integrated with MLflow for model tracking and management.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Key Features
st.markdown("### 🎯 Key Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <h4>💰 Real-Time Predictions</h4>
        <p>Get instant EMI eligibility assessments and maximum EMI amount predictions using dual ML models:</p>
        <ul>
            <li><strong>Classification Model (LightGBM):</strong> 97.16% accuracy</li>
            <li><strong>Regression Model (XGBoost):</strong> R² score of 0.9817</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
with col2:
    st.markdown("""
    <div class="feature-card">
        <h4>📊 Interactive Data Explorer</h4>
        <p>Explore the training dataset with interactive visualizations:</p>
        <ul>
            <li>Distributions & statistics of numerical features</li>
            <li>Distributions & statistics of categorical features</li>
            <li>Correlation analysis</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <h4>📈 Model Performance Dashboard</h4>
        <p>Monitor model metrics and performance:</p>
        <ul>
            <li>Comprehensive evaluation metrics</li>
            <li>Confusion matrices and ROC curves</li>
            <li>Feature importance analysis</li>
            <li>MLflow experiment tracking integration</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Model Performance Summary
st.markdown("### 🏆 Model Performance Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Classification Accuracy",
        value="97.16%",
        delta="High Performance"
    )

with col2:
    st.metric(
        label="Classification F1-Score",
        value="0.9707",
        delta="Excellent"
    )

with col3:
    st.metric(
        label="Regression R² Score",
        value="0.9817",
        delta="Outstanding"
    )

with col4:
    st.metric(
        label="Regression RMSE",
        value="996.55",
        delta="Low Error"
    )

st.markdown("---")

# Technology Stack
st.markdown("### 🛠️ Technology Stack")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **Machine Learning:**
    - XGBoost (Regression)
    - LightGBM (Classification)
    - Scikit-learn
    - SMOTE (Imbalanced data handling)
    """)

with col2:
    st.markdown("""
    **MLOps & Tracking:**
    - MLflow Model Registry
    - DagsHub Integration
    - Model Versioning
    - Experiment Tracking
    """)

with col3:
    st.markdown("""
    **Web Application:**
    - Streamlit
    - Pandas & NumPy
    - Matplotlib & Seaborn
    - Python 3.10+
    """)

st.markdown("---")

# System Information
st.sidebar.markdown("### 🖥️ System Information")


st.sidebar.markdown("""
**Application Info**
- **Version:** 1.0
- **Framework:** Streamlit
- **Python Version:** 3.10+
- **MLflow Tracking:** Enabled
- **DagsHub Integration:** Active
""")

st.sidebar.markdown("""
**Model Registry**
- **Classification Model:** LightGBM v1.0
- **Regression Model:** XGBoost v1.0
- **Model Status:** Production
- **Last Updated:** Check MLflow Registry
""")

# Quick Start
st.markdown("### 🚀 Quick Start")

st.markdown("""
1. **Make a Prediction:** Navigate to the 💰 **Predict EMI** page to get instant eligibility and EMI amount predictions
2. **Explore Data:** Visit the 📊 **Data Explorer** to understand the dataset and feature distributions
3. **View Performance:** Check the 📈 **Model Performance** page for detailed metrics and visualizations

Use the sidebar navigation to explore different sections of the application.
""")

st.markdown("---")

# Footer
st.markdown("""
<div style="text-align: center; color: #B0B0B0; padding: 2rem 0;">
    <p>Built with ❤️ using Streamlit | Powered by MLflow & DagsHub</p>
    <p><em>EMI Prediction Platform v1.0</em></p>
</div>
""", unsafe_allow_html=True)
