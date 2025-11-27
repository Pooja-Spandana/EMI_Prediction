"""
Data Explorer Page - Interactive Dataset Exploration
"""

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.exception import CustomException

# Page configuration
st.set_page_config(
    page_title="Data Explorer",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .explorer-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF4B4B;
        margin-bottom: 1rem;
    }
    .stat-card {
        padding: 1rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="explorer-header">📊 Data Explorer</div>', unsafe_allow_html=True)
st.markdown("Explore the training dataset, visualize feature distributions, and analyze correlations.")

st.markdown("---")

# Load Data
@st.cache_data
def load_data():
    import os
    try:
        df = None
        source = None
        
        # Always load Sample Data from GitHub (Cloud Compatible)
        github_raw_url = "https://raw.githubusercontent.com/Pooja-Spandana/EMI_Prediction/main/Data/Raw/sample_data.csv"
        try:
            df = pd.read_csv(github_raw_url)
            source = "Sample Data (GitHub)"
        except Exception as github_error:
            # Fallback to local if GitHub fails (CDN caching or network issues)
            if Path("Data/Raw/sample_data.csv").exists():
                df = pd.read_csv("Data/Raw/sample_data.csv")
                source = "Sample Data (Local Fallback)"
            else:
                return None, f"Error loading from GitHub: {str(github_error)}"
            
        return df, source
    except Exception as e:
        return None, str(e)

with st.spinner("Loading dataset..."):
    df, source = load_data()

# Add Download Button and Instructions
st.sidebar.markdown("### 📥 Get Raw Data")
st.sidebar.link_button("Download Raw Dataset", "https://drive.google.com/file/d/1C7tcEdnRIlxwIsFnsN6F0jkpU1FRlieS/view")

st.sidebar.markdown("""
    1. **Download** the raw dataset from the link above.
    2. **Place** the file in `Data/Raw/emi_prediction_dataset.csv`.
    3. **Run Data Ingestion**: ```python src/components/data_ingestion.py```
    4. **Run Data Cleaning**: ```python src/components/data_cleaning.py```  
    5. **Run Feature Engineering**: ```python src/components/feature_engineering.py```
    """)
    
st.markdown("#### 📄 Raw Data Preview (from GitHub)")
try:
    github_raw_url = "https://raw.githubusercontent.com/Pooja-Spandana/EMI_Prediction/main/Data/Raw/sample_data.csv"
    df_raw_preview = pd.read_csv(github_raw_url)
    st.dataframe(df_raw_preview.head(10), width='stretch')
    st.caption(f"Showing 10 rows from {github_raw_url}")
except Exception as e:
    st.warning(f"Could not load raw data preview: {e}")

if df is None:
    st.error(f"❌ Error loading data: {source}")
    st.stop()

st.success(f"✅ Loaded {source} successfully! Shape: {df.shape}")

# Dataset Overview
st.markdown("### 📋 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Records", len(df))
with col2:
    st.metric("Total Features", len(df.columns))
with col3:
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    st.metric("Numerical columns", f"{len(num_cols)}")
with col4:
    cat_cols = df.select_dtypes(include=['O']).columns.tolist()
    st.metric("Categorical columns", f"{len(cat_cols)}")

st.markdown("---")

# Visualizations
st.markdown("### 📈 Feature Analysis")

tabs = st.tabs(["Univariate Analysis", "Bivariate Analysis", "Correlation"])

with tabs[0]:
    st.markdown("#### Distribution of Numerical Features")
    
    # Select numerical columns
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    selected_num_col = st.selectbox("Select Feature to Visualize", num_cols, index=0)
    
    if selected_num_col:
        fig = px.histogram(df, x=selected_num_col, nbins=50, title=f"Distribution of {selected_num_col}",
                          color_discrete_sequence=['#FF4B4B'])
        st.plotly_chart(fig, width='stretch')
        
        # Stats
        st.markdown(f"**Statistics for {selected_num_col}:**")
        desc = df[selected_num_col].describe()
        st.dataframe(desc.to_frame().T, width='stretch')

with tabs[1]:
    st.markdown("#### Relationship between Features")
    
    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("X-Axis", num_cols, index=0, key="bi_x")
    with col2:
        y_axis = st.selectbox("Y-Axis", num_cols, index=1, key="bi_y")
    
    color_col = st.selectbox("Color By (Categorical)", ["None"] + df.select_dtypes(include=['object']).columns.tolist())
    
    if x_axis and y_axis:
        if color_col != "None":
            fig = px.scatter(df.sample(min(1000, len(df))), x=x_axis, y=y_axis, color=color_col,
                            title=f"{x_axis} vs {y_axis} (Sampled)", opacity=0.7)
        else:
            fig = px.scatter(df.sample(min(1000, len(df))), x=x_axis, y=y_axis,
                            title=f"{x_axis} vs {y_axis} (Sampled)", opacity=0.7)
        st.plotly_chart(fig, width='stretch')

with tabs[2]:
    st.markdown("#### Correlation Matrix")
    
    if len(num_cols) > 1:
        corr = df[num_cols].corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r",
                       title="Feature Correlation Heatmap")
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("Not enough numerical columns for correlation analysis.")

# Footer
st.markdown("---")
st.caption("Data source: Local training artifacts")
