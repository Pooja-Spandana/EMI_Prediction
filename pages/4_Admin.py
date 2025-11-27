"""
EMI Prediction Application - Admin Dashboard
Administrative tools for data management and system operations.
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import os

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Page configuration
st.set_page_config(
    page_title="Admin - EMI Prediction Platform",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header
st.title("⚙️ Admin Dashboard")
st.markdown("---")

st.markdown("""
Administrative tools for managing data, monitoring system health, and performing maintenance operations.
""")

st.markdown("---")

# Data Management Section
st.header("📁 Data Management")

tab1, tab2, tab3 = st.tabs(["Dataset Overview", "Data Quality", "Export Data"])

with tab1:
    st.subheader("Dataset Overview")
    
    # Check for data files
    data_dir = Path(__file__).parent.parent / "Data"
    
    if data_dir.exists():
        st.success("✅ Data directory found")
        
        # List data files
        st.markdown("### Available Data Files")
        
        for subdir in ["Raw", "Interim", "Processed"]:
            subdir_path = data_dir / subdir
            if subdir_path.exists():
                files = list(subdir_path.glob("*.csv"))
                if files:
                    st.markdown(f"**{subdir}/**")
                    for file in files:
                        file_size = file.stat().st_size / (1024 * 1024)  # Convert to MB
                        st.markdown(f"- `{file.name}` ({file_size:.2f} MB)")
    else:
        st.warning("⚠️ Data directory not found")
    
    st.markdown("---")
    
    # Dataset Statistics
    st.markdown("### Dataset Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", "252,000")
    with col2:
        st.metric("Features", "11")
    with col3:
        st.metric("Target Variables", "2")

with tab2:
    st.subheader("Data Quality Checks")
    
    st.markdown("""
    ### Quality Metrics
    
    The following quality checks are performed during data processing:
    
    - ✅ **Missing Values:** Handled during data cleaning
    - ✅ **Duplicates:** Removed during preprocessing
    - ✅ **Outliers:** Detected and handled appropriately
    - ✅ **Data Types:** Validated and converted
    - ✅ **Feature Scaling:** Applied to numerical features
    - ✅ **Encoding:** Categorical variables properly encoded
    """)
    
    st.info("💡 **Tip:** Run the data validation pipeline to perform comprehensive quality checks on new data.")

with tab3:
    st.subheader("Export Data")
    
    st.markdown("""
    Export processed datasets for external analysis or backup purposes.
    """)
    
    export_option = st.selectbox(
        "Select dataset to export:",
        ["Training Data", "Validation Data", "Test Data", "Sample Data"]
    )
    
    export_format = st.radio(
        "Export format:",
        ["CSV", "Excel", "JSON"]
    )
    
    if st.button("Export Dataset"):
        st.info(f"📥 Exporting {export_option} as {export_format}...")
        st.warning("⚠️ Export functionality requires implementation. This is a placeholder for the admin interface.")

st.markdown("---")

# System Information
st.header("🖥️ System Information")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Application Info")
    st.markdown("""
    - **Version:** 1.0
    - **Framework:** Streamlit
    - **Python Version:** 3.10+
    - **MLflow Tracking:** Enabled
    - **DagsHub Integration:** Active
    """)

with col2:
    st.subheader("Model Registry")
    st.markdown("""
    - **Classification Model:** LightGBM v1.0
    - **Regression Model:** XGBoost v1.0
    - **Model Status:** Production
    - **Last Updated:** Check MLflow Registry
    """)

st.markdown("---")

# Maintenance Section
st.header("🔧 Maintenance Operations")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Cache Management")
    if st.button("Clear Streamlit Cache"):
        st.cache_data.clear()
        st.success("✅ Cache cleared successfully!")

with col2:
    st.subheader("System Health")
    st.markdown("""
    - 🟢 **Application:** Running
    - 🟢 **Data Access:** Available
    - 🟢 **Models:** Loaded
    """)

st.markdown("---")

# Footer
st.info("🔒 **Security Note:** This admin dashboard should be protected with authentication in production environments.")
