"""
EMI Prediction Page - Real-time EMI Eligibility and Amount Predictions
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.utils import load_models, make_predictions, format_currency, validate_user_input, perform_feature_engineering
from src.exception import CustomException

# ... (rest of imports and config) ...

# Page configuration
st.set_page_config(
    page_title="EMI Prediction",
    page_icon="💰",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .prediction-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF4B4B;
        margin-bottom: 1rem;
    }
    .result-card {
        padding: 2rem;
        border-radius: 15px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .eligible {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .not-eligible {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="prediction-header">💰 EMI Prediction</div>', unsafe_allow_html=True)
st.markdown("Get instant EMI eligibility assessment and maximum EMI amount prediction")

st.markdown("---")

# Load models with caching
@st.cache_resource
def get_models():
    return load_models()

try:
    with st.spinner("Loading models..."):
        regressor, classifier, preprocessor = get_models()
    
    if regressor is None or classifier is None or preprocessor is None:
        st.error("⚠️ Failed to load models. Please check the configuration and try again.")
        st.stop()
    
    st.success("✅ Models loaded successfully!")
except CustomException as ce:
    st.error(f"⚠️ Model Loading Error: {str(ce)}")
    st.info("💡 **Troubleshooting Tips:**\n- Check if model files exist in `artifacts/models/`\n- Verify MLflow credentials in `.env` file\n- Ensure models are registered in MLflow Model Registry")
    st.stop()
except Exception as e:
    st.error(f"⚠️ Unexpected error while loading models: {str(e)}")
    st.stop()

st.markdown("---")

# Input Form
st.markdown("### 📝 Enter Your Financial Information")

with st.form("prediction_form"):
    # 1. Personal Demographics
    st.markdown("#### 👤 Personal Demographics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        age = st.number_input("Age", min_value=25, max_value=60, value=30, step=1, help="Customer age (25-60 years)")
    with col2:
        gender = st.selectbox("Gender", ["M", "F"])
    with col3:
        marital_status = st.selectbox("Marital Status", ["Single", "Married"])
    with col4:
        education = st.selectbox("Education", ["High School", "Graduate", "Post Graduate", "Professional"])

    st.markdown("---")

    # 2. Employment and Income
    st.markdown("#### 💼 Employment and Income")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        monthly_salary = st.number_input("Monthly Salary (₹)", min_value=15000, max_value=200000, value=50000, step=1000, help="Monthly gross salary (15K-200K INR)")
    with col2:
        employment_type = st.selectbox("Employment Type", ["Private", "Government", "Self-employed"])
    with col3:
        years_employed = st.number_input("Years of Employment", min_value=0.0, max_value=40.0, value=3.0, step=0.5)
    with col4:
        company_type = st.selectbox("Company Type", ["Large Indian", "MNC", "Mid-size", "Startup", "Small"])

    st.markdown("---")

    # 3. Housing and Family
    st.markdown("#### 🏠 Housing and Family")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        house_type = st.selectbox("House Type", ["Rented", "Own", "Family"])
    with col2:
        monthly_rent = st.number_input("Monthly Rent (₹)", min_value=0, value=0, step=500)
    with col3:
        family_size = st.number_input("Family Size", min_value=1, max_value=20, value=3, step=1)
    with col4:
        dependents = st.number_input("Dependents", min_value=0, max_value=10, value=1, step=1)

    st.markdown("---")

    # 4. Monthly Financial Obligations
    st.markdown("#### 💸 Monthly Financial Obligations")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        school_fees = st.number_input("School Fees (₹)", min_value=0, value=0, step=500)
    with col2:
        college_fees = st.number_input("College Fees (₹)", min_value=0, value=0, step=500)
    with col3:
        travel_expenses = st.number_input("Travel Expenses (₹)", min_value=0, value=2000, step=500)
    with col4:
        groceries = st.number_input("Groceries & Utilities (₹)", min_value=0, value=5000, step=500)
    with col5:
        other_expenses = st.number_input("Other Expenses (₹)", min_value=0, value=2000, step=500)

    st.markdown("---")

    # 5. Financial Status and Credit History
    st.markdown("#### 💳 Financial Status and Credit History")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        existing_loans = st.selectbox("Existing Loans", ["Yes", "No"])
    with col2:
        current_emi = st.number_input("Current EMI Amount (₹)", min_value=0, value=0, step=500)
    with col3:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=750, step=10, help="Credit worthiness score (300-850)")
    with col4:
        bank_balance = st.number_input("Bank Balance (₹)", min_value=0, value=10000, step=1000)
    with col5:
        emergency_fund = st.number_input("Emergency Fund (₹)", min_value=0, value=5000, step=1000)

    st.markdown("---")

    # 6. Loan Application Details
    st.markdown("#### 🏦 Loan Application Details")
    col1, col2, col3 = st.columns(3)
    with col1:
        emi_scenario = st.selectbox("EMI Scenario", ["E-commerce Shopping EMI", "Education EMI", "Home Appliances EMI", "Personal Loan EMI", "Vehicle EMI"])
    with col2:
        requested_amount = st.number_input("Requested Loan Amount (₹)", min_value=10000, value=500000, step=10000)
    with col3:
        requested_tenure = st.number_input("Requested Tenure (Months)", min_value=6, max_value=360, value=60, step=6)

    st.markdown("---")
    
    # Submit button
    submitted = st.form_submit_button("🔮 Predict EMI Eligibility", width='stretch')

# Process prediction
if submitted:
    # Prepare input data with ALL raw fields
    input_data = {
        # Numerical Columns (Raw)
        'Age': age,
        'Monthly_Salary': monthly_salary,
        'Years_Of_Employment': years_employed,
        'Current_Emi_Amount': current_emi,
        'Credit_Score': credit_score,
        'Bank_Balance': bank_balance,
        'Emergency_Fund': emergency_fund,
        'Requested_Amount': requested_amount,
        'Requested_Tenure': requested_tenure,
        
        # Expense Columns (for Feature Engineering)
        'Monthly_Rent': monthly_rent,
        'School_Fees': school_fees,
        'College_Fees': college_fees,
        'Travel_Expenses': travel_expenses,
        'Groceries_Utilities': groceries,
        'Other_Monthly_Expenses': other_expenses,
        'Family_Size': family_size,
        'Dependents': dependents,
        
        # Categorical Columns
        'Gender': gender,
        'Marital_Status': marital_status,
        'Education': education,
        'Employment_Type': employment_type,
        'Company_Type': company_type,
        'House_Type': house_type,
        'Existing_Loans': existing_loans,
        'Emi_Scenario': emi_scenario
    }
    
    # Create DataFrame
    raw_df = pd.DataFrame([input_data])
    
    # Make prediction
    try:
        with st.spinner("Processing data & making predictions..."):
            # 1. Feature Engineering
            engineered_df = perform_feature_engineering(raw_df)
            
            # 2. Prediction
            result = make_predictions(preprocessor, regressor, classifier, engineered_df)
    except CustomException as ce:
        st.error(f"❌ Prediction Error: {str(ce)}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Unexpected error occurred: {str(e)}")
        st.stop()
    
    if result is None:
        st.error("❌ Prediction failed. Please try again.")
    else:
        st.markdown("---")
        st.markdown("### 🎯 Prediction Results")
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            # Eligibility result
            label = result['eligibility_label']
            confidence = result['confidence']
            
            if label == "Eligible":
                st.markdown(f"""
                <div class="result-card eligible">
                    <h2>✅ ELIGIBLE</h2>
                    <p style="font-size: 1.2rem;">You are eligible for an EMI loan!</p>
                    <p style="font-size: 3rem; font-weight: bold;">{confidence*100:.1f}%</p>
                    <p>Confidence Level</p>
                </div>
                """, unsafe_allow_html=True)
            elif label == "High_Risk":
                st.markdown(f"""
                <div class="result-card" style="background: linear-gradient(135deg, #f2994a 0%, #f2c94c 100%);">
                    <h2>⚠️ HIGH RISK</h2>
                    <p style="font-size: 1.2rem;">Eligible with conditions (High Risk)</p>
                    <p style="font-size: 3rem; font-weight: bold;">{confidence*100:.1f}%</p>
                    <p>Confidence Level</p>
                </div>
                """, unsafe_allow_html=True)
            else:  # Not Eligible
                st.markdown(f"""
                <div class="result-card not-eligible">
                    <h2>❌ NOT ELIGIBLE</h2>
                    <p style="font-size: 1.2rem;">You do not meet the eligibility criteria</p>
                    <p style="font-size: 3rem; font-weight: bold;">{confidence*100:.1f}%</p>
                    <p>Confidence Level</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # EMI Amount
            if label in ["Eligible", "High_Risk"]:
                st.markdown(f"""
                <div class="result-card">
                    <h3>Maximum EMI Amount</h3>
                    <p style="font-size: 3rem; font-weight: bold;">{format_currency(result['emi_amount'])}</p>
                    <p style="font-size: 1rem;">per month</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Additional metrics
                st.metric("Loan-to-Income Ratio", f"{(result['emi_amount']/monthly_salary)*100:.1f}%")
                st.metric("Requested Amount", format_currency(requested_amount))
                
                if label == "High_Risk":
                    st.warning("⚠️ Note: Due to high risk status, interest rates may be higher.")
            else:
                st.info("💡 **Tip:** Improve your eligibility by:\n- Increasing your income\n- Reducing existing debts\n- Improving your credit score\n- Reducing monthly expenses")
        
        # Detailed breakdown
        if label in ["Eligible", "High_Risk"]:
            st.markdown("---")
            st.markdown("### 📊 Loan Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Monthly EMI", format_currency(result['emi_amount']))
            
            with col2:
                total_payment = result['emi_amount'] * requested_tenure
                st.metric("Total Payment", format_currency(total_payment))
            
            with col3:
                interest = total_payment - requested_amount
                st.metric("Total Interest", format_currency(interest))
            
            with col4:
                st.metric("Tenure", f"{requested_tenure} months")
            
            # Download report
            st.markdown("---")
            st.download_button(
                label="📥 Download Prediction Report",
                data=f"""EMI Prediction Report
                
Applicant Details:
- Age: {age}
- Income: {format_currency(monthly_salary)}
- Credit Score: {credit_score}

Prediction Results:
- Eligibility: {label.replace('_', ' ').upper()}
- Confidence: {confidence*100:.1f}%
- Maximum EMI: {format_currency(result['emi_amount'])}
- Requested Loan: {format_currency(requested_amount)}
- Tenure: {requested_tenure} months
- Total Payment: {format_currency(result['emi_amount'] * requested_tenure)}
                """,
                file_name="emi_prediction_report.txt",
                mime="text/plain"
            )

# Information section
st.markdown("---")
st.markdown("### ℹ️ How It Works")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Classification Model (LightGBM):**
    - Predicts EMI eligibility
    - Classes: Eligible, High Risk, Not Eligible
    - Accuracy: 97.16%
    """)

with col2:
    st.markdown("""
    **Regression Model (XGBoost):**
    - Predicts maximum EMI amount
    - R² Score: 0.9817
    - RMSE: 996.55
    """)
