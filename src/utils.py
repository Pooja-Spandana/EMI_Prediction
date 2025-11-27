import logging
import numpy as np
import pandas as pd
from pathlib import Path
from exception import CustomException
import sys

# Function to ensures the parent directory for a given file path exists
def ensure_parent_dir(path):
    """
    Ensures the parent directory for a given file path exists.
    Example:
        ensure_parent_dir("artifacts/ingested_raw.csv")
    """
    path = Path(path)
    parent = path.parent

    if not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created directory: {parent}")

# Function to extract numeric values from strings
def extract_number(series: pd.Series) -> pd.Series:
    """
    Extracts numeric values from string columns like:
    '₹ 12,345', '45 years', '1,200.50', 'salary: 50000'
    """
    return (
        series.astype(str)
        .str.extract(r"([\d\.,]+)")[0]
        .str.replace(",", "", regex=False)
    )

# Function to convert to numeric values
def to_numeric(series: pd.Series) -> pd.Series:
    """
    Safe conversion to numeric. Returns NaN for invalid values.
    """
    return pd.to_numeric(series, errors="coerce")

# Function to save df to csv
def save_csv(df: pd.DataFrame, path: str, index: bool = False):
    """
    Save DataFrame to CSV, ensuring parent directory exists.
    """
    ensure_parent_dir(path)
    df.to_csv(path, index=index)
    logging.info(f"Saved CSV to {path}")

# Function to evaluate classification models
def evaluate_classification_model(model, X_train, y_train, X_test, y_test, model_name="Model"):
    """
    Evaluate classification model and return metrics.
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    def compute_metrics(X, y):
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)

        return {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y, y_pred, average='weighted', zero_division=0),
            "f1_score": f1_score(y, y_pred, average='weighted', zero_division=0),
            "roc_auc": roc_auc_score(y, y_pred_proba, multi_class='ovr', average='weighted')
        }
    train_metrics = compute_metrics(X_train, y_train)
    test_metrics  = compute_metrics(X_test,  y_test)

    logging.info(f"{model_name} — Train Accuracy: {train_metrics['accuracy']:.4f}, Train F1: {train_metrics['f1_score']:.4f}, Train ROC-AUC: {train_metrics['roc_auc']:.4f}")
    logging.info(f"{model_name} — Test Accuracy: {test_metrics['accuracy']:.4f}, Test F1: {test_metrics['f1_score']:.4f}, Test ROC-AUC: {test_metrics['roc_auc']:.4f}")

    return {"train": train_metrics, "test": test_metrics}

# Function to evaluate regression models
def evaluate_regression_model(model, X_train, y_train, X_test, y_test, model_name="Model"):
    """
    Evaluate regression model and return metrics.
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import numpy as np

    def compute_metrics(X, y):
        y_pred = model.predict(X)

        return {
            "rmse": np.sqrt(mean_squared_error(y, y_pred)),
            "mae": mean_absolute_error(y, y_pred),
            "r2_score": r2_score(y, y_pred),
            "mape": np.mean(np.abs((y - y_pred) / (y + 1e-10))) * 100
        }

    train_metrics = compute_metrics(X_train, y_train)
    test_metrics  = compute_metrics(X_test,  y_test)

    logging.info(f"{model_name} — Train RMSE: {train_metrics['rmse']:.2f}, Train MAE: {train_metrics['mae']:.2f}, Train R2: {train_metrics['r2_score']:.4f}")
    logging.info(f"{model_name} — Test RMSE: {test_metrics['rmse']:.2f}, Test MAE: {test_metrics['mae']:.2f}, Test R2: {test_metrics['r2_score']:.4f}")

    return {" train": train_metrics, "test": test_metrics}


# ============================================================================
# STREAMLIT APPLICATION UTILITIES
# ============================================================================

def load_models_local():
    """
    Load models and preprocessor from local artifacts folder.
    Used for local development and testing.
    
    Returns:
        tuple: (regressor, classifier, preprocessor)
    """
    import joblib
    import streamlit as st
    
    try:
        regressor = joblib.load('artifacts/models/final_regressor.pkl')
        classifier = joblib.load('artifacts/models/final_classifier.pkl')
        preprocessor = joblib.load('artifacts/preprocessor/preprocessor.pkl')
        logging.info("✓ Models loaded from local artifacts")
        return regressor, classifier, preprocessor
    except Exception as e:
        error_msg = f"Error loading local models: {e}"
        st.error(error_msg)
        logging.error(error_msg)
        raise CustomException(e, sys)
        return None, None, None


def load_models_from_mlflow():
    """
    Load models and preprocessor from MLflow Model Registry.
    Used for Streamlit Cloud deployment.
    
    Returns:
        tuple: (regressor, classifier, preprocessor)
    """
    import mlflow
    import os
    import streamlit as st
    
    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        
        # Load registered models from Production stage
        regressor = mlflow.pyfunc.load_model("models:/EMI_Regressor/Production")
        classifier = mlflow.pyfunc.load_model("models:/EMI_Classifier/Production")
        
        # Load preprocessor from the regressor's run artifacts
        # Get the run_id from the registered model
        client = mlflow.tracking.MlflowClient()
        reg_model = client.get_registered_model("EMI_Regressor")
        latest_version = reg_model.latest_versions[0]
        run_id = latest_version.run_id
        
        # Download preprocessor artifact
        preprocessor_path = mlflow.artifacts.download_artifacts(
            f"runs:/{run_id}/preprocessor/preprocessor.pkl"
        )
        import joblib
        preprocessor = joblib.load(preprocessor_path)
        
        logging.info("✓ Models loaded from MLflow Model Registry")
        return regressor, classifier, preprocessor
    except Exception as e:
        error_msg = f"Error loading models from MLflow: {e}"
        st.error(error_msg)
        logging.error(error_msg)
        raise CustomException(e, sys)
        return None, None, None


def load_models():
    """
    Smart model loader that checks environment and loads from appropriate source.
    Uses local files for development, MLflow for cloud deployment.
    
    Returns:
        tuple: (regressor, classifier, preprocessor)
    """
    import os
    import streamlit as st
    
    # Check if running on Streamlit Cloud
    if os.getenv("STREAMLIT_CLOUD") == "true":
        st.info("🌐 Loading models from MLflow (Cloud mode)...")
        return load_models_from_mlflow()
    else:
        st.info("💻 Loading models from local files (Development mode)...")
        return load_models_local()


def validate_user_input(input_data):
    """
    Validate user input data for predictions.
    
    Args:
        input_data (dict): User input data
        
    Returns:
        tuple: (is_valid, error_message)
    """
    errors = []
    
    # Check for required fields
    required_fields = ['Age', 'Income', 'Loan_Amount_Request', 'Loan_Tenure_Months']
    for field in required_fields:
        if field not in input_data or input_data[field] is None:
            errors.append(f"Missing required field: {field}")
    
    # Validate numeric ranges
    if 'Age' in input_data and (input_data['Age'] < 18 or input_data['Age'] > 100):
        errors.append("Age must be between 18 and 100")
    
    if 'Income' in input_data and input_data['Income'] <= 0:
        errors.append("Income must be greater than 0")
    
    if 'Loan_Amount_Request' in input_data and input_data['Loan_Amount_Request'] <= 0:
        errors.append("Loan amount must be greater than 0")
    
    if errors:
        return False, "; ".join(errors)
    return True, ""


from src.components.feature_engineering import FeatureEngineer

def perform_feature_engineering(df):
    """
    Replicate the feature engineering logic using the FeatureEngineer class.
    
    Args:
        df (pd.DataFrame): Raw input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with engineered features
    """
    try:
        fe = FeatureEngineer()
        # Create features
        df = fe.create_features(df)
        # Drop redundant columns (optional, but ensures consistency with training)
        df = fe.drop_redundant(df) 
        return df
    except Exception as e:
        logging.error(f"Feature engineering error: {e}")
        raise CustomException(e, sys)


def make_predictions(preprocessor, regressor, classifier, input_df):
    """
    Make predictions using both models.
    
    Args:
        preprocessor: Dictionary containing 'scaler', 'ohe', 'num_cols', 'cat_cols'
        regressor: Trained regression model
        classifier: Trained classification model
        input_df (pd.DataFrame): Input data
        
    Returns:
        dict: Prediction results with eligibility, probability, and EMI amount
    """
    try:
        # Extract components from the preprocessor dictionary
        scaler = preprocessor['scaler']
        ohe = preprocessor['ohe']
        num_cols = preprocessor['num_cols']
        cat_cols = preprocessor['cat_cols']
        
        # 1. Scale Numerical Features
        # Ensure input_df has the correct numerical columns
        X_num = input_df[num_cols].copy()
        X_num_scaled = scaler.transform(X_num)
        
        # 2. Encode Categorical Features
        # Ensure input_df has the correct categorical columns
        X_cat = input_df[cat_cols].copy()
        X_cat_encoded = ohe.transform(X_cat)
        
        # 3. Handle Remainder Columns (Engineered Features)
        # These are columns that were not scaled or encoded (e.g. Total_Expenses, Ratios)
        # They correspond to the 'remainder' part in FeatureEngineer.transform
        remainder_cols = [c for c in input_df.columns if c not in num_cols and c not in cat_cols]
        
        # We need to ensure the order matches training, but since we don't have the exact list,
        # we rely on the fact that input_df comes from perform_feature_engineering which should match.
        # However, input_df might contain extra columns if drop_redundant didn't remove everything.
        # Let's explicitly select the known engineered features to be safe, if we can.
        # But dynamic is better if we trust drop_redundant.
        # Let's try dynamic first, but log the columns.
        
        if remainder_cols:
            X_remainder = input_df[remainder_cols].values
            # 4. Combine Features
            X_processed = np.hstack((X_num_scaled, X_cat_encoded, X_remainder))
        else:
            X_processed = np.hstack((X_num_scaled, X_cat_encoded))
            
        logging.info(f"Prediction features: {X_processed.shape[1]} (Num: {len(num_cols)}, Cat: {X_cat_encoded.shape[1]}, Rem: {len(remainder_cols)})")
        
        # Classification prediction (eligibility)
        # Classes: 0=Eligible, 1=High_Risk, 2=Not_Eligible
        eligibility_pred = classifier.predict(X_processed)[0]
        eligibility_proba = classifier.predict_proba(X_processed)[0]
        
        # Get probability of the predicted class
        confidence = float(eligibility_proba[eligibility_pred])
        
        # Map class index to label
        class_mapping = {0: "Eligible", 1: "High_Risk", 2: "Not_Eligible"}
        predicted_label = class_mapping.get(int(eligibility_pred), "Unknown")
        
        # Regression prediction (EMI amount)
        # Only relevant if Eligible (0) or High_Risk (1) - arguably High_Risk might get a lower amount or higher interest
        emi_amount = regressor.predict(X_processed)[0]
        
        return {
            "eligibility_class": int(eligibility_pred),
            "eligibility_label": predicted_label,
            "confidence": confidence,
            "emi_amount": float(emi_amount) if int(eligibility_pred) in [0, 1] else 0.0,
            "probabilities": {
                "Eligible": float(eligibility_proba[0]),
                "High_Risk": float(eligibility_proba[1]),
                "Not_Eligible": float(eligibility_proba[2])
            }
        }
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise CustomException(e, sys)


def format_currency(amount):
    """
    Format amount as Indian currency (₹).
    
    Args:
        amount (float): Amount to format
        
    Returns:
        str: Formatted currency string
    """
    return f"₹ {amount:,.2f}"


def create_metric_card(label, value, delta=None):
    """
    Create a metric display card (wrapper for st.metric).
    
    Args:
        label (str): Metric label
        value: Metric value
        delta: Optional delta value
    """
    import streamlit as st
    st.metric(label=label, value=value, delta=delta)


def plot_confusion_matrix(y_true, y_pred, labels=None):
    """
    Plot confusion matrix heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
        
    Returns:
        matplotlib figure
    """
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=labels or ['Not Eligible', 'Eligible'],
                yticklabels=labels or ['Not Eligible', 'Eligible'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    
    return fig


def plot_feature_importance(model, feature_names, top_n=10):
    """
    Plot feature importance for tree-based models.
    
    Args:
        model: Trained model with feature_importances_
        feature_names: List of feature names
        top_n: Number of top features to display
        
    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(indices)), importances[indices], color='skyblue')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Feature Importances')
    plt.tight_layout()
    
    return fig


def plot_actual_vs_predicted(y_true, y_pred):
    """
    Plot actual vs predicted scatter plot for regression.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, s=10)
    
    # Add perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual EMI Amount')
    ax.set_ylabel('Predicted EMI Amount')
    ax.set_title('Actual vs Predicted EMI Amount')
    ax.legend()
    plt.tight_layout()
    
    return fig


def plot_residuals(y_true, y_pred):
    """
    Plot residuals for regression model.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt
    
    residuals = y_true - y_pred
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_pred, residuals, alpha=0.5, s=10)
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('Predicted EMI Amount')
    ax.set_ylabel('Residuals')
    ax.set_title('Residual Plot')
    plt.tight_layout()
    
    return fig
