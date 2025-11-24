import logging
import numpy as np
import pandas as pd
from pathlib import Path

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

    return {"train": train_metrics, "test": test_metrics}


