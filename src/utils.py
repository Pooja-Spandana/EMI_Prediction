import logging
import numpy as np
import pandas as pd
from pathlib import Path

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


def to_numeric(series: pd.Series) -> pd.Series:
    """
    Safe conversion to numeric. Returns NaN for invalid values.
    """
    return pd.to_numeric(series, errors="coerce")


def save_csv(df: pd.DataFrame, path: str, index: bool = False):
    """
    Save DataFrame to CSV, ensuring parent directory exists.
    """
    ensure_parent_dir(path)
    df.to_csv(path, index=index)
    logging.info(f"Saved CSV to {path}")


def evaluate_classification_model(model, X, y, model_name="Model"):
    """
    Evaluate classification model and return metrics.
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)
    
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y, y_pred, average='weighted', zero_division=0),
        "f1_score": f1_score(y, y_pred, average='weighted', zero_division=0),
        "roc_auc": roc_auc_score(y, y_pred_proba, multi_class='ovr', average='weighted')
    }
    
    logging.info(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
    
    return metrics


def evaluate_regression_model(model, X, y, model_name="Model"):
    """
    Evaluate regression model and return metrics.
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    y_pred = model.predict(X)
    
    metrics = {
        "rmse": np.sqrt(mean_squared_error(y, y_pred)),
        "mae": mean_absolute_error(y, y_pred),
        "r2_score": r2_score(y, y_pred),
        "mape": np.mean(np.abs((y - y_pred) / (y + 1e-10))) * 100  # Adding small value to avoid division by zero
    }
    
    logging.info(f"{model_name} - RMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}, R2: {metrics['r2_score']:.4f}")
    
    return metrics


