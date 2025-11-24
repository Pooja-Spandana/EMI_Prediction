"""
Final Training Pipeline

This script:
1. Loads best hyperparameters from JSON files
2. Combines train and validation datasets
3. Trains final models on the combined dataset
4. Saves final models to disk
5. Logs training details to MLflow
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from imblearn.over_sampling import SMOTE

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBRegressor, XGBClassifier

from src.logger import logging
from src.exception import CustomException
from src.config import (
    PROCESSED_DATA_DIR, 
    ARTIFACTS_DIR, 
    BEST_PARAMS_DIR, 
    RANDOM_STATE
)
from src.mlflow_config import (
    get_mlflow_config, 
    EXPERIMENT_REGRESSION, 
    EXPERIMENT_CLASSIFICATION
)


def load_best_params(task_type):
    """Load best parameters from JSON file."""
    try:
        filename = f"{task_type}_best_params.json"
        filepath = os.path.join(BEST_PARAMS_DIR, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Best params file not found: {filepath}")
            
        with open(filepath, 'r') as f:
            params_data = json.load(f)
            
        logging.info(f"Loaded best params for {task_type}: {params_data['model_name']}")
        return params_data
    except Exception as e:
        raise CustomException(e, sys)


def load_and_combine_data(data_dir, target_file):
    """Load and combine train and validation data."""
    try:
        # Load train
        X_train = pd.read_csv(os.path.join(data_dir, "train", "X.csv"))
        y_train = pd.read_csv(os.path.join(data_dir, "train", target_file)).values.ravel()
        
        # Load val
        X_val = pd.read_csv(os.path.join(data_dir, "val", "X.csv"))
        y_val = pd.read_csv(os.path.join(data_dir, "val", target_file)).values.ravel()
        
        # Combine
        X_combined = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
        y_combined = np.concatenate([y_train, y_val])
        
        logging.info(f"Combined data shape: {X_combined.shape}")
        return X_combined, y_combined
        
    except Exception as e:
        raise CustomException(e, sys)


def get_model_instance(model_name, task_type, params):
    """Instantiate model with parameters."""
    try:
        if task_type == "regression":
            if model_name == "Linear_Regression":
                return LinearRegression(**params)
            elif model_name == "LightGBM":
                return LGBMRegressor(random_state=RANDOM_STATE, verbose=-1, **params)
            elif model_name == "XGBoost":
                return XGBRegressor(random_state=RANDOM_STATE, **params)
            elif model_name == "Gradient_Boosting":
                return GradientBoostingRegressor(random_state=RANDOM_STATE, **params)
        
        elif task_type == "classification":
            if model_name == "Logistic_Regression":
                return LogisticRegression(random_state=RANDOM_STATE, **params)
            elif model_name == "LightGBM":
                return LGBMClassifier(random_state=RANDOM_STATE, verbose=-1, **params)
            elif model_name == "XGBoost":
                return XGBClassifier(random_state=RANDOM_STATE, **params)
            elif model_name == "Gradient_Boosting":
                return GradientBoostingClassifier(random_state=RANDOM_STATE, **params)
                
        raise ValueError(f"Unknown model {model_name} for task {task_type}")
        
    except Exception as e:
        raise CustomException(e, sys)


def train_final_regression_model():
    """Train final regression model."""
    try:
        logging.info("Starting final regression model training...")
        
        # Load params
        params_data = load_best_params("regression")
        model_name = params_data["model_name"]
        params = params_data["parameters"]
        
        # Load data
        X, y = load_and_combine_data(PROCESSED_DATA_DIR, "y_reg.csv")
        
        # Initialize MLflow
        mlflow_config = get_mlflow_config()
        mlflow_config.get_or_create_experiment(EXPERIMENT_REGRESSION)
        mlflow.set_experiment(EXPERIMENT_REGRESSION)
        
        with mlflow.start_run(run_name="Final_Training_Combined_Data", nested=False):
            mlflow.set_tag("stage", "final_training")
            mlflow.set_tag("data", "train+val")
            mlflow.set_tag("model_type", model_name)
            
            # Log params
            mlflow_config.log_params_from_dict(params)
            mlflow.log_param("training_samples", len(y))
            
            # Train
            model = get_model_instance(model_name, "regression", params)
            model.fit(X, y)
            
            # Save model
            save_dir = os.path.join(ARTIFACTS_DIR, "models")
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, "final_regressor.pkl")
            joblib.dump(model, model_path)
            
            # Log model artifact
            mlflow_config.log_model_with_signature(
                model, 
                "final_model", 
                X_sample=X[:5]
            )
            
            logging.info(f"Final regression model saved to {model_path}")
            
    except Exception as e:
        logging.error(f"Final regression training failed: {e}")
        raise


def train_final_classification_model():
    """Train final classification model."""
    try:
        logging.info("Starting final classification model training...")
        
        # Load params
        params_data = load_best_params("classification")
        model_name = params_data["model_name"]
        params = params_data["parameters"]
        smote_applied = params_data.get("smote_applied", False)
        
        # Load data
        X, y = load_and_combine_data(PROCESSED_DATA_DIR, "y_clf.csv")
        
        # Apply SMOTE if needed
        if smote_applied:
            logging.info("Applying SMOTE to combined dataset...")
            smote = SMOTE(random_state=RANDOM_STATE)
            X, y = smote.fit_resample(X, y)
            logging.info(f"Resampled data shape: {X.shape}")
        
        # Initialize MLflow
        mlflow_config = get_mlflow_config()
        mlflow_config.get_or_create_experiment(EXPERIMENT_CLASSIFICATION)
        mlflow.set_experiment(EXPERIMENT_CLASSIFICATION)
        
        with mlflow.start_run(run_name="Final_Training_Combined_Data", nested=False):
            mlflow.set_tag("stage", "final_training")
            mlflow.set_tag("data", "train+val")
            mlflow.set_tag("model_type", model_name)
            mlflow.set_tag("smote_applied", str(smote_applied))
            
            # Log params
            mlflow_config.log_params_from_dict(params)
            mlflow.log_param("training_samples", len(y))
            
            # Train
            model = get_model_instance(model_name, "classification", params)
            model.fit(X, y)
            
            # Save model
            save_dir = os.path.join(ARTIFACTS_DIR, "models")
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, "final_classifier.pkl")
            joblib.dump(model, model_path)
            
            # Log model artifact
            mlflow_config.log_model_with_signature(
                model, 
                "final_model", 
                X_sample=X[:5]
            )
            
            logging.info(f"Final classification model saved to {model_path}")
            
    except Exception as e:
        logging.error(f"Final classification training failed: {e}")
        raise


if __name__ == "__main__":
    try:
        # Check if best params exist
        reg_params_exist = os.path.exists(os.path.join(BEST_PARAMS_DIR, "regression_best_params.json"))
        clf_params_exist = os.path.exists(os.path.join(BEST_PARAMS_DIR, "classification_best_params.json"))
        
        if reg_params_exist:
            train_final_regression_model()
        else:
            logging.warning("Regression best params not found. Skipping regression training.")
            
        if clf_params_exist:
            train_final_classification_model()
        else:
            logging.warning("Classification best params not found. Skipping classification training.")
            
        print("\n✅ Final training completed!")
        
    except Exception as e:
        logging.error(f"Final training pipeline failed: {e}")
        sys.exit(1)
