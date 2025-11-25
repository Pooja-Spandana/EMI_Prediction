"""
Final Model Training and Evaluation Component

This script:
1. Loads best hyperparameters from JSON files
2. Trains final models on combined train+val dataset
3. Evaluates on both training and test datasets
4. Logs all metrics to MLflow in a single run
5. Registers models to MLflow Model Registry if thresholds are met
6. Saves final models to disk
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix

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
    RANDOM_STATE,
    CLASSIFICATION_ACCURACY_THRESHOLD,
    REGRESSION_RMSE_THRESHOLD
)
from src.mlflow_config import (
    get_mlflow_config, 
    EXPERIMENT_REGRESSION, 
    EXPERIMENT_CLASSIFICATION,
    MODEL_NAME_REGRESSION,
    MODEL_NAME_CLASSIFICATION
)
from src.utils import evaluate_regression_model, evaluate_classification_model


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


def load_train_val_data(data_dir, target_file):
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
        
        logging.info(f"Combined train+val data shape: {X_combined.shape}")
        return X_combined, y_combined
        
    except Exception as e:
        raise CustomException(e, sys)


def load_test_data(data_dir, target_file):
    """Load test data."""
    try:
        X_test = pd.read_csv(os.path.join(data_dir, "test", "X.csv"))
        y_test = pd.read_csv(os.path.join(data_dir, "test", target_file)).values.ravel()
        
        logging.info(f"Test data shape: {X_test.shape}")
        return X_test, y_test
        
    except Exception as e:
        raise CustomException(e, sys)


def get_model_instance(model_name, task_type, params):
    """Instantiate model with parameters."""
    try:
        # Ensure random_state is in params for models that need it
        if model_name != "Linear_Regression" and "random_state" not in params:
             params["random_state"] = RANDOM_STATE

        if task_type == "regression":
            if model_name == "Linear_Regression":
                return LinearRegression(**params)
            elif model_name == "LightGBM":
                if "verbose" not in params: params["verbose"] = -1
                return LGBMRegressor(**params)
            elif model_name == "XGBoost":
                return XGBRegressor(**params)
            elif model_name == "Gradient_Boosting":
                return GradientBoostingRegressor(**params)
        
        elif task_type == "classification":
            if model_name == "Logistic_Regression":
                return LogisticRegression(**params)
            elif model_name == "LightGBM":
                if "verbose" not in params: params["verbose"] = -1
                return LGBMClassifier(**params)
            elif model_name == "XGBoost":
                return XGBClassifier(**params)
            elif model_name == "Gradient_Boosting":
                return GradientBoostingClassifier(**params)
                
        raise ValueError(f"Unknown model {model_name} for task {task_type}")
        
    except Exception as e:
        raise CustomException(e, sys)


def train_and_evaluate_regression():
    """Train and evaluate final regression model."""
    try:
        logging.info("Starting final regression model training and evaluation...")
        
        # Load params
        params_data = load_best_params("regression")
        model_name = params_data["model_name"]
        params = params_data["parameters"]
        
        # Load data
        X_train, y_train = load_train_val_data(PROCESSED_DATA_DIR, "y_reg.csv")
        X_test, y_test = load_test_data(PROCESSED_DATA_DIR, "y_reg.csv")
        
        # Load preprocessor
        preprocessor_path = os.path.join(ARTIFACTS_DIR, "preprocessor", "preprocessor.pkl")
        if os.path.exists(preprocessor_path):
            preprocessor = joblib.load(preprocessor_path)
            logging.info(f"Loaded preprocessor from {preprocessor_path}")
        else:
            logging.warning("Preprocessor not found. Model will only work with preprocessed data.")
            preprocessor = None
        
        # Initialize MLflow
        mlflow_config = get_mlflow_config()
        mlflow_config.get_or_create_experiment(EXPERIMENT_REGRESSION)
        mlflow.set_experiment(EXPERIMENT_REGRESSION)
        
        with mlflow.start_run(run_name="Final_Model_Training_and_Evaluation", nested=False) as run:
            mlflow.set_tag("stage", "final_model")
            mlflow.set_tag("model_type", model_name)
            
            # Log params
            mlflow_config.log_params_from_dict(params)
            mlflow.log_param("training_samples", len(y_train))
            mlflow.log_param("test_samples", len(y_test))
            
            # Train
            logging.info(f"Training {model_name} on combined train+val data...")
            model = get_model_instance(model_name, "regression", params)
            model.fit(X_train, y_train)
            
            # Evaluate on both train and test data
            metrics_dict = evaluate_regression_model(model, X_train, y_train, X_test, y_test, model_name)
            train_metrics = metrics_dict['train']
            test_metrics = metrics_dict['test']
            
            # Log metrics
            mlflow_config.log_metrics_from_dict(train_metrics, prefix="train_")
            mlflow_config.log_metrics_from_dict(test_metrics, prefix="test_")
            
            # Save model
            save_dir = os.path.join(ARTIFACTS_DIR, "models")
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, "final_regressor.pkl")
            joblib.dump(model, model_path)
            logging.info(f"Model saved to {model_path}")
            
            # Log model artifact
            mlflow_config.log_model_with_signature(
                model, 
                "final_model", 
                X_sample=X_test[:5]
            )
            
            # Log preprocessor as artifact
            if preprocessor is not None:
                mlflow.log_artifact(preprocessor_path, "preprocessor")
                logging.info("Preprocessor logged to MLflow")
            
            # Save predictions
            y_pred = model.predict(X_test)
            preds_df = pd.DataFrame({
                'Actual': y_test,
                'Predicted': y_pred
            })
            preds_dir = os.path.join(ARTIFACTS_DIR, "predictions")
            os.makedirs(preds_dir, exist_ok=True)
            preds_path = os.path.join(preds_dir, "regression_predictions.csv")
            preds_df.to_csv(preds_path, index=False)
            mlflow.log_artifact(preds_path)
            
            # Check threshold for registration
            rmse = test_metrics['rmse']
            if rmse < REGRESSION_RMSE_THRESHOLD:
                logging.info(f"RMSE {rmse:.2f} < Threshold {REGRESSION_RMSE_THRESHOLD}. Registering model...")
                
                model_uri = f"runs:/{run.info.run_id}/final_model"
                tags = {
                    "rmse": f"{rmse:.2f}",
                    "r2_score": f"{test_metrics['r2_score']:.4f}",
                    "stage": "production_candidate"
                }
                mlflow_config.register_model(model_uri, MODEL_NAME_REGRESSION, tags)
            else:
                logging.warning(f"RMSE {rmse:.2f} >= Threshold {REGRESSION_RMSE_THRESHOLD}. Model NOT registered.")
            
    except Exception as e:
        logging.error(f"Regression training/evaluation failed: {e}")
        raise


def train_and_evaluate_classification():
    """Train and evaluate final classification model."""
    try:
        logging.info("Starting final classification model training and evaluation...")
        
        # Load params
        params_data = load_best_params("classification")
        model_name = params_data["model_name"]
        params = params_data["parameters"]
        smote_applied = params_data.get("smote_applied", False)
        
        # Load data
        X_train, y_train = load_train_val_data(PROCESSED_DATA_DIR, "y_clf.csv")
        X_test, y_test = load_test_data(PROCESSED_DATA_DIR, "y_clf.csv")
        
        # Load preprocessor
        preprocessor_path = os.path.join(ARTIFACTS_DIR, "preprocessor", "preprocessor.pkl")
        if os.path.exists(preprocessor_path):
            preprocessor = joblib.load(preprocessor_path)
            logging.info(f"Loaded preprocessor from {preprocessor_path}")
        else:
            logging.warning("Preprocessor not found. Model will only work with preprocessed data.")
            preprocessor = None
        
        # Apply SMOTE if needed
        if smote_applied:
            logging.info("Applying SMOTE to combined dataset...")
            smote = SMOTE(random_state=RANDOM_STATE)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            logging.info(f"Resampled data shape: {X_train.shape}")
        
        # Initialize MLflow
        mlflow_config = get_mlflow_config()
        mlflow_config.get_or_create_experiment(EXPERIMENT_CLASSIFICATION)
        mlflow.set_experiment(EXPERIMENT_CLASSIFICATION)
        
        with mlflow.start_run(run_name="Final_Model_Training_and_Evaluation", nested=False) as run:
            mlflow.set_tag("stage", "final_model")
            mlflow.set_tag("model_type", model_name)
            mlflow.set_tag("smote_applied", str(smote_applied))
            
            # Log params
            mlflow_config.log_params_from_dict(params)
            mlflow.log_param("training_samples", len(y_train))
            mlflow.log_param("test_samples", len(y_test))
            
            # Train
            logging.info(f"Training {model_name} on combined train+val data...")
            model = get_model_instance(model_name, "classification", params)
            model.fit(X_train, y_train)
            
            # Evaluate on both train and test data
            metrics_dict = evaluate_classification_model(model, X_train, y_train, X_test, y_test, model_name)
            train_metrics = metrics_dict['train']
            test_metrics = metrics_dict['test']
            
            # Log metrics
            mlflow_config.log_metrics_from_dict(train_metrics, prefix="train_")
            mlflow_config.log_metrics_from_dict(test_metrics, prefix="test_")
            
            # Save model
            save_dir = os.path.join(ARTIFACTS_DIR, "models")
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, "final_classifier.pkl")
            joblib.dump(model, model_path)
            logging.info(f"Model saved to {model_path}")
            
            # Log model artifact
            mlflow_config.log_model_with_signature(
                model, 
                "final_model", 
                X_sample=X_test[:5]
            )
            
            # Log preprocessor as artifact
            if preprocessor is not None:
                mlflow.log_artifact(preprocessor_path, "preprocessor")
                logging.info("Preprocessor logged to MLflow")
            
            # Save predictions
            y_pred = model.predict(X_test)
            preds_df = pd.DataFrame({
                'Actual': y_test,
                'Predicted': y_pred
            })
            preds_dir = os.path.join(ARTIFACTS_DIR, "predictions")
            os.makedirs(preds_dir, exist_ok=True)
            preds_path = os.path.join(preds_dir, "classification_predictions.csv")
            preds_df.to_csv(preds_path, index=False)
            mlflow.log_artifact(preds_path)
            
            # Log confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            cm_path = os.path.join(preds_dir, "confusion_matrix.png")
            plt.savefig(cm_path)
            plt.close()
            mlflow.log_artifact(cm_path)
            
            # Check threshold for registration
            accuracy = test_metrics['accuracy']
            if accuracy > CLASSIFICATION_ACCURACY_THRESHOLD:
                logging.info(f"Accuracy {accuracy:.4f} > Threshold {CLASSIFICATION_ACCURACY_THRESHOLD}. Registering model...")
                
                model_uri = f"runs:/{run.info.run_id}/final_model"
                tags = {
                    "accuracy": f"{accuracy:.4f}",
                    "f1_score": f"{test_metrics['f1_score']:.4f}",
                    "stage": "production_candidate"
                }
                mlflow_config.register_model(model_uri, MODEL_NAME_CLASSIFICATION, tags)
            else:
                logging.warning(f"Accuracy {accuracy:.4f} <= Threshold {CLASSIFICATION_ACCURACY_THRESHOLD}. Model NOT registered.")
            
    except Exception as e:
        logging.error(f"Classification training/evaluation failed: {e}")
        raise


if __name__ == "__main__":
    try:
        # Check if best params exist
        reg_params_exist = os.path.exists(os.path.join(BEST_PARAMS_DIR, "regression_best_params.json"))
        clf_params_exist = os.path.exists(os.path.join(BEST_PARAMS_DIR, "classification_best_params.json"))
        
        if reg_params_exist:
            train_and_evaluate_regression()
        else:
            logging.warning("Regression best params not found. Skipping regression.")
            
        if clf_params_exist:
            train_and_evaluate_classification()
        else:
            logging.warning("Classification best params not found. Skipping classification.")
            
        print("\n✅ Final model training and evaluation completed!")
        print("📊 View experiments at: https://dagshub.com/Pooja-Spandana/EMI_Prediction/experiments")
        
    except Exception as e:
        logging.error(f"Final training pipeline failed: {e}")
        sys.exit(1)
