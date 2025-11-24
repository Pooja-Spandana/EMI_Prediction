"""
Evaluation and Registration Pipeline

This script:
1. Loads final trained models
2. Loads test dataset
3. Evaluates models on test data
4. Logs metrics to MLflow
5. Registers models to MLflow Model Registry if thresholds are met
"""

import os
import sys
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.logger import logging
from src.exception import CustomException
from src.config import (
    PROCESSED_DATA_DIR, 
    ARTIFACTS_DIR, 
    CLASSIFICATION_ACCURACY_THRESHOLD,
    REGRESSION_RMSE_THRESHOLD
)
from src.utils import evaluate_regression_model, evaluate_classification_model
from src.mlflow_config import (
    get_mlflow_config, 
    EXPERIMENT_REGRESSION, 
    EXPERIMENT_CLASSIFICATION,
    MODEL_NAME_REGRESSION,
    MODEL_NAME_CLASSIFICATION
)


def load_test_data(data_dir, target_file):
    """Load test data."""
    try:
        X_test = pd.read_csv(os.path.join(data_dir, "test", "X.csv"))
        y_test = pd.read_csv(os.path.join(data_dir, "test", target_file)).values.ravel()
        
        logging.info(f"Test data loaded: {X_test.shape}")
        return X_test, y_test
        
    except Exception as e:
        raise CustomException(e, sys)


def load_model(task_type):
    """Load final trained model."""
    try:
        if task_type == "regression":
            filename = "final_regressor.pkl"
        else:
            filename = "final_classifier.pkl"
            
        filepath = os.path.join(ARTIFACTS_DIR, "models", filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        model = joblib.load(filepath)
        logging.info(f"Loaded {task_type} model from {filepath}")
        return model
        
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_and_register_regression():
    """Evaluate and register regression model."""
    try:
        logging.info("Starting regression evaluation...")
        
        # Load model and data
        model = load_model("regression")
        X_test, y_test = load_test_data(PROCESSED_DATA_DIR, "y_reg.csv")
        
        # Initialize MLflow
        mlflow_config = get_mlflow_config()
        mlflow_config.get_or_create_experiment(EXPERIMENT_REGRESSION)
        mlflow.set_experiment(EXPERIMENT_REGRESSION)
        
        with mlflow.start_run(run_name="Final_Evaluation_Test_Set", nested=False) as run:
            mlflow.set_tag("stage", "final_evaluation")
            mlflow.set_tag("data", "test_set")
            
            # Evaluate
            metrics_dict = evaluate_regression_model(model, X_test, y_test, X_test, y_test, "final_regressor")
            test_metrics = metrics_dict['test']
            
            # Log metrics
            mlflow_config.log_metrics_from_dict(test_metrics, prefix="test_")
            
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
            
            logging.info(f"Regression Test Metrics: {test_metrics}")
            
            # Check threshold for registration
            rmse = test_metrics['rmse']
            if rmse < REGRESSION_RMSE_THRESHOLD:
                logging.info(f"RMSE {rmse:.2f} < Threshold {REGRESSION_RMSE_THRESHOLD}. Registering model...")
                
                # We need to log the model again in this run to register it from this run
                mlflow_config.log_model_with_signature(
                    model, 
                    "final_model", 
                    X_sample=X_test[:5]
                )
                
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
        logging.error(f"Regression evaluation failed: {e}")
        raise


def evaluate_and_register_classification():
    """Evaluate and register classification model."""
    try:
        logging.info("Starting classification evaluation...")
        
        # Load model and data
        model = load_model("classification")
        X_test, y_test = load_test_data(PROCESSED_DATA_DIR, "y_clf.csv")
        
        # Initialize MLflow
        mlflow_config = get_mlflow_config()
        mlflow_config.get_or_create_experiment(EXPERIMENT_CLASSIFICATION)
        mlflow.set_experiment(EXPERIMENT_CLASSIFICATION)
        
        with mlflow.start_run(run_name="Final_Evaluation_Test_Set", nested=False) as run:
            mlflow.set_tag("stage", "final_evaluation")
            mlflow.set_tag("data", "test_set")
            
            # Evaluate
            metrics_dict = evaluate_classification_model(model, X_test, y_test, X_test, y_test, "final_classifier")
            test_metrics = metrics_dict['test']
            
            # Log metrics
            mlflow_config.log_metrics_from_dict(test_metrics, prefix="test_")
            
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
            
            logging.info(f"Classification Test Metrics: {test_metrics}")
            
            # Check threshold for registration
            accuracy = test_metrics['accuracy']
            if accuracy > CLASSIFICATION_ACCURACY_THRESHOLD:
                logging.info(f"Accuracy {accuracy:.4f} > Threshold {CLASSIFICATION_ACCURACY_THRESHOLD}. Registering model...")
                
                # We need to log the model again in this run to register it from this run
                mlflow_config.log_model_with_signature(
                    model, 
                    "final_model", 
                    X_sample=X_test[:5]
                )
                
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
        logging.error(f"Classification evaluation failed: {e}")
        raise


if __name__ == "__main__":
    try:
        # Check if models exist
        reg_model_exists = os.path.exists(os.path.join(ARTIFACTS_DIR, "models", "final_regressor.pkl"))
        clf_model_exists = os.path.exists(os.path.join(ARTIFACTS_DIR, "models", "final_classifier.pkl"))
        
        if reg_model_exists:
            evaluate_and_register_regression()
        else:
            logging.warning("Regression model not found. Skipping evaluation.")
            
        if clf_model_exists:
            evaluate_and_register_classification()
        else:
            logging.warning("Classification model not found. Skipping evaluation.")
            
        print("\n✅ Evaluation and registration completed!")
        
    except Exception as e:
        logging.error(f"Evaluation pipeline failed: {e}")
        sys.exit(1)
