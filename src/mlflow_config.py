"""
MLflow Configuration Module for DagsHub Integration

This module handles MLflow setup, experiment management, and logging utilities
for tracking machine learning experiments with DagsHub.
"""

import os
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file in project root
project_root = Path(__file__).parent.parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)


class MLflowConfig:
    """MLflow configuration and helper utilities for DagsHub integration."""
    
    def __init__(self):
        """Initialize MLflow configuration with DagsHub credentials."""
        self.dagshub_username = os.getenv("DAGSHUB_USERNAME")
        self.dagshub_repo = os.getenv("DAGSHUB_REPO_NAME")
        self.dagshub_token = os.getenv("DAGSHUB_TOKEN")
        self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        
        # Validate credentials
        if not all([self.dagshub_username, self.dagshub_repo, self.dagshub_token, self.tracking_uri]):
            raise ValueError(
                "Missing DagsHub credentials. Please ensure .env file contains: "
                "DAGSHUB_USERNAME, DAGSHUB_REPO_NAME, DAGSHUB_TOKEN, MLFLOW_TRACKING_URI"
            )
        
        # Set up MLflow
        self.setup_mlflow()
    
    def setup_mlflow(self):
        """Configure MLflow tracking URI and authentication."""
        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Set DagsHub credentials for authentication
        os.environ['MLFLOW_TRACKING_USERNAME'] = self.dagshub_username
        os.environ['MLFLOW_TRACKING_PASSWORD'] = self.dagshub_token
        
        print(f"[OK] MLflow configured with DagsHub tracking URI: {self.tracking_uri}")
    
    def get_or_create_experiment(self, experiment_name):
        """
        Get existing experiment or create new one.
        
        Args:
            experiment_name (str): Name of the experiment
            
        Returns:
            str: Experiment ID
        """
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f"[OK] Created new experiment: {experiment_name} (ID: {experiment_id})")
        else:
            experiment_id = experiment.experiment_id
            print(f"[OK] Using existing experiment: {experiment_name} (ID: {experiment_id})")
        
        return experiment_id
    
    @staticmethod
    def log_params_from_dict(params_dict):
        """
        Log parameters from a dictionary.
        
        Args:
            params_dict (dict): Dictionary of parameters to log
        """
        for key, value in params_dict.items():
            try:
                mlflow.log_param(key, value)
            except Exception as e:
                print(f"Warning: Could not log parameter {key}: {e}")
    
    @staticmethod
    def log_metrics_from_dict(metrics_dict, prefix=""):
        """
        Log metrics from a dictionary.
        
        Args:
            metrics_dict (dict): Dictionary of metrics to log
            prefix (str): Optional prefix for metric names (e.g., 'train_', 'val_')
        """
        for key, value in metrics_dict.items():
            try:
                metric_name = f"{prefix}{key}" if prefix else key
                mlflow.log_metric(metric_name, value)
            except Exception as e:
                print(f"Warning: Could not log metric {key}: {e}")
    
    @staticmethod
    def log_model_with_signature(model, artifact_path, X_sample=None):
        """
        Log model with input/output signature.
        
        Args:
            model: Trained sklearn model
            artifact_path (str): Path to store model artifact
            X_sample: Sample input data for signature inference
        """
        try:
            if X_sample is not None:
                from mlflow.models.signature import infer_signature
                signature = infer_signature(X_sample, model.predict(X_sample))
                mlflow.sklearn.log_model(model, artifact_path, signature=signature)
            else:
                mlflow.sklearn.log_model(model, artifact_path)
            print(f"[OK] Model logged to artifact path: {artifact_path}")
        except Exception as e:
            print(f"Warning: Could not log model: {e}")
    
    @staticmethod
    def register_model(model_uri, model_name, tags=None):
        """
        Register model to MLflow Model Registry.
        
        Args:
            model_uri (str): URI of the model (e.g., 'runs:/<run_id>/model')
            model_name (str): Name for the registered model
            tags (dict): Optional tags for the model version
            
        Returns:
            ModelVersion: Registered model version
        """
        try:
            model_version = mlflow.register_model(model_uri, model_name)
            print(f"[OK] Model registered: {model_name} (Version: {model_version.version})")
            
            # Add tags if provided
            if tags:
                client = mlflow.tracking.MlflowClient()
                for key, value in tags.items():
                    client.set_model_version_tag(
                        model_name, 
                        model_version.version, 
                        key, 
                        str(value)
                    )
            
            return model_version
        except Exception as e:
            print(f"Warning: Could not register model: {e}")
            return None


# Experiment names
EXPERIMENT_REGRESSION = "EMI_Regression"
EXPERIMENT_CLASSIFICATION = "EMI_classification"

# Model names for registry
MODEL_NAME_REGRESSION = "emi_max_monthly_predictor"
MODEL_NAME_CLASSIFICATION = "emi_eligibility_classifier"


def get_mlflow_config():
    """
    Get MLflow configuration instance.
    
    Returns:
        MLflowConfig: Configured MLflow instance
    """
    return MLflowConfig()
