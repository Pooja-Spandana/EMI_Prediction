import os
import sys
import json
import datetime
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn

from src.logger import logging
from src.exception import CustomException
from src.config import RANDOM_STATE, BEST_PARAMS_DIR
from src.utils import evaluate_classification_model
from src.mlflow_config import (
    get_mlflow_config, 
    EXPERIMENT_CLASSIFICATION
)


class ClassificationTrainer:
    """
    Trains multiple classification models with MLflow tracking.
    Supports SMOTE for imbalanced data and hyperparameter tuning with RandomizedSearchCV.
    """
    
    def __init__(self, use_mlflow=True):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.use_mlflow = use_mlflow
        self.mlflow_config = None
        self.experiment_id = None
        
        # Initialize MLflow if enabled
        if self.use_mlflow:
            try:
                self.mlflow_config = get_mlflow_config()
                self.experiment_id = self.mlflow_config.get_or_create_experiment(EXPERIMENT_CLASSIFICATION)
                mlflow.set_experiment(EXPERIMENT_CLASSIFICATION)
                logging.info("MLflow tracking enabled for classification training")
            except Exception as e:
                logging.warning(f"MLflow initialization failed: {e}. Continuing without MLflow tracking.")
                self.use_mlflow = False
        
    def load_data(self, data_dir):
        """Load train, val, test data from processed directory."""
        try:
            logging.info(f"Loading data from {data_dir}")
            
            X_train = pd.read_csv(os.path.join(data_dir, "train", "X.csv"))
            y_train = pd.read_csv(os.path.join(data_dir, "train", "y_clf.csv")).values.ravel()
            
            X_val = pd.read_csv(os.path.join(data_dir, "val", "X.csv"))
            y_val = pd.read_csv(os.path.join(data_dir, "val", "y_clf.csv")).values.ravel()
            
            X_test = pd.read_csv(os.path.join(data_dir, "test", "X.csv"))
            y_test = pd.read_csv(os.path.join(data_dir, "test", "y_clf.csv")).values.ravel()
            
            logging.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
            logging.info(f"Target dtype: {y_train.dtype}, unique values: {np.unique(y_train)}")
            
            return X_train, y_train, X_val, y_val, X_test, y_test
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def apply_smote(self, X_train, y_train):
        """Apply SMOTE to handle class imbalance."""
        try:
            logging.info("Applying SMOTE to handle class imbalance...")
            logging.info(f"Before SMOTE - Class distribution: {np.bincount(y_train)}")
            
            smote = SMOTE(random_state=RANDOM_STATE)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train) # type: ignore
            
            logging.info(f"After SMOTE - Class distribution: {np.bincount(y_resampled)}")
            logging.info(f"Training data shape after SMOTE: {X_resampled.shape}")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def train_baseline_models(self, X_train, y_train, X_val, y_val):
        """Train baseline models without tuning."""
        try:
            logging.info("Training baseline classification models...")
            
            # Apply SMOTE
            X_train_resampled, y_train_resampled = self.apply_smote(X_train, y_train)
            
            # Define models
            models = {
                "Logistic_Regression": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
                "LightGBM": LGBMClassifier(random_state=RANDOM_STATE, n_estimators=100, verbose=-1),
                "XGBoost": XGBClassifier(random_state=RANDOM_STATE, eval_metric='mlogloss'),
                "Gradient_Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE)
            }
            
            # Start parent MLflow run for baseline models
            if self.use_mlflow:
                parent_run = mlflow.start_run(run_name="Baseline_Models", nested=False)
                mlflow.set_tag("stage", "baseline")
                mlflow.set_tag("model_count", len(models))
                mlflow.set_tag("smote_applied", "True")
                
                # Log SMOTE info
                mlflow.log_param("smote_strategy", "auto")
                mlflow.log_param("original_train_size", len(y_train))
                mlflow.log_param("resampled_train_size", len(y_train_resampled))
            
            for model_name, model in models.items():
                logging.info(f"Training {model_name}...")
                
                # Start child run for each model
                if self.use_mlflow:
                    child_run = mlflow.start_run(run_name=model_name, nested=True)
                    mlflow.set_tag("model_type", model_name)
                    mlflow.set_tag("tuned", "False")
                
                try:
                    # Train on SMOTE-resampled data
                    model.fit(X_train_resampled, y_train_resampled)
                    
                    # Log model parameters
                    if self.use_mlflow and self.mlflow_config:
                        params = model.get_params()
                        self.mlflow_config.log_params_from_dict(params)
                    
                    # Evaluate on validation (original, not resampled)
                    metrics_dict = evaluate_classification_model(model, X_train_resampled, y_train_resampled, X_val, y_val, model_name)
                    
                    # Log metrics to MLflow
                    if self.use_mlflow and self.mlflow_config:
                        self.mlflow_config.log_metrics_from_dict(metrics_dict['train'], prefix="train_")
                        self.mlflow_config.log_metrics_from_dict(metrics_dict['test'], prefix="val_")
                        
                        # Log model artifact
                        self.mlflow_config.log_model_with_signature(
                            model, 
                            f"model_{model_name}", 
                            X_sample=X_val[:5]
                        )
                    
                    # Store results (use test metrics which are validation metrics here)
                    self.models[model_name] = model
                    self.results[model_name] = metrics_dict['test']
                    
                finally:
                    if self.use_mlflow:
                        mlflow.end_run()  # End child run
                        
            if self.use_mlflow:
                mlflow.end_run()  # End parent run
                    
            logging.info("Baseline models training completed.")
            
        except Exception as e:
            if self.use_mlflow and mlflow.active_run():
                mlflow.end_run(status="FAILED")
            raise CustomException(e, sys)
    
    def tune_top_models(self, X_train, y_train, X_val, y_val, top_n=2):
        """Hyperparameter tuning for top N models using RandomizedSearchCV."""
        try:
            logging.info(f"Tuning top {top_n} models...")
            
            # Apply SMOTE
            X_train_resampled, y_train_resampled = self.apply_smote(X_train, y_train)
            
            # Sort models by F1 score
            sorted_models = sorted(self.results.items(), key=lambda x: x[1]['f1_score'], reverse=True)
            top_models = [name for name, _ in sorted_models[:top_n]]
            
            logging.info(f"Top models for tuning: {top_models}")
            
            # Define expanded hyperparameter distributions
            param_distributions = {
                "LightGBM": {
                    'n_estimators': [100, 200, 300, 500],
                    'max_depth': [10, 20, 30, -1],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'num_leaves': [31, 50, 70, 100],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0]
                },
                "XGBoost": {
                    'n_estimators': [100, 200, 300, 500],
                    'max_depth': [3, 5, 7, 9],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0]
                },
                "Gradient_Boosting": {
                    'n_estimators': [100, 200, 300, 500],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'subsample': [0.6, 0.8, 1.0],
                    'min_samples_split': [2, 5, 10]
                },
                "Logistic_Regression": {
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            }
            
            # Start parent MLflow run for hyperparameter tuning
            if self.use_mlflow:
                parent_run = mlflow.start_run(run_name="Hyperparameter_Tuning", nested=False)
                mlflow.set_tag("stage", "tuning")
                mlflow.set_tag("top_n", top_n)
            
            for model_name in top_models:
                if model_name not in param_distributions:
                    logging.warning(f"No param distribution for {model_name}, skipping tuning.")
                    continue
                
                logging.info(f"Tuning {model_name}...")
                
                # Start child run for tuning
                if self.use_mlflow:
                    child_run = mlflow.start_run(run_name=f"{model_name}_Tuned", nested=True)
                    mlflow.set_tag("model_type", model_name)
                    mlflow.set_tag("tuned", "True")
                    
                    # Log search space
                    mlflow.log_param("search_space", str(param_distributions[model_name]))
                    mlflow.log_param("n_iter", 20)
                    mlflow.log_param("cv_folds", 3)
                
                try:
                    base_model = self.models[model_name]
                    
                    # RandomizedSearchCV
                    random_search = RandomizedSearchCV(
                        base_model,
                        param_distributions[model_name],
                        n_iter=20,
                        cv=3,
                        scoring='f1_weighted',
                        n_jobs=-1,
                        verbose=1,
                        random_state=RANDOM_STATE
                    )
                    
                    random_search.fit(X_train_resampled, y_train_resampled) # type: ignore
                    
                    # Best model
                    tuned_model = random_search.best_estimator_
                    
                    logging.info(f"Best params for {model_name}: {random_search.best_params_}")
                    
                    # Log best parameters
                    if self.use_mlflow and self.mlflow_config:
                        self.mlflow_config.log_params_from_dict(random_search.best_params_)
                        mlflow.log_metric("cv_best_score", random_search.best_score_)
                    
                    # Evaluate
                    metrics_dict = evaluate_classification_model(tuned_model, X_train_resampled, y_train_resampled, X_val, y_val, f"{model_name}_tuned")
                    
                    # Log metrics
                    if self.use_mlflow and self.mlflow_config:
                        self.mlflow_config.log_metrics_from_dict(metrics_dict['train'], prefix="train_")
                        self.mlflow_config.log_metrics_from_dict(metrics_dict['test'], prefix="val_")
                        
                        # Log tuned model
                        self.mlflow_config.log_model_with_signature(
                            tuned_model, 
                            f"model_{model_name}_tuned", 
                            X_sample=X_val[:5]
                        )
                    
                    # Update if better (use test metrics which are validation metrics here)
                    if metrics_dict['test']['f1_score'] > self.results[model_name]['f1_score']:
                        self.models[model_name] = tuned_model
                        self.results[model_name] = metrics_dict['test']
                        logging.info(f"{model_name} improved after tuning!")
                        if self.use_mlflow:
                            mlflow.set_tag("improved", "True")
                    else:
                        logging.info(f"{model_name} did not improve after tuning.")
                        if self.use_mlflow:
                            mlflow.set_tag("improved", "False")
                
                finally:
                    if self.use_mlflow:
                        mlflow.end_run()  # End child run
                        
            if self.use_mlflow:
                mlflow.end_run()  # End parent run
                    
            logging.info("Hyperparameter tuning completed.")
            
        except Exception as e:
            if self.use_mlflow and mlflow.active_run():
                mlflow.end_run(status="FAILED")
            raise CustomException(e, sys)
    
    def select_best_model(self):
        """Select the best model based on F1 score."""
        try:
            best_name = max(self.results, key=lambda x: self.results[x]['f1_score'])
            self.best_model = self.models[best_name]
            self.best_model_name = best_name
            
            logging.info(f"Best model: {best_name} with F1: {self.results[best_name]['f1_score']:.4f}")
            
            return self.best_model, self.best_model_name
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def save_best_params(self):
        """Save the best model parameters to JSON."""
        try:
            os.makedirs(BEST_PARAMS_DIR, exist_ok=True)
            
            # Get best model parameters
            if self.best_model is not None and hasattr(self.best_model, 'get_params'):
                params = self.best_model.get_params()
            else:
                params = {}
                
            # Create params dictionary
            params_data = {
                "model_name": self.best_model_name,
                "parameters": params,
                "validation_metrics": self.results[self.best_model_name],
                "smote_applied": True,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            # Save to JSON
            params_path = os.path.join(BEST_PARAMS_DIR, "classification_best_params.json")
            with open(params_path, 'w') as f:
                json.dump(params_data, f, indent=4)
            
            logging.info(f"Best parameters saved to {params_path}")
            return params_path
            
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        trainer = ClassificationTrainer(use_mlflow=True)
        
        # Load data
        X_train, y_train, X_val, y_val, X_test, y_test = trainer.load_data("Data/Processed")
        
        # Train baseline models
        trainer.train_baseline_models(X_train, y_train, X_val, y_val)
        
        # Tune top 2 models
        trainer.tune_top_models(X_train, y_train, X_val, y_val, top_n=2)
        
        # Select best
        best_model, best_name = trainer.select_best_model()
        
        # Save best parameters
        trainer.save_best_params()
        
        print(f"\nClassification hyperparameter search completed!")
        print(f"Best model: {best_name}")
        print(f"Best F1 Score (Validation): {trainer.results[best_name]['f1_score']:.4f}")
        print(f"Best Accuracy (Validation): {trainer.results[best_name]['accuracy']:.4f}")
        
        if trainer.use_mlflow:
            print(f"\n✅ View experiments at: https://dagshub.com/Pooja-Spandana/EMI_Prediction/experiments")
        
    except Exception as e:
        logging.error(f"Classification training failed: {str(e)}")
        raise
