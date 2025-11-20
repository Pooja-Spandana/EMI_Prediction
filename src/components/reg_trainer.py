import os
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
import joblib

from src.logger import logging
from src.exception import CustomException
from src.config import ARTIFACTS_DIR, RANDOM_STATE
from src.utils import evaluate_regression_model


class RegressionTrainer:
    """
    Trains multiple regression models.
    Supports hyperparameter tuning with RandomizedSearchCV.
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def load_data(self, data_dir):
        """Load train, val, test data from processed directory."""
        try:
            logging.info(f"Loading data from {data_dir}")
            
            X_train = pd.read_csv(os.path.join(data_dir, "train", "X.csv"))
            y_train = pd.read_csv(os.path.join(data_dir, "train", "y_reg.csv")).values.ravel()
            
            X_val = pd.read_csv(os.path.join(data_dir, "val", "X.csv"))
            y_val = pd.read_csv(os.path.join(data_dir, "val", "y_reg.csv")).values.ravel()
            
            X_test = pd.read_csv(os.path.join(data_dir, "test", "X.csv"))
            y_test = pd.read_csv(os.path.join(data_dir, "test", "y_reg.csv")).values.ravel()
            
            logging.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
            logging.info(f"Target dtype: {y_train.dtype}, min: {y_train.min():.2f}, max: {y_train.max():.2f}")
            
            return X_train, y_train, X_val, y_val, X_test, y_test
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def train_baseline_models(self, X_train, y_train, X_val, y_val):
        """Train baseline models without tuning."""
        try:
            logging.info("Training baseline regression models...")
            
            # Define models
            models = {
                "Linear_Regression": LinearRegression(),
                "Random_Forest": RandomForestRegressor(random_state=RANDOM_STATE, n_estimators=100),
                "XGBoost": XGBRegressor(random_state=RANDOM_STATE, eval_metric='rmse'),
                "Gradient_Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE)
            }
            
            for model_name, model in models.items():
                logging.info(f"Training {model_name}...")
                
                # Train
                model.fit(X_train, y_train)
                
                # Evaluate on validation
                val_metrics = evaluate_regression_model(model, X_val, y_val, model_name)
                
                # Store results
                self.models[model_name] = model
                self.results[model_name] = val_metrics
                    
            logging.info("Baseline models training completed.")
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def tune_top_models(self, X_train, y_train, X_val, y_val, top_n=2):
        """Hyperparameter tuning for top N models using RandomizedSearchCV."""
        try:
            logging.info(f"Tuning top {top_n} models...")
            
            # Sort models by R2 score
            sorted_models = sorted(self.results.items(), key=lambda x: x[1]['r2_score'], reverse=True)
            top_models = [name for name, _ in sorted_models[:top_n]]
            
            logging.info(f"Top models for tuning: {top_models}")
            
            # Define expanded hyperparameter distributions
            param_distributions = {
                "Random_Forest": {
                    'n_estimators': [100, 200, 300, 500],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
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
                "Linear_Regression": {
                    'fit_intercept': [True, False]
                }
            }
            
            for model_name in top_models:
                if model_name not in param_distributions:
                    logging.warning(f"No param distribution for {model_name}, skipping tuning.")
                    continue
                
                logging.info(f"Tuning {model_name}...")
                
                base_model = self.models[model_name]
                
                # RandomizedSearchCV
                random_search = RandomizedSearchCV(
                    base_model,
                    param_distributions[model_name],
                    n_iter=20,  # Number of parameter settings sampled
                    cv=3,
                    scoring='r2',
                    n_jobs=-1,
                    verbose=1,
                    random_state=RANDOM_STATE
                )
                
                random_search.fit(X_train, y_train)
                
                # Best model
                tuned_model = random_search.best_estimator_
                
                logging.info(f"Best params for {model_name}: {random_search.best_params_}")
                
                # Evaluate
                val_metrics = evaluate_regression_model(tuned_model, X_val, y_val, f"{model_name}_tuned")
                
                # Update if better
                if val_metrics['r2_score'] > self.results[model_name]['r2_score']:
                    self.models[model_name] = tuned_model
                    self.results[model_name] = val_metrics
                    logging.info(f"{model_name} improved after tuning!")
                else:
                    logging.info(f"{model_name} did not improve after tuning.")
                    
            logging.info("Hyperparameter tuning completed.")
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def select_best_model(self):
        """Select the best model based on R2 score."""
        try:
            best_name = max(self.results, key=lambda x: self.results[x]['r2_score'])
            self.best_model = self.models[best_name]
            self.best_model_name = best_name
            
            logging.info(f"Best model: {best_name} with R2: {self.results[best_name]['r2_score']:.4f}")
            
            return self.best_model, self.best_model_name
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def save_best_model(self, save_dir=None):
        """Save the best model to disk."""
        try:
            if save_dir is None:
                save_dir = os.path.join(ARTIFACTS_DIR, "models")
            
            os.makedirs(save_dir, exist_ok=True)
            
            model_path = os.path.join(save_dir, f"best_regressor_{self.best_model_name}.pkl")
            joblib.dump(self.best_model, model_path)
            
            logging.info(f"Best model saved to {model_path}")
            
            return model_path
            
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        trainer = RegressionTrainer()
        
        # Load data
        X_train, y_train, X_val, y_val, X_test, y_test = trainer.load_data("Data/Processed")
        
        # Train baseline models
        trainer.train_baseline_models(X_train, y_train, X_val, y_val)
        
        # Tune top 2 models
        trainer.tune_top_models(X_train, y_train, X_val, y_val, top_n=2)
        
        # Select best
        best_model, best_name = trainer.select_best_model()
        
        # Evaluate on test
        test_metrics = evaluate_regression_model(best_model, X_test, y_test, f"{best_name}_test")
        logging.info(f"Test metrics: {test_metrics}")
        
        # Save
        trainer.save_best_model()
        
        print(f"\nRegression training completed!")
        print(f"Best model: {best_name}")
        print(f"Test R2: {test_metrics['r2_score']:.4f}")
        print(f"Test RMSE: {test_metrics['rmse']:.2f}")
        
    except Exception as e:
        logging.error(f"Regression training failed: {str(e)}")
        raise
