# src/components/feature_engineering.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler, LabelEncoder
import joblib
import os
import sys

from src.config import (
    NUMERIC_COLS, CATEGORICAL_COLS, COLS_TO_DROP,
    TARGET_REG, TARGET_CLF, TEMP_FRACTION, TEST_FRACTION,
    RANDOM_STATE, ARTIFACTS_DIR
)


from src.logger import logging
from src.config import PROCESSED_DATA_DIR
from src.exception import CustomException


class FeatureEngineer:
    """
    Very simple, beginner-friendly feature engineering class.
    Handles:
      - Creating engineered features
      - Dropping redundant columns
      - 70/30 then 50/50 train-val-test split
      - Fit & transform OHE + Scaler
    """

    def __init__(self):
        self.ohe = None
        self.scaler = None
        self.label_encoder = None
        self.num_cols = []
        self.cat_cols = []


    # ----------------------------------------------------------
    # 1. CREATE ROW-WISE FEATURES  (Safe before split)
    # ----------------------------------------------------------
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Creating row-wise engineered features...")
        df = df.copy()

        expense_cols = [
            "Monthly_Rent","School_Fees","College_Fees",
            "Travel_Expenses","Groceries_Utilities","Other_Monthly_Expenses"
        ]
        expense_cols = [c for c in expense_cols if c in df.columns]

        df["Total_Expenses"] = df[expense_cols].sum(axis=1)

        df["Expense_Salary_Ratio"] = df["Total_Expenses"] / (df["Monthly_Salary"] + 1)
        df["Emi_Salary_Ratio"] = df["Current_Emi_Amount"] / (df["Monthly_Salary"] + 1)
        df["Liquidity_Score"] = (df["Bank_Balance"] + df["Emergency_Fund"]) / (df["Monthly_Salary"] + 1)
        df["Household_Load"] = df["Family_Size"] + 1.5 * df["Dependents"]
        df["Requested_Amount_Ratio"] = df["Requested_Amount"] / (df["Monthly_Salary"] + 1)
        df["Expected_Emi_If_Approved"] = df["Requested_Amount"] / (df["Requested_Tenure"] + 1)

        return df


    # ----------------------------------------------------------
    # 2. DROP REDUNDANT COLUMNS  (Safe before split)
    # ----------------------------------------------------------
    def drop_redundant(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Dropping redundant columns: {COLS_TO_DROP}")
        cols_present = [c for c in COLS_TO_DROP if c in df.columns]
        return df.drop(columns=cols_present)


    # ----------------------------------------------------------
    # 3. TRAIN / VAL / TEST SPLIT (70/30 → 50/50 temp split)
    # ----------------------------------------------------------
    def split_data(self, df: pd.DataFrame):
        logging.info("Performing 70/15/15 split...")

        # --- First split: 70% train, 30% temp ---
        train_df, temp_df = train_test_split(
            df,
            test_size=TEMP_FRACTION,
            random_state=RANDOM_STATE,
            shuffle=True
        )

        # --- Second split: temp → 50% val, 50% test ---
        val_df, test_df = train_test_split(
            temp_df,
            test_size=TEST_FRACTION,
            random_state=RANDOM_STATE,
            shuffle=True
        )

        logging.info(f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")

        return train_df, val_df, test_df


    # ----------------------------------------------------------
    # 4. FIT SCALER + OHE ON TRAIN ONLY
    # ----------------------------------------------------------
    def fit(self, X_train: pd.DataFrame):
        logging.info("Fitting encoders & scaler on TRAIN ONLY...")

        self.num_cols = [c for c in NUMERIC_COLS if c in X_train.columns]
        self.cat_cols = [c for c in CATEGORICAL_COLS if c in X_train.columns]

        # Fit OHE
        if self.cat_cols:
            self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            self.ohe.fit(X_train[self.cat_cols])

        # Fit Scaler
        if self.num_cols:
            self.scaler = RobustScaler()
            self.scaler.fit(X_train[self.num_cols])

        logging.info("Fitting complete.")


    # ----------------------------------------------------------
    # 5. TRANSFORM (TRAIN / VAL / TEST)
    # ----------------------------------------------------------
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logging.info("Transforming dataset...")
        X = X.copy()

        parts = []

        # numeric scaled
        if self.scaler and self.num_cols:
            scaled = self.scaler.transform(X[self.num_cols])
            parts.append(pd.DataFrame(scaled, columns=self.num_cols, index=X.index))
        else:
            parts.append(X[self.num_cols])

        # one-hot categorical
        if self.ohe and self.cat_cols:
            ohe_arr = self.ohe.transform(X[self.cat_cols])
            ohe_cols = list(self.ohe.get_feature_names_out(self.cat_cols))
            parts.append(pd.DataFrame(ohe_arr, columns=ohe_cols, index=X.index))
        else:
            parts.append(X[self.cat_cols])

        # remainder columns (neither numeric nor categorical)
        remainder = [c for c in X.columns if c not in self.num_cols + self.cat_cols]
        if remainder:
            parts.append(X[remainder])

        final_df = pd.concat(parts, axis=1)
        return final_df


    # ----------------------------------------------------------
    # 6. FULL PIPELINE CALL (optional helper)
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    # 6. FULL PIPELINE CALL (optional helper)
    # ----------------------------------------------------------
    def run(self, df: pd.DataFrame):
        from src.utils import save_csv
        logging.info("Running full Feature Engineering pipeline...")

        # 1. create features
        df = self.create_features(df)

        # 2. drop redundant
        df = self.drop_redundant(df)
        
        # Save feature-engineered dataset before splitting
        fe_dataset_path = os.path.join("Data/Interim", "FE_Dataset.csv")
        save_csv(df, fe_dataset_path)

        # 3. split (X still contains both targets)
        train_df, val_df, test_df = self.split_data(df)

        # separate X/y for both targets
        X_train = train_df.drop(columns=[TARGET_REG, TARGET_CLF])
        y_train_reg = train_df[TARGET_REG]
        y_train_clf = train_df[TARGET_CLF]

        X_val = val_df.drop(columns=[TARGET_REG, TARGET_CLF])
        y_val_reg = val_df[TARGET_REG]
        y_val_clf = val_df[TARGET_CLF]

        X_test = test_df.drop(columns=[TARGET_REG, TARGET_CLF])
        y_test_reg = test_df[TARGET_REG]
        y_test_clf = test_df[TARGET_CLF]
        
        # Encode classification target if it's string
        if y_train_clf.dtype == 'object' or y_train_clf.dtype == 'O':
            logging.info("Encoding classification target labels...")
            self.label_encoder = LabelEncoder()
            y_train_clf = pd.Series(self.label_encoder.fit_transform(y_train_clf), index=y_train_clf.index)
            y_val_clf = pd.Series(self.label_encoder.transform(y_val_clf), index=y_val_clf.index)
            y_test_clf = pd.Series(self.label_encoder.transform(y_test_clf), index=y_test_clf.index)
            logging.info(f"Label mapping: {dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}")

        # 4. fit on TRAIN ONLY
        self.fit(X_train)

        # 5. transform all
        X_train_p = self.transform(X_train)
        X_val_p = self.transform(X_val)
        X_test_p = self.transform(X_test)
        
        # Save preprocessor
        self.save_preprocessor()

        return {
            "train": (X_train_p, y_train_reg, y_train_clf),
            "val": (X_val_p, y_val_reg, y_val_clf),
            "test": (X_test_p, y_test_reg, y_test_clf),
        }
    
    def save_preprocessor(self, save_dir=None):
        """Save the preprocessor (scaler, ohe, label_encoder) to disk."""
        try:
            if save_dir is None:
                save_dir = os.path.join(ARTIFACTS_DIR, "preprocessor")
            
            os.makedirs(save_dir, exist_ok=True)
            
            preprocessor = {
                'scaler': self.scaler,
                'ohe': self.ohe,
                'label_encoder': self.label_encoder,
                'num_cols': self.num_cols,
                'cat_cols': self.cat_cols
            }
            
            preprocessor_path = os.path.join(save_dir, "preprocessor.pkl")
            joblib.dump(preprocessor, preprocessor_path)
            
            logging.info(f"Preprocessor saved to {preprocessor_path}")
            
            return preprocessor_path
            
        except Exception as e:
            logging.error(f"Failed to save preprocessor: {str(e)}")
            raise CustomException(e, sys) from e
    
    def load_cleaned_data(self, interim_dir="Data/Interim"):
        """Load the cleaned data from Interim folder."""
        try:
            cleaned_path = os.path.join(interim_dir, "cleaned_data.csv")
            
            if not os.path.exists(cleaned_path):
                raise FileNotFoundError(f"Cleaned data not found at {cleaned_path}. Please run data_cleaning.py first.")
            
            logging.info(f"Loading cleaned data from {cleaned_path}")
            df = pd.read_csv(cleaned_path)
            logging.info(f"Loaded cleaned data | Shape: {df.shape}")
            
            return df
            
        except Exception as e:
            logging.error(f"Failed to load cleaned data: {str(e)}")
            raise CustomException(e, sys) from e


if __name__ == "__main__":
    try:
        from src.utils import save_csv
        fe = FeatureEngineer()
        
        # Load cleaned data
        cleaned_df = fe.load_cleaned_data()
        
        # Run feature engineering
        data_splits = fe.run(cleaned_df)
        
        # Save splits
        for split_name, (X, y_reg, y_clf) in data_splits.items():
            save_dir = os.path.join(PROCESSED_DATA_DIR, split_name)
            os.makedirs(save_dir, exist_ok=True)
            
            save_csv(X, os.path.join(save_dir, "X.csv"))
            save_csv(y_reg, os.path.join(save_dir, "y_reg.csv"))
            save_csv(y_clf, os.path.join(save_dir, "y_clf.csv"))
            
            print(f"Saved {split_name} split - X: {X.shape}, y_reg: {y_reg.shape}, y_clf: {y_clf.shape}")
        
        print("Feature engineering pipeline completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
