# src/components/data_cleaning.py
from pathlib import Path
from typing import Optional, Dict
import sys

import numpy as np
import pandas as pd

from src.logger import logging
from src.exception import CustomException
from src.config import INTERIM_DATA_DIR, EMI_RULES
from src.utils import ensure_parent_dir, extract_number, to_numeric, save_csv


class DataCleaner:
    """
    Simple DataCleaner class implementing the cleaning steps you had in the notebook.
    Usage:
        cleaner = DataCleaner(emirules=EMI_RULES)
        clean_df = cleaner.clean(raw_df, save_snapshot=True, snapshot_dir="artifacts/processed")
    """
    def __init__(self, emirules: Optional[Dict] = None):
        self.rules = emirules or EMI_RULES

    # ---------- steps ----------
    def drop_row_na(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Dropping rows with missing values.")
        return df.dropna(axis=0).copy()

    def normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = df.columns.str.replace(r"\s+", "_", regex=True).str.title()
        logging.info("Normalized column names.")
        return df

    def parse_and_cast(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "age" in [c.lower() for c in df.columns]:
            # assume 'Age' after normalization; handle both cases
            if "Age" in df.columns:
                df["Age"] = extract_number(df["Age"])
                df["Age"] = to_numeric(df["Age"])
                logging.info("Parsed 'Age' to numeric.")

        if "bank_balance" in [c.lower() for c in df.columns]:
            if "Bank_Balance" in df.columns:
                df["Bank_Balance"] = extract_number(df["Bank_Balance"])
                df["Bank_Balance"] = to_numeric(df["Bank_Balance"])
                logging.info("Parsed 'Bank_Balance' to numeric.")

        if "monthly_salary" in [c.lower() for c in df.columns]:
            if "Monthly_Salary" in df.columns:
                df["Monthly_Salary"] = to_numeric(df["Monthly_Salary"])
                logging.info("Converted 'Monthly_Salary' to numeric.")

        if "Gender" in df.columns:
            df["Gender"] = df["Gender"].astype(str).str.upper().str.strip().str[0]
            logging.info("Normalized 'Gender'.")

        return df

    def validate_and_cap_emi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cap Requested_Amount and Requested_Tenure to scenario bounds where needed.
        """
        df = df.copy()
        required = {"Emi_Scenario", "Requested_Amount", "Requested_Tenure"}
        if not required.issubset(df.columns):
            logging.warning("Skipping EMI validation — required columns missing.")
            return df

        for scen, rule in self.rules.items():
            mask = df["Emi_Scenario"] == scen
            if not mask.any():
                continue

            # Amount
            amt_min, amt_max = rule["amount_min"], rule["amount_max"]
            too_low = mask & (df["Requested_Amount"] < amt_min)
            too_high = mask & (df["Requested_Amount"] > amt_max)
            if too_low.any() or too_high.any():
                logging.info(f"{scen}: capping Requested_Amount low={too_low.sum()}, high={too_high.sum()}")
                df.loc[too_low, "Requested_Amount"] = amt_min
                df.loc[too_high, "Requested_Amount"] = amt_max

            # Tenure
            t_min, t_max = rule["tenure_min"], rule["tenure_max"]
            t_low = mask & (df["Requested_Tenure"] < t_min)
            t_high = mask & (df["Requested_Tenure"] > t_max)
            if t_low.any() or t_high.any():
                logging.info(f"{scen}: capping Requested_Tenure low={t_low.sum()}, high={t_high.sum()}")
                df.loc[t_low, "Requested_Tenure"] = t_min
                df.loc[t_high, "Requested_Tenure"] = t_max

        return df

    def treat_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "Credit_Score" in df.columns:
            df["Credit_Score"] = df["Credit_Score"].clip(300, 850)
        if "Years_Of_Employment" in df.columns:
            df["Years_Of_Employment"] = df["Years_Of_Employment"].clip(lower=0)
        if "Requested_Tenure" in df.columns:
            df["Requested_Tenure"] = df["Requested_Tenure"].clip(1, 120)
        if "Max_Monthly_Emi" in df.columns:
            df["Max_Monthly_Emi"] = df["Max_Monthly_Emi"].clip(500, 50_000)

        perc_cols = [
            "Monthly_Salary", "Monthly_Rent", "College_Fees",
            "Travel_Expenses", "Groceries_Utilities", "Other_Monthly_Expenses",
            "Current_Emi_Amount", "Bank_Balance", "Emergency_Fund",
            "Requested_Amount", "Max_Monthly_Emi"
        ]
        for col in perc_cols:
            if col not in df.columns:
                continue
            upper = df[col].quantile(0.99)
            df[col] = np.where(df[col] > upper, upper, df[col])
            df[col] = df[col].clip(lower=0)
            logging.debug(f"Capped {col} at 99th percentile -> {upper}")

        logging.info("Outlier treatment applied.")
        return df

    def initiate_data_cleaning(self, df: pd.DataFrame, save_snapshot: bool = False, snapshot_dir: Optional[str] = None) -> pd.DataFrame:
        """
        Run end-to-end cleaning pipeline (returns cleaned df).
        If save_snapshot=True the cleaned CSV will be written to snapshot_dir or PROCESSED_DATA_DIR.
        """
        try:
            logging.info(f"Starting cleaning | input shape: {df.shape}")

            out = df.copy()
            logging.info(f"Start dataset cleaning...")
            logging.info(f"Shape before cleaning: {out.shape}")
            out = self.drop_row_na(out)
            out = self.normalize_column_names(out)
            out = self.parse_and_cast(out)
            out = out.dropna(axis=0) # drop rows created NA after casting
            out = self.validate_and_cap_emi(out)
            out = self.treat_outliers(out)
            logging.info(f"Shape after cleaning: {out.shape}")

            if save_snapshot:
                snap_dir = Path(snapshot_dir) if snapshot_dir else Path(INTERIM_DATA_DIR)
                snap_path = str(snap_dir / "cleaned_data.csv")
                save_csv(out, snap_path)

            logging.info(f"Cleaning completed...")
            return out

        except Exception as e:
            raise CustomException(e, sys) from e
    
    def load_ingested_data(self, artifacts_dir="artifacts"):
        """Load the ingested raw data from artifacts folder."""
        try:
            from pathlib import Path
            ingested_path = Path(artifacts_dir) / "ingested_raw.csv"
            
            if not ingested_path.exists():
                raise FileNotFoundError(f"Ingested data not found at {ingested_path}. Please run data_ingestion.py first.")
            
            logging.info(f"Loading ingested data from {ingested_path}")
            df = pd.read_csv(ingested_path, low_memory=False)
            logging.info(f"Loaded ingested data | Shape: {df.shape}")
            
            return df
            
        except Exception as e:
            raise CustomException(e, sys) from e


if __name__ == "__main__":
    try:
        cleaner = DataCleaner()
        
        # Load ingested data
        raw_df = cleaner.load_ingested_data()
        
        # Clean it
        cleaned_df = cleaner.initiate_data_cleaning(raw_df, save_snapshot=True)
        
        print(f"Data cleaning completed. Shape: {cleaned_df.shape}")
    except Exception as e:
        print(f"Error: {e}")
