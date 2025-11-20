import sys
import pandas as pd
from pathlib import Path
from typing import Optional, Union

from src.logger import logging
from src.utils import ensure_parent_dir, save_csv
from src.exception import CustomException
from src.config import RAW_DATA_DIR, ARTIFACTS_DIR


class DataIngestor:
    """
    Simple Data Ingestor class.
    Usage:
        ingestor = DataIngestor(raw_dir="Data/Raw", artifacts_dir="artifacts")
        df = ingestor.initiate_data_ingestion("emi_prediction_dataset.csv")
    """
    def __init__(self, raw_dir: Union[str, Path] = RAW_DATA_DIR, artifacts_dir: Union[str, Path] = ARTIFACTS_DIR):
        self.raw_dir = Path(raw_dir)
        self.artifacts_dir = Path(artifacts_dir)
        # ensure artifacts directory exists
        ensure_parent_dir(self.artifacts_dir / "dummy")  # creates parent if needed

    def initiate_data_ingestion(self, filename: str, nrows: Optional[int] = None) -> pd.DataFrame:
        """
        Read raw CSV from raw_dir/filename and save a snapshot artifacts/ingested_raw.csv.
        Returns the loaded DataFrame.
        """
        try:
            logging.info("Starting data ingestion...")
            raw_path = self.raw_dir / filename

            if not raw_path.exists():
                msg = f"Raw file not found: {raw_path}"
                logging.error(msg)
                raise FileNotFoundError(msg)

            df = pd.read_csv(raw_path, low_memory=False, nrows=nrows)
            logging.info(f"Loaded raw dataset from {raw_path} | Rows={df.shape[0]}, Cols={df.shape[1]}")

            # save snapshot
            snapshot_path = str(self.artifacts_dir / "ingested_raw.csv")
            save_csv(df, snapshot_path)
            logging.info(f"Saved ingested snapshot -> {snapshot_path}")

            return df

        except Exception as e:
            raise CustomException(e, sys) from e


if __name__ == "__main__":
    try:
        ingestor = DataIngestor()
        df = ingestor.initiate_data_ingestion(filename="emi_prediction_dataset.csv")
        print(f"Data ingestion completed. Shape: {df.shape}")
    except Exception as e:
        print(f"Error: {e}")
