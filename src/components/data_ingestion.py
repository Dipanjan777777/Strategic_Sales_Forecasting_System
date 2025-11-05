import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        try:
            df = pd.read_csv('notebooks/dataset/restaurant_sales_data.csv')
            logging.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            ## Basic preprocessing steps must be done here
            logging.info("Basic preprocessing initiated")
            ## Fix the date conversion
            df['date'] = pd.to_datetime(df['date'])
            df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
            ## Aggregate sales by date
            df = df.groupby('date')['quantity_sold'].sum().reset_index()
            # Renaming columns to 'ds' and 'y' for Prophet
            df = df.rename(columns={'date': 'ds', 'quantity_sold': 'y'}) 
            logging.info("Basic preprocessing done")
            
            logging.info("Train test split initiated")
            # Time series split: Use last 30 days for test
            test_size = 30
            if len(df) <= test_size:
                raise CustomException(f"Not enough data for test split. Need > {test_size} rows, but got {len(df)}", sys)

            train_set = df.iloc[:-test_size]
            test_set = df.iloc[-test_size:]

            # saving the train and test files
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info(f"Train set saved to: {self.ingestion_config.train_data_path}")
            logging.info(f"Test set saved to: {self.ingestion_config.test_data_path}")
            logging.info("Data ingestion completed")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.info("Exception occurred in the data ingestion method")
            raise CustomException(e, sys)

# This block allows you to run this script directly for testing
if __name__ == "__main__":
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()
    print(f"Data ingestion complete. Train file at: {train_path}, Test file at: {test_path}")       