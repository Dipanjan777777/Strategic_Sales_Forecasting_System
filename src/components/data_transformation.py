import os
import sys
import pandas as pd
import numpy as np
from scipy import stats
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object  

@dataclass
class DataTransformationConfig:
    # We are not saving any CSV files, so this is empty
    pass

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self, train_path, test_path):
        '''
        This function applies outlier removal to the training data
        and returns the cleaned train_df and original test_df in memory.
        '''
        logging.info("Entered data transformation method")
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info(f"Read train data ({len(train_df)} rows) and test data ({len(test_df)} rows)")


            # This is only applied to the training set to improve tuning.
            logging.info("Applying z-score outlier removal to training data")
            
            df_clean = train_df.copy()
            # Use 'y' as defined in your data_ingestion.py
            z = np.abs(stats.zscore(df_clean['y'])) 
            outlier_index = np.where(z > 2.7)[0] # 1 ~68%, 2 ~95%, 2.7 ~99%

            logging.info(f"Total training data points: {len(df_clean)}")
            logging.info(f"Dropping {len(outlier_index)} outlier rows (z-score > 2.7)")

            train_df_cleaned = df_clean.drop(df_clean.index[outlier_index]).reset_index(drop=True)
            logging.info(f"Remaining training data points: {len(train_df_cleaned)}")
            logging.info("Data transformation completed.")

            # Return both dataframes
            return (
                train_df_cleaned,
                test_df 
            )

        except Exception as e:
            raise CustomException(e, sys)

# This block allows you to run this script directly for testing
if __name__ == "__main__":
    # Add the project root to the Python path
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    from src.components.data_ingestion import DataIngestion

    # First, run data ingestion to get the initial train/test split
    ingestion_obj = DataIngestion()
    train_path, test_path = ingestion_obj.initiate_data_ingestion()


    # Now, run data transformation
    logging.info("Running Data Transformation...")
    transformation_obj = DataTransformation()
    
    # FIX 3: Pass both paths and accept both returned dataframes
    train_df, test_df = transformation_obj.initiate_data_transformation(train_path, test_path)
    