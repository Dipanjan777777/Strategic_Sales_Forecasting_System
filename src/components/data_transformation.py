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
    cleaned_train_data_path: str = os.path.join('artifacts', 'train_cleaned.csv')
    cleaned_test_data_path: str = os.path.join('artifacts', 'test_cleaned.csv')

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self, train_path, test_path):
        '''
        This function applies outlier removal based on
        the training and test sets to prevent data leakage.
        ''' 
        logging.info("Entered data transformation method")
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info(f"Read train data ({len(train_df)} rows) and test data ({len(test_df)} rows)")

            # Outlier Removal using Z-Score 
            logging.info("Calculating outlier boundaries using training data only")

            # Calculate boundaries (mean/std) from the training data
            train_y = train_df['y']
            train_mean = train_y.mean()
            train_std = train_y.std()
            threshold = 13.59  # ~95.44% of data
            
            logging.info(f"Train data: Mean={train_mean:.2f}, Std={train_std:.2f}, Threshold={threshold}")

            # Apply boundaries to the training data
            z_train = np.abs((train_df['y'] - train_mean) / train_std)
            outlier_index_train = np.where(z_train > threshold)[0]

            logging.info(f"Total training data points: {len(train_df)}")
            logging.info(f"Dropping {len(outlier_index_train)} training outlier rows (z-score > {threshold})")
            train_df_cleaned = train_df.drop(train_df.index[outlier_index_train]).reset_index(drop=True)
            logging.info(f"Remaining training data points: {len(train_df_cleaned)}")

            # Apply the SAME boundaries to the test data
            logging.info("Applying same z-score boundaries to test data")
            # Use train_mean and train_std to prevent data leakage
            z_test = np.abs((test_df['y'] - train_mean) / train_std) 
            outlier_index_test = np.where(z_test > threshold)[0]

            logging.info(f"Total test data points: {len(test_df)}")
            logging.info(f"Dropping {len(outlier_index_test)} test outlier rows (z-score > {threshold})")
            test_df_cleaned = test_df.drop(test_df.index[outlier_index_test]).reset_index(drop=True) 
            logging.info(f"Remaining test data points: {len(test_df_cleaned)}") 


            train_df_cleaned.to_csv(self.transformation_config.cleaned_train_data_path, index=False)
            test_df_cleaned.to_csv(self.transformation_config.cleaned_test_data_path, index=False) 

            logging.info("Data transformation completed. Saved cleaned files to artifacts.")

            # Return the paths to the new files
            return (
                self.transformation_config.cleaned_train_data_path,
                self.transformation_config.cleaned_test_data_path, 
            )

        except Exception as e:
            raise CustomException(e, sys)

# This block allows you to run this script directly for testing
if __name__ == "__main__":
    # Add the project root to the Python path
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    from src.components.data_ingestion import DataIngestion


    ingestion_obj = DataIngestion()
    train_path, test_path = ingestion_obj.initiate_data_ingestion()


    transformation_obj = DataTransformation()
    
    # This will now correctly receive the new paths
    cleaned_train_path, cleaned_test_path = transformation_obj.initiate_data_transformation(train_path, test_path)
    
