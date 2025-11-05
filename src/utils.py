import os
import sys
import numpy as np
import pandas as pd
import dill
import holidays
from datetime import datetime
from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Saved object to {file_path}")
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def mape(actual, pred): 
    '''
    Mean Absolute Percentage Error (MAPE) Function
    '''
    actual, pred = np.array(actual), np.array(pred)
    # Remove zeros from actual to avoid division by zero
    nonzero_mask = actual != 0
    actual = actual[nonzero_mask]
    pred = pred[nonzero_mask]
    
    if len(actual) == 0:
        return 0.0  # Or np.nan, depending on how you want to handle this
        
    return np.mean(np.abs((actual - pred) / actual)) * 100

def get_singapore_holidays():
    '''
    This function creates and returns the Singapore holiday dataframe
    for the model trainer to use.
    '''
    logging.info("Preparing Singapore holiday dataframe")
    try:
        singapore_holidays_df = pd.DataFrame([])
        
        # We can hardcode the years or make it dynamic
        # Let's make it dynamic for the next 3 years
        current_year = datetime.now().year
        years_range = list(range(current_year - 1, current_year + 3)) # e.g., 2023 to 2026

        logging.info(f"Generating Singapore holidays for years: {years_range}")

        for date_, name in sorted(holidays.SG(years=years_range).items()):
            singapore_holidays_df = pd.concat([
                singapore_holidays_df, 
                pd.DataFrame({
                    'ds': date_, 
                    'holiday': 'SG-Holiday',
                    'lower_window': -2,
                    'upper_window': 1
                }, index=[0])
            ], ignore_index=True)

        singapore_holidays_df['ds'] = pd.to_datetime(singapore_holidays_df['ds'])
        logging.info(f"Total Singapore holidays loaded: {len(singapore_holidays_df)}")
        
        return singapore_holidays_df

    except Exception as e:
        raise CustomException(e, sys)