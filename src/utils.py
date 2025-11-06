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
        return 0.0  
        
    return np.mean(np.abs((actual - pred) / actual)) * 100

def get_holidays(): ## as we dont know which custom holidays to add, we will create a function for future use
    '''
    This function creates and returns a dataframe of
    Malaysia (MY) holidays.
    '''
    logging.info("Preparing MY holiday dataframe")
    try:
        my_holidays_df = pd.DataFrame([])
        
        current_year = datetime.now().year
        years_range = list(range(current_year - 1, current_year + 3)) 

        logging.info(f"Generating MY holidays for years: {years_range}")
        
        # Get Malaysia Holidays
        for date_, name in sorted(holidays.MY(years=years_range).items()):
            my_holidays_df = pd.concat([
                my_holidays_df, 
                pd.DataFrame({
                    'ds': date_, 
                    'holiday': 'MY-Holiday',
                    'lower_window': -2,
                    'upper_window': 1
                }, index=[0])
            ], ignore_index=True)

        my_holidays_df['ds'] = pd.to_datetime(my_holidays_df['ds'])
        
        # Remove any duplicate dates
        my_holidays_df = my_holidays_df.drop_duplicates(subset=['ds'])
        
        logging.info(f"Total Malaysia holidays loaded: {len(my_holidays_df)}")
        
        return my_holidays_df

    except Exception as e:
        raise CustomException(e, sys)