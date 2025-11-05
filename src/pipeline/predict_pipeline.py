import os
import sys
import pandas as pd
from dataclasses import dataclass


from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

@dataclass
class PredictPipelineConfig:
    model_path: str = os.path.join("artifacts", "model.pkl")
    train_data_path: str = os.path.join("artifacts", "train_cleaned.csv")

class PredictPipeline:
    def __init__(self):
        try:
            self.config = PredictPipelineConfig()
            
            logging.info("Loading model and training data for prediction...")
            self.model = load_object(file_path=self.config.model_path)
            
            # We need the training data to get historical regressor values
            self.train_df = pd.read_csv(self.config.train_data_path)
            self.train_df['ds'] = pd.to_datetime(self.train_df['ds'])
            
            # Get regressor names from the loaded data
            self.regressor_cols = [col for col in self.train_df.columns if col not in ['ds', 'y']]
            logging.info(f"Loaded model and {len(self.regressor_cols)} regressors.")
            
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features):
        try:
            # 'features' dataframe comes from CustomData.get_data_as_data_frame()
            # It just contains the number of periods to predict
            periods = int(features['periods'].iloc[0])
            
            logging.info(f"Making future dataframe for {periods} periods.")
            
            # 1. Create future dataframe
            future = self.model.make_future_dataframe(periods=periods)
            
            # 2. Merge regressor columns into the future dataframe
            # This gets all the *historical* values for the regressors
            future_with_regressors = pd.merge(
                future,
                self.train_df[['ds'] + self.regressor_cols],
                on='ds',
                how='left'
            )
            
            # 3. Fill future regressor values (for the new 'periods' days) with 0
            # This assumes no promotions, events, etc. for the future forecast
            future_with_regressors[self.regressor_cols] = future_with_regressors[self.regressor_cols].fillna(0)

            # 4. Predict
            logging.info("Generating forecast...")
            forecast = self.model.predict(future_with_regressors)
            
            # Return the full forecast (not just future) so we can plot historical data
            return forecast

        except Exception as e:
            raise CustomException(e, sys)
    
    def get_model(self):
        """Return the trained Prophet model for plotting."""
        return self.model

class CustomData:
    '''
    This class takes the number of periods to forecast
    and prepares it for the PredictPipeline.
    '''
    def __init__(self, periods: int):
        self.periods = periods

    def get_data_as_data_frame(self):
        try:
            # Create a dataframe with the 'periods' feature
            custom_data_input_dict = {
                "periods": [self.periods]
            }
            
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

# This block allows you to run this script directly for testing
if __name__ == "__main__":
    logging.info("Starting prediction pipeline test...")
    
    # 1. Create custom data: We want to predict the next 30 days
    custom_data = CustomData(periods=30)
    
    # 2. Convert to dataframe
    features_df = custom_data.get_data_as_data_frame()
    
    # 3. Initialize pipeline and predict
    pipeline = PredictPipeline()
    forecast = pipeline.predict(features_df)
    
    print("\n--- 30-Day Forecast ---")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_string())
    logging.info("Prediction pipeline test complete.")