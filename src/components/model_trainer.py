import os
import sys
import pandas as pd
import numpy as np
import time
import itertools
import warnings
from dataclasses import dataclass
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, get_holidays, mape

# --- NEW: Import the standard logging module ---
import logging as py_logging 

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, cleaned_train_path, cleaned_test_path):
        '''
        This function loads cleaned data, finds the best hyperparameters 
        (including holiday effects) via cross-validation, then trains 
        the single final model and saves it.
        '''
        logging.info("Entered model training method")
        try:
            cleaned_train_df = pd.read_csv(cleaned_train_path)
            test_df = pd.read_csv(cleaned_test_path)
            logging.info(f"Read cleaned train data ({len(cleaned_train_df)} rows) and test data ({len(test_df)} rows)")

            # Suppress Prophet's stan logs
            warnings.filterwarnings('ignore')

            # --- FIXED: Set log levels to ERROR to hide INFO and WARNINGS ---
            py_logging.getLogger('cmdstanpy').setLevel(py_logging.ERROR)
            py_logging.getLogger('prophet').setLevel(py_logging.ERROR)
            
            # --- Get Holidays (Because I don't know which country's data this is, so I am not using it in model training)
            logging.info("Loading Malaysia holiday data from utils")
            holidays_df = get_holidays()

            # HYPERPARAMETER TUNING WITH CROSS-VALIDATION
            logging.info("Starting hyperparameter tuning with cross-validation")
            
            param_grid = {
                'changepoint_prior_scale': [0.01, 0.1, 0.5],
                'seasonality_prior_scale': [0.1, 1.0, 10.0],
                'holidays_prior_scale': [0.1, 1.0, 10.0],
                'seasonality_mode': ['additive', 'multiplicative']
            }
            
            all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
            logging.info(f"Testing {len(all_params)} parameter combinations")

            total_days = len(cleaned_train_df)
            cv_initial = f'{int(total_days * 0.74)} days' 
            cv_period = '30 days'
            cv_horizon = '30 days'
            logging.info(f"CV settings: Initial={cv_initial}, Period={cv_period}, Horizon={cv_horizon}")

            results = []
            start_time = time.time()

            # Run Tuning Loop
            for idx, params in enumerate(all_params, 1):
                try:
                    m = Prophet(
                        changepoint_prior_scale=params['changepoint_prior_scale'],
                        seasonality_prior_scale=params['seasonality_prior_scale'],
                        seasonality_mode=params['seasonality_mode'],
                        #holidays=holidays_df, # Holidays commented out as in your code
                        holidays_prior_scale=params['holidays_prior_scale'],
                        daily_seasonality=False,
                        weekly_seasonality=True,
                        yearly_seasonality=True
                    )
                    m.fit(cleaned_train_df) 
                    
                    df_cv = cross_validation(m, initial=cv_initial, period=cv_period, horizon=cv_horizon, parallel=None)
                    df_p = performance_metrics(df_cv)
                    
                    result = params.copy()
                    result['mape'] = df_p['mape'].mean()
                    results.append(result)
                    
                    # Log progress
                    if idx % 5 == 0 or idx == len(all_params):
                        logging.info(f"Completed {idx}/{len(all_params)} combinations. Current best MAPE: {min([r['mape'] for r in results]):.4f}")

                except Exception as e:
                    logging.warning(f"Failed tuning combo {params}: {e}")
					
            # Get Best Parameters
            results_df = pd.DataFrame(results).sort_values('mape').reset_index(drop=True)
            if results_df.empty:
                raise CustomException("All hyperparameter tuning combinations failed.", sys)
                
            best_params = results_df.iloc[0].to_dict()
            best_mape = best_params.pop('mape')
            
            logging.info(f"Tuning complete in {time.time() - start_time:.2f}s. Best CV MAPE: {best_mape:.4f}")
            logging.info(f"Best parameters found: {best_params}")

            #  TRAIN FINAL MODEL
            logging.info("Training final model on the full cleaned training dataset...")
            
            final_model = Prophet(
                changepoint_prior_scale=best_params['changepoint_prior_scale'],
                seasonality_prior_scale=best_params['seasonality_prior_scale'],
                seasonality_mode=best_params['seasonality_mode'],
                #holidays=holidays_df,
                holidays_prior_scale=best_params['holidays_prior_scale'],
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True
            )
            
            final_model.fit(cleaned_train_df) 
            logging.info("Final model trained.")

            # Save the Model Artifact
            logging.info(f"Saving final model to {self.model_trainer_config.trained_model_file_path}")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=final_model
            )

            # Evaluate on Test Set
            logging.info("Evaluating final model on unseen test data")
            
            future = final_model.make_future_dataframe(periods=len(test_df))
            forecast = final_model.predict(future)
            
            test_predictions = forecast.iloc[-len(test_df):]['yhat']
            test_actuals = test_df['y']

            test_mape = mape(test_actuals, test_predictions)
            test_accuracy = 100 - test_mape
            
            logging.info(f"Test Set MAPE: {test_mape:.4f}")
            logging.info(f"Test Set Accuracy: {test_accuracy:.2f}%")

            # Keeping the 25% MAPE threshold
            if test_mape > 25: 
                raise CustomException(f"Model accuracy ({test_accuracy}%) is too low (Threshold > 25% MAPE).", sys)

            return (
                test_accuracy,
                self.model_trainer_config.trained_model_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)

# This block allows you to run this script directly for testing
if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation import DataTransformation

    # 1. Run Data Ingestion

    ingestion_obj = DataIngestion()
    train_path, test_path = ingestion_obj.initiate_data_ingestion()

    # 2. Run Data Transformation

    transformation_obj = DataTransformation()
    cleaned_train_path, cleaned_test_path = transformation_obj.initiate_data_transformation(train_path, test_path)

    # 3. Run Model Trainer

    model_trainer = ModelTrainer()
    accuracy, model_path = model_trainer.initiate_model_training(cleaned_train_path, cleaned_test_path)
    
