import os
import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging

class TrainPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def run_pipeline(self):
        try:
            logging.info("Starting the training pipeline...")

            # Step 1: Data Ingestion
            train_path, test_path = self.data_ingestion.initiate_data_ingestion()


            # Step 2: Data Transformation
            cleaned_train_path, cleaned_test_path = self.data_transformation.initiate_data_transformation(train_path, test_path)

            # Step 3: Model Training
            accuracy, model_path = self.model_trainer.initiate_model_training(cleaned_train_path, cleaned_test_path)

            logging.info("Training pipeline finished successfully.")

        except Exception as e:
            logging.error("Exception occurred in the training pipeline")
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        pipeline = TrainPipeline()
        pipeline.run_pipeline()
    except Exception as e:
        logging.critical(f"Pipeline failed: {e}")
        sys.exit(1)