from dataclasses import dataclass
from pathlib import Path
import os
import urllib.request as request
from src.textSummarizer.logging.logging_config import setup_logger
from datasets import DatasetDict
from src.textSummarizer.entity import (DataIngestionConfig)
from random import seed

# Initialize the logger
logger = setup_logger("logs/running_logs.log")  # Specify the log file path
# Example usage of the logger
logger.info("Application started.")

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        logger.info("DataIngestion initialized with config: %s", self.config)
    
    def download_data(self) -> DatasetDict:
        """
        Downloads the dataset from Hugging Face and saves it locally
        Returns:
            DatasetDict: The loaded dataset containing train, validation and test splits
        """
        # Create directories if they don't exist
        os.makedirs(self.config.local_data_dir, exist_ok=True)
        logger.info("Starting data download...")
        try:
            # Load dataset from Hugging Face
            from datasets import load_dataset
            dataset = load_dataset(
                path=self.config.dataset_name,
                name=self.config.dataset_version,
                cache_dir=self.config.local_data_dir
            )

            logger.debug("Loading dataset from %s", self.config.dataset_name)
            
            # Save datasets to local files
            seed(42)
            dataset['train'].shuffle(seed=42).select(range(10000)).to_csv(os.path.join(self.config.local_data_dir, 'train.csv'))
            dataset['validation'].shuffle(seed=42).select(range(1000)).to_csv(os.path.join(self.config.local_data_dir, 'validation.csv')) 
            dataset['test'].shuffle(seed=42).select(range(1000)).to_csv(os.path.join(self.config.local_data_dir, 'test.csv'))
            
            logger.info("Data download completed successfully.")

            return dataset
            
        except Exception as e:
            # raise Exception(f"Error downloading dataset: {str(e)}")
            logger.error("Error downloading data: %s", str(e))

    def get_data(self) -> DatasetDict:
        """
        Public method to get the downloaded data
        Returns:
            DatasetDict: The dataset dictionary
        """
        return self.download_data()
