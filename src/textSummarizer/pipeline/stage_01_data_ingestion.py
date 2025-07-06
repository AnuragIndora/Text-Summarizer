from src.textSummarizer.config.configration import ConfigurationManager
from src.textSummarizer.components.data_ingestion import DataIngestion
from src.textSummarizer.logging import logger


class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config_manager = ConfigurationManager()
        data_ingestion_config = config_manager.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.get_data()