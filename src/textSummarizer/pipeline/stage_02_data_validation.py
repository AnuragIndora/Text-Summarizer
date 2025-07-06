from src.textSummarizer.config.configration import ConfigurationManager
from src.textSummarizer.components.data_validation import DataValidation
from src.textSummarizer.logging import logger


class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config_manager = ConfigurationManager()
        validation_config = config_manager.get_data_validation_config()
        validator = DataValidation(validation_config)
        validator.run_validation()