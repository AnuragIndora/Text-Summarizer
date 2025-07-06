from src.textSummarizer.config.configration import ConfigurationManager
from src.textSummarizer.components.data_transformation import DataTransformation
from src.textSummarizer.logging import logger


class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config_manager = ConfigurationManager()
        data_transformation_config = config_manager.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.transform()