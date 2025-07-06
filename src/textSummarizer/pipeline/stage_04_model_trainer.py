from src.textSummarizer.config.configration import ConfigurationManager
from src.textSummarizer.components.model_trainer import ModelTrainer
from src.textSummarizer.logging import logger


class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        # Run training
        config_manager = ConfigurationManager()
        train_config = config_manager.get_model_trainer_config()
        ModelTrainer(train_config).train()