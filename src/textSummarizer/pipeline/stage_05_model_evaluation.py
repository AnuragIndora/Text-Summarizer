from src.textSummarizer.config.configration import ConfigurationManager
from src.textSummarizer.components.model_evolution import ModelEvaluation 
from src.textSummarizer.logging import logger



class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config_manager = ConfigurationManager()
        # Get configurations
        eval_config = config_manager.get_model_evaluation_config()
        training_args_config = config_manager.get_training_args_config()
        # Now you can use these configurations in your ModelEvaluation class
        evaluator = ModelEvaluation(eval_config, training_args_config)
        results = evaluator.evaluate()
        print("ROUGE Scores:", results)