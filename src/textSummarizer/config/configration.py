from src.textSummarizer.constants import *
from src.textSummarizer.utils.common import read_yaml, create_directories
from src.textSummarizer.entity import (DataIngestionConfig, 
                                        DataValidationConfig,
                                        DataTransformationConfig,
                                        ModelTrainerConfig,
                                        ModelEvaluationConfig,
                                        Seq2SeqTrainingArgumentsConfig)
from pathlib import Path

class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    # ------------- Data Ingestion Config ------------- #
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        return DataIngestionConfig(
            root_dir=Path(config.root_dir),
            dataset_name=config.dataset_name,
            dataset_version=config.dataset_version,
            local_data_dir=Path(config.local_data_dir)
        )
    
    # ------------- Data Validation Config ------------- #
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        create_directories([config.root_dir])

        return DataValidationConfig(
            root_dir=Path(config.root_dir),
            dataset_dir=Path(config.dataset_dir),
            STATUS_FILE=Path(config.STATUS_FILE),
            REQUIRED_FILES=config.REQUIRED_FILES,
            DATA_FORMATS=config.DATA_FORMATS
        )
    
    # ------------- Data Transformation Config ------------- #
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        create_directories([config.root_dir])

        return DataTransformationConfig(
            root_dir=Path(config.root_dir),
            dataset_dir=Path(config.dataset_dir),
            model_name=config.model_name,
            tokenizer_name=config.tokenizer_name,
            max_input_length=config.max_input_length,
            max_target_length=config.max_target_length
        )
    
    # ------------- Model Trainer Config ------------- #
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        training_params = self.params.Seq2SeqTrainingArguments

        create_directories([config.root_dir])

        return ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            model_name=config.model_name,
            tokenizer_name=config.tokenizer_name,
            num_train_epochs=training_params.num_train_epochs,
            per_device_train_batch_size=training_params.per_device_train_batch_size,
            per_device_eval_batch_size=training_params.per_device_eval_batch_size,
            gradient_accumulation_steps=training_params.gradient_accumulation_steps,
            fp16=training_params.fp16,
            gradient_checkpointing=training_params.gradient_checkpointing,
        )
    
    # ------------- Model Evaluation Config ------------- #
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        """Load model evaluation configuration from the loaded config."""
        config = self.config.model_evaluation

        create_directories([config.root_dir])
        
        return ModelEvaluationConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            model_path=Path(config.model_path),
            tokenizer_path=Path(config.tokenizer_path),
            metric_file_name=config.metric_file_name
        )

    # ------------- Training Arguments Config ------------- #
    def get_training_args_config(self) -> Seq2SeqTrainingArgumentsConfig:
        """Load training arguments configuration from the loaded params."""
        training_args = self.params.Seq2SeqTrainingArguments

        return Seq2SeqTrainingArgumentsConfig(
            output_dir=Path(training_args.output_dir),
            eval_strategy=training_args.eval_strategy,
            eval_steps=training_args.eval_steps,
            logging_steps=training_args.logging_steps,
            save_steps=training_args.save_steps,
            per_device_train_batch_size=training_args.per_device_train_batch_size,
            per_device_eval_batch_size=training_args.per_device_eval_batch_size,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            num_train_epochs=training_args.num_train_epochs,
            save_total_limit=training_args.save_total_limit,
            predict_with_generate=training_args.predict_with_generate,
            fp16=training_args.fp16,
            logging_dir=Path(training_args.logging_dir),
            report_to=training_args.report_to,
            learning_rate=training_args.learning_rate,
            weight_decay=training_args.weight_decay,
            warmup_steps=training_args.warmup_steps,
            gradient_checkpointing=training_args.gradient_checkpointing
        )
