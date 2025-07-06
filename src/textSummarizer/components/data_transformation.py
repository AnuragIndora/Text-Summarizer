from src.textSummarizer.constants import *
from pathlib import Path
import logging
from pathlib import Path
import logging
from transformers import BartTokenizer, BartForConditionalGeneration
from datasets import load_dataset
from src.textSummarizer.entity import DataTransformationConfig


logger = logging.getLogger(__name__)

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = BartTokenizer.from_pretrained(self.config.tokenizer_name)
        self.model = BartForConditionalGeneration.from_pretrained(
            self.config.model_name,
            gradient_checkpointing=True  # Saves memory during training
        )
        logger.info("Initialized DataTransformation with model: %s", 
                    self.config.model_name)

    def convert_examples_to_features(self, example_batch):
        """
        Convert raw text to tokenized features
        """
        try:
            # Tokenize inputs and targets
            input_encodings = self.tokenizer(
                example_batch['article'], 
                max_length=self.config.max_input_length,
                truncation=True,
                padding='max_length'
            )
            
            with self.tokenizer.as_target_tokenizer():
                target_encodings = self.tokenizer(
                    example_batch['highlights'],
                    max_length=self.config.max_target_length,
                    truncation=True,
                    padding='max_length'
                )
            
            return {
                'input_ids': input_encodings['input_ids'],
                'attention_mask': input_encodings['attention_mask'],
                'labels': target_encodings['input_ids']
            }
        except Exception as e:
            logger.error("Error in converting examples: %s", str(e))
            raise e
        

    def transform(self):
        """
        Complete data transformation pipeline for train, test, and validation datasets.
        """
        try:
            logger.info("Starting data transformation")
            
            # Load datasets from CSV files
            dataset_files = {
                "train": str(Path(self.config.dataset_dir) / "train.csv"),
                "test": str(Path(self.config.dataset_dir) / "test.csv"),
                "validation": str(Path(self.config.dataset_dir) / "validation.csv")
            }

            datasets = load_dataset('csv', data_files=dataset_files)

            logger.info("Loaded datasets: train (%d samples), test (%d samples), validation (%d samples)", 
                        len(datasets['train']), len(datasets['test']), len(datasets['validation']))

            # Apply transformation to each dataset
            transformed_train_dataset = datasets['train'].map(
                self.convert_examples_to_features,
                batched=True,
                remove_columns=['article', 'highlights', 'id']
            )
            transformed_test_dataset = datasets['test'].map(
                self.convert_examples_to_features,
                batched=True,
                remove_columns=['article', 'highlights', 'id']
            )
            transformed_validation_dataset = datasets['validation'].map(
                self.convert_examples_to_features,
                batched=True,
                remove_columns=['article', 'highlights', 'id']
            )

            logger.info("Dataset transformation completed")

            # Save transformed data
            transformed_train_path = Path(self.config.root_dir) / "transformed_train_data"
            transformed_test_path = Path(self.config.root_dir) / "transformed_test_data"
            transformed_validation_path = Path(self.config.root_dir) / "transformed_validation_data"

            transformed_train_dataset.save_to_disk(transformed_train_path)
            transformed_test_dataset.save_to_disk(transformed_test_path)
            transformed_validation_dataset.save_to_disk(transformed_validation_path)

            logger.info("Transformed datasets saved to: %s, %s, %s", 
                        transformed_train_path, transformed_test_path, transformed_validation_path)

            return {
                "train": transformed_train_dataset,
                "test": transformed_test_dataset,
                "validation": transformed_validation_dataset
            }
            
        except Exception as e:
            logger.exception("Data transformation failed: %s", str(e))
            raise e