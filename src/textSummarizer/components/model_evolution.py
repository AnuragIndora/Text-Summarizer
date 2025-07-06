import logging
from pathlib import Path
from transformers import BartForConditionalGeneration, BartTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_from_disk
import pandas as pd
import torch
from src.textSummarizer.utils.common import create_directories
from typing import Dict, Tuple
import evaluate 
from src.textSummarizer.entity import (ModelEvaluationConfig,
                                       Seq2SeqTrainingArgumentsConfig)

logger = logging.getLogger(__name__)


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig, training_args_config: Seq2SeqTrainingArgumentsConfig):
        self.config = config
        self.training_args_config = training_args_config  
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Evaluation will run on: {self.device}")

    def load_components(self) -> Tuple[BartForConditionalGeneration, BartTokenizer]:
        """Load model and tokenizer"""
        try:
            model = BartForConditionalGeneration.from_pretrained(str(self.config.model_path))
            model = model.to(self.device)
            
            tokenizer = BartTokenizer.from_pretrained(str(self.config.tokenizer_path))
            
            return model, tokenizer
        except Exception as e:
            logger.error(f"Error loading components: {str(e)}")
            raise

    def load_datasets(self):
        """Load the test dataset for evaluation"""
        try:
            test_path = Path(self.config.data_path) / "transformed_test_data"
            test_dataset = load_from_disk(test_path)
            logger.info(f"Loaded test dataset with {len(test_dataset)} samples")
            return test_dataset
        except Exception as e:
            logger.error(f"Error loading datasets: {str(e)}")
            raise

    def compute_metrics(self, eval_pred, tokenizer) -> Dict[str, float]:
        """Compute evaluation metrics (ROUGE)"""
        rouge = evaluate.load("rouge")
        predictions, labels = eval_pred

        # Replace -100 in labels with pad_token_id for decoding
        labels = [[(label if label != -100 else tokenizer.pad_token_id) for label in l] for l in labels]

        # Decode predicted and reference summaries
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Clean text
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        # Compute ROUGE
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}

        return result

    def evaluate(self):
        """Complete evaluation pipeline"""
        try:
            model, tokenizer = self.load_components()
            test_dataset = self.load_datasets()
            
            # Prepare data collator
            data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

            # Prepare the trainer using training arguments from the configuration
            trainer = Seq2SeqTrainer(
                model=model,
                args=Seq2SeqTrainingArguments(
                    output_dir=self.training_args_config.output_dir,
                    per_device_eval_batch_size=self.training_args_config.per_device_eval_batch_size,
                    predict_with_generate=self.training_args_config.predict_with_generate,
                    logging_dir=self.training_args_config.logging_dir,
                ),
                data_collator=data_collator,
                compute_metrics=lambda eval_pred: self.compute_metrics(eval_pred, tokenizer),  # Pass tokenizer here
            )

            logger.info("Starting evaluation...")
            metrics = trainer.evaluate(eval_dataset=test_dataset)
            logger.info(f"Evaluation metrics: {metrics}")

            # Save metrics
            metrics_path = Path(self.config.root_dir) / self.config.metric_file_name
            pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
            logger.info(f"Metrics saved to {metrics_path}")

            return metrics
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise