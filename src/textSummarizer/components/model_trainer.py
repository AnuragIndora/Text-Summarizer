from logging import getLogger
from pathlib import Path
from dataclasses import dataclass
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from src.textSummarizer.entity import (ModelTrainerConfig)
import torch
from datasets import load_from_disk
from src.textSummarizer.constants import *

logger = getLogger(__name__)

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Training on: {self.device} (VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB)")

    def get_model(self):
        """Load BART with gradient checkpointing for memory efficiency"""
        model = BartForConditionalGeneration.from_pretrained(
            self.config.model_name,
            gradient_checkpointing=self.config.gradient_checkpointing
        )
        model = model.to(self.device)
        if self.config.gradient_checkpointing:
            logger.info("Gradient Checkpointing enabled (slower, but saves VRAM)")
        return model

    def get_tokenizer(self):
        """Load tokenizer"""
        return BartTokenizer.from_pretrained(self.config.tokenizer_name)

    def load_datasets(self):
        """Load tokenized datasets from disk"""
        train_path = Path(self.config.data_path) / "transformed_train_data"
        val_path = Path(self.config.data_path) / "transformed_validation_data"
        train_dataset = load_from_disk(train_path)
        val_dataset = load_from_disk(val_path)
        logger.info(f"Train: {len(train_dataset)} samples | Val: {len(val_dataset)}")
        return train_dataset, val_dataset

    def train(self):
        """Optimized training loop for 4GB GPU"""
        model = self.get_model()
        tokenizer = self.get_tokenizer()
        train_dataset, val_dataset = self.load_datasets()

        training_args = Seq2SeqTrainingArguments(
            output_dir=self.config.root_dir,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            num_train_epochs=self.config.num_train_epochs,
            fp16=self.config.fp16,     # 🟠 Mixed Precision (4GB GPU requirement)
            logging_steps=100,
            eval_strategy="steps",
            save_total_limit=2,
            report_to="none",  # Disable WandB to save memory
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer,
            data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        )

        logger.info("Starting training (VRAM optimized)")
        trainer.train()

        # Save the model
        output_dir = Path(self.config.root_dir) / "final_model"
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Model saved to {output_dir}")
