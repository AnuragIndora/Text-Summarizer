import os
import logging
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from typing import List
from src.textSummarizer.config.configration import ConfigurationManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextSummarizer:
    def __init__(self, device: str = "cuda"):

        self.device = device if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        # Load model and tokenizer
        self.config = ConfigurationManager().get_model_evaluation_config()
        self.model = BartForConditionalGeneration.from_pretrained(self.config.model_path).to(self.device)
        self.tokenizer = BartTokenizer.from_pretrained(self.config.tokenizer_path)

    def summarize(self, texts: List[str], max_length: int = 150, num_beams: int = 4) -> List[str]:
        """Generate summaries for the given texts."""
        logger.info("Generating summaries...")
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(self.device)

        # Generate summaries
        summaries = self.model.generate(
            inputs['input_ids'],
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True
        )

        # Decode summaries
        decoded_summaries = self.tokenizer.batch_decode(summaries, skip_special_tokens=True)
        return decoded_summaries