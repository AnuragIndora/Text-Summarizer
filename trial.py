import os
import logging
from pathlib import Path
from dataclasses import dataclass
from transformers import BartForConditionalGeneration, BartTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_from_disk, load_metric
import pandas as pd
import torch
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    data_path: Path
    model_path: Path
    tokenizer_path: Path
    metric_file_name: str
    batch_size: int = 8
    max_length: int = 128
    num_beams: int = 4

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
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
        rouge = load_metric("rouge")
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

            # Prepare the trainer
            trainer = Seq2SeqTrainer(
                model=model,
                args=Seq2SeqTrainingArguments(
                    output_dir=self.config.root_dir,
                    per_device_eval_batch_size=self.config.batch_size,
                    predict_with_generate=True,
                    logging_dir=os.path.join(self.config.root_dir, "logs"),
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



# ---------------------------------------------------- 
import logging
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from typing import List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextSummarizer:
    def __init__(self, model_path: str, tokenizer_path: str, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        # Load model and tokenizer
        self.model = BartForConditionalGeneration.from_pretrained(model_path).to(self.device)
        self.tokenizer = BartTokenizer.from_pretrained(tokenizer_path)

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

def main():
    # Define paths for model and tokenizer
    MODEL_PATH = "path/to/your/model"  # Update with your model path
    TOKENIZER_PATH = "path/to/your/tokenizer"  # Update with your tokenizer path

    # Initialize the summarizer
    summarizer = TextSummarizer(model_path=MODEL_PATH, tokenizer_path=TOKENIZER_PATH)

    while True:
        # Take user input
        input_text = input("Enter the text to summarize (or type 'exit' to quit): ")
        if input_text.lower() == 'exit':
            break

        # Generate summary
        summary = summarizer.summarize([input_text])[0]

        # Display the input and summary
        print(f"\nInput: {input_text}")
        print(f"Summary: {summary}\n")

if __name__ == "__main__":
    main()


# ==============================

# Example 
"""
Editor's note: In our Behind the Scenes series, CNN correspondents share their experiences in covering news and analyze the stories behind the events. Here, Soledad O'Brien takes users inside a jail where many of the inmates are mentally ill. An inmate housed on the "forgotten floor," where many mentally ill inmates are housed in Miami before trial. MIAMI, Florida (CNN) -- The ninth floor of the Miami-Dade pretrial detention facility is dubbed the "forgotten floor." Here, inmates with the most severe mental illnesses are incarcerated until they're ready to appear in court. Most often, they face drug charges or charges of assaulting an officer --charges that Judge Steven Leifman says are usually "avoidable felonies." He says the arrests often result from confrontations with police. Mentally ill people often won't do what they're told when police arrive on the scene -- confrontation seems to exacerbate their illness and they become more paranoid, delusional, and less likely to follow directions, according to Leifman. So, they end up on the ninth floor severely mentally disturbed, but not getting any real help because they're in jail. We toured the jail with Leifman. He is well known in Miami as an advocate for justice and the mentally ill. Even though we were not exactly welcomed with open arms by the guards, we were given permission to shoot videotape and tour the floor. Go inside the 'forgotten floor' » . At first, it's hard to determine where the people are. The prisoners are wearing sleeveless robes. Imagine cutting holes for arms and feet in a heavy wool sleeping bag -- that's kind of what they look like. They're designed to keep the mentally ill patients from injuring themselves. That's also why they have no shoes, laces or mattresses. Leifman says about one-third of all people in Miami-Dade county jails are mentally ill. So, he says, the sheer volume is overwhelming the system, and the result is what we see on the ninth floor. Of course, it is a jail, so it's not supposed to be warm and comforting, but the lights glare, the cells are tiny and it's loud. We see two, sometimes three men -- sometimes in the robes, sometimes naked, lying or sitting in their cells. "I am the son of the president. You need to get me out of here!" one man shouts at me. He is absolutely serious, convinced that help is on the way -- if only he could reach the White House. Leifman tells me that these prisoner-patients will often circulate through the system, occasionally stabilizing in a mental hospital, only to return to jail to face their charges. It's brutally unjust, in his mind, and he has become a strong advocate for changing things in Miami. Over a meal later, we talk about how things got this way for mental patients. Leifman says 200 years ago people were considered "lunatics" and they were locked up in jails even if they had no charges against them. They were just considered unfit to be in society. Over the years, he says, there was some public outcry, and the mentally ill were moved out of jails and into hospitals. But Leifman says many of these mental hospitals were so horrible they were shut down. Where did the patients go? Nowhere. The streets. They became, in many cases, the homeless, he says. They never got treatment. Leifman says in 1955 there were more than half a million people in state mental hospitals, and today that number has been reduced 90 percent, and 40,000 to 50,000 people are in mental hospitals. The judge says he's working to change this. Starting in 2008, many inmates who would otherwise have been brought to the "forgotten floor" will instead be sent to a new mental health facility -- the first step on a journey toward long-term treatment, not just punishment. Leifman says it's not the complete answer, but it's a start. Leifman says the best part is that it's a win-win solution. The patients win, the families are relieved, and the state saves money by simply not cycling these prisoners through again and again. And, for Leifman, justice is served. E-mail to a friend .
"""