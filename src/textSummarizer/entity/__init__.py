from dataclasses import dataclass
from pathlib import Path

# Data Ingestion
@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    dataset_name: str
    dataset_version: str
    local_data_dir: Path

# Data Validation
@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    dataset_dir: Path
    STATUS_FILE: Path
    REQUIRED_FILES: list[str]
    DATA_FORMATS: list[str]


# Data Transformation
@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    dataset_dir: Path
    model_name: str
    tokenizer_name: str
    max_input_length: int 
    max_target_length: int 

# Model Trainer
@dataclass
class ModelTrainerConfig:
    root_dir: Path
    data_path: Path
    model_name: str
    tokenizer_name: str
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2   
    per_device_eval_batch_size: int = 2   
    gradient_accumulation_steps: int = 4   
    fp16: bool = True                      
    gradient_checkpointing: bool = True    

# model evolution 
@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    data_path: Path
    model_path: Path
    tokenizer_path: Path
    metric_file_name: str


@dataclass
class Seq2SeqTrainingArgumentsConfig:
    output_dir: Path
    eval_strategy: str
    eval_steps: int
    logging_steps: int
    save_steps: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    num_train_epochs: int
    save_total_limit: int
    predict_with_generate: bool
    fp16: bool
    logging_dir: Path
    report_to: str
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    gradient_checkpointing: bool