from src.textSummarizer.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.textSummarizer.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from src.textSummarizer.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from src.textSummarizer.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline
from src.textSummarizer.pipeline.stage_05_model_evaluation import ModelEvaluationTrainingPipeline
from src.textSummarizer.pipeline.predication import TextSummarizer
from src.textSummarizer.logging import logger

def run_pipeline():
    STAGES = [
        ("Data Ingestion Stage", DataIngestionTrainingPipeline),
        ("Data Validation Stage", DataValidationTrainingPipeline),
        ("Data Transformation Stage", DataTransformationTrainingPipeline),
        ("Model Training Stage", ModelTrainerTrainingPipeline),
        ("Model Evaluation Stage", ModelEvaluationTrainingPipeline)
    ]
    
    for stage_name, stage_class in STAGES:
        try:
            logger.info(f"\n\n===== {stage_name} started =====\n")
            pipeline = stage_class()
            pipeline.main()
            logger.info(f"\n===== {stage_name} completed =====\n\n")
        except Exception as e:
            logger.error(f"Error during {stage_name}: {e}")
            raise RuntimeError(f"{stage_name} failed") from e

def main():
    # Initialize the summarizer
    summarizer = TextSummarizer()
    while True:
        # Take user input
        input_text = input("Enter the text to summarize (or type 'exit' to quit): ")
        if input_text.lower() == 'exit':
            break
        # Generate summary
        summary = summarizer.summarize([input_text])[0]
        # Display the input and summary
        print("=====================" * 10)
        print(f"Summary: {summary}\n")

if __name__ == "__main__":
    # Run the data processing pipeline first
    run_pipeline()
    
    # Then enter the summarization loop
    main()
