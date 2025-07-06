from pathlib import Path
from src.textSummarizer.logging.logging_config import setup_logger
from src.textSummarizer.utils.common import get_size
from src.textSummarizer.constants import *
from src.textSummarizer.utils.common import read_yaml, create_directories
from src.textSummarizer.entity import (DataValidationConfig)
from ensure import ensure_annotations
import logging
import pandas as pd 
logger = logging.getLogger(__name__)



class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config
        create_directories([self.config.root_dir])
        self.status_file = Path(self.config.STATUS_FILE)
        logger.info("Initialized DataValidation with config: %s", self.config)

    @ensure_annotations
    def validate_files_exist(self) -> bool:
        """Validate all required files exist in the specified directory"""
        try:
            validation_status = True
            missing_files = []
            
            logger.info("Validating file existence in: %s", self.config.root_dir)
            
            for file in self.config.REQUIRED_FILES:
                file_path = Path(self.config.dataset_dir) / file
                if not file_path.exists():
                    validation_status = False
                    missing_files.append(str(file_path))
                    logger.warning("File missing: %s", file_path)

            # Write status to file
            with open(self.status_file, 'w') as f:
                f.write(f"FILE_EXISTENCE_VALIDATION: {'PASS' if validation_status else 'FAIL'}\n")
                if not validation_status:
                    f.write(f"MISSING_FILES: {','.join(missing_files)}\n")

            return validation_status
            
        except Exception as e:
            logger.error("Error during file existence validation: %s", str(e))
            raise

    @ensure_annotations
    def validate_file_formats(self) -> bool:
        """Validate all files contain required columns"""
        validation_status = True
        invalid_files = []
        
        try:
            logger.info("Validating file formats")
            
            for file in self.config.REQUIRED_FILES:
                file_path = Path(self.config.dataset_dir) / file
                
                try:
                    # Handle both CSV and JSON files
                    if file.endswith('.csv'):
                        df = pd.read_csv(file_path)
                    elif file.endswith('.json'):
                        df = pd.read_json(file_path)
                    else:
                        logger.error("Unsupported file format for file: %s", file)
                        validation_status = False
                        invalid_files.append(f"{file} (error: Unsupported file format)")
                        continue
                    
                    # Check required columns
                    missing_cols = [col for col in self.config.DATA_FORMATS if col not in df.columns]
                    
                    if missing_cols:
                        validation_status = False
                        invalid_files.append(f"{file} (missing: {','.join(missing_cols)})")
                        logger.warning("Missing columns in %s: %s", file, missing_cols)
                        
                except Exception as e:
                    validation_status = False
                    invalid_files.append(f"{file} (error: {str(e)})")
                    logger.error("Error reading %s: %s", file, str(e))

            # Update status file
            with open(self.status_file, 'a') as f:
                f.write(f"FORMAT_VALIDATION: {'PASS' if validation_status else 'FAIL'}\n")
                if not validation_status:
                    f.write(f"INVALID_FILES: {'; '.join(invalid_files)}\n")

            return validation_status
            
        except Exception as e:
            logger.error("Error during format validation: %s", str(e))
            raise

    def run_validation(self) -> bool:
        """Run complete validation pipeline"""
        logger.info("Starting comprehensive data validation")
        
        file_check = self.validate_files_exist()
        content_check = self.validate_file_formats() if file_check else False
        
        final_status = file_check and content_check
        
        with open(self.status_file, 'a') as f:
            f.write(f"OVERALL_VALIDATION: {'PASS' if final_status else 'FAIL'}\n")
        
        logger.info("Validation completed. Final status: %s", "PASS" if final_status else "FAIL")
        return final_status
