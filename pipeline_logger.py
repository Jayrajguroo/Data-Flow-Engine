import logging
import os
from datetime import datetime
from typing import Optional


class PipelineLogger:
    """Custom logger for ETL pipeline with database integration"""
    
    def __init__(self, pipeline_run_id: Optional[int] = None, log_level: str = "INFO"):
        self.pipeline_run_id = pipeline_run_id
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        
        # Set up standard Python logger
        self.logger = logging.getLogger(f"etl_pipeline_{pipeline_run_id}")
        self.logger.setLevel(self.log_level)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.log_level)
            
            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            
            # File handler (optional)
            log_file = os.getenv('ETL_LOG_FILE')
            if log_file:
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(self.log_level)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
    
    def _log_to_database(self, level: str, message: str, step: Optional[str] = None):
        """Log message to database if pipeline_run_id is available"""
        if self.pipeline_run_id is None:
            return
        
        try:
            # Import here to avoid circular dependency
            from database import db
            from models import PipelineLog
            
            log_entry = PipelineLog(
                pipeline_run_id=self.pipeline_run_id,
                level=level,
                message=message,
                step=step,
                timestamp=datetime.utcnow()
            )
            db.session.add(log_entry)
            db.session.commit()
        except Exception as e:
            # If database logging fails, log to standard logger
            self.logger.error(f"Failed to log to database: {str(e)}")
    
    def debug(self, message: str, step: Optional[str] = None):
        """Log debug message"""
        self.logger.debug(message)
        self._log_to_database("DEBUG", message, step)
    
    def info(self, message: str, step: Optional[str] = None):
        """Log info message"""
        self.logger.info(message)
        self._log_to_database("INFO", message, step)
    
    def warning(self, message: str, step: Optional[str] = None):
        """Log warning message"""
        self.logger.warning(message)
        self._log_to_database("WARNING", message, step)
    
    def error(self, message: str, step: Optional[str] = None):
        """Log error message"""
        self.logger.error(message)
        self._log_to_database("ERROR", message, step)
    
    def critical(self, message: str, step: Optional[str] = None):
        """Log critical message"""
        self.logger.critical(message)
        self._log_to_database("CRITICAL", message, step)
    
    def log_data_summary(self, data_summary: dict, step: str):
        """Log data summary information"""
        shape = data_summary.get('shape', (0, 0))
        memory_usage = data_summary.get('memory_usage', 0)
        null_counts = data_summary.get('null_counts', {})
        
        summary_message = (
            f"Data Summary - Shape: {shape}, "
            f"Memory: {memory_usage / (1024*1024):.2f} MB, "
            f"Columns with nulls: {len([col for col, count in null_counts.items() if count > 0])}"
        )
        
        self.info(summary_message, step)
        
        # Log detailed null information if significant nulls exist
        significant_nulls = {col: count for col, count in null_counts.items() if count > 0}
        if significant_nulls:
            null_message = f"Null values by column: {significant_nulls}"
            self.info(null_message, step)
    
    def log_validation_results(self, validation_results: list, step: str):
        """Log validation results"""
        total_rules = len(validation_results)
        passed_rules = len([r for r in validation_results if r['passed']])
        failed_rules = total_rules - passed_rules
        
        summary_message = f"Validation Results - Total: {total_rules}, Passed: {passed_rules}, Failed: {failed_rules}"
        
        if failed_rules == 0:
            self.info(summary_message, step)
        else:
            self.warning(summary_message, step)
        
        # Log failed validations
        for result in validation_results:
            if not result['passed']:
                level = "ERROR" if result.get('critical', False) else "WARNING"
                message = f"Validation Failed: {result['rule']} - {result['message']}"
                
                if level == "ERROR":
                    self.error(message, step)
                else:
                    self.warning(message, step)
    
    def log_performance_metrics(self, metrics: dict, step: str):
        """Log performance metrics"""
        duration = metrics.get('duration', 0)
        records_processed = metrics.get('records_processed', 0)
        
        if duration > 0:
            records_per_second = records_processed / duration
            message = (
                f"Performance Metrics - Duration: {duration:.2f}s, "
                f"Records: {records_processed}, "
                f"Rate: {records_per_second:.2f} records/sec"
            )
        else:
            message = f"Performance Metrics - Records: {records_processed}"
        
        self.info(message, step)
    
    def log_memory_usage(self, current_usage_mb: float, step: str):
        """Log memory usage information"""
        message = f"Memory Usage: {current_usage_mb:.2f} MB"
        
        # Warn if memory usage is high (>1GB)
        if current_usage_mb > 1024:
            self.warning(f"High {message}", step)
        else:
            self.info(message, step)
    
    def log_transformation_step(self, step_name: str, input_shape: tuple, output_shape: tuple, step: str):
        """Log transformation step details"""
        message = (
            f"Transformation '{step_name}' completed - "
            f"Input shape: {input_shape}, Output shape: {output_shape}"
        )
        
        if input_shape != output_shape:
            self.info(message, step)
        else:
            self.debug(message, step)
    
    def log_extraction_source(self, source_name: str, records_extracted: int, step: str):
        """Log data extraction details"""
        message = f"Extracted {records_extracted} records from source '{source_name}'"
        self.info(message, step)
    
    def log_loading_target(self, target_name: str, records_loaded: int, step: str):
        """Log data loading details"""
        message = f"Loaded {records_loaded} records to target '{target_name}'"
        self.info(message, step)
    
    def log_error_with_context(self, error: Exception, context: dict, step: str):
        """Log error with additional context information"""
        error_message = f"Error: {str(error)}"
        context_message = f"Context: {context}"
        
        self.error(error_message, step)
        self.debug(context_message, step)
    
    def log_configuration_info(self, config_summary: dict):
        """Log pipeline configuration summary"""
        sources_count = len(config_summary.get('sources', []))
        transformations_count = len(config_summary.get('transformations', []))
        targets_count = len(config_summary.get('targets', []))
        
        message = (
            f"Pipeline Configuration - Sources: {sources_count}, "
            f"Transformations: {transformations_count}, Targets: {targets_count}"
        )
        
        self.info(message, "pipeline")
    
    def close(self):
        """Close logger and cleanup handlers"""
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)


class BatchProgressLogger:
    """Helper class for logging batch processing progress"""
    
    def __init__(self, logger: PipelineLogger, total_batches: int, step: str):
        self.logger = logger
        self.total_batches = total_batches
        self.step = step
        self.current_batch = 0
        self.start_time = datetime.utcnow()
    
    def log_batch_progress(self, batch_size: int):
        """Log progress for a completed batch"""
        self.current_batch += 1
        progress_percentage = (self.current_batch / self.total_batches) * 100
        
        elapsed_time = (datetime.utcnow() - self.start_time).total_seconds()
        if elapsed_time > 0:
            batches_per_second = self.current_batch / elapsed_time
            estimated_remaining = (self.total_batches - self.current_batch) / batches_per_second
            
            message = (
                f"Batch {self.current_batch}/{self.total_batches} completed "
                f"({progress_percentage:.1f}%) - Records: {batch_size}, "
                f"ETA: {estimated_remaining:.0f}s"
            )
        else:
            message = (
                f"Batch {self.current_batch}/{self.total_batches} completed "
                f"({progress_percentage:.1f}%) - Records: {batch_size}"
            )
        
        # Log every 10% or every 10 batches, whichever is less frequent
        log_interval = max(1, min(10, self.total_batches // 10))
        if self.current_batch % log_interval == 0 or self.current_batch == self.total_batches:
            self.logger.info(message, self.step)
    
    def log_batch_error(self, batch_number: int, error: Exception):
        """Log batch processing error"""
        message = f"Batch {batch_number} failed: {str(error)}"
        self.logger.error(message, self.step)
    
    def log_completion(self, total_records: int):
        """Log batch processing completion"""
        elapsed_time = (datetime.utcnow() - self.start_time).total_seconds()
        records_per_second = total_records / elapsed_time if elapsed_time > 0 else 0
        
        message = (
            f"Batch processing completed - Total records: {total_records}, "
            f"Duration: {elapsed_time:.2f}s, Rate: {records_per_second:.2f} records/sec"
        )
        
        self.logger.info(message, self.step)


def setup_pipeline_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup global logging configuration for ETL pipeline"""
    
    # Set root logger level
    logging.getLogger().setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Configure format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    console_handler.setFormatter(formatter)
    
    # File handler if specified
    handlers = [console_handler]
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    root_logger = logging.getLogger()
    for handler in handlers:
        root_logger.addHandler(handler)
    
    return root_logger
