import pandas as pd
import numpy as np
from datetime import datetime
import traceback
import time
from typing import Dict, Any, List, Optional

from extractors import DataExtractor
from transformers import DataTransformer
from loaders import DataLoader
from validators import DataValidator
from pipeline_logger import PipelineLogger


class ETLPipeline:
    """Main ETL Pipeline orchestrator"""
    
    def __init__(self, config: Dict[str, Any], pipeline_run_id: Optional[int] = None):
        """
        Initialize ETL Pipeline
        
        Args:
            config: Pipeline configuration dictionary
            pipeline_run_id: ID of the pipeline run for logging
        """
        self.config = config
        self.pipeline_run_id = pipeline_run_id
        self.logger = PipelineLogger(pipeline_run_id)
        
        # Initialize components
        self.extractor = DataExtractor(config.get('extraction', {}), self.logger)
        self.transformer = DataTransformer(config.get('transformation', {}), self.logger)
        self.loader = DataLoader(config.get('loading', {}), self.logger)
        self.validator = DataValidator(config.get('validation', {}), self.logger)
        
        # Pipeline state
        self.data = None
        self.records_processed = 0
        self.start_time = None
        self.end_time = None
    
    def run(self) -> Dict[str, Any]:
        """
        Execute the complete ETL pipeline
        
        Returns:
            Dictionary containing execution results and metrics
        """
        self.start_time = datetime.utcnow()
        self.logger.info("Starting ETL Pipeline execution", "pipeline")
        
        try:
            # Step 1: Data Extraction
            self.logger.info("Starting data extraction phase", "extract")
            self.data = self._extract_data()
            
            if self.data is None or self.data.empty:
                raise Exception("No data extracted from sources")
            
            self.records_processed = len(self.data)
            self.logger.info(f"Extracted {self.records_processed} records", "extract")
            
            # Step 2: Data Validation (Pre-processing)
            if self.config.get('validation', {}).get('pre_processing', True):
                self.logger.info("Starting pre-processing validation", "validate")
                self._validate_data("pre_processing")
            
            # Step 3: Data Transformation
            self.logger.info("Starting data transformation phase", "transform")
            self.data = self._transform_data()
            
            # Step 4: Data Validation (Post-processing)
            if self.config.get('validation', {}).get('post_processing', True):
                self.logger.info("Starting post-processing validation", "validate")
                self._validate_data("post_processing")
            
            # Step 5: Data Loading
            self.logger.info("Starting data loading phase", "load")
            self._load_data()
            
            self.end_time = datetime.utcnow()
            duration = (self.end_time - self.start_time).total_seconds()
            
            self.logger.info(f"Pipeline completed successfully in {duration:.2f} seconds", "pipeline")
            
            return {
                'success': True,
                'records_processed': self.records_processed,
                'duration': duration,
                'start_time': self.start_time.isoformat(),
                'end_time': self.end_time.isoformat()
            }
            
        except Exception as e:
            self.end_time = datetime.utcnow()
            error_msg = f"Pipeline failed: {str(e)}"
            self.logger.error(error_msg, "pipeline")
            self.logger.error(f"Stack trace: {traceback.format_exc()}", "pipeline")
            
            return {
                'success': False,
                'error_message': error_msg,
                'records_processed': self.records_processed,
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'end_time': self.end_time.isoformat() if self.end_time else None
            }
    
    def _extract_data(self) -> pd.DataFrame:
        """Extract data from configured sources"""
        extraction_config = self.config.get('extraction', {})
        sources = extraction_config.get('sources', [])
        
        if not sources:
            raise Exception("No data sources configured")
        
        all_data = []
        
        for source in sources:
            try:
                source_data = self.extractor.extract(source)
                if source_data is not None and not source_data.empty:
                    # Add source identifier column
                    source_data['_source'] = source.get('name', 'unknown')
                    all_data.append(source_data)
                    self.logger.info(f"Extracted {len(source_data)} records from {source.get('name', 'unknown')}", "extract")
                else:
                    self.logger.warning(f"No data extracted from source: {source.get('name', 'unknown')}", "extract")
                    
            except Exception as e:
                error_msg = f"Failed to extract from source {source.get('name', 'unknown')}: {str(e)}"
                if extraction_config.get('fail_on_source_error', True):
                    raise Exception(error_msg)
                else:
                    self.logger.warning(error_msg, "extract")
        
        if not all_data:
            raise Exception("No data extracted from any source")
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True, sort=False)
        
        # Apply global extraction filters
        if extraction_config.get('filters'):
            combined_data = self._apply_filters(combined_data, extraction_config['filters'])
        
        return combined_data
    
    def _transform_data(self) -> pd.DataFrame:
        """Transform the extracted data"""
        transformation_config = self.config.get('transformation', {})
        
        if not transformation_config:
            self.logger.info("No transformations configured, returning original data", "transform")
            return self.data
        
        transformed_data = self.data.copy()
        
        # Apply transformations in order
        steps = transformation_config.get('steps', [])
        
        for step_config in steps:
            step_name = step_config.get('name', 'unknown')
            self.logger.info(f"Applying transformation step: {step_name}", "transform")
            
            try:
                transformed_data = self.transformer.apply_transformation(transformed_data, step_config)
                self.logger.info(f"Completed transformation step: {step_name}", "transform")
                
            except Exception as e:
                error_msg = f"Failed at transformation step {step_name}: {str(e)}"
                if transformation_config.get('fail_on_step_error', True):
                    raise Exception(error_msg)
                else:
                    self.logger.warning(error_msg, "transform")
        
        self.logger.info(f"Transformation completed. Final dataset shape: {transformed_data.shape}", "transform")
        return transformed_data
    
    def _validate_data(self, phase: str):
        """Validate data quality"""
        validation_config = self.config.get('validation', {})
        phase_config = validation_config.get(phase, {})
        
        if not phase_config:
            return
        
        validation_results = self.validator.validate(self.data, phase_config)
        
        # Log validation results
        for result in validation_results:
            if result['passed']:
                self.logger.info(f"Validation passed: {result['rule']}", "validate")
            else:
                message = f"Validation failed: {result['rule']} - {result['message']}"
                if result.get('critical', False):
                    self.logger.error(message, "validate")
                    raise Exception(f"Critical validation failed: {result['rule']}")
                else:
                    self.logger.warning(message, "validate")
    
    def _load_data(self):
        """Load transformed data to target destinations"""
        loading_config = self.config.get('loading', {})
        targets = loading_config.get('targets', [])
        
        if not targets:
            self.logger.warning("No loading targets configured", "load")
            return
        
        for target in targets:
            try:
                target_name = target.get('name', 'unknown')
                self.logger.info(f"Loading data to target: {target_name}", "load")
                
                self.loader.load(self.data, target)
                self.logger.info(f"Successfully loaded data to target: {target_name}", "load")
                
            except Exception as e:
                error_msg = f"Failed to load to target {target.get('name', 'unknown')}: {str(e)}"
                if loading_config.get('fail_on_target_error', True):
                    raise Exception(error_msg)
                else:
                    self.logger.warning(error_msg, "load")
    
    def _apply_filters(self, data: pd.DataFrame, filters: List[Dict]) -> pd.DataFrame:
        """Apply filtering conditions to data"""
        filtered_data = data.copy()
        
        for filter_config in filters:
            column = filter_config.get('column')
            condition = filter_config.get('condition')
            value = filter_config.get('value')
            
            if not all([column, condition, value is not None]):
                self.logger.warning(f"Invalid filter configuration: {filter_config}", "extract")
                continue
            
            if column not in filtered_data.columns:
                self.logger.warning(f"Filter column '{column}' not found in data", "extract")
                continue
            
            initial_count = len(filtered_data)
            
            try:
                if condition == 'equals':
                    filtered_data = filtered_data[filtered_data[column] == value]
                elif condition == 'not_equals':
                    filtered_data = filtered_data[filtered_data[column] != value]
                elif condition == 'greater_than':
                    filtered_data = filtered_data[filtered_data[column] > value]
                elif condition == 'less_than':
                    filtered_data = filtered_data[filtered_data[column] < value]
                elif condition == 'contains':
                    filtered_data = filtered_data[filtered_data[column].astype(str).str.contains(str(value), na=False)]
                elif condition == 'in':
                    filtered_data = filtered_data[filtered_data[column].isin(value)]
                elif condition == 'not_null':
                    filtered_data = filtered_data[filtered_data[column].notna()]
                elif condition == 'is_null':
                    filtered_data = filtered_data[filtered_data[column].isna()]
                
                filtered_count = len(filtered_data)
                self.logger.info(f"Filter applied: {initial_count} -> {filtered_count} records", "extract")
                
            except Exception as e:
                self.logger.warning(f"Failed to apply filter {filter_config}: {str(e)}", "extract")
        
        return filtered_data
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the current data"""
        if self.data is None:
            return {}
        
        summary = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'memory_usage': self.data.memory_usage(deep=True).sum(),
            'null_counts': self.data.isnull().sum().to_dict(),
            'duplicate_rows': self.data.duplicated().sum()
        }
        
        # Add numeric column statistics
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            summary['numeric_stats'] = self.data[numeric_columns].describe().to_dict()
        
        return summary
