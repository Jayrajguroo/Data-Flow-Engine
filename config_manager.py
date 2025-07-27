import os
import yaml
import json
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path


class ConfigManager:
    """Manages pipeline configuration loading, validation, and storage"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Schema for validating pipeline configurations
        self.config_schema = {
            'required_sections': ['extraction', 'transformation', 'loading'],
            'optional_sections': ['validation', 'logging'],
            'extraction_schema': {
                'required': ['sources'],
                'optional': ['filters', 'fail_on_source_error']
            },
            'transformation_schema': {
                'required': [],
                'optional': ['steps', 'fail_on_step_error']
            },
            'loading_schema': {
                'required': ['targets'],
                'optional': ['fail_on_target_error']
            },
            'validation_schema': {
                'required': [],
                'optional': ['pre_processing', 'post_processing', 'schema', 'quality', 'business', 'statistical']
            }
        }
    
    def load_config(self, config_name: str = "default") -> Dict[str, Any]:
        """
        Load pipeline configuration from file
        
        Args:
            config_name: Name of the configuration file (without extension)
            
        Returns:
            Dictionary containing pipeline configuration
        """
        config_file = self.config_dir / f"{config_name}.yaml"
        
        if not config_file.exists():
            # Try .json extension
            config_file = self.config_dir / f"{config_name}.json"
            
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_name}")
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                if config_file.suffix.lower() == '.json':
                    config = json.load(f)
                else:
                    config = yaml.safe_load(f)
            
            # Validate configuration
            is_valid, errors = self.validate_config(config)
            if not is_valid:
                raise ValueError(f"Invalid configuration: {'; '.join(errors)}")
            
            # Apply environment variable substitution
            config = self._substitute_env_vars(config)
            
            return config
            
        except Exception as e:
            if isinstance(e, (ValueError, FileNotFoundError)):
                raise
            raise Exception(f"Failed to load configuration {config_name}: {str(e)}")
    
    def save_config(self, config: Dict[str, Any], config_name: str, format: str = "yaml") -> str:
        """
        Save pipeline configuration to file
        
        Args:
            config: Configuration dictionary to save
            config_name: Name for the configuration file
            format: File format ('yaml' or 'json')
            
        Returns:
            Path to saved configuration file
        """
        # Validate configuration before saving
        is_valid, errors = self.validate_config(config)
        if not is_valid:
            raise ValueError(f"Invalid configuration: {'; '.join(errors)}")
        
        if format.lower() == 'json':
            config_file = self.config_dir / f"{config_name}.json"
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        else:
            config_file = self.config_dir / f"{config_name}.yaml"
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
        
        return str(config_file)
    
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate pipeline configuration structure and content
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if not isinstance(config, dict):
            errors.append("Configuration must be a dictionary")
            return False, errors
        
        # Check required sections
        for section in self.config_schema['required_sections']:
            if section not in config:
                errors.append(f"Required section '{section}' is missing")
        
        # Validate extraction section
        if 'extraction' in config:
            extraction_errors = self._validate_extraction_config(config['extraction'])
            errors.extend(extraction_errors)
        
        # Validate transformation section
        if 'transformation' in config:
            transformation_errors = self._validate_transformation_config(config['transformation'])
            errors.extend(transformation_errors)
        
        # Validate loading section
        if 'loading' in config:
            loading_errors = self._validate_loading_config(config['loading'])
            errors.extend(loading_errors)
        
        # Validate validation section
        if 'validation' in config:
            validation_errors = self._validate_validation_config(config['validation'])
            errors.extend(validation_errors)
        
        return len(errors) == 0, errors
    
    def _validate_extraction_config(self, extraction_config: Dict[str, Any]) -> List[str]:
        """Validate extraction configuration"""
        errors = []
        
        if not isinstance(extraction_config, dict):
            errors.append("Extraction configuration must be a dictionary")
            return errors
        
        # Check required fields
        if 'sources' not in extraction_config:
            errors.append("Extraction section must contain 'sources'")
            return errors
        
        sources = extraction_config['sources']
        if not isinstance(sources, list) or len(sources) == 0:
            errors.append("Extraction sources must be a non-empty list")
            return errors
        
        # Validate each source
        for i, source in enumerate(sources):
            source_errors = self._validate_source_config(source, i)
            errors.extend(source_errors)
        
        return errors
    
    def _validate_source_config(self, source: Dict[str, Any], index: int) -> List[str]:
        """Validate individual source configuration"""
        errors = []
        source_prefix = f"Source {index}"
        
        if not isinstance(source, dict):
            errors.append(f"{source_prefix}: must be a dictionary")
            return errors
        
        # Check required fields
        if 'type' not in source:
            errors.append(f"{source_prefix}: 'type' is required")
            return errors
        
        source_type = source['type']
        valid_types = ['csv', 'json', 'excel', 'api', 'database', 'parquet']
        
        if source_type not in valid_types:
            errors.append(f"{source_prefix}: type must be one of {valid_types}")
        
        # Type-specific validation
        if source_type in ['csv', 'json', 'excel', 'parquet']:
            if 'path' not in source:
                errors.append(f"{source_prefix}: 'path' is required for {source_type} type")
        
        elif source_type == 'api':
            if 'url' not in source:
                errors.append(f"{source_prefix}: 'url' is required for API type")
        
        elif source_type == 'database':
            if 'query' not in source:
                errors.append(f"{source_prefix}: 'query' is required for database type")
            
            if not source.get('connection_string') and not source.get('db_type'):
                errors.append(f"{source_prefix}: 'connection_string' or 'db_type' is required for database type")
        
        return errors
    
    def _validate_transformation_config(self, transformation_config: Dict[str, Any]) -> List[str]:
        """Validate transformation configuration"""
        errors = []
        
        if not isinstance(transformation_config, dict):
            errors.append("Transformation configuration must be a dictionary")
            return errors
        
        # Validate transformation steps if present
        if 'steps' in transformation_config:
            steps = transformation_config['steps']
            if not isinstance(steps, list):
                errors.append("Transformation steps must be a list")
                return errors
            
            for i, step in enumerate(steps):
                step_errors = self._validate_transformation_step(step, i)
                errors.extend(step_errors)
        
        return errors
    
    def _validate_transformation_step(self, step: Dict[str, Any], index: int) -> List[str]:
        """Validate individual transformation step"""
        errors = []
        step_prefix = f"Transformation step {index}"
        
        if not isinstance(step, dict):
            errors.append(f"{step_prefix}: must be a dictionary")
            return errors
        
        if 'type' not in step:
            errors.append(f"{step_prefix}: 'type' is required")
            return errors
        
        step_type = step['type']
        valid_types = [
            'clean', 'impute', 'scale', 'encode', 'feature_engineering',
            'outlier_detection', 'feature_selection', 'dimensionality_reduction', 'custom'
        ]
        
        if step_type not in valid_types:
            errors.append(f"{step_prefix}: type must be one of {valid_types}")
        
        return errors
    
    def _validate_loading_config(self, loading_config: Dict[str, Any]) -> List[str]:
        """Validate loading configuration"""
        errors = []
        
        if not isinstance(loading_config, dict):
            errors.append("Loading configuration must be a dictionary")
            return errors
        
        if 'targets' not in loading_config:
            errors.append("Loading section must contain 'targets'")
            return errors
        
        targets = loading_config['targets']
        if not isinstance(targets, list) or len(targets) == 0:
            errors.append("Loading targets must be a non-empty list")
            return errors
        
        # Validate each target
        for i, target in enumerate(targets):
            target_errors = self._validate_target_config(target, i)
            errors.extend(target_errors)
        
        return errors
    
    def _validate_target_config(self, target: Dict[str, Any], index: int) -> List[str]:
        """Validate individual target configuration"""
        errors = []
        target_prefix = f"Target {index}"
        
        if not isinstance(target, dict):
            errors.append(f"{target_prefix}: must be a dictionary")
            return errors
        
        if 'type' not in target:
            errors.append(f"{target_prefix}: 'type' is required")
            return errors
        
        target_type = target['type']
        valid_types = ['csv', 'json', 'excel', 'parquet', 'database', 'api', 'multiple_files']
        
        if target_type not in valid_types:
            errors.append(f"{target_prefix}: type must be one of {valid_types}")
        
        # Type-specific validation
        if target_type in ['csv', 'json', 'excel', 'parquet']:
            if 'path' not in target:
                errors.append(f"{target_prefix}: 'path' is required for {target_type} type")
        
        elif target_type == 'api':
            if 'url' not in target:
                errors.append(f"{target_prefix}: 'url' is required for API type")
        
        elif target_type == 'database':
            if 'table_name' not in target:
                errors.append(f"{target_prefix}: 'table_name' is required for database type")
        
        elif target_type == 'multiple_files':
            if 'group_by' not in target:
                errors.append(f"{target_prefix}: 'group_by' is required for multiple_files type")
        
        return errors
    
    def _validate_validation_config(self, validation_config: Dict[str, Any]) -> List[str]:
        """Validate validation configuration"""
        errors = []
        
        if not isinstance(validation_config, dict):
            errors.append("Validation configuration must be a dictionary")
            return errors
        
        # Validate each validation phase
        for phase in ['pre_processing', 'post_processing']:
            if phase in validation_config:
                phase_config = validation_config[phase]
                if not isinstance(phase_config, dict):
                    errors.append(f"Validation {phase} must be a dictionary")
                    continue
                
                # Validate rule types
                for rule_type in ['schema', 'quality', 'business', 'statistical']:
                    if rule_type in phase_config:
                        rules = phase_config[rule_type]
                        if not isinstance(rules, list):
                            errors.append(f"Validation {phase}.{rule_type} must be a list")
        
        return errors
    
    def _substitute_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Substitute environment variables in configuration"""
        def substitute_recursive(obj):
            if isinstance(obj, dict):
                return {key: substitute_recursive(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [substitute_recursive(item) for item in obj]
            elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
                # Extract environment variable name and default value
                env_expr = obj[2:-1]  # Remove ${ and }
                if ':' in env_expr:
                    env_var, default_value = env_expr.split(':', 1)
                    return os.getenv(env_var.strip(), default_value.strip())
                else:
                    return os.getenv(env_expr.strip(), obj)
            else:
                return obj
        
        return substitute_recursive(config)
    
    def list_available_configs(self) -> List[str]:
        """List all available configuration files"""
        configs = []
        
        for file_path in self.config_dir.glob("*.yaml"):
            configs.append(file_path.stem)
        
        for file_path in self.config_dir.glob("*.json"):
            if f"{file_path.stem}.yaml" not in [f"{p.stem}.yaml" for p in self.config_dir.glob("*.yaml")]:
                configs.append(file_path.stem)
        
        return sorted(configs)
    
    def delete_config(self, config_name: str) -> bool:
        """
        Delete a configuration file
        
        Args:
            config_name: Name of the configuration to delete
            
        Returns:
            True if deleted successfully, False if not found
        """
        yaml_file = self.config_dir / f"{config_name}.yaml"
        json_file = self.config_dir / f"{config_name}.json"
        
        deleted = False
        
        if yaml_file.exists():
            yaml_file.unlink()
            deleted = True
        
        if json_file.exists():
            json_file.unlink()
            deleted = True
        
        return deleted
    
    def get_config_template(self) -> Dict[str, Any]:
        """Get a template configuration with common settings"""
        return {
            "extraction": {
                "sources": [
                    {
                        "name": "sample_csv",
                        "type": "csv",
                        "path": "data/input.csv",
                        "encoding": "utf-8",
                        "separator": ",",
                        "header": 0
                    }
                ],
                "fail_on_source_error": True
            },
            "transformation": {
                "steps": [
                    {
                        "name": "clean_data",
                        "type": "clean",
                        "remove_duplicates": True,
                        "drop_columns": [],
                        "dtype_conversions": {}
                    },
                    {
                        "name": "handle_missing",
                        "type": "impute",
                        "column_strategies": {
                            "numeric_column": "mean",
                            "categorical_column": "mode"
                        }
                    }
                ],
                "fail_on_step_error": True
            },
            "loading": {
                "targets": [
                    {
                        "name": "output_csv",
                        "type": "csv",
                        "path": "data/output.csv",
                        "include_index": False,
                        "include_header": True
                    }
                ],
                "fail_on_target_error": True
            },
            "validation": {
                "pre_processing": {
                    "schema": [
                        {
                            "name": "check_required_columns",
                            "type": "required_columns",
                            "columns": ["id", "name", "value"],
                            "critical": True
                        }
                    ],
                    "quality": [
                        {
                            "name": "check_nulls",
                            "type": "null_check",
                            "columns": ["id"],
                            "allow_nulls": False,
                            "critical": True
                        }
                    ]
                },
                "post_processing": {
                    "quality": [
                        {
                            "name": "check_completeness",
                            "type": "completeness",
                            "min_completeness": 95,
                            "critical": False
                        }
                    ]
                }
            },
            "logging": {
                "level": "INFO",
                "log_to_database": True,
                "log_to_file": False
            }
        }
