import pandas as pd
import os
import json
from typing import Dict, Any, Optional
from sqlalchemy import create_engine, text
import sqlite3
from datetime import datetime


class DataLoader:
    """Handles loading data to various destinations"""
    
    def __init__(self, config: Dict[str, Any], logger):
        self.config = config
        self.logger = logger
    
    def load(self, data: pd.DataFrame, target_config: Dict[str, Any]):
        """
        Load data to a single target destination
        
        Args:
            data: DataFrame to load
            target_config: Configuration for the target destination
        """
        target_type = target_config.get('type')
        
        if target_type == 'csv':
            self._load_to_csv(data, target_config)
        elif target_type == 'json':
            self._load_to_json(data, target_config)
        elif target_type == 'excel':
            self._load_to_excel(data, target_config)
        elif target_type == 'parquet':
            self._load_to_parquet(data, target_config)
        elif target_type == 'database':
            self._load_to_database(data, target_config)
        elif target_type == 'api':
            self._load_to_api(data, target_config)
        elif target_type == 'multiple_files':
            self._load_to_multiple_files(data, target_config)
        else:
            raise ValueError(f"Unsupported target type: {target_type}")
    
    def _load_to_csv(self, data: pd.DataFrame, config: Dict[str, Any]):
        """Load data to CSV file"""
        file_path = config.get('path')
        if not file_path:
            raise ValueError("CSV file path is required")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # CSV writing options
        options = {
            'sep': config.get('separator', ','),
            'encoding': config.get('encoding', 'utf-8'),
            'index': config.get('include_index', False),
            'header': config.get('include_header', True),
            'mode': config.get('mode', 'w'),
            'na_rep': config.get('na_representation', ''),
            'float_format': config.get('float_format'),
            'date_format': config.get('date_format')
        }
        
        # Remove None values
        options = {k: v for k, v in options.items() if v is not None}
        
        try:
            data.to_csv(file_path, **options)
            self.logger.info(f"Successfully saved {len(data)} records to CSV: {file_path}")
            
            # Add timestamp suffix if configured
            if config.get('add_timestamp', False):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                base_path, ext = os.path.splitext(file_path)
                timestamped_path = f"{base_path}_{timestamp}{ext}"
                data.to_csv(timestamped_path, **options)
                self.logger.info(f"Also saved timestamped copy: {timestamped_path}")
                
        except Exception as e:
            raise Exception(f"Failed to save CSV file {file_path}: {str(e)}")
    
    def _load_to_json(self, data: pd.DataFrame, config: Dict[str, Any]):
        """Load data to JSON file"""
        file_path = config.get('path')
        if not file_path:
            raise ValueError("JSON file path is required")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        try:
            # Convert DataFrame to desired JSON format
            orient = config.get('orient', 'records')
            indent = config.get('indent', 2)
            
            json_data = data.to_json(orient=orient, date_format='iso', indent=indent)
            
            # If we want pretty printed JSON, parse and re-dump
            if indent:
                json_obj = json.loads(json_data)
                with open(file_path, 'w', encoding=config.get('encoding', 'utf-8')) as f:
                    json.dump(json_obj, f, indent=indent, ensure_ascii=False)
            else:
                with open(file_path, 'w', encoding=config.get('encoding', 'utf-8')) as f:
                    f.write(json_data)
            
            self.logger.info(f"Successfully saved {len(data)} records to JSON: {file_path}")
            
        except Exception as e:
            raise Exception(f"Failed to save JSON file {file_path}: {str(e)}")
    
    def _load_to_excel(self, data: pd.DataFrame, config: Dict[str, Any]):
        """Load data to Excel file"""
        file_path = config.get('path')
        if not file_path:
            raise ValueError("Excel file path is required")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        try:
            # Excel writing options
            sheet_name = config.get('sheet_name', 'Sheet1')
            index = config.get('include_index', False)
            header = config.get('include_header', True)
            
            # Check if we need to append to existing file
            mode = config.get('mode', 'w')
            if mode == 'a' and os.path.exists(file_path):
                with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
                    data.to_excel(writer, sheet_name=sheet_name, index=index, header=header)
            else:
                with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                    data.to_excel(writer, sheet_name=sheet_name, index=index, header=header)
            
            self.logger.info(f"Successfully saved {len(data)} records to Excel: {file_path}")
            
        except Exception as e:
            raise Exception(f"Failed to save Excel file {file_path}: {str(e)}")
    
    def _load_to_parquet(self, data: pd.DataFrame, config: Dict[str, Any]):
        """Load data to Parquet file"""
        file_path = config.get('path')
        if not file_path:
            raise ValueError("Parquet file path is required")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        try:
            # Parquet writing options
            compression = config.get('compression', 'snappy')
            index = config.get('include_index', False)
            
            data.to_parquet(file_path, compression=compression, index=index)
            self.logger.info(f"Successfully saved {len(data)} records to Parquet: {file_path}")
            
        except Exception as e:
            raise Exception(f"Failed to save Parquet file {file_path}: {str(e)}")
    
    def _load_to_database(self, data: pd.DataFrame, config: Dict[str, Any]):
        """Load data to database table"""
        connection_string = config.get('connection_string')
        table_name = config.get('table_name')
        
        if not table_name:
            raise ValueError("Database table name is required")
        
        if not connection_string:
            # Try to build connection string from environment variables
            db_type = config.get('db_type', 'postgresql')
            if db_type == 'postgresql':
                connection_string = self._build_postgres_connection_string(config)
            elif db_type == 'sqlite':
                db_path = config.get('database', 'database.db')
                connection_string = f"sqlite:///{db_path}"
            else:
                raise ValueError("Database connection string or configuration required")
        
        try:
            engine = create_engine(connection_string)
            
            # Database loading options
            if_exists = config.get('if_exists', 'replace')  # replace, append, fail
            index = config.get('include_index', False)
            chunksize = config.get('chunksize', None)
            method = config.get('method', None)
            
            # Load data to database
            data.to_sql(
                name=table_name,
                con=engine,
                if_exists=if_exists,
                index=index,
                chunksize=chunksize,
                method=method
            )
            
            self.logger.info(f"Successfully loaded {len(data)} records to database table: {table_name}")
            
            # Execute post-load SQL if provided
            post_load_sql = config.get('post_load_sql')
            if post_load_sql:
                with engine.connect() as conn:
                    conn.execute(text(post_load_sql))
                    conn.commit()
                self.logger.info("Executed post-load SQL statements")
                
        except Exception as e:
            raise Exception(f"Database loading failed: {str(e)}")
    
    def _load_to_api(self, data: pd.DataFrame, config: Dict[str, Any]):
        """Load data to API endpoint"""
        import requests
        
        url = config.get('url')
        if not url:
            raise ValueError("API URL is required")
        
        method = config.get('method', 'POST').upper()
        headers = config.get('headers', {})
        timeout = config.get('timeout', 30)
        batch_size = config.get('batch_size', 100)
        
        # Add API key if provided
        api_key = config.get('api_key') or os.getenv('API_KEY')
        if api_key:
            auth_type = config.get('auth_type', 'header')
            if auth_type == 'header':
                auth_header = config.get('auth_header', 'Authorization')
                headers[auth_header] = f"Bearer {api_key}" if not api_key.startswith('Bearer') else api_key
        
        try:
            # Convert DataFrame to list of dictionaries
            records = data.to_dict('records')
            
            # Send data in batches
            total_records = len(records)
            successful_records = 0
            
            for i in range(0, total_records, batch_size):
                batch = records[i:i + batch_size]
                
                payload = {
                    'data': batch,
                    'batch_info': {
                        'batch_number': i // batch_size + 1,
                        'batch_size': len(batch),
                        'total_batches': (total_records + batch_size - 1) // batch_size
                    }
                }
                
                if method == 'POST':
                    response = requests.post(url, json=payload, headers=headers, timeout=timeout)
                elif method == 'PUT':
                    response = requests.put(url, json=payload, headers=headers, timeout=timeout)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                response.raise_for_status()
                successful_records += len(batch)
                
                self.logger.info(f"Successfully sent batch {i // batch_size + 1}, records: {successful_records}/{total_records}")
            
            self.logger.info(f"Successfully loaded {successful_records} records to API: {url}")
            
        except Exception as e:
            raise Exception(f"API loading failed: {str(e)}")
    
    def _load_to_multiple_files(self, data: pd.DataFrame, config: Dict[str, Any]):
        """Load data to multiple files based on grouping"""
        group_by = config.get('group_by')
        if not group_by or group_by not in data.columns:
            raise ValueError(f"Group by column '{group_by}' is required and must exist in data")
        
        base_path = config.get('base_path', '.')
        file_format = config.get('file_format', 'csv')
        filename_template = config.get('filename_template', '{group_value}.{format}')
        
        # Ensure base directory exists
        os.makedirs(base_path, exist_ok=True)
        
        try:
            grouped_data = data.groupby(group_by)
            
            for group_value, group_df in grouped_data:
                # Generate filename
                safe_group_value = str(group_value).replace('/', '_').replace('\\', '_')
                filename = filename_template.format(group_value=safe_group_value, format=file_format)
                file_path = os.path.join(base_path, filename)
                
                # Create individual file config
                file_config = config.copy()
                file_config['type'] = file_format
                file_config['path'] = file_path
                
                # Load group data to individual file
                if file_format == 'csv':
                    self._load_to_csv(group_df, file_config)
                elif file_format == 'json':
                    self._load_to_json(group_df, file_config)
                elif file_format == 'excel':
                    self._load_to_excel(group_df, file_config)
                elif file_format == 'parquet':
                    self._load_to_parquet(group_df, file_config)
                
                self.logger.info(f"Saved group '{group_value}' with {len(group_df)} records to {file_path}")
            
            total_groups = len(grouped_data)
            self.logger.info(f"Successfully split data into {total_groups} files based on '{group_by}'")
            
        except Exception as e:
            raise Exception(f"Multiple file loading failed: {str(e)}")
    
    def _build_postgres_connection_string(self, config: Dict[str, Any]) -> str:
        """Build PostgreSQL connection string from config or environment variables"""
        host = config.get('host') or os.getenv('PGHOST', 'localhost')
        port = config.get('port') or os.getenv('PGPORT', '5432')
        database = config.get('database') or os.getenv('PGDATABASE', 'postgres')
        username = config.get('username') or os.getenv('PGUSER', 'postgres')
        password = config.get('password') or os.getenv('PGPASSWORD', '')
        
        return f"postgresql://{username}:{password}@{host}:{port}/{database}"
    
    def validate_target_config(self, target_config: Dict[str, Any]) -> tuple[bool, list]:
        """
        Validate target configuration
        
        Args:
            target_config: Target configuration to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        target_type = target_config.get('type')
        
        if not target_type:
            errors.append("Target type is required")
            return False, errors
        
        # Type-specific validation
        if target_type in ['csv', 'json', 'excel', 'parquet']:
            if not target_config.get('path'):
                errors.append(f"{target_type.upper()} file path is required")
        
        elif target_type == 'database':
            if not target_config.get('table_name'):
                errors.append("Database table name is required")
            
            if not target_config.get('connection_string') and not target_config.get('db_type'):
                errors.append("Database connection string or db_type is required")
        
        elif target_type == 'api':
            if not target_config.get('url'):
                errors.append("API URL is required")
        
        elif target_type == 'multiple_files':
            if not target_config.get('group_by'):
                errors.append("Group by column is required for multiple files")
        
        return len(errors) == 0, errors
