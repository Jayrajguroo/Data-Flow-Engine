import pandas as pd
import requests
import json
import os
from typing import Dict, Any, Optional
from urllib.parse import urljoin
import sqlite3
from sqlalchemy import create_engine
import time


class DataExtractor:
    """Handles data extraction from various sources"""
    
    def __init__(self, config: Dict[str, Any], logger):
        self.config = config
        self.logger = logger
    
    def extract(self, source_config: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        Extract data from a single source
        
        Args:
            source_config: Configuration for the data source
            
        Returns:
            DataFrame containing extracted data or None if extraction fails
        """
        source_type = source_config.get('type')
        
        if source_type == 'csv':
            return self._extract_csv(source_config)
        elif source_type == 'json':
            return self._extract_json(source_config)
        elif source_type == 'excel':
            return self._extract_excel(source_config)
        elif source_type == 'api':
            return self._extract_api(source_config)
        elif source_type == 'database':
            return self._extract_database(source_config)
        elif source_type == 'parquet':
            return self._extract_parquet(source_config)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
    
    def _extract_csv(self, config: Dict[str, Any]) -> pd.DataFrame:
        """Extract data from CSV file"""
        file_path = config.get('path')
        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        # CSV reading options
        options = {
            'sep': config.get('separator', ','),
            'encoding': config.get('encoding', 'utf-8'),
            'header': config.get('header', 0),
            'skiprows': config.get('skip_rows', 0),
            'nrows': config.get('max_rows'),
            'usecols': config.get('columns'),
            'dtype': config.get('dtypes', {}),
            'parse_dates': config.get('date_columns', []),
            'na_values': config.get('na_values', [])
        }
        
        # Remove None values
        options = {k: v for k, v in options.items() if v is not None}
        
        try:
            data = pd.read_csv(file_path, **options)
            self.logger.info(f"Successfully read CSV file: {file_path}, shape: {data.shape}")
            return data
        except Exception as e:
            raise Exception(f"Failed to read CSV file {file_path}: {str(e)}")
    
    def _extract_json(self, config: Dict[str, Any]) -> pd.DataFrame:
        """Extract data from JSON file"""
        file_path = config.get('path')
        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError(f"JSON file not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding=config.get('encoding', 'utf-8')) as f:
                data = json.load(f)
            
            # Handle different JSON structures
            json_path = config.get('json_path')
            if json_path:
                # Navigate to nested data using dot notation
                for key in json_path.split('.'):
                    data = data[key]
            
            # Convert to DataFrame
            if isinstance(data, list):
                df = pd.json_normalize(data)
            elif isinstance(data, dict):
                df = pd.json_normalize([data])
            else:
                raise ValueError("JSON data must be a list or dictionary")
            
            self.logger.info(f"Successfully read JSON file: {file_path}, shape: {df.shape}")
            return df
            
        except Exception as e:
            raise Exception(f"Failed to read JSON file {file_path}: {str(e)}")
    
    def _extract_excel(self, config: Dict[str, Any]) -> pd.DataFrame:
        """Extract data from Excel file"""
        file_path = config.get('path')
        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError(f"Excel file not found: {file_path}")
        
        options = {
            'sheet_name': config.get('sheet_name', 0),
            'header': config.get('header', 0),
            'skiprows': config.get('skip_rows', 0),
            'nrows': config.get('max_rows'),
            'usecols': config.get('columns'),
            'dtype': config.get('dtypes', {}),
            'parse_dates': config.get('date_columns', []),
            'na_values': config.get('na_values', [])
        }
        
        # Remove None values
        options = {k: v for k, v in options.items() if v is not None}
        
        try:
            data = pd.read_excel(file_path, **options)
            self.logger.info(f"Successfully read Excel file: {file_path}, shape: {data.shape}")
            return data
        except Exception as e:
            raise Exception(f"Failed to read Excel file {file_path}: {str(e)}")
    
    def _extract_api(self, config: Dict[str, Any]) -> pd.DataFrame:
        """Extract data from API endpoint"""
        url = config.get('url')
        if not url:
            raise ValueError("API URL is required")
        
        # Request configuration
        method = config.get('method', 'GET').upper()
        headers = config.get('headers', {})
        params = config.get('params', {})
        data = config.get('data', {})
        timeout = config.get('timeout', 30)
        retries = config.get('retries', 3)
        retry_delay = config.get('retry_delay', 1)
        
        # Add API key if provided
        api_key = config.get('api_key') or os.getenv('API_KEY')
        if api_key:
            auth_type = config.get('auth_type', 'header')
            if auth_type == 'header':
                auth_header = config.get('auth_header', 'Authorization')
                headers[auth_header] = f"Bearer {api_key}" if not api_key.startswith('Bearer') else api_key
            elif auth_type == 'param':
                auth_param = config.get('auth_param', 'api_key')
                params[auth_param] = api_key
        
        # Pagination support
        pagination = config.get('pagination', {})
        all_data = []
        page = pagination.get('start_page', 1)
        max_pages = pagination.get('max_pages', 1)
        
        for page_num in range(max_pages):
            current_params = params.copy()
            
            # Add pagination parameters
            if pagination.get('enabled', False):
                page_param = pagination.get('page_param', 'page')
                size_param = pagination.get('size_param', 'size')
                page_size = pagination.get('page_size', 100)
                
                current_params[page_param] = page + page_num
                current_params[size_param] = page_size
            
            # Make request with retries
            for attempt in range(retries):
                try:
                    if method == 'GET':
                        response = requests.get(url, headers=headers, params=current_params, timeout=timeout)
                    elif method == 'POST':
                        response = requests.post(url, headers=headers, params=current_params, json=data, timeout=timeout)
                    else:
                        raise ValueError(f"Unsupported HTTP method: {method}")
                    
                    response.raise_for_status()
                    break
                    
                except requests.exceptions.RequestException as e:
                    if attempt == retries - 1:
                        raise Exception(f"API request failed after {retries} attempts: {str(e)}")
                    self.logger.warning(f"API request attempt {attempt + 1} failed, retrying in {retry_delay}s: {str(e)}")
                    time.sleep(retry_delay)
            
            # Process response
            try:
                response_data = response.json()
            except json.JSONDecodeError as e:
                raise Exception(f"Failed to parse API response as JSON: {str(e)}")
            
            # Extract data from response using JSON path
            json_path = config.get('data_path', 'data')
            if json_path:
                for key in json_path.split('.'):
                    if key in response_data:
                        response_data = response_data[key]
                    else:
                        response_data = []
                        break
            
            if not response_data:
                break
            
            all_data.extend(response_data if isinstance(response_data, list) else [response_data])
            
            # Check if we should continue paginating
            if not pagination.get('enabled', False):
                break
            
            # Check for end of data
            if len(response_data) < pagination.get('page_size', 100):
                break
        
        if not all_data:
            self.logger.warning(f"No data received from API: {url}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.json_normalize(all_data)
        self.logger.info(f"Successfully extracted data from API: {url}, shape: {df.shape}")
        return df
    
    def _extract_database(self, config: Dict[str, Any]) -> pd.DataFrame:
        """Extract data from database"""
        connection_string = config.get('connection_string')
        query = config.get('query')
        
        if not connection_string:
            # Try to build connection string from environment variables
            db_type = config.get('db_type', 'postgresql')
            if db_type == 'postgresql':
                connection_string = self._build_postgres_connection_string(config)
            elif db_type == 'sqlite':
                connection_string = f"sqlite:///{config.get('database', 'database.db')}"
            else:
                raise ValueError("Database connection string or configuration required")
        
        if not query:
            raise ValueError("Database query is required")
        
        try:
            engine = create_engine(connection_string)
            
            # Execute query with parameters
            params = config.get('params', {})
            df = pd.read_sql(query, engine, params=params)
            
            self.logger.info(f"Successfully executed database query, shape: {df.shape}")
            return df
            
        except Exception as e:
            raise Exception(f"Database extraction failed: {str(e)}")
    
    def _extract_parquet(self, config: Dict[str, Any]) -> pd.DataFrame:
        """Extract data from Parquet file"""
        file_path = config.get('path')
        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError(f"Parquet file not found: {file_path}")
        
        try:
            columns = config.get('columns')
            df = pd.read_parquet(file_path, columns=columns)
            self.logger.info(f"Successfully read Parquet file: {file_path}, shape: {df.shape}")
            return df
        except Exception as e:
            raise Exception(f"Failed to read Parquet file {file_path}: {str(e)}")
    
    def _build_postgres_connection_string(self, config: Dict[str, Any]) -> str:
        """Build PostgreSQL connection string from config or environment variables"""
        host = config.get('host') or os.getenv('PGHOST', 'localhost')
        port = config.get('port') or os.getenv('PGPORT', '5432')
        database = config.get('database') or os.getenv('PGDATABASE', 'postgres')
        username = config.get('username') or os.getenv('PGUSER', 'postgres')
        password = config.get('password') or os.getenv('PGPASSWORD', '')
        
        return f"postgresql://{username}:{password}@{host}:{port}/{database}"
