import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
import re
from datetime import datetime


class DataValidator:
    """Handles data validation and quality checks"""
    
    def __init__(self, config: Dict[str, Any], logger):
        self.config = config
        self.logger = logger
    
    def validate(self, data: pd.DataFrame, validation_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Validate data based on configuration
        
        Args:
            data: DataFrame to validate
            validation_config: Validation rules configuration
            
        Returns:
            List of validation results
        """
        results = []
        
        # Schema validation
        schema_rules = validation_config.get('schema', [])
        for rule in schema_rules:
            result = self._validate_schema(data, rule)
            results.append(result)
        
        # Data quality rules
        quality_rules = validation_config.get('quality', [])
        for rule in quality_rules:
            result = self._validate_quality(data, rule)
            results.append(result)
        
        # Business rules
        business_rules = validation_config.get('business', [])
        for rule in business_rules:
            result = self._validate_business_rule(data, rule)
            results.append(result)
        
        # Statistical validation
        statistical_rules = validation_config.get('statistical', [])
        for rule in statistical_rules:
            result = self._validate_statistical(data, rule)
            results.append(result)
        
        return results
    
    def _validate_schema(self, data: pd.DataFrame, rule: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data schema"""
        rule_type = rule.get('type')
        rule_name = rule.get('name', f'schema_{rule_type}')
        
        try:
            if rule_type == 'required_columns':
                return self._validate_required_columns(data, rule, rule_name)
            elif rule_type == 'column_types':
                return self._validate_column_types(data, rule, rule_name)
            elif rule_type == 'column_count':
                return self._validate_column_count(data, rule, rule_name)
            elif rule_type == 'row_count':
                return self._validate_row_count(data, rule, rule_name)
            else:
                return {
                    'rule': rule_name,
                    'passed': False,
                    'message': f'Unknown schema validation type: {rule_type}',
                    'critical': rule.get('critical', False)
                }
        except Exception as e:
            return {
                'rule': rule_name,
                'passed': False,
                'message': f'Schema validation error: {str(e)}',
                'critical': rule.get('critical', False)
            }
    
    def _validate_required_columns(self, data: pd.DataFrame, rule: Dict[str, Any], rule_name: str) -> Dict[str, Any]:
        """Validate that required columns are present"""
        required_columns = rule.get('columns', [])
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        passed = len(missing_columns) == 0
        message = 'All required columns are present'
        
        if not passed:
            message = f'Missing required columns: {missing_columns}'
        
        return {
            'rule': rule_name,
            'passed': passed,
            'message': message,
            'critical': rule.get('critical', True),
            'details': {
                'required_columns': required_columns,
                'missing_columns': missing_columns,
                'present_columns': [col for col in required_columns if col in data.columns]
            }
        }
    
    def _validate_column_types(self, data: pd.DataFrame, rule: Dict[str, Any], rule_name: str) -> Dict[str, Any]:
        """Validate column data types"""
        expected_types = rule.get('types', {})
        type_mismatches = []
        
        for column, expected_type in expected_types.items():
            if column not in data.columns:
                continue
            
            actual_type = str(data[column].dtype)
            
            # Check type compatibility
            if not self._is_compatible_type(actual_type, expected_type):
                type_mismatches.append({
                    'column': column,
                    'expected': expected_type,
                    'actual': actual_type
                })
        
        passed = len(type_mismatches) == 0
        message = 'All column types match expectations'
        
        if not passed:
            message = f'Type mismatches found: {type_mismatches}'
        
        return {
            'rule': rule_name,
            'passed': passed,
            'message': message,
            'critical': rule.get('critical', False),
            'details': {
                'expected_types': expected_types,
                'type_mismatches': type_mismatches
            }
        }
    
    def _validate_column_count(self, data: pd.DataFrame, rule: Dict[str, Any], rule_name: str) -> Dict[str, Any]:
        """Validate number of columns"""
        actual_count = len(data.columns)
        min_columns = rule.get('min')
        max_columns = rule.get('max')
        exact_columns = rule.get('exact')
        
        passed = True
        messages = []
        
        if exact_columns is not None:
            if actual_count != exact_columns:
                passed = False
                messages.append(f'Expected exactly {exact_columns} columns, got {actual_count}')
        else:
            if min_columns is not None and actual_count < min_columns:
                passed = False
                messages.append(f'Expected at least {min_columns} columns, got {actual_count}')
            
            if max_columns is not None and actual_count > max_columns:
                passed = False
                messages.append(f'Expected at most {max_columns} columns, got {actual_count}')
        
        message = 'Column count is valid' if passed else '; '.join(messages)
        
        return {
            'rule': rule_name,
            'passed': passed,
            'message': message,
            'critical': rule.get('critical', False),
            'details': {
                'actual_count': actual_count,
                'min_columns': min_columns,
                'max_columns': max_columns,
                'exact_columns': exact_columns
            }
        }
    
    def _validate_row_count(self, data: pd.DataFrame, rule: Dict[str, Any], rule_name: str) -> Dict[str, Any]:
        """Validate number of rows"""
        actual_count = len(data)
        min_rows = rule.get('min')
        max_rows = rule.get('max')
        exact_rows = rule.get('exact')
        
        passed = True
        messages = []
        
        if exact_rows is not None:
            if actual_count != exact_rows:
                passed = False
                messages.append(f'Expected exactly {exact_rows} rows, got {actual_count}')
        else:
            if min_rows is not None and actual_count < min_rows:
                passed = False
                messages.append(f'Expected at least {min_rows} rows, got {actual_count}')
            
            if max_rows is not None and actual_count > max_rows:
                passed = False
                messages.append(f'Expected at most {max_rows} rows, got {actual_count}')
        
        message = 'Row count is valid' if passed else '; '.join(messages)
        
        return {
            'rule': rule_name,
            'passed': passed,
            'message': message,
            'critical': rule.get('critical', False),
            'details': {
                'actual_count': actual_count,
                'min_rows': min_rows,
                'max_rows': max_rows,
                'exact_rows': exact_rows
            }
        }
    
    def _validate_quality(self, data: pd.DataFrame, rule: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data quality"""
        rule_type = rule.get('type')
        rule_name = rule.get('name', f'quality_{rule_type}')
        
        try:
            if rule_type == 'null_check':
                return self._validate_null_values(data, rule, rule_name)
            elif rule_type == 'duplicate_check':
                return self._validate_duplicates(data, rule, rule_name)
            elif rule_type == 'value_range':
                return self._validate_value_range(data, rule, rule_name)
            elif rule_type == 'regex_pattern':
                return self._validate_regex_pattern(data, rule, rule_name)
            elif rule_type == 'uniqueness':
                return self._validate_uniqueness(data, rule, rule_name)
            elif rule_type == 'completeness':
                return self._validate_completeness(data, rule, rule_name)
            else:
                return {
                    'rule': rule_name,
                    'passed': False,
                    'message': f'Unknown quality validation type: {rule_type}',
                    'critical': rule.get('critical', False)
                }
        except Exception as e:
            return {
                'rule': rule_name,
                'passed': False,
                'message': f'Quality validation error: {str(e)}',
                'critical': rule.get('critical', False)
            }
    
    def _validate_null_values(self, data: pd.DataFrame, rule: Dict[str, Any], rule_name: str) -> Dict[str, Any]:
        """Validate null value constraints"""
        columns = rule.get('columns', [])
        max_null_percentage = rule.get('max_null_percentage', 0)
        allow_nulls = rule.get('allow_nulls', False)
        
        violations = []
        
        for column in columns:
            if column not in data.columns:
                continue
            
            null_count = data[column].isnull().sum()
            null_percentage = (null_count / len(data)) * 100 if len(data) > 0 else 0
            
            if not allow_nulls and null_count > 0:
                violations.append({
                    'column': column,
                    'issue': 'nulls_not_allowed',
                    'null_count': null_count,
                    'null_percentage': null_percentage
                })
            elif null_percentage > max_null_percentage:
                violations.append({
                    'column': column,
                    'issue': 'exceeds_max_null_percentage',
                    'null_count': null_count,
                    'null_percentage': null_percentage,
                    'max_allowed': max_null_percentage
                })
        
        passed = len(violations) == 0
        message = 'Null value constraints satisfied' if passed else f'Null value violations: {violations}'
        
        return {
            'rule': rule_name,
            'passed': passed,
            'message': message,
            'critical': rule.get('critical', False),
            'details': {
                'columns_checked': columns,
                'violations': violations
            }
        }
    
    def _validate_duplicates(self, data: pd.DataFrame, rule: Dict[str, Any], rule_name: str) -> Dict[str, Any]:
        """Validate duplicate constraints"""
        columns = rule.get('columns', [])
        allow_duplicates = rule.get('allow_duplicates', True)
        max_duplicate_percentage = rule.get('max_duplicate_percentage', 100)
        
        if not columns:
            # Check for duplicate rows across all columns
            duplicate_count = data.duplicated().sum()
        else:
            # Check for duplicates in specified columns
            existing_columns = [col for col in columns if col in data.columns]
            if not existing_columns:
                return {
                    'rule': rule_name,
                    'passed': False,
                    'message': f'None of the specified columns exist: {columns}',
                    'critical': rule.get('critical', False)
                }
            duplicate_count = data.duplicated(subset=existing_columns).sum()
        
        duplicate_percentage = (duplicate_count / len(data)) * 100 if len(data) > 0 else 0
        
        passed = True
        messages = []
        
        if not allow_duplicates and duplicate_count > 0:
            passed = False
            messages.append(f'Duplicates not allowed, found {duplicate_count} duplicates')
        
        if duplicate_percentage > max_duplicate_percentage:
            passed = False
            messages.append(f'Duplicate percentage {duplicate_percentage:.2f}% exceeds maximum {max_duplicate_percentage}%')
        
        message = 'Duplicate constraints satisfied' if passed else '; '.join(messages)
        
        return {
            'rule': rule_name,
            'passed': passed,
            'message': message,
            'critical': rule.get('critical', False),
            'details': {
                'columns_checked': columns,
                'duplicate_count': duplicate_count,
                'duplicate_percentage': duplicate_percentage,
                'total_rows': len(data)
            }
        }
    
    def _validate_value_range(self, data: pd.DataFrame, rule: Dict[str, Any], rule_name: str) -> Dict[str, Any]:
        """Validate value ranges for numeric columns"""
        columns = rule.get('columns', [])
        min_value = rule.get('min_value')
        max_value = rule.get('max_value')
        
        violations = []
        
        for column in columns:
            if column not in data.columns:
                continue
            
            if not pd.api.types.is_numeric_dtype(data[column]):
                violations.append({
                    'column': column,
                    'issue': 'not_numeric',
                    'dtype': str(data[column].dtype)
                })
                continue
            
            column_violations = []
            
            if min_value is not None:
                below_min = (data[column] < min_value).sum()
                if below_min > 0:
                    column_violations.append(f'{below_min} values below minimum {min_value}')
            
            if max_value is not None:
                above_max = (data[column] > max_value).sum()
                if above_max > 0:
                    column_violations.append(f'{above_max} values above maximum {max_value}')
            
            if column_violations:
                violations.append({
                    'column': column,
                    'issue': 'value_range',
                    'violations': column_violations
                })
        
        passed = len(violations) == 0
        message = 'Value range constraints satisfied' if passed else f'Value range violations: {violations}'
        
        return {
            'rule': rule_name,
            'passed': passed,
            'message': message,
            'critical': rule.get('critical', False),
            'details': {
                'columns_checked': columns,
                'min_value': min_value,
                'max_value': max_value,
                'violations': violations
            }
        }
    
    def _validate_regex_pattern(self, data: pd.DataFrame, rule: Dict[str, Any], rule_name: str) -> Dict[str, Any]:
        """Validate text patterns using regex"""
        columns = rule.get('columns', [])
        pattern = rule.get('pattern')
        
        if not pattern:
            return {
                'rule': rule_name,
                'passed': False,
                'message': 'Regex pattern is required',
                'critical': rule.get('critical', False)
            }
        
        violations = []
        
        try:
            compiled_pattern = re.compile(pattern)
        except re.error as e:
            return {
                'rule': rule_name,
                'passed': False,
                'message': f'Invalid regex pattern: {str(e)}',
                'critical': rule.get('critical', False)
            }
        
        for column in columns:
            if column not in data.columns:
                continue
            
            # Convert to string and check pattern
            string_series = data[column].astype(str)
            matches = string_series.str.match(compiled_pattern, na=False)
            non_matching_count = (~matches).sum()
            
            if non_matching_count > 0:
                violations.append({
                    'column': column,
                    'non_matching_count': non_matching_count,
                    'total_count': len(data),
                    'match_percentage': (matches.sum() / len(data)) * 100
                })
        
        passed = len(violations) == 0
        message = 'Regex pattern constraints satisfied' if passed else f'Pattern violations: {violations}'
        
        return {
            'rule': rule_name,
            'passed': passed,
            'message': message,
            'critical': rule.get('critical', False),
            'details': {
                'columns_checked': columns,
                'pattern': pattern,
                'violations': violations
            }
        }
    
    def _validate_uniqueness(self, data: pd.DataFrame, rule: Dict[str, Any], rule_name: str) -> Dict[str, Any]:
        """Validate uniqueness constraints"""
        columns = rule.get('columns', [])
        
        violations = []
        
        for column in columns:
            if column not in data.columns:
                continue
            
            unique_count = data[column].nunique()
            total_count = len(data)
            duplicate_count = total_count - unique_count
            
            if duplicate_count > 0:
                violations.append({
                    'column': column,
                    'unique_count': unique_count,
                    'total_count': total_count,
                    'duplicate_count': duplicate_count,
                    'uniqueness_percentage': (unique_count / total_count) * 100 if total_count > 0 else 100
                })
        
        passed = len(violations) == 0
        message = 'Uniqueness constraints satisfied' if passed else f'Uniqueness violations: {violations}'
        
        return {
            'rule': rule_name,
            'passed': passed,
            'message': message,
            'critical': rule.get('critical', True),
            'details': {
                'columns_checked': columns,
                'violations': violations
            }
        }
    
    def _validate_completeness(self, data: pd.DataFrame, rule: Dict[str, Any], rule_name: str) -> Dict[str, Any]:
        """Validate data completeness"""
        min_completeness = rule.get('min_completeness', 100)  # Percentage
        columns = rule.get('columns', [])
        
        if not columns:
            columns = data.columns.tolist()
        
        violations = []
        
        for column in columns:
            if column not in data.columns:
                continue
            
            non_null_count = data[column].notna().sum()
            total_count = len(data)
            completeness_percentage = (non_null_count / total_count) * 100 if total_count > 0 else 100
            
            if completeness_percentage < min_completeness:
                violations.append({
                    'column': column,
                    'completeness_percentage': completeness_percentage,
                    'min_required': min_completeness,
                    'non_null_count': non_null_count,
                    'total_count': total_count
                })
        
        passed = len(violations) == 0
        message = 'Completeness constraints satisfied' if passed else f'Completeness violations: {violations}'
        
        return {
            'rule': rule_name,
            'passed': passed,
            'message': message,
            'critical': rule.get('critical', False),
            'details': {
                'columns_checked': columns,
                'min_completeness': min_completeness,
                'violations': violations
            }
        }
    
    def _validate_business_rule(self, data: pd.DataFrame, rule: Dict[str, Any]) -> Dict[str, Any]:
        """Validate custom business rules"""
        rule_name = rule.get('name', 'business_rule')
        expression = rule.get('expression')
        
        if not expression:
            return {
                'rule': rule_name,
                'passed': False,
                'message': 'Business rule expression is required',
                'critical': rule.get('critical', False)
            }
        
        try:
            # Create a safe environment for eval
            safe_dict = {
                'data': data,
                'np': np,
                'pd': pd,
                'len': len,
                '__builtins__': {}
            }
            
            # Evaluate the expression
            result = eval(expression, safe_dict)
            
            if isinstance(result, bool):
                passed = result
                message = rule.get('success_message', 'Business rule satisfied') if passed else rule.get('failure_message', 'Business rule violated')
            elif isinstance(result, (pd.Series, np.ndarray)):
                # Result is a boolean series/array
                passed = result.all()
                violation_count = (~result).sum() if hasattr(result, 'sum') else 0
                message = f'Business rule satisfied' if passed else f'Business rule violated for {violation_count} records'
            else:
                passed = False
                message = f'Business rule expression must return boolean or boolean series, got {type(result)}'
            
            return {
                'rule': rule_name,
                'passed': passed,
                'message': message,
                'critical': rule.get('critical', False),
                'details': {
                    'expression': expression
                }
            }
            
        except Exception as e:
            return {
                'rule': rule_name,
                'passed': False,
                'message': f'Business rule evaluation error: {str(e)}',
                'critical': rule.get('critical', False)
            }
    
    def _validate_statistical(self, data: pd.DataFrame, rule: Dict[str, Any]) -> Dict[str, Any]:
        """Validate statistical properties"""
        rule_type = rule.get('type')
        rule_name = rule.get('name', f'statistical_{rule_type}')
        
        try:
            if rule_type == 'distribution':
                return self._validate_distribution(data, rule, rule_name)
            elif rule_type == 'correlation':
                return self._validate_correlation(data, rule, rule_name)
            elif rule_type == 'outliers':
                return self._validate_outliers(data, rule, rule_name)
            else:
                return {
                    'rule': rule_name,
                    'passed': False,
                    'message': f'Unknown statistical validation type: {rule_type}',
                    'critical': rule.get('critical', False)
                }
        except Exception as e:
            return {
                'rule': rule_name,
                'passed': False,
                'message': f'Statistical validation error: {str(e)}',
                'critical': rule.get('critical', False)
            }
    
    def _validate_distribution(self, data: pd.DataFrame, rule: Dict[str, Any], rule_name: str) -> Dict[str, Any]:
        """Validate statistical distribution properties"""
        column = rule.get('column')
        expected_mean = rule.get('expected_mean')
        expected_std = rule.get('expected_std')
        tolerance = rule.get('tolerance', 0.1)  # 10% tolerance by default
        
        if not column or column not in data.columns:
            return {
                'rule': rule_name,
                'passed': False,
                'message': f'Column {column} not found',
                'critical': rule.get('critical', False)
            }
        
        if not pd.api.types.is_numeric_dtype(data[column]):
            return {
                'rule': rule_name,
                'passed': False,
                'message': f'Column {column} is not numeric',
                'critical': rule.get('critical', False)
            }
        
        actual_mean = data[column].mean()
        actual_std = data[column].std()
        
        violations = []
        
        if expected_mean is not None:
            mean_diff = abs(actual_mean - expected_mean) / expected_mean if expected_mean != 0 else abs(actual_mean)
            if mean_diff > tolerance:
                violations.append(f'Mean deviation {mean_diff:.3f} exceeds tolerance {tolerance}')
        
        if expected_std is not None:
            std_diff = abs(actual_std - expected_std) / expected_std if expected_std != 0 else abs(actual_std)
            if std_diff > tolerance:
                violations.append(f'Standard deviation deviation {std_diff:.3f} exceeds tolerance {tolerance}')
        
        passed = len(violations) == 0
        message = 'Distribution properties satisfied' if passed else f'Distribution violations: {violations}'
        
        return {
            'rule': rule_name,
            'passed': passed,
            'message': message,
            'critical': rule.get('critical', False),
            'details': {
                'column': column,
                'expected_mean': expected_mean,
                'actual_mean': actual_mean,
                'expected_std': expected_std,
                'actual_std': actual_std,
                'tolerance': tolerance,
                'violations': violations
            }
        }
    
    def _validate_correlation(self, data: pd.DataFrame, rule: Dict[str, Any], rule_name: str) -> Dict[str, Any]:
        """Validate correlation between columns"""
        columns = rule.get('columns', [])
        expected_correlation = rule.get('expected_correlation')
        tolerance = rule.get('tolerance', 0.1)
        
        if len(columns) != 2:
            return {
                'rule': rule_name,
                'passed': False,
                'message': 'Exactly 2 columns required for correlation validation',
                'critical': rule.get('critical', False)
            }
        
        col1, col2 = columns
        if col1 not in data.columns or col2 not in data.columns:
            return {
                'rule': rule_name,
                'passed': False,
                'message': f'Columns {columns} not found',
                'critical': rule.get('critical', False)
            }
        
        if not pd.api.types.is_numeric_dtype(data[col1]) or not pd.api.types.is_numeric_dtype(data[col2]):
            return {
                'rule': rule_name,
                'passed': False,
                'message': 'Both columns must be numeric for correlation validation',
                'critical': rule.get('critical', False)
            }
        
        actual_correlation = data[col1].corr(data[col2])
        
        if expected_correlation is not None:
            correlation_diff = abs(actual_correlation - expected_correlation)
            passed = correlation_diff <= tolerance
            message = f'Correlation satisfied' if passed else f'Correlation difference {correlation_diff:.3f} exceeds tolerance {tolerance}'
        else:
            passed = True
            message = f'Correlation calculated: {actual_correlation:.3f}'
        
        return {
            'rule': rule_name,
            'passed': passed,
            'message': message,
            'critical': rule.get('critical', False),
            'details': {
                'columns': columns,
                'expected_correlation': expected_correlation,
                'actual_correlation': actual_correlation,
                'tolerance': tolerance
            }
        }
    
    def _validate_outliers(self, data: pd.DataFrame, rule: Dict[str, Any], rule_name: str) -> Dict[str, Any]:
        """Validate outlier constraints"""
        columns = rule.get('columns', [])
        max_outlier_percentage = rule.get('max_outlier_percentage', 5)
        method = rule.get('method', 'iqr')  # iqr, zscore
        
        if not columns:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        violations = []
        
        for column in columns:
            if column not in data.columns:
                continue
            
            if not pd.api.types.is_numeric_dtype(data[column]):
                continue
            
            if method == 'iqr':
                Q1 = data[column].quantile(0.25)
                Q3 = data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = (data[column] < lower_bound) | (data[column] > upper_bound)
            elif method == 'zscore':
                threshold = rule.get('zscore_threshold', 3)
                z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
                outliers = z_scores > threshold
            else:
                continue
            
            outlier_count = outliers.sum()
            outlier_percentage = (outlier_count / len(data)) * 100 if len(data) > 0 else 0
            
            if outlier_percentage > max_outlier_percentage:
                violations.append({
                    'column': column,
                    'outlier_count': outlier_count,
                    'outlier_percentage': outlier_percentage,
                    'max_allowed': max_outlier_percentage
                })
        
        passed = len(violations) == 0
        message = 'Outlier constraints satisfied' if passed else f'Outlier violations: {violations}'
        
        return {
            'rule': rule_name,
            'passed': passed,
            'message': message,
            'critical': rule.get('critical', False),
            'details': {
                'columns_checked': columns,
                'method': method,
                'max_outlier_percentage': max_outlier_percentage,
                'violations': violations
            }
        }
    
    def _is_compatible_type(self, actual_type: str, expected_type: str) -> bool:
        """Check if actual data type is compatible with expected type"""
        type_mappings = {
            'int': ['int64', 'int32', 'int16', 'int8'],
            'float': ['float64', 'float32', 'float16'],
            'string': ['object', 'string'],
            'datetime': ['datetime64', 'datetime64[ns]'],
            'bool': ['bool'],
            'category': ['category']
        }
        
        # Direct match
        if actual_type == expected_type:
            return True
        
        # Check mappings
        for expected, actuals in type_mappings.items():
            if expected_type == expected and actual_type in actuals:
                return True
        
        return False
    
    def generate_data_profile(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data profile"""
        profile = {
            'basic_info': {
                'shape': data.shape,
                'columns': list(data.columns),
                'dtypes': data.dtypes.to_dict(),
                'memory_usage_mb': data.memory_usage(deep=True).sum() / (1024 * 1024)
            },
            'missing_values': {
                'total_missing': data.isnull().sum().sum(),
                'missing_by_column': data.isnull().sum().to_dict(),
                'missing_percentage_by_column': (data.isnull().sum() / len(data) * 100).to_dict()
            },
            'duplicates': {
                'duplicate_rows': data.duplicated().sum(),
                'duplicate_percentage': (data.duplicated().sum() / len(data) * 100) if len(data) > 0 else 0
            }
        }
        
        # Numeric column statistics
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            profile['numeric_stats'] = data[numeric_columns].describe().to_dict()
        
        # Categorical column statistics
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns
        if len(categorical_columns) > 0:
            profile['categorical_stats'] = {}
            for col in categorical_columns:
                profile['categorical_stats'][col] = {
                    'unique_count': data[col].nunique(),
                    'most_frequent': data[col].mode().iloc[0] if not data[col].mode().empty else None,
                    'value_counts': data[col].value_counts().head(10).to_dict()
                }
        
        return profile
