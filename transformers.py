import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from typing import Dict, Any, List, Optional, Union
import re
from datetime import datetime, timedelta


class DataTransformer:
    """Handles data transformation and feature engineering"""
    
    def __init__(self, config: Dict[str, Any], logger):
        self.config = config
        self.logger = logger
        self.fitted_transformers = {}
    
    def apply_transformation(self, data: pd.DataFrame, step_config: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply a single transformation step
        
        Args:
            data: Input DataFrame
            step_config: Configuration for the transformation step
            
        Returns:
            Transformed DataFrame
        """
        step_type = step_config.get('type')
        
        if step_type == 'clean':
            return self._clean_data(data, step_config)
        elif step_type == 'impute':
            return self._impute_missing_values(data, step_config)
        elif step_type == 'scale':
            return self._scale_features(data, step_config)
        elif step_type == 'encode':
            return self._encode_categorical(data, step_config)
        elif step_type == 'feature_engineering':
            return self._engineer_features(data, step_config)
        elif step_type == 'outlier_detection':
            return self._detect_outliers(data, step_config)
        elif step_type == 'feature_selection':
            return self._select_features(data, step_config)
        elif step_type == 'dimensionality_reduction':
            return self._reduce_dimensions(data, step_config)
        elif step_type == 'custom':
            return self._apply_custom_transformation(data, step_config)
        else:
            raise ValueError(f"Unsupported transformation type: {step_type}")
    
    def _clean_data(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Clean data by handling duplicates, formatting, etc."""
        cleaned_data = data.copy()
        
        # Remove duplicates
        if config.get('remove_duplicates', False):
            initial_shape = cleaned_data.shape
            subset = config.get('duplicate_subset')
            keep = config.get('duplicate_keep', 'first')
            cleaned_data = cleaned_data.drop_duplicates(subset=subset, keep=keep)
            self.logger.info(f"Removed duplicates: {initial_shape} -> {cleaned_data.shape}")
        
        # Drop columns
        drop_columns = config.get('drop_columns', [])
        if drop_columns:
            existing_columns = [col for col in drop_columns if col in cleaned_data.columns]
            if existing_columns:
                cleaned_data = cleaned_data.drop(columns=existing_columns)
                self.logger.info(f"Dropped columns: {existing_columns}")
        
        # Rename columns
        rename_columns = config.get('rename_columns', {})
        if rename_columns:
            cleaned_data = cleaned_data.rename(columns=rename_columns)
            self.logger.info(f"Renamed columns: {rename_columns}")
        
        # Convert data types
        dtype_conversions = config.get('dtype_conversions', {})
        for column, dtype in dtype_conversions.items():
            if column in cleaned_data.columns:
                try:
                    if dtype == 'datetime':
                        date_format = config.get('date_format', {}).get(column)
                        cleaned_data[column] = pd.to_datetime(cleaned_data[column], format=date_format, errors='coerce')
                    else:
                        cleaned_data[column] = cleaned_data[column].astype(dtype)
                    self.logger.info(f"Converted {column} to {dtype}")
                except Exception as e:
                    self.logger.warning(f"Failed to convert {column} to {dtype}: {str(e)}")
        
        # Text cleaning
        text_columns = config.get('text_cleaning', {})
        for column, operations in text_columns.items():
            if column in cleaned_data.columns:
                cleaned_data[column] = self._clean_text_column(cleaned_data[column], operations)
        
        # Filter rows
        row_filters = config.get('row_filters', [])
        for filter_config in row_filters:
            cleaned_data = self._apply_row_filter(cleaned_data, filter_config)
        
        return cleaned_data
    
    def _clean_text_column(self, series: pd.Series, operations: List[str]) -> pd.Series:
        """Clean text data in a series"""
        cleaned_series = series.copy()
        
        for operation in operations:
            if operation == 'lowercase':
                cleaned_series = cleaned_series.str.lower()
            elif operation == 'uppercase':
                cleaned_series = cleaned_series.str.upper()
            elif operation == 'strip':
                cleaned_series = cleaned_series.str.strip()
            elif operation == 'remove_punctuation':
                cleaned_series = cleaned_series.str.replace(r'[^\w\s]', '', regex=True)
            elif operation == 'remove_digits':
                cleaned_series = cleaned_series.str.replace(r'\d', '', regex=True)
            elif operation == 'remove_extra_spaces':
                cleaned_series = cleaned_series.str.replace(r'\s+', ' ', regex=True)
        
        return cleaned_series
    
    def _apply_row_filter(self, data: pd.DataFrame, filter_config: Dict[str, Any]) -> pd.DataFrame:
        """Apply row-level filtering"""
        column = filter_config.get('column')
        condition = filter_config.get('condition')
        value = filter_config.get('value')
        
        if not all([column, condition, value is not None]) or column not in data.columns:
            return data
        
        initial_count = len(data)
        
        if condition == 'equals':
            filtered_data = data[data[column] == value]
        elif condition == 'not_equals':
            filtered_data = data[data[column] != value]
        elif condition == 'greater_than':
            filtered_data = data[data[column] > value]
        elif condition == 'less_than':
            filtered_data = data[data[column] < value]
        elif condition == 'contains':
            filtered_data = data[data[column].astype(str).str.contains(str(value), na=False)]
        elif condition == 'in':
            filtered_data = data[data[column].isin(value)]
        elif condition == 'not_null':
            filtered_data = data[data[column].notna()]
        elif condition == 'is_null':
            filtered_data = data[data[column].isna()]
        else:
            return data
        
        self.logger.info(f"Row filter applied: {initial_count} -> {len(filtered_data)} records")
        return filtered_data
    
    def _impute_missing_values(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Impute missing values using various strategies"""
        imputed_data = data.copy()
        
        # Column-specific imputation
        column_strategies = config.get('column_strategies', {})
        
        for column, strategy in column_strategies.items():
            if column not in imputed_data.columns:
                continue
            
            missing_count = imputed_data[column].isnull().sum()
            if missing_count == 0:
                continue
            
            if strategy == 'drop':
                imputed_data = imputed_data.dropna(subset=[column])
                self.logger.info(f"Dropped {missing_count} rows with missing values in {column}")
                
            elif strategy in ['mean', 'median', 'mode', 'constant']:
                if strategy == 'constant':
                    fill_value = config.get('fill_values', {}).get(column, 0)
                    imputed_data[column] = imputed_data[column].fillna(fill_value)
                else:
                    imputer = SimpleImputer(strategy=strategy)
                    imputed_data[column] = imputer.fit_transform(imputed_data[[column]]).ravel()
                self.logger.info(f"Imputed {missing_count} missing values in {column} using {strategy}")
                
            elif strategy == 'knn':
                # KNN imputation for numeric columns
                if imputed_data[column].dtype in ['int64', 'float64']:
                    n_neighbors = config.get('knn_neighbors', 5)
                    numeric_columns = imputed_data.select_dtypes(include=[np.number]).columns
                    
                    if len(numeric_columns) > 1:
                        imputer = KNNImputer(n_neighbors=n_neighbors)
                        imputed_data[numeric_columns] = imputer.fit_transform(imputed_data[numeric_columns])
                        self.logger.info(f"Applied KNN imputation to numeric columns including {column}")
                    else:
                        # Fall back to median for single numeric column
                        imputer = SimpleImputer(strategy='median')
                        imputed_data[column] = imputer.fit_transform(imputed_data[[column]]).ravel()
                
            elif strategy == 'forward_fill':
                imputed_data[column] = imputed_data[column].fillna(method='ffill')
                self.logger.info(f"Applied forward fill to {column}")
                
            elif strategy == 'backward_fill':
                imputed_data[column] = imputed_data[column].fillna(method='bfill')
                self.logger.info(f"Applied backward fill to {column}")
        
        # Global strategy for remaining missing values
        global_strategy = config.get('global_strategy')
        if global_strategy:
            remaining_missing = imputed_data.isnull().sum().sum()
            if remaining_missing > 0:
                if global_strategy == 'drop_rows':
                    imputed_data = imputed_data.dropna()
                elif global_strategy == 'drop_columns':
                    imputed_data = imputed_data.dropna(axis=1)
                self.logger.info(f"Applied global strategy {global_strategy} for remaining missing values")
        
        return imputed_data
    
    def _scale_features(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Scale numerical features"""
        scaled_data = data.copy()
        
        columns = config.get('columns', [])
        if not columns:
            # Auto-select numeric columns
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter existing columns
        columns = [col for col in columns if col in scaled_data.columns]
        
        if not columns:
            self.logger.warning("No columns found for scaling")
            return scaled_data
        
        scaler_type = config.get('method', 'standard')
        
        # Initialize scaler
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            feature_range = config.get('feature_range', (0, 1))
            scaler = MinMaxScaler(feature_range=feature_range)
        elif scaler_type == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unsupported scaling method: {scaler_type}")
        
        # Fit and transform
        scaled_data[columns] = scaler.fit_transform(scaled_data[columns])
        
        # Store fitted scaler for inverse transformation if needed
        scaler_key = f"scaler_{config.get('name', 'default')}"
        self.fitted_transformers[scaler_key] = {
            'scaler': scaler,
            'columns': columns,
            'type': scaler_type
        }
        
        self.logger.info(f"Applied {scaler_type} scaling to columns: {columns}")
        return scaled_data
    
    def _encode_categorical(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Encode categorical variables"""
        encoded_data = data.copy()
        
        columns = config.get('columns', [])
        if not columns:
            # Auto-select categorical columns
            columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Filter existing columns
        columns = [col for col in columns if col in encoded_data.columns]
        
        if not columns:
            self.logger.warning("No columns found for encoding")
            return encoded_data
        
        encoding_method = config.get('method', 'onehot')
        
        for column in columns:
            if encoding_method == 'label':
                encoder = LabelEncoder()
                encoded_data[column] = encoder.fit_transform(encoded_data[column].astype(str))
                
                # Store encoder
                encoder_key = f"label_encoder_{column}"
                self.fitted_transformers[encoder_key] = encoder
                
                self.logger.info(f"Applied label encoding to {column}")
                
            elif encoding_method == 'onehot':
                # Get dummies
                prefix = config.get('prefix', column)
                dummy_df = pd.get_dummies(
                    encoded_data[column], 
                    prefix=prefix,
                    drop_first=config.get('drop_first', False)
                )
                
                # Replace original column with dummy columns
                encoded_data = encoded_data.drop(columns=[column])
                encoded_data = pd.concat([encoded_data, dummy_df], axis=1)
                
                self.logger.info(f"Applied one-hot encoding to {column}, created {len(dummy_df.columns)} new columns")
                
            elif encoding_method == 'target':
                # Target encoding (requires target column)
                target_column = config.get('target_column')
                if target_column and target_column in encoded_data.columns:
                    target_means = encoded_data.groupby(column)[target_column].mean()
                    encoded_data[f"{column}_target_encoded"] = encoded_data[column].map(target_means)
                    
                    if config.get('drop_original', True):
                        encoded_data = encoded_data.drop(columns=[column])
                    
                    self.logger.info(f"Applied target encoding to {column}")
                else:
                    self.logger.warning(f"Target column not specified or not found for target encoding of {column}")
        
        return encoded_data
    
    def _engineer_features(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Create new features through feature engineering"""
        feature_data = data.copy()
        
        features = config.get('features', [])
        
        for feature_config in features:
            feature_name = feature_config.get('name')
            feature_type = feature_config.get('type')
            
            try:
                if feature_type == 'arithmetic':
                    feature_data = self._create_arithmetic_feature(feature_data, feature_config)
                elif feature_type == 'datetime':
                    feature_data = self._create_datetime_features(feature_data, feature_config)
                elif feature_type == 'text':
                    feature_data = self._create_text_features(feature_data, feature_config)
                elif feature_type == 'interaction':
                    feature_data = self._create_interaction_features(feature_data, feature_config)
                elif feature_type == 'binning':
                    feature_data = self._create_binned_features(feature_data, feature_config)
                elif feature_type == 'polynomial':
                    feature_data = self._create_polynomial_features(feature_data, feature_config)
                
                self.logger.info(f"Created feature: {feature_name}")
                
            except Exception as e:
                self.logger.warning(f"Failed to create feature {feature_name}: {str(e)}")
        
        return feature_data
    
    def _create_arithmetic_feature(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Create arithmetic features"""
        feature_name = config.get('name')
        operation = config.get('operation')
        columns = config.get('columns', [])
        
        if len(columns) < 1:
            return data
        
        if operation == 'sum':
            data[feature_name] = data[columns].sum(axis=1)
        elif operation == 'mean':
            data[feature_name] = data[columns].mean(axis=1)
        elif operation == 'difference' and len(columns) >= 2:
            data[feature_name] = data[columns[0]] - data[columns[1]]
        elif operation == 'ratio' and len(columns) >= 2:
            data[feature_name] = data[columns[0]] / (data[columns[1]] + 1e-8)  # Add small value to avoid division by zero
        elif operation == 'product':
            data[feature_name] = data[columns].prod(axis=1)
        elif operation == 'max':
            data[feature_name] = data[columns].max(axis=1)
        elif operation == 'min':
            data[feature_name] = data[columns].min(axis=1)
        
        return data
    
    def _create_datetime_features(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Extract datetime features"""
        column = config.get('column')
        features = config.get('extract_features', ['year', 'month', 'day'])
        
        if column not in data.columns:
            return data
        
        # Ensure column is datetime
        if not pd.api.types.is_datetime64_any_dtype(data[column]):
            data[column] = pd.to_datetime(data[column], errors='coerce')
        
        # Extract features
        for feature in features:
            if feature == 'year':
                data[f"{column}_year"] = data[column].dt.year
            elif feature == 'month':
                data[f"{column}_month"] = data[column].dt.month
            elif feature == 'day':
                data[f"{column}_day"] = data[column].dt.day
            elif feature == 'dayofweek':
                data[f"{column}_dayofweek"] = data[column].dt.dayofweek
            elif feature == 'quarter':
                data[f"{column}_quarter"] = data[column].dt.quarter
            elif feature == 'hour':
                data[f"{column}_hour"] = data[column].dt.hour
            elif feature == 'minute':
                data[f"{column}_minute"] = data[column].dt.minute
            elif feature == 'is_weekend':
                data[f"{column}_is_weekend"] = (data[column].dt.dayofweek >= 5).astype(int)
        
        return data
    
    def _create_text_features(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Create text-based features"""
        column = config.get('column')
        features = config.get('extract_features', ['length'])
        
        if column not in data.columns:
            return data
        
        text_series = data[column].astype(str)
        
        for feature in features:
            if feature == 'length':
                data[f"{column}_length"] = text_series.str.len()
            elif feature == 'word_count':
                data[f"{column}_word_count"] = text_series.str.split().str.len()
            elif feature == 'char_count':
                data[f"{column}_char_count"] = text_series.str.replace(' ', '').str.len()
            elif feature == 'uppercase_count':
                data[f"{column}_uppercase_count"] = text_series.str.count(r'[A-Z]')
            elif feature == 'digit_count':
                data[f"{column}_digit_count"] = text_series.str.count(r'\d')
            elif feature == 'punctuation_count':
                data[f"{column}_punctuation_count"] = text_series.str.count(r'[^\w\s]')
        
        return data
    
    def _create_interaction_features(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Create interaction features between columns"""
        columns = config.get('columns', [])
        interaction_type = config.get('interaction_type', 'multiply')
        
        if len(columns) < 2:
            return data
        
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                col1, col2 = columns[i], columns[j]
                if col1 in data.columns and col2 in data.columns:
                    feature_name = f"{col1}_{interaction_type}_{col2}"
                    
                    if interaction_type == 'multiply':
                        data[feature_name] = data[col1] * data[col2]
                    elif interaction_type == 'add':
                        data[feature_name] = data[col1] + data[col2]
                    elif interaction_type == 'subtract':
                        data[feature_name] = data[col1] - data[col2]
                    elif interaction_type == 'divide':
                        data[feature_name] = data[col1] / (data[col2] + 1e-8)
        
        return data
    
    def _create_binned_features(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Create binned/discretized features"""
        column = config.get('column')
        bins = config.get('bins', 5)
        labels = config.get('labels')
        strategy = config.get('strategy', 'equal_width')
        
        if column not in data.columns:
            return data
        
        feature_name = config.get('name', f"{column}_binned")
        
        if strategy == 'equal_width':
            data[feature_name] = pd.cut(data[column], bins=bins, labels=labels)
        elif strategy == 'equal_frequency':
            data[feature_name] = pd.qcut(data[column], q=bins, labels=labels, duplicates='drop')
        
        return data
    
    def _create_polynomial_features(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Create polynomial features"""
        from sklearn.preprocessing import PolynomialFeatures
        
        columns = config.get('columns', [])
        degree = config.get('degree', 2)
        include_bias = config.get('include_bias', False)
        
        # Filter existing columns
        columns = [col for col in columns if col in data.columns]
        
        if not columns:
            return data
        
        # Create polynomial features
        poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
        poly_features = poly.fit_transform(data[columns])
        
        # Create feature names
        feature_names = poly.get_feature_names_out(columns)
        
        # Add polynomial features to dataframe
        poly_df = pd.DataFrame(poly_features, columns=feature_names, index=data.index)
        
        # Remove original columns from polynomial features to avoid duplication
        original_columns = [col for col in feature_names if col in columns]
        poly_df = poly_df.drop(columns=original_columns)
        
        # Combine with original data
        result_data = pd.concat([data, poly_df], axis=1)
        
        return result_data
    
    def _detect_outliers(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Detect and handle outliers"""
        method = config.get('method', 'isolation_forest')
        columns = config.get('columns', [])
        action = config.get('action', 'flag')  # flag, remove, cap
        
        if not columns:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        columns = [col for col in columns if col in data.columns]
        
        if not columns:
            return data
        
        outlier_data = data.copy()
        
        if method == 'isolation_forest':
            contamination = config.get('contamination', 0.1)
            clf = IsolationForest(contamination=contamination, random_state=42)
            outliers = clf.fit_predict(outlier_data[columns])
            outlier_mask = outliers == -1
            
        elif method == 'iqr':
            outlier_mask = pd.Series([False] * len(outlier_data), index=outlier_data.index)
            
            for col in columns:
                Q1 = outlier_data[col].quantile(0.25)
                Q3 = outlier_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                col_outliers = (outlier_data[col] < lower_bound) | (outlier_data[col] > upper_bound)
                outlier_mask = outlier_mask | col_outliers
        
        elif method == 'zscore':
            threshold = config.get('threshold', 3)
            outlier_mask = pd.Series([False] * len(outlier_data), index=outlier_data.index)
            
            for col in columns:
                z_scores = np.abs((outlier_data[col] - outlier_data[col].mean()) / outlier_data[col].std())
                col_outliers = z_scores > threshold
                outlier_mask = outlier_mask | col_outliers
        
        # Handle outliers based on action
        outlier_count = outlier_mask.sum()
        
        if action == 'flag':
            outlier_data['is_outlier'] = outlier_mask
            self.logger.info(f"Flagged {outlier_count} outliers using {method}")
            
        elif action == 'remove':
            outlier_data = outlier_data[~outlier_mask]
            self.logger.info(f"Removed {outlier_count} outliers using {method}")
            
        elif action == 'cap':
            # Cap outliers at percentiles
            lower_percentile = config.get('lower_percentile', 5)
            upper_percentile = config.get('upper_percentile', 95)
            
            for col in columns:
                lower_cap = outlier_data[col].quantile(lower_percentile / 100)
                upper_cap = outlier_data[col].quantile(upper_percentile / 100)
                
                outlier_data[col] = outlier_data[col].clip(lower=lower_cap, upper=upper_cap)
            
            self.logger.info(f"Capped outliers at {lower_percentile}th and {upper_percentile}th percentiles")
        
        return outlier_data
    
    def _select_features(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Select important features"""
        method = config.get('method', 'variance')
        target_column = config.get('target_column')
        k = config.get('k', 10)
        
        feature_columns = config.get('feature_columns', [])
        if not feature_columns:
            # Auto-select numeric columns, excluding target
            feature_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            if target_column and target_column in feature_columns:
                feature_columns.remove(target_column)
        
        feature_columns = [col for col in feature_columns if col in data.columns]
        
        if not feature_columns:
            return data
        
        selected_data = data.copy()
        
        if method == 'variance':
            # Remove low variance features
            threshold = config.get('variance_threshold', 0.01)
            variances = selected_data[feature_columns].var()
            high_variance_cols = variances[variances > threshold].index.tolist()
            
            # Keep non-feature columns and high variance feature columns
            keep_columns = [col for col in selected_data.columns if col not in feature_columns or col in high_variance_cols]
            selected_data = selected_data[keep_columns]
            
            self.logger.info(f"Selected {len(high_variance_cols)} features with variance > {threshold}")
            
        elif method == 'univariate' and target_column and target_column in data.columns:
            # Univariate feature selection
            problem_type = config.get('problem_type', 'classification')
            
            if problem_type == 'classification':
                score_func = f_classif
            else:
                score_func = f_regression
            
            selector = SelectKBest(score_func=score_func, k=min(k, len(feature_columns)))
            X = selected_data[feature_columns]
            y = selected_data[target_column]
            
            X_selected = selector.fit_transform(X, y)
            selected_features = [feature_columns[i] for i in selector.get_support(indices=True)]
            
            # Keep non-feature columns and selected feature columns
            keep_columns = [col for col in selected_data.columns if col not in feature_columns or col in selected_features]
            selected_data = selected_data[keep_columns]
            
            self.logger.info(f"Selected top {len(selected_features)} features using univariate selection")
        
        return selected_data
    
    def _reduce_dimensions(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Reduce dimensionality of features"""
        method = config.get('method', 'pca')
        columns = config.get('columns', [])
        n_components = config.get('n_components', 2)
        
        if not columns:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        columns = [col for col in columns if col in data.columns]
        
        if not columns or len(columns) < n_components:
            return data
        
        reduced_data = data.copy()
        
        if method == 'pca':
            pca = PCA(n_components=n_components)
            reduced_features = pca.fit_transform(reduced_data[columns])
            
            # Create new column names
            component_names = [f"pca_component_{i+1}" for i in range(n_components)]
            
            # Replace original columns with PCA components
            reduced_data = reduced_data.drop(columns=columns)
            for i, col_name in enumerate(component_names):
                reduced_data[col_name] = reduced_features[:, i]
            
            # Store PCA transformer
            pca_key = f"pca_{config.get('name', 'default')}"
            self.fitted_transformers[pca_key] = {
                'transformer': pca,
                'original_columns': columns,
                'component_names': component_names,
                'explained_variance_ratio': pca.explained_variance_ratio_
            }
            
            total_variance = pca.explained_variance_ratio_.sum()
            self.logger.info(f"Applied PCA: {len(columns)} -> {n_components} components, explaining {total_variance:.3f} of variance")
        
        return reduced_data
    
    def _apply_custom_transformation(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Apply custom transformation using eval (use with caution)"""
        expression = config.get('expression')
        new_column = config.get('new_column')
        
        if not expression:
            return data
        
        try:
            # Create a safe environment for eval
            safe_dict = {
                'data': data,
                'np': np,
                'pd': pd,
                '__builtins__': {}
            }
            
            result = eval(expression, safe_dict)
            
            if new_column:
                data[new_column] = result
            else:
                # Assume expression returns a DataFrame
                data = result
            
            self.logger.info(f"Applied custom transformation: {expression}")
            
        except Exception as e:
            self.logger.error(f"Failed to apply custom transformation: {str(e)}")
            raise
        
        return data
