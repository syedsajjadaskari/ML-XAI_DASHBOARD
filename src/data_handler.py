"""
Data Handler Module
Handles data loading, preprocessing, and validation (No PyCaret dependency)
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Any, Optional, Union
import logging
from pathlib import Path
import io

logger = logging.getLogger(__name__)

class DataHandler:
    """Handles all data operations without PyCaret dependency."""
    
    def __init__(self, config: Dict[str, Any]):
        # Handle None config gracefully
        if config is None:
            config = {}
        
        self.config = config
        
        # Get app config with defaults
        app_config = config.get('app', {})
        self.max_file_size = app_config.get('max_file_size', 200) * 1024 * 1024  # Convert to bytes
        self.supported_formats = app_config.get('supported_formats', ['csv', 'xlsx', 'parquet'])
    
    def load_data(self, uploaded_file) -> pd.DataFrame:
        """Load data from uploaded file."""
        try:
            # Check file size
            if uploaded_file.size > self.max_file_size:
                raise ValueError(f"File size ({uploaded_file.size / 1024 / 1024:.1f}MB) exceeds limit ({self.max_file_size / 1024 / 1024:.0f}MB)")
            
            # Get file extension
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension not in self.supported_formats:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            # Load based on file type
            if file_extension == 'csv':
                # Try different encodings
                try:
                    data = pd.read_csv(uploaded_file, encoding='utf-8')
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    data = pd.read_csv(uploaded_file, encoding='latin-1')
            
            elif file_extension in ['xlsx', 'xls']:
                data = pd.read_excel(uploaded_file)
            
            elif file_extension == 'parquet':
                data = pd.read_parquet(uploaded_file)
            
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            # Basic validation
            if data.empty:
                raise ValueError("File is empty")
            
            if len(data.columns) < 2:
                raise ValueError("Dataset must have at least 2 columns")
            
            # Clean column names
            data.columns = self._clean_column_names(data.columns)
            
            logger.info(f"Data loaded successfully: {data.shape}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def load_sample_data(self, dataset_name: str) -> pd.DataFrame:
        """Load sample datasets without PyCaret."""
        try:
            # Create synthetic data instead of using PyCaret
            data = self._create_synthetic_data(dataset_name)
            
            # Clean column names
            data.columns = self._clean_column_names(data.columns)
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading sample data: {e}")
            return self._create_synthetic_data(dataset_name)
    
    def _create_synthetic_data(self, dataset_name: str) -> pd.DataFrame:
        """Create synthetic data as sample datasets."""
        np.random.seed(42)
        
        if dataset_name == 'titanic':
            n_samples = 891
            data = pd.DataFrame({
                'Age': np.random.normal(30, 12, n_samples),
                'Fare': np.random.exponential(20, n_samples),
                'Sex': np.random.choice(['male', 'female'], n_samples),
                'Pclass': np.random.choice([1, 2, 3], n_samples),
                'SibSp': np.random.poisson(0.5, n_samples),
                'Parch': np.random.poisson(0.4, n_samples),
                'Embarked': np.random.choice(['C', 'Q', 'S'], n_samples),
                'Survived': np.random.choice([0, 1], n_samples)
            })
        
        elif dataset_name == 'boston':
            n_samples = 506
            # Create correlated features for more realistic data
            noise = np.random.normal(0, 1, (n_samples, 13))
            
            data = pd.DataFrame({
                'CRIM': np.abs(noise[:, 0] * 3),
                'ZN': np.abs(noise[:, 1] * 10),
                'INDUS': noise[:, 2] * 7 + 11,
                'CHAS': np.random.choice([0, 1], n_samples, p=[0.93, 0.07]),
                'NOX': noise[:, 3] * 0.12 + 0.55,
                'RM': noise[:, 4] * 0.7 + 6.3,
                'AGE': np.abs(noise[:, 5] * 28 + 68),
                'DIS': np.abs(noise[:, 6] * 2.1 + 3.8),
                'RAD': np.random.choice(range(1, 25), n_samples),
                'TAX': noise[:, 7] * 169 + 408,
                'PTRATIO': noise[:, 8] * 2.2 + 18.5,
                'B': noise[:, 9] * 91 + 356,
                'LSTAT': np.abs(noise[:, 10] * 7.1 + 12.6)
            })
            
            # Create realistic target (house prices)
            medv = (
                25 - data['LSTAT'] * 0.5 + data['RM'] * 3 - 
                data['CRIM'] * 0.1 + np.random.normal(0, 3, n_samples)
            )
            data['medv'] = np.clip(medv, 5, 50)
        
        elif dataset_name == 'diabetes':
            n_samples = 442
            # Create realistic diabetes dataset
            features = np.random.normal(0, 1, (n_samples, 10))
            
            data = pd.DataFrame({
                'age': features[:, 0],
                'sex': features[:, 1],
                'bmi': features[:, 2],
                'bp': features[:, 3],
                's1': features[:, 4],
                's2': features[:, 5],
                's3': features[:, 6],
                's4': features[:, 7],
                's5': features[:, 8],
                's6': features[:, 9]
            })
            
            # Create target (diabetes progression)
            target = (
                features.sum(axis=1) * 50 + 150 + 
                np.random.normal(0, 30, n_samples)
            )
            data['target'] = target
            
        elif dataset_name == 'wine':
            n_samples = 1599
            # Create wine quality dataset
            data = pd.DataFrame({
                'fixed_acidity': np.random.normal(8.3, 1.7, n_samples),
                'volatile_acidity': np.random.gamma(2, 0.15, n_samples),
                'citric_acid': np.random.beta(2, 5, n_samples),
                'residual_sugar': np.random.exponential(2.5, n_samples),
                'chlorides': np.random.gamma(1.5, 0.05, n_samples),
                'free_sulfur_dioxide': np.random.normal(15, 10, n_samples),
                'total_sulfur_dioxide': np.random.normal(46, 32, n_samples),
                'density': np.random.normal(0.997, 0.002, n_samples),
                'pH': np.random.normal(3.3, 0.15, n_samples),
                'sulphates': np.random.gamma(2, 0.3, n_samples),
                'alcohol': np.random.normal(10.4, 1.1, n_samples)
            })
            
            # Create quality scores (3-8)
            quality_score = (
                data['alcohol'] * 0.3 + 
                data['volatile_acidity'] * -2 +
                data['citric_acid'] * 1 +
                np.random.normal(0, 0.5, n_samples)
            )
            data['quality'] = np.clip(np.round(quality_score + 6), 3, 8).astype(int)
        
        else:
            # Generic dataset
            n_samples = 1000
            data = pd.DataFrame({
                'feature_1': np.random.normal(0, 1, n_samples),
                'feature_2': np.random.normal(0, 1, n_samples),
                'feature_3': np.random.choice(['A', 'B', 'C'], n_samples),
                'feature_4': np.random.uniform(0, 100, n_samples),
                'target': np.random.choice([0, 1], n_samples)
            })
        
        return data
    
    def get_column_info(self, data: pd.DataFrame) -> pd.DataFrame:
        """Get detailed column information."""
        info_dict = {
            'Column': data.columns,
            'Type': data.dtypes.astype(str),
            'Non-Null': data.count(),
            'Null': data.isnull().sum(),
            'Null %': (data.isnull().sum() / len(data) * 100).round(2),
            'Unique': data.nunique(),
            'Memory (KB)': (data.memory_usage(deep=True)[1:] / 1024).round(2)
        }
        
        return pd.DataFrame(info_dict)
    
    def detect_problem_type(self, data: pd.DataFrame, target_column: str) -> str:
        """Auto-detect problem type based on target column."""
        try:
            target_series = data[target_column]
            
            # Check if target is numeric
            if pd.api.types.is_numeric_dtype(target_series):
                # Check if it looks like classification
                unique_values = target_series.nunique()
                total_values = len(target_series)
                
                # If less than 10 unique values or less than 5% unique values, likely classification
                if unique_values <= 10 or (unique_values / total_values) < 0.05:
                    return 'classification'
                else:
                    return 'regression'
            else:
                # Non-numeric target is classification
                return 'classification'
                
        except Exception:
            # Default to classification
            return 'classification'
    
    def apply_preprocessing(self, data: pd.DataFrame, target_column: str, config: Dict[str, Any]) -> pd.DataFrame:
        """Apply basic preprocessing steps to data."""
        processed_data = data.copy()
        
        try:
            # Handle missing values
            missing_strategy = config.get('missing_strategy', 'none')
            
            if 'drop' in missing_strategy.lower():
                processed_data = processed_data.dropna()
            elif 'mean' in missing_strategy.lower():
                numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
                processed_data[numeric_columns] = processed_data[numeric_columns].fillna(
                    processed_data[numeric_columns].mean()
                )
                # Fill categorical with mode
                categorical_columns = processed_data.select_dtypes(include=['object', 'category']).columns
                for col in categorical_columns:
                    if col != target_column:
                        mode_val = processed_data[col].mode()
                        if len(mode_val) > 0:
                            processed_data[col] = processed_data[col].fillna(mode_val[0])
                            
            elif 'median' in missing_strategy.lower():
                numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
                processed_data[numeric_columns] = processed_data[numeric_columns].fillna(
                    processed_data[numeric_columns].median()
                )
                # Fill categorical with mode
                categorical_columns = processed_data.select_dtypes(include=['object', 'category']).columns
                for col in categorical_columns:
                    if col != target_column:
                        mode_val = processed_data[col].mode()
                        if len(mode_val) > 0:
                            processed_data[col] = processed_data[col].fillna(mode_val[0])
            
            # Handle categorical encoding (basic)
            categorical_columns = processed_data.select_dtypes(include=['object', 'category']).columns
            categorical_columns = [col for col in categorical_columns if col != target_column]
            
            encoding_method = config.get('encoding_method', 'none')
            if 'one-hot' in encoding_method.lower() and len(categorical_columns) > 0:
                # Limit categories to prevent explosion
                for col in categorical_columns:
                    if processed_data[col].nunique() > 10:
                        top_categories = processed_data[col].value_counts().head(10).index
                        processed_data[col] = processed_data[col].apply(
                            lambda x: x if x in top_categories else 'Other'
                        )
                
                processed_data = pd.get_dummies(processed_data, columns=categorical_columns, prefix=categorical_columns)
            
            elif 'label' in encoding_method.lower() and len(categorical_columns) > 0:
                from sklearn.preprocessing import LabelEncoder
                for col in categorical_columns:
                    le = LabelEncoder()
                    processed_data[col] = le.fit_transform(processed_data[col].astype(str))
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            # Return original data if preprocessing fails
            return data
    
    def _clean_column_names(self, columns) -> list:
        """Clean column names for better compatibility."""
        cleaned = []
        for col in columns:
            # Convert to string and clean
            clean_col = str(col).strip()
            # Replace spaces and special characters
            clean_col = clean_col.replace(' ', '_')
            clean_col = ''.join(c for c in clean_col if c.isalnum() or c == '_')
            # Ensure it doesn't start with a number
            if clean_col and clean_col[0].isdigit():
                clean_col = 'col_' + clean_col
            cleaned.append(clean_col)
        
        return cleaned
    
    def validate_data_for_training(self, data: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Validate data for ML training."""
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': []
        }
        
        try:
            # Check minimum requirements
            if len(data) < 10:
                validation_results['errors'].append("Dataset too small (minimum 10 rows required)")
                validation_results['is_valid'] = False
            
            if len(data.columns) < 2:
                validation_results['errors'].append("Dataset must have at least 2 columns")
                validation_results['is_valid'] = False
            
            if target_column not in data.columns:
                validation_results['errors'].append(f"Target column '{target_column}' not found")
                validation_results['is_valid'] = False
                return validation_results
            
            # Check target column
            target_null_pct = data[target_column].isnull().sum() / len(data) * 100
            if target_null_pct > 50:
                validation_results['errors'].append(f"Target column has {target_null_pct:.1f}% missing values")
                validation_results['is_valid'] = False
            elif target_null_pct > 0:
                validation_results['warnings'].append(f"Target column has {target_null_pct:.1f}% missing values")
            
            # Check for high cardinality categorical columns
            categorical_columns = data.select_dtypes(include=['object', 'category']).columns
            for col in categorical_columns:
                if col != target_column:
                    unique_ratio = data[col].nunique() / len(data)
                    if unique_ratio > 0.9:
                        validation_results['warnings'].append(f"Column '{col}' has high cardinality ({unique_ratio:.1%})")
            
            # Check for constant columns
            constant_columns = []
            for col in data.columns:
                if col != target_column and data[col].nunique() <= 1:
                    constant_columns.append(col)
            
            if constant_columns:
                validation_results['warnings'].append(f"Constant columns found: {constant_columns}")
            
            return validation_results
            
        except Exception as e:
            validation_results['errors'].append(f"Validation error: {str(e)}")
            validation_results['is_valid'] = False
            return validation_results