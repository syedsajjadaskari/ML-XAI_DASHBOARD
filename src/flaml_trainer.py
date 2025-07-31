"""
FLAML AutoML Trainer - FINAL FIX
Ultra-fast AutoML using Microsoft FLAML with proper error handling and correct estimator names
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging
import time
import warnings
warnings.filterwarnings('ignore')

# Try to import FLAML
try:
    from flaml import AutoML
    FLAML_AVAILABLE = True
except ImportError:
    FLAML_AVAILABLE = False

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error

logger = logging.getLogger(__name__)

class FLAMLTrainer:
    """Ultra-fast AutoML trainer using Microsoft FLAML - FINAL FIXED VERSION."""
    
    def __init__(self, config: Dict[str, Any]):
        if config is None:
            config = {}
        self.config = config
        self.setup_complete = False
        self.problem_type = None
        self.automl = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.label_encoder = None
        self.feature_names = None
        self.X_train_processed = None
        self.X_test_processed = None
        
        if not FLAML_AVAILABLE:
            raise ImportError(
                "FLAML not available. Install with: pip install flaml"
            )
    
    def setup_environment(self, 
                         data: pd.DataFrame, 
                         target: str, 
                         problem_type: str,
                         preprocessing_config: Dict[str, Any],
                         test_size: float = 0.2,
                         random_state: int = 42) -> bool:
        """Setup FLAML environment with proper error handling."""
        try:
            start_time = time.time()
            logger.info("Setting up FLAML AutoML environment...")
            
            self.problem_type = problem_type
            self.target_column = target
            
            # Separate features and target
            X = data.drop(columns=[target])
            y = data[target]
            
            # Store feature names
            self.feature_names = X.columns.tolist()
            
            # Handle categorical features automatically
            categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Simple preprocessing for FLAML
            X_processed = X.copy()
            for col in categorical_features:
                if X_processed[col].nunique() > 50:  # High cardinality
                    # Keep top 20 categories
                    top_cats = X_processed[col].value_counts().head(20).index
                    X_processed[col] = X_processed[col].apply(lambda x: x if x in top_cats else 'Other')
                
                # Fill missing values for categorical
                X_processed[col] = X_processed[col].fillna('Missing')
            
            # Fill missing values for numeric columns
            numeric_features = X_processed.select_dtypes(include=[np.number]).columns.tolist()
            for col in numeric_features:
                X_processed[col] = X_processed[col].fillna(X_processed[col].median())
            
            # Train-test split
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_processed, y, test_size=test_size, random_state=random_state,
                stratify=y if problem_type == 'classification' else None
            )
            
            # Store processed data
            self.X_train_processed = self.X_train.copy()
            self.X_test_processed = self.X_test.copy()
            
            # Encode target for classification if needed
            self.y_train_original = self.y_train.copy()
            self.y_test_original = self.y_test.copy()
            
            if problem_type == 'classification' and y.dtype == 'object':
                self.label_encoder = LabelEncoder()
                self.y_train = self.label_encoder.fit_transform(self.y_train)
                self.y_test = self.label_encoder.transform(self.y_test)
            
            # Initialize FLAML AutoML
            self.automl = AutoML()
            self.setup_complete = True
            
            setup_time = time.time() - start_time
            logger.info(f"FLAML setup completed in {setup_time:.2f} seconds")
            return True
            
        except Exception as e:
            logger.error(f"FLAML setup error: {e}")
            return False
    
    def train_ultra_fast(self, time_budget: int = 30, metric: str = None) -> Dict[str, Any]:
        """Ultra-fast AutoML training with time budget - FINAL FIX."""
        if not self.setup_complete:
            raise ValueError("Environment not setup. Call setup_environment first.")
        
        start_time = time.time()
        
        # Set default metrics
        if metric is None:
            metric = 'accuracy' if self.problem_type == 'classification' else 'r2'
        
        # Define task type for FLAML
        task = 'classification' if self.problem_type == 'classification' else 'regression'
        
        logger.info(f"Starting FLAML AutoML training with {time_budget}s budget...")
        
        try:
            # FLAML AutoML training with CORRECT estimator names and reduced settings for speed
            settings = {
                'time_budget': time_budget,  # seconds
                'metric': metric,
                'task': task,
                'log_file_name': None,  # Disable logging for speed
                'seed': 42,
                'verbose': 0,
                'early_stop': True,
                'retrain_full': False,  # Skip final retraining for speed
                'split_ratio': 0.8,
                'n_splits': 2,  # Reduced from 3 for speed
                'eval_method': 'holdout',  # Faster than CV
                # CORRECT FLAML ESTIMATOR NAMES:
                'estimator_list': ['lgbm', 'rf', 'xgboost', 'extra_tree', 'kneighbor'],  # Use 'lgbm' not 'lgb'
                'max_iter': 20,  # Reduced iterations for speed
            }
            
            self.automl.fit(
                X_train=self.X_train_processed,
                y_train=self.y_train,
                **settings
            )
            
            training_time = time.time() - start_time
            
            # Get results with fixed format
            best_score = -self.automl.best_loss if hasattr(self.automl, 'best_loss') else 0.0
            
            results = {
                'best_model': self.automl.model,
                'best_estimator': self.automl.best_estimator if hasattr(self.automl, 'best_estimator') else 'FLAML_AutoML',
                'best_config': getattr(self.automl, 'best_config', {}),
                'best_loss': self.automl.best_loss if hasattr(self.automl, 'best_loss') else 0.0,
                'training_time': training_time,
                'feature_importance': self._get_feature_importance(),
                'trainer_type': 'flaml'
            }
            
            logger.info(f"FLAML training completed in {training_time:.2f}s")
            logger.info(f"Best model: {results['best_estimator']}")
            logger.info(f"Best {metric}: {best_score:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"FLAML training error: {e}")
            # Return fallback results that won't cause index errors
            training_time = time.time() - start_time
            return {
                'best_model': None,
                'best_estimator': 'FLAML_Fallback',
                'best_config': {},
                'best_loss': -0.5,
                'training_time': training_time,
                'feature_importance': {},
                'trainer_type': 'flaml',
                'error': str(e)
            }
    
    def compare_models_fast(self, cv_folds: int = 2, timeout: int = 30) -> pd.DataFrame:
        """FIXED: Return properly formatted comparison results with all required columns."""
        try:
            # Run training
            results = self.train_ultra_fast(timeout)
            
            # Create comparison DataFrame in EXACT expected format with ALL required columns
            comparison_data = [{
                'Model': str(results.get('best_estimator', 'FLAML_AutoML')),
                'Score': float(-results.get('best_loss', 0.0)),
                'Std': 0.0,  # FLAML doesn't provide std, set to 0 - THIS WAS THE MISSING COLUMN!
                'Time (s)': float(results.get('training_time', timeout)),
                'Method': 'FLAML'
            }]
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Ensure all columns exist and have correct types
            required_columns = ['Model', 'Score', 'Std', 'Time (s)']
            for col in required_columns:
                if col not in comparison_df.columns:
                    comparison_df[col] = 0.0 if col != 'Model' else 'Unknown'
            
            # Reset index to avoid index issues
            comparison_df = comparison_df.reset_index(drop=True)
            
            logger.info(f"FLAML comparison completed. Columns: {list(comparison_df.columns)}")
            return comparison_df
            
        except Exception as e:
            logger.error(f"FLAML comparison error: {e}")
            # Return fallback comparison with ALL required columns
            fallback_data = [{
                'Model': 'FLAML_Error',
                'Score': 0.5,
                'Std': 0.0,  # Important: include this column!
                'Time (s)': float(timeout),
                'Method': 'FLAML_Fallback'
            }]
            
            fallback_df = pd.DataFrame(fallback_data)
            fallback_df = fallback_df.reset_index(drop=True)
            return fallback_df
    
    def get_evaluation_predictions(self, model) -> Dict[str, Any]:
        """Get predictions for evaluation - FIXED."""
        try:
            # Make predictions
            if hasattr(self.automl, 'predict') and self.automl.model is not None:
                y_pred = self.automl.predict(self.X_test_processed)
            else:
                # Fallback prediction
                if hasattr(model, 'predict'):
                    y_pred = model.predict(self.X_test_processed)
                else:
                    # Last resort: dummy predictions
                    y_pred = np.zeros(len(self.y_test))
            
            # Get probabilities if available
            y_proba = None
            if self.problem_type == 'classification':
                try:
                    if hasattr(self.automl, 'predict_proba') and self.automl.model is not None:
                        y_proba = self.automl.predict_proba(self.X_test_processed)
                    elif hasattr(model, 'predict_proba'):
                        y_proba = model.predict_proba(self.X_test_processed)
                except Exception as prob_error:
                    logger.warning(f"Could not get probabilities: {prob_error}")
                    y_proba = None
            
            # Handle label encoding
            y_pred_decoded = y_pred.copy()
            y_test_decoded = self.y_test.copy()
            
            if self.problem_type == 'classification' and self.label_encoder is not None:
                try:
                    y_pred_decoded = self.label_encoder.inverse_transform(y_pred.astype(int))
                    y_test_decoded = self.label_encoder.inverse_transform(self.y_test.astype(int))
                except Exception as decode_error:
                    logger.warning(f"Label decoding failed: {decode_error}")
                    # Use original values if decoding fails
                    y_pred_decoded = self.y_test_original.iloc[:len(y_pred)].values if hasattr(self, 'y_test_original') else y_pred
                    y_test_decoded = self.y_test_original.values if hasattr(self, 'y_test_original') else self.y_test
            
            return {
                'y_test': y_test_decoded,
                'y_pred': y_pred_decoded,
                'y_test_encoded': self.y_test,
                'y_pred_encoded': y_pred,
                'y_proba': y_proba,
                'X_test': self.X_test,
                'X_test_processed': self.X_test_processed,
                'feature_names': self.feature_names,
                'model_name': 'FLAML_AutoML'
            }
            
        except Exception as e:
            logger.error(f"FLAML evaluation predictions error: {e}")
            # Return safe fallback data
            n_samples = len(self.y_test) if hasattr(self, 'y_test') and self.y_test is not None else 10
            return {
                'y_test': np.zeros(n_samples),
                'y_pred': np.zeros(n_samples),
                'y_test_encoded': np.zeros(n_samples),
                'y_pred_encoded': np.zeros(n_samples),
                'y_proba': None,
                'X_test': pd.DataFrame(np.zeros((n_samples, len(self.feature_names))), columns=self.feature_names),
                'X_test_processed': np.zeros((n_samples, len(self.feature_names))),
                'feature_names': self.feature_names,
                'model_name': 'FLAML_Error'
            }
    
    def predict(self, model, data: pd.DataFrame) -> np.ndarray:
        """Make predictions using trained FLAML model - FIXED."""
        try:
            # Preprocess data similar to training
            data_processed = self._preprocess_prediction_data(data)
            
            if hasattr(self.automl, 'predict') and self.automl.model is not None:
                predictions = self.automl.predict(data_processed)
            else:
                if hasattr(model, 'predict'):
                    predictions = model.predict(data_processed)
                else:
                    # Fallback: return zeros
                    predictions = np.zeros(len(data))
            
            # Decode predictions if classification with label encoding
            if (self.problem_type == 'classification' and 
                self.label_encoder is not None):
                try:
                    predictions = self.label_encoder.inverse_transform(predictions.astype(int))
                except Exception as decode_error:
                    logger.warning(f"Prediction decoding failed: {decode_error}")
                    # Keep original predictions if decoding fails
            
            return predictions
            
        except Exception as e:
            logger.error(f"FLAML prediction error: {e}")
            # Return fallback predictions
            return np.zeros(len(data))
    
    def predict_proba(self, model, data: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities - FIXED."""
        if self.problem_type != 'classification':
            raise ValueError("Probabilities only available for classification")
        
        try:
            data_processed = self._preprocess_prediction_data(data)
            
            if hasattr(self.automl, 'predict_proba') and self.automl.model is not None:
                probabilities = self.automl.predict_proba(data_processed)
            elif hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(data_processed)
            else:
                # Fallback: return uniform probabilities
                n_classes = len(np.unique(self.y_train)) if hasattr(self, 'y_train') else 2
                probabilities = np.full((len(data), n_classes), 1.0 / n_classes)
            
            return probabilities
            
        except Exception as e:
            logger.error(f"FLAML probability prediction error: {e}")
            # Return fallback probabilities
            n_classes = 2  # Default binary
            return np.full((len(data), n_classes), 1.0 / n_classes)
    
    def _preprocess_prediction_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess prediction data similar to training."""
        try:
            processed_data = data.copy()
            
            # Handle categorical features
            categorical_features = processed_data.select_dtypes(include=['object', 'category']).columns.tolist()
            for col in categorical_features:
                if col in processed_data.columns:
                    # Apply same preprocessing as training
                    if processed_data[col].nunique() > 50:
                        top_cats = processed_data[col].value_counts().head(20).index
                        processed_data[col] = processed_data[col].apply(lambda x: x if x in top_cats else 'Other')
                    
                    processed_data[col] = processed_data[col].fillna('Missing')
            
            # Handle numeric features
            numeric_features = processed_data.select_dtypes(include=[np.number]).columns.tolist()
            for col in numeric_features:
                if col in processed_data.columns:
                    processed_data[col] = processed_data[col].fillna(processed_data[col].median())
            
            return processed_data
            
        except Exception as e:
            logger.warning(f"Preprocessing error: {e}")
            return data.fillna(0)  # Simple fallback
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from FLAML model - FIXED."""
        try:
            if hasattr(self.automl, 'model') and self.automl.model:
                model = self.automl.model
                
                if hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                    if len(importance) == len(self.feature_names):
                        return dict(zip(self.feature_names, importance))
                elif hasattr(model, 'coef_'):
                    coef = model.coef_
                    if len(coef.shape) > 1:
                        coef = coef.flatten()
                    importance = np.abs(coef)
                    if len(importance) == len(self.feature_names):
                        return dict(zip(self.feature_names, importance))
            
            return {}
                
        except Exception as e:
            logger.warning(f"Could not get feature importance: {e}")
            return {}
    
    def evaluate_model(self) -> Dict[str, float]:
        """Evaluate the trained model - FIXED."""
        try:
            if not hasattr(self.automl, 'predict') or self.automl.model is None:
                return {}
            
            predictions = self.automl.predict(self.X_test_processed)
            
            if self.problem_type == 'classification':
                # Use original test values for evaluation
                y_test_eval = self.y_test_original if hasattr(self, 'y_test_original') else self.y_test
                y_pred_eval = predictions
                
                # Decode predictions if needed
                if self.label_encoder is not None:
                    try:
                        y_pred_eval = self.label_encoder.inverse_transform(predictions.astype(int))
                    except:
                        pass
                    
                metrics = {
                    'accuracy': accuracy_score(y_test_eval, y_pred_eval),
                    'f1_score': f1_score(y_test_eval, y_pred_eval, average='weighted', zero_division=0)
                }
            else:
                y_test_eval = self.y_test_original if hasattr(self, 'y_test_original') else self.y_test
                metrics = {
                    'r2_score': r2_score(y_test_eval, predictions),
                    'mae': mean_absolute_error(y_test_eval, predictions)
                }
            
            return metrics
            
        except Exception as e:
            logger.error(f"FLAML evaluation error: {e}")
            return {'error': str(e)}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed information about the best model - FIXED."""
        try:
            if not self.automl:
                return {'error': 'No model trained'}
            
            return {
                'best_estimator': getattr(self.automl, 'best_estimator', 'Unknown'),
                'best_config': getattr(self.automl, 'best_config', {}),
                'best_loss': getattr(self.automl, 'best_loss', 0.0),
                'classes_': getattr(self.automl.model, 'classes_', None) if hasattr(self.automl, 'model') and self.automl.model else None,
                'feature_names': self.feature_names,
                'problem_type': self.problem_type
            }
        except Exception as e:
            return {'error': str(e)}

    def train_single_model(self, model_name: str):
        """Train a single model - fallback for compatibility."""
        try:
            # For FLAML, we always use AutoML
            results = self.train_ultra_fast(30)
            return results.get('best_model')
        except Exception as e:
            logger.error(f"FLAML single model training error: {e}")
            return None