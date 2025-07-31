"""
Model Compatibility Utilities
Handles compatibility between different training methods and ensures 
consistent evaluation across FastModelTrainer, FLAML, and other methods.
"""

import pandas as pd
import numpy as np
import streamlit as st
import logging
from typing import Dict, Any, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ModelCompatibilityHandler:
    """Handles compatibility between different model types and training methods."""
    
    def __init__(self):
        self.supported_trainers = ['fast_sklearn', 'flaml', 'h2o', 'pycaret']
    
    def get_standardized_model_info(self) -> Dict[str, Any]:
        """Extract standardized model information from session state."""
        try:
            # Get basic information
            model_info = {
                'model': st.session_state.get('trained_model'),
                'problem_type': st.session_state.get('problem_type'),
                'target_column': st.session_state.get('target_column'),
                'trainer': st.session_state.get('fast_trainer'),
                'training_results': st.session_state.get('fast_training_results', {}),
                'original_data': st.session_state.get('data'),
                'processed_data': st.session_state.get('preview_data'),
                'trainer_type': self._detect_trainer_type(),
                'test_data': None,
                'predictions': None,
                'probabilities': None
            }
            
            # Get test data and predictions using the appropriate method
            test_data = self._extract_test_data(model_info)
            if test_data:
                model_info.update(test_data)
                
                # Generate predictions
                predictions_data = self._generate_predictions(model_info)
                if predictions_data:
                    model_info.update(predictions_data)
            
            return model_info
            
        except Exception as e:
            logger.error(f"Error extracting standardized model info: {e}")
            return self._create_fallback_model_info()
    
    def _detect_trainer_type(self) -> str:
        """Detect the type of trainer used."""
        try:
            # Check training results for trainer type
            training_results = st.session_state.get('fast_training_results', {})
            if 'trainer_type' in training_results:
                return training_results['trainer_type']
            
            # Check for specific trainer objects
            trainer = st.session_state.get('fast_trainer')
            if trainer:
                trainer_class = type(trainer).__name__
                if 'FLAML' in trainer_class:
                    return 'flaml'
                elif 'Fast' in trainer_class:
                    return 'fast_sklearn'
                elif 'H2O' in trainer_class:
                    return 'h2o'
                elif 'PyCaret' in trainer_class or 'ModelTrainer' in trainer_class:
                    return 'pycaret'
            
            # Check session state for other indicators
            if hasattr(st.session_state, 'model_results'):
                return 'pycaret'
            
            return 'unknown'
            
        except Exception as e:
            logger.warning(f"Could not detect trainer type: {e}")
            return 'unknown'
    
    def _extract_test_data(self, model_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract test data from various trainer types."""
        try:
            trainer = model_info['trainer']
            trainer_type = model_info['trainer_type']
            
            test_data = {}
            
            if trainer_type == 'fast_sklearn' and trainer:
                # FastModelTrainer stores test data directly
                if hasattr(trainer, 'X_test_processed') and hasattr(trainer, 'y_test'):
                    test_data['X_test'] = trainer.X_test_processed
                    test_data['y_test'] = trainer.y_test
                    if hasattr(trainer, 'X_train_processed'):
                        test_data['X_train'] = trainer.X_train_processed
                        test_data['y_train'] = trainer.y_train
                    return test_data
            
            elif trainer_type == 'flaml' and trainer:
                # FLAML trainer test data
                if hasattr(trainer, 'X_test') and hasattr(trainer, 'y_test'):
                    test_data['X_test'] = trainer.X_test
                    test_data['y_test'] = trainer.y_test
                    if hasattr(trainer, 'X_train'):
                        test_data['X_train'] = trainer.X_train
                        test_data['y_train'] = trainer.y_train
                    return test_data
            
            elif trainer_type == 'pycaret':
                # For PyCaret, we need to use pull() to get test data
                try:
                    # This is a placeholder - PyCaret test data extraction would go here
                    logger.info("PyCaret test data extraction not implemented")
                except:
                    pass
            
            # Fallback: create test data from available data
            return self._create_test_data_fallback(model_info)
            
        except Exception as e:
            logger.warning(f"Could not extract test data: {e}")
            return self._create_test_data_fallback(model_info)
    
    def _create_test_data_fallback(self, model_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create test data as fallback when trainer doesn't provide it."""
        try:
            # Use processed data if available, otherwise original data
            data = model_info['processed_data'] if model_info['processed_data'] is not None else model_info['original_data']
            target_column = model_info['target_column']
            
            if data is None or target_column is None:
                return None
            
            # Simple train-test split
            from sklearn.model_selection import train_test_split
            
            X = data.drop(columns=[target_column])
            y = data[target_column]
            
            # Use a small test size for evaluation
            test_size = min(0.3, max(0.1, 100 / len(data)))
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42,
                stratify=y if model_info['problem_type'] == 'classification' else None
            )
            
            # Simple preprocessing for compatibility
            X_test_processed = self._simple_preprocessing(X_test, X_train)
            X_train_processed = self._simple_preprocessing(X_train, X_train)
            
            return {
                'X_test': X_test_processed,
                'y_test': y_test,
                'X_train': X_train_processed,
                'y_train': y_train
            }
            
        except Exception as e:
            logger.error(f"Could not create fallback test data: {e}")
            return None
    
    def _simple_preprocessing(self, X: pd.DataFrame, X_reference: pd.DataFrame) -> np.ndarray:
        """Simple preprocessing for compatibility."""
        try:
            # Handle categorical variables
            X_processed = X.copy()
            
            # Fill missing values
            for col in X_processed.columns:
                if X_processed[col].dtype in ['object', 'category']:
                    # Fill categorical with mode
                    mode_val = X_reference[col].mode()
                    fill_val = mode_val[0] if len(mode_val) > 0 else 'Unknown'
                    X_processed[col] = X_processed[col].fillna(fill_val)
                else:
                    # Fill numeric with mean
                    mean_val = X_reference[col].mean()
                    X_processed[col] = X_processed[col].fillna(mean_val)
            
            # Simple encoding for categorical variables
            from sklearn.preprocessing import LabelEncoder
            
            for col in X_processed.columns:
                if X_processed[col].dtype in ['object', 'category']:
                    le = LabelEncoder()
                    # Fit on reference data and transform current data
                    combined_data = pd.concat([X_reference[col], X_processed[col]])
                    le.fit(combined_data.astype(str))
                    X_processed[col] = le.transform(X_processed[col].astype(str))
            
            return X_processed.values
            
        except Exception as e:
            logger.warning(f"Simple preprocessing failed: {e}")
            # Return as-is, converted to numeric
            return pd.get_dummies(X, drop_first=True).fillna(0).values
    
    def _generate_predictions(self, model_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate predictions using the appropriate method."""
        try:
            model = model_info['model']
            X_test = model_info.get('X_test')
            
            if model is None or X_test is None:
                return None
            
            predictions_data = {}
            
            # Try different prediction methods
            predictions = self._safe_predict(model, X_test, model_info)
            if predictions is not None:
                predictions_data['predictions'] = predictions
            
            # Try to get probabilities for classification
            if model_info['problem_type'] == 'classification':
                probabilities = self._safe_predict_proba(model, X_test, model_info)
                if probabilities is not None:
                    predictions_data['probabilities'] = probabilities
            
            return predictions_data if predictions_data else None
            
        except Exception as e:
            logger.error(f"Could not generate predictions: {e}")
            return None
    
    def _safe_predict(self, model, X_test, model_info: Dict[str, Any]) -> Optional[np.ndarray]:
        """Safely make predictions with error handling."""
        try:
            trainer = model_info.get('trainer')
            trainer_type = model_info.get('trainer_type')
            
            # Method 1: Use trainer's predict method
            if trainer and hasattr(trainer, 'predict'):
                try:
                    # Convert to DataFrame if needed
                    if not isinstance(X_test, pd.DataFrame):
                        if hasattr(trainer, 'feature_names') and len(trainer.feature_names) == X_test.shape[1]:
                            X_test_df = pd.DataFrame(X_test, columns=trainer.feature_names)
                        else:
                            X_test_df = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(X_test.shape[1])])
                    else:
                        X_test_df = X_test
                    
                    return trainer.predict(model, X_test_df)
                except Exception as e:
                    logger.warning(f"Trainer predict failed: {e}")
            
            # Method 2: Direct model prediction
            if hasattr(model, 'predict'):
                try:
                    return model.predict(X_test)
                except Exception as e:
                    logger.warning(f"Direct model predict failed: {e}")
            
            # Method 3: FLAML specific
            if trainer_type == 'flaml' and hasattr(trainer, 'automl'):
                try:
                    X_test_df = pd.DataFrame(X_test) if not isinstance(X_test, pd.DataFrame) else X_test
                    return trainer.automl.predict(X_test_df)
                except Exception as e:
                    logger.warning(f"FLAML predict failed: {e}")
            
            return None
            
        except Exception as e:
            logger.error(f"All prediction methods failed: {e}")
            return None
    
    def _safe_predict_proba(self, model, X_test, model_info: Dict[str, Any]) -> Optional[np.ndarray]:
        """Safely get prediction probabilities with error handling."""
        try:
            trainer = model_info.get('trainer')
            trainer_type = model_info.get('trainer_type')
            
            # Method 1: Use trainer's predict_proba method
            if trainer and hasattr(trainer, 'predict_proba'):
                try:
                    # Convert to DataFrame if needed
                    if not isinstance(X_test, pd.DataFrame):
                        if hasattr(trainer, 'feature_names') and len(trainer.feature_names) == X_test.shape[1]:
                            X_test_df = pd.DataFrame(X_test, columns=trainer.feature_names)
                        else:
                            X_test_df = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(X_test.shape[1])])
                    else:
                        X_test_df = X_test
                    
                    return trainer.predict_proba(model, X_test_df)
                except Exception as e:
                    logger.warning(f"Trainer predict_proba failed: {e}")
            
            # Method 2: Direct model prediction probabilities
            if hasattr(model, 'predict_proba'):
                try:
                    return model.predict_proba(X_test)
                except Exception as e:
                    logger.warning(f"Direct model predict_proba failed: {e}")
            
            # Method 3: FLAML specific
            if trainer_type == 'flaml' and hasattr(trainer, 'automl'):
                try:
                    X_test_df = pd.DataFrame(X_test) if not isinstance(X_test, pd.DataFrame) else X_test
                    return trainer.automl.predict_proba(X_test_df)
                except Exception as e:
                    logger.warning(f"FLAML predict_proba failed: {e}")
            
            # Method 4: Try decision_function for SVM-like models
            if hasattr(model, 'decision_function'):
                try:
                    decision_scores = model.decision_function(X_test)
                    # Convert to probabilities using sigmoid for binary classification
                    if len(decision_scores.shape) == 1:  # Binary classification
                        from scipy.special import expit
                        prob_positive = expit(decision_scores)
                        return np.column_stack([1 - prob_positive, prob_positive])
                except Exception as e:
                    logger.warning(f"Decision function conversion failed: {e}")
            
            return None
            
        except Exception as e:
            logger.error(f"All probability prediction methods failed: {e}")
            return None
    
    def _create_fallback_model_info(self) -> Dict[str, Any]:
        """Create fallback model info when extraction fails."""
        return {
            'model': st.session_state.get('trained_model'),
            'problem_type': st.session_state.get('problem_type', 'classification'),
            'target_column': st.session_state.get('target_column'),
            'trainer': None,
            'training_results': {},
            'original_data': st.session_state.get('data'),
            'processed_data': None,
            'trainer_type': 'unknown',
            'X_test': None,
            'y_test': None,
            'X_train': None,
            'y_train': None,
            'predictions': None,
            'probabilities': None
        }
    
    def validate_model_compatibility(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate model compatibility and return status."""
        validation = {
            'is_compatible': True,
            'warnings': [],
            'errors': [],
            'capabilities': {
                'can_predict': False,
                'can_predict_proba': False,
                'has_feature_importance': False,
                'has_test_data': False
            }
        }
        
        try:
            model = model_info.get('model')
            
            # Check basic model availability
            if model is None:
                validation['errors'].append("No trained model available")
                validation['is_compatible'] = False
                return validation
            
            # Check prediction capability
            if hasattr(model, 'predict') or (model_info.get('trainer') and hasattr(model_info['trainer'], 'predict')):
                validation['capabilities']['can_predict'] = True
            else:
                validation['warnings'].append("Model does not support prediction")
            
            # Check probability prediction capability
            if (hasattr(model, 'predict_proba') or 
                (model_info.get('trainer') and hasattr(model_info['trainer'], 'predict_proba')) or
                hasattr(model, 'decision_function')):
                validation['capabilities']['can_predict_proba'] = True
            
            # Check feature importance capability
            if (hasattr(model, 'feature_importances_') or 
                hasattr(model, 'coef_') or
                (model_info.get('trainer') and hasattr(model_info['trainer'], 'get_feature_importance'))):
                validation['capabilities']['has_feature_importance'] = True
            
            # Check test data availability
            if model_info.get('X_test') is not None and model_info.get('y_test') is not None:
                validation['capabilities']['has_test_data'] = True
            else:
                validation['warnings'].append("No test data available - some evaluations will be limited")
            
            # Check problem type consistency
            problem_type = model_info.get('problem_type')
            if problem_type not in ['classification', 'regression']:
                validation['warnings'].append(f"Unknown problem type: {problem_type}")
            
            # Trainer-specific validations
            trainer_type = model_info.get('trainer_type')
            if trainer_type == 'unknown':
                validation['warnings'].append("Could not determine trainer type - some features may not work")
            
            return validation
            
        except Exception as e:
            validation['errors'].append(f"Validation error: {str(e)}")
            validation['is_compatible'] = False
            return validation
    
    def get_feature_names_universal(self, model_info: Dict[str, Any]) -> list:
        """Get feature names from any trainer type."""
        try:
            trainer = model_info.get('trainer')
            
            # Method 1: Trainer feature names
            if trainer and hasattr(trainer, 'feature_names'):
                return trainer.feature_names
            
            # Method 2: From processed data
            if model_info.get('X_test') is not None:
                X_test = model_info['X_test']
                if hasattr(X_test, 'columns'):
                    return X_test.columns.tolist()
                elif hasattr(X_test, 'shape'):
                    n_features = X_test.shape[1]
                    return [f'Feature_{i}' for i in range(n_features)]
            
            # Method 3: From original data
            original_data = model_info.get('original_data')
            target_column = model_info.get('target_column')
            if original_data is not None and target_column is not None:
                feature_columns = [col for col in original_data.columns if col != target_column]
                return feature_columns
            
            # Fallback
            return [f'Feature_{i}' for i in range(10)]
            
        except Exception as e:
            logger.warning(f"Could not get feature names: {e}")
            return [f'Feature_{i}' for i in range(10)]
    
    def get_model_parameters_universal(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Get model parameters from any model type."""
        try:
            model = model_info.get('model')
            parameters = {}
            
            if model is None:
                return parameters
            
            # Try get_params method (sklearn-like)
            if hasattr(model, 'get_params'):
                try:
                    all_params = model.get_params()
                    # Filter to important parameters
                    important_params = [
                        'n_estimators', 'max_depth', 'learning_rate', 'random_state',
                        'C', 'gamma', 'kernel', 'alpha', 'l1_ratio', 'n_neighbors',
                        'criterion', 'max_features', 'min_samples_split', 'min_samples_leaf'
                    ]
                    
                    for param in important_params:
                        if param in all_params:
                            parameters[param] = all_params[param]
                            
                except Exception as e:
                    logger.warning(f"Could not get model parameters: {e}")
            
            # Add model-specific information
            parameters['model_type'] = type(model).__name__
            parameters['trainer_type'] = model_info.get('trainer_type', 'unknown')
            
            return parameters
            
        except Exception as e:
            logger.error(f"Error getting model parameters: {e}")
            return {'model_type': 'Unknown'}
    
    def create_evaluation_summary(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create a comprehensive evaluation summary."""
        try:
            summary = {
                'model_info': {
                    'model_type': type(model_info['model']).__name__ if model_info['model'] else 'None',
                    'problem_type': model_info.get('problem_type', 'unknown'),
                    'trainer_type': model_info.get('trainer_type', 'unknown'),
                    'target_column': model_info.get('target_column', 'unknown')
                },
                'data_info': {},
                'performance_metrics': {},
                'capabilities': {}
            }
            
            # Data information
            if model_info.get('X_test') is not None and model_info.get('y_test') is not None:
                summary['data_info']['test_samples'] = len(model_info['y_test'])
                summary['data_info']['num_features'] = (
                    model_info['X_test'].shape[1] if hasattr(model_info['X_test'], 'shape') else 0
                )
            
            if model_info.get('original_data') is not None:
                summary['data_info']['original_shape'] = list(model_info['original_data'].shape)
            
            # Performance metrics
            if model_info.get('predictions') is not None and model_info.get('y_test') is not None:
                y_test = model_info['y_test']
                y_pred = model_info['predictions']
                
                if model_info['problem_type'] == 'classification':
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                    summary['performance_metrics'] = {
                        'accuracy': float(accuracy_score(y_test, y_pred)),
                        'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
                        'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
                        'f1_score': float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
                    }
                else:
                    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
                    summary['performance_metrics'] = {
                        'r2_score': float(r2_score(y_test, y_pred)),
                        'mae': float(mean_absolute_error(y_test, y_pred)),
                        'mse': float(mean_squared_error(y_test, y_pred)),
                        'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred)))
                    }
            
            # Capabilities
            validation = self.validate_model_compatibility(model_info)
            summary['capabilities'] = validation['capabilities']
            
            # Training information
            training_results = model_info.get('training_results', {})
            summary['training_info'] = {
                'training_time': training_results.get('training_time', 0),
                'models_compared': training_results.get('models_compared', 1)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error creating evaluation summary: {e}")
            return {'error': str(e)}

# Global instance for easy access
compatibility_handler = ModelCompatibilityHandler()

def get_compatible_model_info() -> Dict[str, Any]:
    """Get standardized model information for evaluation."""
    return compatibility_handler.get_standardized_model_info()

def validate_model_for_evaluation() -> Dict[str, Any]:
    """Validate if model is ready for evaluation."""
    model_info = get_compatible_model_info()
    return compatibility_handler.validate_model_compatibility(model_info)

def safe_model_predict(model, X_test, model_info: Dict[str, Any]) -> Optional[np.ndarray]:
    """Safely predict with any model type."""
    return compatibility_handler._safe_predict(model, X_test, model_info)

def safe_model_predict_proba(model, X_test, model_info: Dict[str, Any]) -> Optional[np.ndarray]:
    """Safely get probabilities with any model type."""
    return compatibility_handler._safe_predict_proba(model, X_test, model_info)