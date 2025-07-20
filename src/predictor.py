"""
Predictor Module
Handles model predictions and inference (No PyCaret dependency)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, List
import logging
import streamlit as st

logger = logging.getLogger(__name__)

class Predictor:
    """Handles model predictions and inference without PyCaret."""
    
    def __init__(self, config: Dict[str, Any]):
        # Handle None config gracefully
        if config is None:
            config = {}
            
        self.config = config
    
    def predict_single(self, model, input_data: pd.DataFrame) -> Union[float, str, int]:
        """Make prediction for a single record using fast trainer."""
        try:
            # Use the fast trainer to make predictions
            trainer = st.session_state.get('fast_trainer')
            
            if trainer and hasattr(trainer, 'predict'):
                prediction = trainer.predict(model, input_data)
                return prediction[0] if isinstance(prediction, np.ndarray) else prediction
            else:
                # Fallback: direct model prediction
                # Prepare data using the same preprocessing as training
                processed_data = self._preprocess_for_prediction(input_data)
                prediction = model.predict(processed_data)
                return prediction[0] if isinstance(prediction, np.ndarray) else prediction
            
        except Exception as e:
            logger.error(f"Single prediction error: {e}")
            raise
    
    def predict_batch(self, model, input_data: pd.DataFrame) -> np.ndarray:
        """Make predictions for batch data using fast trainer."""
        try:
            # Use the fast trainer to make predictions
            trainer = st.session_state.get('fast_trainer')
            
            if trainer and hasattr(trainer, 'predict'):
                predictions = trainer.predict(model, input_data)
                return predictions
            else:
                # Fallback: direct model prediction
                processed_data = self._preprocess_for_prediction(input_data)
                predictions = model.predict(processed_data)
                return predictions
            
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            raise
    
    def predict_probability(self, model, input_data: pd.DataFrame) -> float:
        """Get prediction probability for classification models."""
        try:
            trainer = st.session_state.get('fast_trainer')
            
            if (st.session_state.problem_type == 'classification' and 
                hasattr(model, 'predict_proba')):
                
                if trainer and hasattr(trainer, 'predict_proba'):
                    probabilities = trainer.predict_proba(model, input_data)
                else:
                    # Fallback: direct model prediction
                    processed_data = self._preprocess_for_prediction(input_data)
                    probabilities = model.predict_proba(processed_data)
                
                # Return the maximum probability
                if len(probabilities.shape) > 1:
                    max_prob = np.max(probabilities[0])
                else:
                    max_prob = probabilities[0]
                
                return float(max_prob)
            else:
                # Not applicable for regression or models without probabilities
                return 0.0
                
        except Exception as e:
            logger.error(f"Probability prediction error: {e}")
            return 0.0
    
    def _preprocess_for_prediction(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess input data for prediction."""
        try:
            # Get the preprocessing configuration used during training
            preprocessing_config = st.session_state.get('preprocessing_config', {})
            
            # Apply the same preprocessing as training
            from src.data_handler import DataHandler
            data_handler = DataHandler(self.config)
            
            # Create a temporary dataset with target column for preprocessing
            temp_data = input_data.copy()
            
            # Add a dummy target column if it doesn't exist
            target_column = st.session_state.get('target_column')
            if target_column and target_column not in temp_data.columns:
                temp_data[target_column] = 0  # Dummy values
            
            # Apply preprocessing
            processed_data = data_handler.apply_preprocessing(
                temp_data, 
                target_column, 
                preprocessing_config
            )
            
            # Remove the target column if it was added
            if target_column and target_column in processed_data.columns:
                processed_data = processed_data.drop(columns=[target_column])
            
            return processed_data
            
        except Exception as e:
            logger.warning(f"Preprocessing failed, using original data: {e}")
            return input_data
    
    def predict_with_explanation(self, model, input_data: pd.DataFrame) -> Dict[str, Any]:
        """Make prediction with model explanation."""
        try:
            # Get basic prediction
            prediction = self.predict_single(model, input_data)
            
            # Get probability if classification
            probability = None
            if st.session_state.problem_type == 'classification':
                probability = self.predict_probability(model, input_data)
            
            # Try to get feature importance for explanation
            explanation = None
            try:
                explanation = self._get_prediction_explanation(model, input_data)
            except:
                logger.warning("Could not generate prediction explanation")
            
            result = {
                'prediction': prediction,
                'probability': probability,
                'explanation': explanation,
                'input_data': input_data.to_dict('records')[0]
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction with explanation error: {e}")
            raise
    
    def _get_prediction_explanation(self, model, input_data: pd.DataFrame) -> Dict[str, Any]:
        """Get explanation for prediction using feature importance."""
        try:
            # Get feature importance from the model
            feature_importance = None
            
            if hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                feature_importance = np.abs(model.coef_).flatten()
            
            if feature_importance is not None:
                # Get feature names
                processed_data = self._preprocess_for_prediction(input_data)
                feature_names = processed_data.columns.tolist()
                
                # Ensure lengths match
                if len(feature_importance) == len(feature_names):
                    # Create explanation
                    feature_values = processed_data.iloc[0].values
                    
                    explanation = {
                        'feature_importance': dict(zip(feature_names, feature_importance)),
                        'feature_values': dict(zip(feature_names, feature_values)),
                        'top_features': sorted(
                            zip(feature_names, feature_importance), 
                            key=lambda x: abs(x[1]), 
                            reverse=True
                        )[:5]
                    }
                    
                    return explanation
            
            return None
            
        except Exception as e:
            logger.error(f"Explanation generation error: {e}")
            return None
    
    def batch_predict_with_confidence(self, model, input_data: pd.DataFrame) -> pd.DataFrame:
        """Make batch predictions with confidence scores."""
        try:
            # Get predictions
            predictions = self.predict_batch(model, input_data)
            
            # Create results dataframe
            results_df = input_data.copy()
            results_df['prediction'] = predictions
            
            # Add confidence scores for classification
            if st.session_state.problem_type == 'classification' and hasattr(model, 'predict_proba'):
                try:
                    trainer = st.session_state.get('fast_trainer')
                    
                    if trainer and hasattr(trainer, 'predict_proba'):
                        probabilities = trainer.predict_proba(model, input_data)
                    else:
                        processed_data = self._preprocess_for_prediction(input_data)
                        probabilities = model.predict_proba(processed_data)
                    
                    # Add confidence as maximum probability
                    confidence = np.max(probabilities, axis=1)
                    results_df['confidence'] = confidence
                    
                except Exception as e:
                    logger.warning(f"Could not compute confidence scores: {e}")
                    results_df['confidence'] = 0.5
            
            return results_df
            
        except Exception as e:
            logger.error(f"Batch prediction with confidence error: {e}")
            raise
    
    def validate_input_data(self, input_data: pd.DataFrame, training_columns: List[str]) -> Dict[str, Any]:
        """Validate input data against training data schema."""
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': []
        }
        
        try:
            # Check if all required columns are present
            input_columns = set(input_data.columns)
            required_columns = set(training_columns)
            
            missing_columns = required_columns - input_columns
            extra_columns = input_columns - required_columns
            
            if missing_columns:
                validation_results['errors'].append(f"Missing columns: {list(missing_columns)}")
                validation_results['is_valid'] = False
            
            if extra_columns:
                validation_results['warnings'].append(f"Extra columns will be ignored: {list(extra_columns)}")
            
            # Check data types (simplified check)
            for column in input_columns.intersection(required_columns):
                if input_data[column].isnull().all():
                    validation_results['warnings'].append(f"Column '{column}' contains only null values")
            
            # Check for empty dataframe
            if input_data.empty:
                validation_results['errors'].append("Input data is empty")
                validation_results['is_valid'] = False
            
            return validation_results
            
        except Exception as e:
            validation_results['errors'].append(f"Validation error: {str(e)}")
            validation_results['is_valid'] = False
            return validation_results
    
    def preprocess_input_for_prediction(self, input_data: pd.DataFrame, training_columns: List[str]) -> pd.DataFrame:
        """Preprocess input data to match training data format."""
        try:
            processed_data = input_data.copy()
            
            # Select only training columns that exist
            available_columns = [col for col in training_columns if col in processed_data.columns]
            processed_data = processed_data[available_columns]
            
            # Add missing columns with default values
            for col in training_columns:
                if col not in processed_data.columns:
                    # Add column with default value (0 for numeric, most frequent for categorical)
                    processed_data[col] = 0  # Simplified - should use proper defaults
            
            # Reorder columns to match training data
            processed_data = processed_data[training_columns]
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Input preprocessing error: {e}")
            raise
    
    def get_prediction_intervals(self, model, input_data: pd.DataFrame, confidence_level: float = 0.95) -> Dict[str, Any]:
        """Get prediction intervals for regression models."""
        try:
            if st.session_state.problem_type == 'regression':
                # Get basic prediction
                prediction = self.predict_single(model, input_data)
                
                # Mock prediction intervals (replace with proper implementation)
                uncertainty = abs(prediction) * 0.1  # 10% uncertainty
                alpha = 1 - confidence_level
                
                lower_bound = prediction - uncertainty
                upper_bound = prediction + uncertainty
                
                return {
                    'prediction': prediction,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'confidence_level': confidence_level,
                    'uncertainty': uncertainty
                }
            else:
                raise ValueError("Prediction intervals only available for regression models")
                
        except Exception as e:
            logger.error(f"Prediction intervals error: {e}")
            raise
    
    def compare_predictions(self, models: List[Any], input_data: pd.DataFrame) -> pd.DataFrame:
        """Compare predictions from multiple models."""
        try:
            predictions = {}
            
            for i, model in enumerate(models):
                try:
                    pred = self.predict_single(model, input_data)
                    predictions[f'Model_{i+1}'] = pred
                except Exception as e:
                    logger.warning(f"Prediction failed for model {i+1}: {e}")
                    predictions[f'Model_{i+1}'] = None
            
            # Create comparison dataframe
            comparison_df = pd.DataFrame([predictions])
            
            # Add input data
            for col, value in input_data.iloc[0].items():
                comparison_df[f'Input_{col}'] = value
            
            return comparison_df
            
        except Exception as e:
            logger.error(f"Model comparison error: {e}")
            raise
    
    def export_predictions(self, predictions_df: pd.DataFrame, format: str = 'csv') -> bytes:
        """Export predictions to various formats."""
        try:
            if format.lower() == 'csv':
                return predictions_df.to_csv(index=False).encode('utf-8')
            elif format.lower() == 'excel':
                from io import BytesIO
                output = BytesIO()
                predictions_df.to_excel(output, index=False)
                return output.getvalue()
            elif format.lower() == 'json':
                return predictions_df.to_json(orient='records').encode('utf-8')
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Export error: {e}")
            raise
    
    def get_model_performance_on_predictions(self, model, input_data: pd.DataFrame, true_values: Optional[pd.Series] = None) -> Dict[str, float]:
        """Evaluate model performance on new predictions if true values are available."""
        try:
            if true_values is None:
                return {"message": "No true values provided for evaluation"}
            
            # Get predictions
            predictions = self.predict_batch(model, input_data)
            
            # Calculate metrics based on problem type
            if st.session_state.problem_type == 'classification':
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                
                metrics = {
                    'accuracy': accuracy_score(true_values, predictions),
                    'precision': precision_score(true_values, predictions, average='weighted', zero_division=0),
                    'recall': recall_score(true_values, predictions, average='weighted', zero_division=0),
                    'f1_score': f1_score(true_values, predictions, average='weighted', zero_division=0)
                }
            else:
                from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                
                metrics = {
                    'mae': mean_absolute_error(true_values, predictions),
                    'mse': mean_squared_error(true_values, predictions),
                    'rmse': np.sqrt(mean_squared_error(true_values, predictions)),
                    'r2': r2_score(true_values, predictions)
                }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Performance evaluation error: {e}")
            return {"error": str(e)}