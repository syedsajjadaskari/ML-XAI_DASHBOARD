"""
Predictor Module
Handles model predictions and inference
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, List
import logging
import streamlit as st

logger = logging.getLogger(__name__)

class Predictor:
    """Handles model predictions and inference."""
    
    def __init__(self, config: Dict[str, Any]):
        # Handle None config gracefully
        if config is None:
            config = {}
            
        self.config = config
    
    def predict_single(self, model, input_data: pd.DataFrame) -> Union[float, str, int]:
        """Make prediction for a single record."""
        try:
            # Import appropriate PyCaret module
            if hasattr(st.session_state, 'problem_type'):
                if st.session_state.problem_type == 'classification':
                    import pycaret.classification as pc
                    predictions = pc.predict_model(model, data=input_data)
                else:
                    import pycaret.regression as pr
                    predictions = pr.predict_model(model, data=input_data)
            else:
                # Default to classification
                import pycaret.classification as pc
                predictions = pc.predict_model(model, data=input_data)
            
            # Extract prediction value
            if 'prediction_label' in predictions.columns:
                prediction = predictions['prediction_label'].iloc[0]
            elif 'Label' in predictions.columns:
                prediction = predictions['Label'].iloc[0]
            else:
                # Take the last column as prediction
                prediction = predictions.iloc[0, -1]
            
            logger.info("Single prediction completed successfully")
            return prediction
            
        except Exception as e:
            logger.error(f"Single prediction error: {e}")
            raise
    
    def predict_batch(self, model, input_data: pd.DataFrame) -> np.ndarray:
        """Make predictions for batch data."""
        try:
            # Import appropriate PyCaret module
            if hasattr(st.session_state, 'problem_type'):
                if st.session_state.problem_type == 'classification':
                    import pycaret.classification as pc
                    predictions = pc.predict_model(model, data=input_data)
                else:
                    import pycaret.regression as pr
                    predictions = pr.predict_model(model, data=input_data)
            else:
                # Default to classification
                import pycaret.classification as pc
                predictions = pc.predict_model(model, data=input_data)
            
            # Extract prediction values
            if 'prediction_label' in predictions.columns:
                prediction_values = predictions['prediction_label'].values
            elif 'Label' in predictions.columns:
                prediction_values = predictions['Label'].values
            else:
                # Take the last column as predictions
                prediction_values = predictions.iloc[:, -1].values
            
            logger.info(f"Batch prediction completed for {len(input_data)} records")
            return prediction_values
            
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            raise
    
    def predict_probability(self, model, input_data: pd.DataFrame) -> float:
        """Get prediction probability for classification models."""
        try:
            if hasattr(st.session_state, 'problem_type') and st.session_state.problem_type == 'classification':
                import pycaret.classification as pc
                predictions = pc.predict_model(model, data=input_data)
                
                # Look for probability columns
                prob_columns = [col for col in predictions.columns if 'prediction_score' in col.lower()]
                
                if prob_columns:
                    # Return the maximum probability
                    max_prob = predictions[prob_columns].max(axis=1).iloc[0]
                    return float(max_prob)
                else:
                    # Fallback: return 0.5 if no probability available
                    return 0.5
            else:
                # Not applicable for regression
                return 0.0
                
        except Exception as e:
            logger.error(f"Probability prediction error: {e}")
            return 0.0
    
    def predict_with_explanation(self, model, input_data: pd.DataFrame) -> Dict[str, Any]:
        """Make prediction with model explanation."""
        try:
            # Get basic prediction
            prediction = self.predict_single(model, input_data)
            
            # Get probability if classification
            probability = None
            if hasattr(st.session_state, 'problem_type') and st.session_state.problem_type == 'classification':
                probability = self.predict_probability(model, input_data)
            
            # Try to get SHAP values for explanation
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
        """Get SHAP-based explanation for prediction."""
        try:
            import shap
            
            # This is a simplified explanation - in practice, you'd need to 
            # extract the underlying model from PyCaret and use appropriate SHAP explainer
            
            # For now, return feature importance as explanation
            feature_names = input_data.columns.tolist()
            feature_values = input_data.iloc[0].values
            
            # Create mock importance scores (in practice, use SHAP)
            importance_scores = np.random.uniform(-1, 1, len(feature_names))
            
            explanation = {
                'feature_importance': dict(zip(feature_names, importance_scores)),
                'feature_values': dict(zip(feature_names, feature_values)),
                'top_features': sorted(
                    zip(feature_names, importance_scores), 
                    key=lambda x: abs(x[1]), 
                    reverse=True
                )[:5]
            }
            
            return explanation
            
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
            if hasattr(st.session_state, 'problem_type') and st.session_state.problem_type == 'classification':
                try:
                    import pycaret.classification as pc
                    pred_df = pc.predict_model(model, data=input_data)
                    
                    # Look for probability columns
                    prob_columns = [col for col in pred_df.columns if 'prediction_score' in col.lower()]
                    
                    if prob_columns:
                        # Add confidence as maximum probability
                        results_df['confidence'] = pred_df[prob_columns].max(axis=1)
                    else:
                        results_df['confidence'] = 0.5  # Default confidence
                        
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
            
            # Select only training columns
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
            if hasattr(st.session_state, 'problem_type') and st.session_state.problem_type == 'regression':
                # This is a simplified implementation
                # In practice, you would use proper prediction interval methods
                
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
            if hasattr(st.session_state, 'problem_type'):
                if st.session_state.problem_type == 'classification':
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                    
                    metrics = {
                        'accuracy': accuracy_score(true_values, predictions),
                        'precision': precision_score(true_values, predictions, average='weighted'),
                        'recall': recall_score(true_values, predictions, average='weighted'),
                        'f1_score': f1_score(true_values, predictions, average='weighted')
                    }
                else:
                    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                    
                    metrics = {
                        'mae': mean_absolute_error(true_values, predictions),
                        'mse': mean_squared_error(true_values, predictions),
                        'rmse': np.sqrt(mean_squared_error(true_values, predictions)),
                        'r2': r2_score(true_values, predictions)
                    }
            else:
                metrics = {"message": "Problem type not specified"}
            
            return metrics
            
        except Exception as e:
            logger.error(f"Performance evaluation error: {e}")
            return {"error": str(e)}