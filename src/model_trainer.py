"""
Model Trainer Module
Handles PyCaret model training and management
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
import logging
import streamlit as st
import warnings
from pathlib import Path
import pickle
import joblib

# Suppress warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Handles model training using PyCaret."""
    
    def __init__(self, config: Dict[str, Any]):
        # Handle None config gracefully
        if config is None:
            config = {}
            
        self.config = config
        self.setup_complete = False
        self.problem_type = None
        self.pycaret_module = None
        self.experiment = None
        self.best_model = None
        
    def get_available_models(self, problem_type: str) -> Dict[str, str]:
        """Get available models for the problem type."""
        if problem_type == 'classification':
            return {
                'lr': 'Logistic Regression',
                'nb': 'Naive Bayes',
                'dt': 'Decision Tree Classifier',
                'rf': 'Random Forest Classifier',
                'et': 'Extra Trees Classifier',
                'ada': 'Ada Boost Classifier',
                'gbc': 'Gradient Boosting Classifier',
                'xgboost': 'Extreme Gradient Boosting',
                'lightgbm': 'Light Gradient Boosting Machine',
                'catboost': 'CatBoost Classifier',
                'svm': 'SVM - Linear Kernel',
                'rbfsvm': 'SVM - Radial Kernel',
                'knn': 'K Neighbors Classifier',
                'mlp': 'MLP Classifier',
                'ridge': 'Ridge Classifier',
                'qda': 'Quadratic Discriminant Analysis',
                'lda': 'Linear Discriminant Analysis'
            }
        else:  # regression
            return {
                'lr': 'Linear Regression',
                'ridge': 'Ridge Regression',
                'lasso': 'Lasso Regression',
                'en': 'Elastic Net',
                'dt': 'Decision Tree Regressor',
                'rf': 'Random Forest Regressor',
                'et': 'Extra Trees Regressor',
                'ada': 'AdaBoost Regressor',
                'gbr': 'Gradient Boosting Regressor',
                'xgboost': 'Extreme Gradient Boosting',
                'lightgbm': 'Light Gradient Boosting Machine',
                'catboost': 'CatBoost Regressor',
                'svm': 'SVM - Linear Kernel',
                'rbfsvm': 'SVM - Radial Kernel',
                'knn': 'K Neighbors Regressor',
                'mlp': 'MLP Regressor',
                'huber': 'Huber Regressor',
                'br': 'Bayesian Ridge',
                'par': 'Passive Aggressive Regressor'
            }
    
    def setup_environment(self, 
                         data: pd.DataFrame, 
                         target: str, 
                         problem_type: str,
                         preprocessing_config: Dict[str, Any],
                         test_size: float = 0.2,
                         cv_folds: int = 5,
                         random_state: int = 42) -> bool:
        """Setup PyCaret environment."""
        try:
            self.problem_type = problem_type
            
            # Import appropriate PyCaret module
            if problem_type == 'classification':
                import pycaret.classification as pc
                self.pycaret_module = pc
            else:
                import pycaret.regression as pr
                self.pycaret_module = pr
            
            # Data validation before setup
            logger.info(f"Setting up environment for {problem_type}")
            logger.info(f"Data shape: {data.shape}")
            logger.info(f"Target column: {target}")
            logger.info(f"Target in columns: {target in data.columns}")
            
            # Validate data
            if data.empty:
                raise ValueError("Dataset is empty")
            
            if target not in data.columns:
                raise ValueError(f"Target column '{target}' not found in data")
            
            if len(data.columns) < 2:
                raise ValueError("Dataset must have at least 2 columns")
            
            # Check for sufficient data
            if len(data) < 10:
                raise ValueError("Dataset must have at least 10 rows")
            
            # Clean data for PyCaret
            clean_data = data.copy()
            
            # Handle any remaining issues with column names
            clean_data.columns = [str(col).replace(' ', '_').replace('-', '_') for col in clean_data.columns]
            
            # Update target name if it was cleaned
            if target != clean_data.columns[data.columns.get_loc(target)]:
                target = clean_data.columns[data.columns.get_loc(target)]
            
            # Remove any completely null columns
            clean_data = clean_data.dropna(axis=1, how='all')
            
            # Ensure target column has valid values
            if clean_data[target].isnull().all():
                raise ValueError("Target column contains only null values")
            
            # For classification, ensure we have at least 2 classes
            if problem_type == 'classification':
                unique_targets = clean_data[target].nunique()
                if unique_targets < 2:
                    raise ValueError(f"Classification target must have at least 2 classes, found {unique_targets}")
            
            # Prepare basic setup parameters that work with PyCaret 3.x
            setup_params = {
                'data': clean_data,
                'target': target,
                'train_size': 1 - test_size,
                'fold': cv_folds,
                'session_id': random_state,
                'verbose': False,
                'use_gpu': False,
                'preprocess': True,
                'n_jobs': 1  # Use single job for stability
            }
            
            # Only add parameters that exist in PyCaret 3.x
            try:
                # Test which parameters are available
                import inspect
                setup_signature = inspect.signature(self.pycaret_module.setup)
                available_params = setup_signature.parameters.keys()
                
                # Add optional parameters if they exist
                optional_params = {
                    'html': False,
                    'log_experiment': False,
                    'experiment_logging': False,
                    'log_plots': False,
                    'log_data': False,
                    'system_log': False
                }
                
                for param, value in optional_params.items():
                    if param in available_params:
                        setup_params[param] = value
                        
            except Exception as e:
                logger.warning(f"Could not inspect setup parameters: {e}")
            
            # Add only essential preprocessing parameters to avoid conflicts
            try:
                self._configure_minimal_preprocessing(setup_params, preprocessing_config)
            except Exception as e:
                logger.warning(f"Preprocessing configuration warning: {e}")
                # Continue with basic setup if preprocessing config fails
            
            # Setup the environment with error handling
            logger.info("Starting PyCaret setup...")
            logger.info(f"Setup parameters: {list(setup_params.keys())}")
            
            self.experiment = self.pycaret_module.setup(**setup_params)
            self.setup_complete = True
            
            logger.info("PyCaret environment setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Setup error: {e}")
            self.setup_complete = False
            
            # Try ultra-minimal setup as fallback
            try:
                logger.info("Attempting ultra-minimal setup as fallback...")
                
                # Only use the most basic parameters
                minimal_params = {
                    'data': data,
                    'target': target,
                    'session_id': random_state,
                    'verbose': False
                }
                
                # Test if preprocess parameter exists
                try:
                    import inspect
                    setup_signature = inspect.signature(self.pycaret_module.setup)
                    if 'preprocess' in setup_signature.parameters:
                        minimal_params['preprocess'] = False
                except:
                    pass
                
                self.experiment = self.pycaret_module.setup(**minimal_params)
                self.setup_complete = True
                logger.info("Ultra-minimal setup successful")
                return True
                
            except Exception as fallback_error:
                logger.error(f"Ultra-minimal setup also failed: {fallback_error}")
                
                # Last resort: try with just data and target
                try:
                    logger.info("Attempting absolute minimal setup...")
                    absolute_minimal = {
                        'data': data,
                        'target': target
                    }
                    
                    self.experiment = self.pycaret_module.setup(**absolute_minimal)
                    self.setup_complete = True
                    logger.info("Absolute minimal setup successful")
                    return True
                    
                except Exception as final_error:
                    logger.error(f"All setup attempts failed. Final error: {final_error}")
                    self.setup_complete = False
                    return False
    
    def _configure_minimal_preprocessing(self, setup_params: Dict[str, Any], config: Dict[str, Any]):
        """Configure minimal preprocessing parameters to avoid conflicts."""
        try:
            # Only add the most basic and reliable preprocessing options
            
            # Missing value strategy - only if explicitly set and safe
            missing_strategy = config.get('missing_strategy')
            if missing_strategy and missing_strategy in ['mean', 'median', 'mode']:
                setup_params['imputation_type'] = missing_strategy
            
            # Normalization - only basic options
            scaling_method = config.get('scaling_method', 'none')
            if scaling_method in ['standard', 'minmax']:
                setup_params['normalize'] = True
                setup_params['normalize_method'] = 'zscore' if scaling_method == 'standard' else 'minmax'
            
            # Feature selection - only if explicitly requested
            if config.get('feature_selection') is True:
                setup_params['feature_selection'] = True
                setup_params['feature_selection_threshold'] = 0.8
            
            # Class imbalance - only for classification and if requested
            if (config.get('balance_data') is True and 
                self.problem_type == 'classification'):
                setup_params['fix_imbalance'] = True
                
        except Exception as e:
            logger.warning(f"Error in minimal preprocessing config: {e}")
            # Don't raise error, just continue without these options
    
    def _configure_preprocessing(self, setup_params: Dict[str, Any], config: Dict[str, Any]):
        """Configure preprocessing parameters."""
        try:
            # Missing value imputation
            if config.get('missing_strategy'):
                strategy = config['missing_strategy']
                if strategy != 'drop':
                    setup_params['imputation_type'] = strategy
                else:
                    setup_params['ignore_low_variance'] = False
            
            # Feature scaling/normalization
            scaling_method = config.get('scaling_method', 'none')
            if scaling_method != 'none':
                setup_params['normalize'] = True
                if scaling_method == 'standard':
                    setup_params['normalize_method'] = 'zscore'
                elif scaling_method == 'minmax':
                    setup_params['normalize_method'] = 'minmax'
                elif scaling_method == 'robust':
                    setup_params['normalize_method'] = 'robust'
            
            # Outlier removal
            if config.get('remove_outliers', False):
                setup_params['remove_outliers'] = True
                setup_params['outliers_threshold'] = 0.05
            
            # Feature selection
            if config.get('feature_selection', False):
                setup_params['feature_selection'] = True
                setup_params['feature_selection_threshold'] = 0.8
            
            # Class imbalance handling (classification only)
            if config.get('balance_data', False) and self.problem_type == 'classification':
                setup_params['fix_imbalance'] = True
                setup_params['fix_imbalance_method'] = 'SMOTE'
            
            # Feature engineering
            if config.get('feature_engineering', False):
                setup_params['feature_interaction'] = True
                setup_params['polynomial_features'] = True
                setup_params['trigonometry_features'] = True
                setup_params['group_features'] = True
                setup_params['bin_numeric_features'] = True
            
            # Categorical encoding
            encoding_method = config.get('categorical_encoding', 'onehot')
            if encoding_method == 'target':
                setup_params['categorical_features'] = None  # Let PyCaret handle automatically
            
        except Exception as e:
            logger.warning(f"Error configuring preprocessing: {e}")
    
    def compare_models(self, 
                      include: Optional[List[str]] = None,
                      exclude: Optional[List[str]] = None,
                      fold: Optional[int] = None,
                      sort: str = None,
                      n_select: int = 15) -> pd.DataFrame:
        """Compare multiple models and return results."""
        try:
            if not self.setup_complete:
                raise ValueError("Environment not setup. Call setup_environment first.")
            
            # Set default sort metric based on problem type
            if sort is None:
                sort = 'Accuracy' if self.problem_type == 'classification' else 'R2'
            
            # Prepare parameters for PyCaret 3.x
            compare_params = {}
            
            # Check which parameters are available in the compare_models function
            try:
                import inspect
                compare_signature = inspect.signature(self.pycaret_module.compare_models)
                available_params = compare_signature.parameters.keys()
                
                # Add parameters only if they exist
                if 'include' in available_params and include is not None:
                    compare_params['include'] = include
                if 'exclude' in available_params and exclude is not None:
                    compare_params['exclude'] = exclude
                if 'fold' in available_params and fold is not None:
                    compare_params['fold'] = fold
                if 'sort' in available_params:
                    compare_params['sort'] = sort
                if 'n_select' in available_params:
                    compare_params['n_select'] = n_select
                if 'verbose' in available_params:
                    compare_params['verbose'] = False
                if 'cross_validation' in available_params:
                    compare_params['cross_validation'] = True
                if 'round' in available_params:
                    compare_params['round'] = 4
                    
            except Exception as e:
                logger.warning(f"Could not inspect compare_models parameters: {e}")
                # Use minimal parameters
                compare_params = {'sort': sort}
            
            logger.info(f"Compare models parameters: {list(compare_params.keys())}")
            
            # Compare models
            comparison_results = self.pycaret_module.compare_models(**compare_params)
            
            # Get the comparison results dataframe
            comparison_df = self.pycaret_module.pull()
            
            # Store the best models
            if isinstance(comparison_results, list):
                self.best_model = comparison_results[0]
            else:
                self.best_model = comparison_results
            
            logger.info(f"Model comparison completed with {len(comparison_df)} models")
            return comparison_df
            
        except Exception as e:
            logger.error(f"Model comparison error: {e}")
            raise
    
    def create_model(self, 
                    model_name: str, 
                    cross_validation: bool = True,
                    fold: int = 10,
                    **kwargs) -> Any:
        """Create and train a single model."""
        try:
            if not self.setup_complete:
                raise ValueError("Environment not setup. Call setup_environment first.")
            
            # Check which parameters are available in create_model
            create_params = {'estimator': model_name}
            
            try:
                import inspect
                create_signature = inspect.signature(self.pycaret_module.create_model)
                available_params = create_signature.parameters.keys()
                
                # Add parameters only if they exist
                if 'cross_validation' in available_params:
                    create_params['cross_validation'] = cross_validation
                if 'fold' in available_params:
                    create_params['fold'] = fold
                if 'verbose' in available_params:
                    create_params['verbose'] = False
                    
                # Add any additional kwargs that are valid
                for key, value in kwargs.items():
                    if key in available_params:
                        create_params[key] = value
                        
            except Exception as e:
                logger.warning(f"Could not inspect create_model parameters: {e}")
                # Use just the model name
                create_params = {'estimator': model_name}
            
            # In PyCaret 3.x, the first parameter might be 'estimator' instead of positional
            try:
                model = self.pycaret_module.create_model(**create_params)
            except TypeError as te:
                # Try with positional argument
                logger.warning(f"Named parameter failed, trying positional: {te}")
                model = self.pycaret_module.create_model(model_name)
            
            logger.info(f"Model {model_name} created successfully")
            return model
            
        except Exception as e:
            logger.error(f"Model creation error: {e}")
            raise
    
    def tune_hyperparameters(self, 
                           model, 
                           optimization: str = None,
                           n_iter: int = 10,
                           search_library: str = 'scikit-learn',
                           search_algorithm: str = 'random') -> Any:
        """Tune model hyperparameters."""
        try:
            if not self.setup_complete:
                raise ValueError("Environment not setup. Call setup_environment first.")
            
            # Set default optimization metric
            if optimization is None:
                optimization = 'Accuracy' if self.problem_type == 'classification' else 'R2'
            
            tuned_model = self.pycaret_module.tune_model(
                model,
                optimize=optimization,
                n_iter=n_iter,
                search_library=search_library,
                search_algorithm=search_algorithm,
                verbose=False,
                fold=5
            )
            
            logger.info("Hyperparameter tuning completed successfully")
            return tuned_model
            
        except Exception as e:
            logger.error(f"Hyperparameter tuning error: {e}")
            raise
    
    def ensemble_models(self, 
                       model_or_models, 
                       method: str = 'Bagging',
                       fold: int = 10) -> Any:
        """Create ensemble of models."""
        try:
            if not self.setup_complete:
                raise ValueError("Environment not setup. Call setup_environment first.")
            
            if method.lower() in ['bagging', 'boosting']:
                # Single model ensemble
                ensemble = self.pycaret_module.ensemble_model(
                    model_or_models, 
                    method=method,
                    fold=fold,
                    verbose=False
                )
            else:
                # Multiple model ensemble (blend/stack)
                if isinstance(model_or_models, list):
                    if method.lower() == 'voting' or method.lower() == 'blend':
                        ensemble = self.pycaret_module.blend_models(
                            model_or_models,
                            fold=fold,
                            verbose=False
                        )
                    elif method.lower() == 'stacking':
                        ensemble = self.pycaret_module.stack_models(
                            model_or_models,
                            fold=fold,
                            verbose=False
                        )
                    else:
                        raise ValueError(f"Unsupported ensemble method: {method}")
                else:
                    raise ValueError("Multiple models required for voting/stacking ensemble")
            
            logger.info(f"Ensemble model created using {method}")
            return ensemble
            
        except Exception as e:
            logger.error(f"Ensemble creation error: {e}")
            raise
    
    def evaluate_model(self, model, fold: int = 10) -> Dict[str, float]:
        """Evaluate model performance."""
        try:
            if not self.setup_complete:
                raise ValueError("Environment not setup. Call setup_environment first.")
            
            # Evaluate model
            self.pycaret_module.evaluate_model(model, fold=fold)
            
            # Get metrics from the evaluation
            metrics_df = self.pycaret_module.pull()
            
            # Convert to dictionary based on problem type
            if self.problem_type == 'classification':
                metrics = {
                    'Accuracy': float(metrics_df.loc[0, 'Accuracy']),
                    'AUC': float(metrics_df.loc[0, 'AUC']),
                    'Recall': float(metrics_df.loc[0, 'Recall']),
                    'Precision': float(metrics_df.loc[0, 'Prec.']),
                    'F1': float(metrics_df.loc[0, 'F1']),
                    'Kappa': float(metrics_df.loc[0, 'Kappa'])
                }
            else:
                metrics = {
                    'MAE': float(metrics_df.loc[0, 'MAE']),
                    'MSE': float(metrics_df.loc[0, 'MSE']),
                    'RMSE': float(metrics_df.loc[0, 'RMSE']),
                    'R2': float(metrics_df.loc[0, 'R2']),
                    'RMSLE': float(metrics_df.loc[0, 'RMSLE']),
                    'MAPE': float(metrics_df.loc[0, 'MAPE'])
                }
            
            logger.info("Model evaluation completed")
            return metrics
            
        except Exception as e:
            logger.error(f"Model evaluation error: {e}")
            return {}
    
    def plot_model(self, model, plot_type: str = 'auc', save: bool = False):
        """Generate model plots."""
        try:
            if not self.setup_complete:
                raise ValueError("Environment not setup. Call setup_environment first.")
            
            # Generate plot
            plot = self.pycaret_module.plot_model(
                model, 
                plot=plot_type, 
                save=save,
                verbose=False,
                display_format='streamlit'
            )
            
            return plot
            
        except Exception as e:
            logger.error(f"Plot generation error: {e}")
            return None
    
    def interpret_model(self, model, plot: str = 'summary', observation: int = None):
        """Generate model interpretation plots using SHAP."""
        try:
            if not self.setup_complete:
                raise ValueError("Environment not setup. Call setup_environment first.")
            
            # Generate interpretation
            if observation is not None:
                interpretation = self.pycaret_module.interpret_model(
                    model,
                    plot=plot,
                    observation=observation,
                    save=False,
                    verbose=False
                )
            else:
                interpretation = self.pycaret_module.interpret_model(
                    model,
                    plot=plot,
                    save=False,
                    verbose=False
                )
            
            return interpretation
            
        except Exception as e:
            logger.error(f"Model interpretation error: {e}")
            return None
    
    def finalize_model(self, model):
        """Finalize model for deployment."""
        try:
            if not self.setup_complete:
                raise ValueError("Environment not setup. Call setup_environment first.")
            
            final_model = self.pycaret_module.finalize_model(model)
            
            logger.info("Model finalized for deployment")
            return final_model
            
        except Exception as e:
            logger.error(f"Model finalization error: {e}")
            raise
    
    def predict_model(self, model, data: Optional[pd.DataFrame] = None):
        """Make predictions using trained model."""
        try:
            if not self.setup_complete:
                raise ValueError("Environment not setup. Call setup_environment first.")
            
            predictions = self.pycaret_module.predict_model(model, data=data)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    def save_model(self, model, model_name: str) -> str:
        """Save trained model to disk."""
        try:
            # Create models directory if it doesn't exist
            models_config = self.config.get('models', {})
            models_dir = Path(models_config.get('save_path', 'models/'))
            models_dir.mkdir(exist_ok=True)
            
            # Save using PyCaret's save function
            model_path = models_dir / model_name
            
            if self.setup_complete:
                # Use PyCaret's built-in save function
                self.pycaret_module.save_model(model, str(model_path))
                saved_path = f"{model_path}.pkl"
            else:
                # Fallback to joblib
                saved_path = f"{model_path}.pkl"
                joblib.dump(model, saved_path)
            
            logger.info(f"Model saved to {saved_path}")
            return saved_path
            
        except Exception as e:
            logger.error(f"Model saving error: {e}")
            raise
    
    def load_model(self, model_path: str):
        """Load saved model from disk."""
        try:
            if self.setup_complete:
                # Use PyCaret's load function
                model = self.pycaret_module.load_model(model_path.replace('.pkl', ''))
            else:
                # Fallback to joblib
                model = joblib.load(model_path)
            
            logger.info(f"Model loaded from {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Model loading error: {e}")
            raise
    
    def get_model_performance(self, model) -> pd.DataFrame:
        """Get detailed model performance metrics."""
        try:
            if not self.setup_complete:
                raise ValueError("Environment not setup. Call setup_environment first.")
            
            # Pull the latest results after model evaluation
            results = self.pycaret_module.pull()
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting model performance: {e}")
            return pd.DataFrame()
    
    def get_leaderboard(self) -> pd.DataFrame:
        """Get model leaderboard from comparison."""
        try:
            if not self.setup_complete:
                raise ValueError("Environment not setup. Call setup_environment first.")
            
            # Get the latest comparison results
            leaderboard = self.pycaret_module.pull()
            
            return leaderboard
            
        except Exception as e:
            logger.error(f"Error getting leaderboard: {e}")
            return pd.DataFrame()
    
    def create_custom_model(self, 
                          estimator,
                          name: str,
                          **kwargs) -> Any:
        """Create custom model from scikit-learn estimator."""
        try:
            if not self.setup_complete:
                raise ValueError("Environment not setup. Call setup_environment first.")
            
            model = self.pycaret_module.create_model(
                estimator,
                verbose=False,
                **kwargs
            )
            
            logger.info(f"Custom model {name} created successfully")
            return model
            
        except Exception as e:
            logger.error(f"Custom model creation error: {e}")
            raise
    
    def check_metric(self, model, metric: str = None) -> float:
        """Check specific metric for a model."""
        try:
            if not self.setup_complete:
                raise ValueError("Environment not setup. Call setup_environment first.")
            
            # Get model metrics
            metrics = self.evaluate_model(model)
            
            if metric is None:
                metric = 'Accuracy' if self.problem_type == 'classification' else 'R2'
            
            return metrics.get(metric, 0.0)
            
        except Exception as e:
            logger.error(f"Error checking metric: {e}")
            return 0.0
    
    def deploy_model(self, 
                    model,
                    model_name: str,
                    platform: str = 'aws',
                    **kwargs):
        """Deploy model to cloud platform."""
        try:
            if not self.setup_complete:
                raise ValueError("Environment not setup. Call setup_environment first.")
            
            # Deploy model using PyCaret
            deployment = self.pycaret_module.deploy_model(
                model,
                model_name=model_name,
                platform=platform,
                **kwargs
            )
            
            logger.info(f"Model deployed to {platform}")
            return deployment
            
        except Exception as e:
            logger.error(f"Deployment error: {e}")
            raise
    
    def automl(self, 
              data: pd.DataFrame,
              target: str,
              problem_type: str,
              time_budget: int = 300,
              optimization_metric: str = None) -> Dict[str, Any]:
        """Automated machine learning pipeline."""
        try:
            # Setup environment
            setup_success = self.setup_environment(
                data=data,
                target=target,
                problem_type=problem_type,
                preprocessing_config={}
            )
            
            if not setup_success:
                raise ValueError("Failed to setup environment for AutoML")
            
            # Compare models
            comparison_df = self.compare_models()
            
            # Get best model
            best_model_name = comparison_df.index[0]
            best_model = self.create_model(best_model_name)
            
            # Tune hyperparameters
            tuned_model = self.tune_hyperparameters(
                best_model, 
                optimization=optimization_metric,
                n_iter=10
            )
            
            # Evaluate final model
            final_metrics = self.evaluate_model(tuned_model)
            
            # Finalize model
            final_model = self.finalize_model(tuned_model)
            
            result = {
                'best_model_name': best_model_name,
                'model': final_model,
                'metrics': final_metrics,
                'comparison_results': comparison_df,
                'tuned': True
            }
            
            logger.info("AutoML pipeline completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"AutoML error: {e}")
            raise
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return {
            'setup_complete': self.setup_complete,
            'problem_type': self.problem_type,
            'config': self.config
        }
    
    def reset_environment(self):
        """Reset the PyCaret environment."""
        try:
            self.setup_complete = False
            self.problem_type = None
            self.pycaret_module = None
            self.experiment = None
            self.best_model = None
            
            logger.info("Environment reset successfully")
            
        except Exception as e:
            logger.error(f"Error resetting environment: {e}")
            
    def get_available_plots(self) -> List[str]:
        """Get available plot types for current problem type."""
        if self.problem_type == 'classification':
            return [
                'auc', 'threshold', 'pr', 'confusion_matrix', 'error',
                'class_report', 'boundary', 'roc', 'lift', 'calibration',
                'dimension', 'manifold', 'feature', 'feature_all', 'parameter'
            ]
        else:
            return [
                'residuals', 'cooks', 'rfe', 'learning', 'validation',
                'manifold', 'feature', 'feature_all', 'parameter', 'tree'
            ]