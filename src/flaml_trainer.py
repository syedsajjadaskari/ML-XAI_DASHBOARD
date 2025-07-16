"""
FLAML AutoML Trainer
Ultra-fast AutoML using Microsoft FLAML
Fastest possible ML training with excellent results
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
    """Ultra-fast AutoML trainer using Microsoft FLAML."""
    
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
        """Setup FLAML environment."""
        try:
            start_time = time.time()
            logger.info("Setting up FLAML AutoML environment...")
            
            self.problem_type = problem_type
            
            # Separate features and target
            X = data.drop(columns=[target])
            y = data[target]
            
            # Store feature names
            self.feature_names = X.columns.tolist()
            
            # Handle categorical features automatically
            categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Simple preprocessing for FLAML
            for col in categorical_features:
                if X[col].nunique() > 50:  # High cardinality
                    # Keep top 20 categories
                    top_cats = X[col].value_counts().head(20).index
                    X[col] = X[col].apply(lambda x: x if x in top_cats else 'Other')
            
            # Train-test split
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state,
                stratify=y if problem_type == 'classification' else None
            )
            
            # Encode target for classification if needed
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
        """Ultra-fast AutoML training with time budget."""
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
            # FLAML AutoML training
            settings = {
                'time_budget': time_budget,  # seconds
                'metric': metric,
                'task': task,
                'log_file_name': 'flaml.log',
                'seed': 42,
                'verbose': 0,
                'early_stop': True,
                'retrain_full': True,
                'split_ratio': 0.8,
                'n_splits': 3,
                'eval_method': 'cv'
            }
            
            self.automl.fit(
                X_train=self.X_train,
                y_train=self.y_train,
                **settings
            )
            
            training_time = time.time() - start_time
            
            # Get results
            results = {
                'best_model': self.automl.model,
                'best_estimator': self.automl.best_estimator,
                'best_config': self.automl.best_config,
                'best_loss': self.automl.best_loss,
                'training_time': training_time,
                'feature_importance': self._get_feature_importance()
            }
            
            logger.info(f"FLAML training completed in {training_time:.2f}s")
            logger.info(f"Best model: {self.automl.best_estimator}")
            logger.info(f"Best {metric}: {-self.automl.best_loss:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"FLAML training error: {e}")
            raise
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make predictions using trained FLAML model."""
        try:
            predictions = self.automl.predict(data)
            
            # Decode predictions if classification with label encoding
            if (self.problem_type == 'classification' and 
                self.label_encoder is not None):
                predictions = self.label_encoder.inverse_transform(predictions)
            
            return predictions
            
        except Exception as e:
            logger.error(f"FLAML prediction error: {e}")
            raise
    
    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        if self.problem_type != 'classification':
            raise ValueError("Probabilities only available for classification")
        
        try:
            probabilities = self.automl.predict_proba(data)
            return probabilities
            
        except Exception as e:
            logger.error(f"FLAML probability prediction error: {e}")
            raise
    
    def evaluate_model(self) -> Dict[str, float]:
        """Evaluate the trained model."""
        try:
            predictions = self.predict(self.X_test)
            
            if self.problem_type == 'classification':
                # Convert back for evaluation if needed
                y_test_eval = self.y_test
                if self.label_encoder is not None:
                    y_test_eval = self.label_encoder.inverse_transform(self.y_test)
                    
                metrics = {
                    'accuracy': accuracy_score(y_test_eval, predictions),
                    'f1_score': f1_score(y_test_eval, predictions, average='weighted')
                }
            else:
                metrics = {
                    'r2_score': r2_score(self.y_test, predictions),
                    'mae': mean_absolute_error(self.y_test, predictions)
                }
            
            return metrics
            
        except Exception as e:
            logger.error(f"FLAML evaluation error: {e}")
            return {}
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from FLAML model."""
        try:
            if hasattr(self.automl.model, 'feature_importances_'):
                importance = self.automl.model.feature_importances_
                return dict(zip(self.feature_names, importance))
            elif hasattr(self.automl.model, 'coef_'):
                importance = np.abs(self.automl.model.coef_).flatten()
                return dict(zip(self.feature_names, importance))
            else:
                return {}
                
        except Exception as e:
            logger.warning(f"Could not get feature importance: {e}")
            return {}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed information about the best model."""
        if not self.automl:
            return {}
        
        return {
            'best_estimator': str(self.automl.best_estimator),
            'best_config': self.automl.best_config,
            'best_loss': self.automl.best_loss,
            'classes_': getattr(self.automl.model, 'classes_', None),
            'feature_names': self.feature_names
        }

# Updated requirements.txt for FLAML
FLAML_REQUIREMENTS = """
# Ultra-Fast AutoML
flaml>=1.2.4
lightgbm>=3.3.0
xgboost>=1.6.0
catboost>=1.1.0

# Optional: Even faster with GPU
# lightgbm[gpu]>=3.3.0
# xgboost[gpu]>=1.6.0
"""