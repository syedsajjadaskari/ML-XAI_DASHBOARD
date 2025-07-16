"""
Hybrid Fast ML Trainer
Combines multiple fast training methods for optimal speed and accuracy
Automatically selects the best approach based on data size and requirements
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
import logging
import time
import warnings
warnings.filterwarnings('ignore')

# Import our fast trainers
from .fast_model_trainer import FastModelTrainer
try:
    from .flaml_trainer import FLAMLTrainer
    FLAML_AVAILABLE = True
except (ImportError, Exception):
    FLAML_AVAILABLE = False

# Additional fast libraries
try:
    import h2o
    from h2o.automl import H2OAutoML
    H2O_AVAILABLE = True
except ImportError:
    H2O_AVAILABLE = False

logger = logging.getLogger(__name__)

class HybridFastTrainer:
    """
    Hybrid trainer that automatically selects the fastest approach:
    - Small datasets (< 1000 rows): FastModelTrainer
    - Medium datasets (1000-10000 rows): FLAML 
    - Large datasets (> 10000 rows): H2O AutoML
    """
    
    def __init__(self, config: Dict[str, Any]):
        if config is None:
            config = {}
        self.config = config
        self.trainer = None
        self.trainer_type = None
        self.setup_complete = False
        
    def setup_environment(self, 
                         data: pd.DataFrame, 
                         target: str, 
                         problem_type: str,
                         preprocessing_config: Dict[str, Any],
                         test_size: float = 0.2,
                         random_state: int = 42,
                         preferred_method: str = 'auto') -> bool:
        """Setup the optimal trainer based on data size and availability."""
        
        try:
            data_size = len(data)
            logger.info(f"Dataset size: {data_size} rows, {len(data.columns)} columns")
            
            # Auto-select best method
            if preferred_method == 'auto':
                method = self._select_optimal_method(data_size)
            else:
                method = preferred_method
            
            logger.info(f"Selected training method: {method}")
            
            # Initialize appropriate trainer
            if method == 'fast_sklearn':
                self.trainer = FastModelTrainer(self.config)
                self.trainer_type = 'fast_sklearn'
                
            elif method == 'flaml' and FLAML_AVAILABLE:
                self.trainer = FLAMLTrainer(self.config)
                self.trainer_type = 'flaml'
                
            elif method == 'h2o' and H2O_AVAILABLE:
                self.trainer = H2OTrainer(self.config)
                self.trainer_type = 'h2o'
                
            else:
                # Fallback to fast sklearn
                logger.warning(f"Method {method} not available, falling back to fast sklearn")
                self.trainer = FastModelTrainer(self.config)
                self.trainer_type = 'fast_sklearn'
            
            # Setup the selected trainer
            success = self.trainer.setup_environment(
                data, target, problem_type, preprocessing_config, test_size, random_state
            )
            
            if success:
                self.setup_complete = True
                logger.info(f"Hybrid trainer setup completed using {self.trainer_type}")
                
            return success
            
        except Exception as e:
            logger.error(f"Hybrid trainer setup error: {e}")
            return False
    
    def _select_optimal_method(self, data_size: int) -> str:
        """Select optimal training method based on data size."""
        
        # Small datasets: Use fast sklearn (fastest setup)
        if data_size < 1000:
            return 'fast_sklearn'
        
        # Medium datasets: Use FLAML if available (best balance)
        elif data_size < 10000:
            if FLAML_AVAILABLE:
                return 'flaml'
            else:
                return 'fast_sklearn'
        
        # Large datasets: Use H2O if available (scales best)
        else:
            if H2O_AVAILABLE:
                return 'h2o'
            elif FLAML_AVAILABLE:
                return 'flaml'
            else:
                return 'fast_sklearn'
    
    def train_lightning_fast(self, time_budget: int = 30) -> Dict[str, Any]:
        """Lightning-fast training with time budget."""
        if not self.setup_complete:
            raise ValueError("Environment not setup")
        
        start_time = time.time()
        logger.info(f"Starting lightning-fast training with {time_budget}s budget")
        
        try:
            if self.trainer_type == 'flaml':
                # FLAML with time budget
                results = self.trainer.train_ultra_fast(time_budget)
                
            elif self.trainer_type == 'h2o':
                # H2O AutoML with time budget
                results = self.trainer.train_automl(max_runtime_secs=time_budget)
                
            else:
                # Fast sklearn with timeout
                comparison_results = self.trainer.compare_models_fast(
                    cv_folds=3, timeout=max(10, time_budget//2)
                )
                
                if len(comparison_results) > 0:
                    best_model = self.trainer.train_best_model(comparison_results)
                    results = {
                        'best_model': best_model,
                        'comparison_results': comparison_results,
                        'trainer_type': 'fast_sklearn'
                    }
                else:
                    # Fallback to single model
                    best_model = self.trainer.train_single_model('rf_fast')
                    results = {
                        'best_model': best_model,
                        'trainer_type': 'fast_sklearn'
                    }
            
            total_time = time.time() - start_time
            results['total_training_time'] = total_time
            results['trainer_type'] = self.trainer_type
            
            logger.info(f"Lightning-fast training completed in {total_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Lightning-fast training error: {e}")
            raise
    
    def compare_models(self, time_budget: int = 60) -> pd.DataFrame:
        """Compare models with time budget."""
        if self.trainer_type == 'fast_sklearn':
            return self.trainer.compare_models_fast(timeout=time_budget)
        else:
            # For FLAML/H2O, run training and return pseudo-comparison
            results = self.train_lightning_fast(time_budget)
            
            # Create comparison-like results
            if self.trainer_type == 'flaml':
                comparison_data = [{
                    'Model': results.get('best_estimator', 'AutoML'),
                    'Score': -results.get('best_loss', 0),
                    'Time': results.get('training_time', 0),
                    'Method': 'FLAML'
                }]
            else:
                comparison_data = [{
                    'Model': 'H2O_AutoML',
                    'Score': 0.0,  # Would need actual metrics
                    'Time': results.get('training_time', 0),
                    'Method': 'H2O'
                }]
            
            return pd.DataFrame(comparison_data)
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model."""
        return self.trainer.predict(data)
    
    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        return self.trainer.predict_proba(data)
    
    def evaluate_model(self) -> Dict[str, float]:
        """Evaluate the trained model."""
        return self.trainer.evaluate_model()
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance."""
        if hasattr(self.trainer, 'get_feature_importance'):
            return self.trainer.get_feature_importance()
        elif hasattr(self.trainer, '_get_feature_importance'):
            return self.trainer._get_feature_importance()
        else:
            return {}
    
    def get_available_methods(self) -> Dict[str, str]:
        """Get available training methods."""
        methods = {
            'fast_sklearn': 'Fast Scikit-learn (Always Available)',
        }
        
        if FLAML_AVAILABLE:
            methods['flaml'] = 'FLAML AutoML (Microsoft - Ultra Fast)'
        
        if H2O_AVAILABLE:
            methods['h2o'] = 'H2O AutoML (Scales to Large Data)'
        
        return methods
    
    def get_trainer_info(self) -> Dict[str, Any]:
        """Get information about the current trainer."""
        return {
            'trainer_type': self.trainer_type,
            'setup_complete': self.setup_complete,
            'available_methods': self.get_available_methods(),
            'trainer_class': str(type(self.trainer).__name__) if self.trainer else None
        }

class H2OTrainer:
    """H2O AutoML trainer for large datasets."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.h2o_initialized = False
        self.automl = None
        self.train_h2o = None
        self.test_h2o = None
        
    def setup_environment(self, data, target, problem_type, preprocessing_config, test_size, random_state):
        """Setup H2O environment."""
        try:
            # Initialize H2O
            h2o.init(nthreads=-1, max_mem_size="4G")
            self.h2o_initialized = True
            
            # Convert to H2O frame
            h2o_frame = h2o.H2OFrame(data)
            
            # Set target as factor for classification
            if problem_type == 'classification':
                h2o_frame[target] = h2o_frame[target].asfactor()
            
            # Split data
            train, test = h2o_frame.split_frame(ratios=[1-test_size], seed=random_state)
            self.train_h2o = train
            self.test_h2o = test
            self.target = target
            self.problem_type = problem_type
            
            return True
            
        except Exception as e:
            logger.error(f"H2O setup error: {e}")
            return False
    
    def train_automl(self, max_runtime_secs=30):
        """Train H2O AutoML."""
        try:
            self.automl = H2OAutoML(
                max_runtime_secs=max_runtime_secs,
                seed=42,
                verbosity="warn"
            )
            
            x = self.train_h2o.columns
            x.remove(self.target)
            
            self.automl.train(x=x, y=self.target, training_frame=self.train_h2o)
            
            return {
                'best_model': self.automl.leader,
                'leaderboard': self.automl.leaderboard.as_data_frame(),
                'training_time': max_runtime_secs
            }
            
        except Exception as e:
            logger.error(f"H2O training error: {e}")
            raise
    
    def predict(self, data):
        """Make predictions with H2O model."""
        h2o_data = h2o.H2OFrame(data)
        predictions = self.automl.predict(h2o_data)
        return predictions.as_data_frame().values.flatten()
    
    def predict_proba(self, data):
        """Get prediction probabilities with H2O."""
        h2o_data = h2o.H2OFrame(data)
        predictions = self.automl.predict(h2o_data)
        # Extract probability columns
        prob_cols = [col for col in predictions.columns if col.startswith('p')]
        return predictions[prob_cols].as_data_frame().values
    
    def evaluate_model(self):
        """Evaluate H2O model."""
        perf = self.automl.leader.model_performance(self.test_h2o)
        
        if self.problem_type == 'classification':
            return {
                'auc': perf.auc()[0][0],
                'accuracy': perf.accuracy()[0][0]
            }
        else:
            return {
                'rmse': perf.rmse(),
                'mae': perf.mae(),
                'r2': perf.r2()
            }

# Installation requirements for each method
INSTALLATION_GUIDE = {
    'flaml': 'pip install flaml[automl]',
    'h2o': 'pip install h2o',
    'lightgbm': 'pip install lightgbm',
    'xgboost': 'pip install xgboost',
    'catboost': 'pip install catboost'
}