"""
Fast Model Trainer - Updated with proper test data storage
Lightning-fast ML training using scikit-learn with optimized pipelines
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging
import time
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Fast ML Models
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Optional: Fast libraries (install if available)
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

logger = logging.getLogger(__name__)

class FastModelTrainer:
    """Lightning-fast ML model trainer using scikit-learn."""
    
    def __init__(self, config: Dict[str, Any]):
        if config is None:
            config = {}
        self.config = config
        self.setup_complete = False
        self.problem_type = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_processed = None
        self.X_test_processed = None
        self.preprocessor = None
        self.trained_models = {}
        self.best_model = None
        self.feature_names = None
        self.original_feature_names = None
        self.label_encoder = None
        
        # Store evaluation data for later use
        self.evaluation_data = None
        
    def setup_environment(self, 
                         data: pd.DataFrame, 
                         target: str, 
                         problem_type: str,
                         preprocessing_config: Dict[str, Any],
                         test_size: float = 0.2,
                         random_state: int = 42) -> bool:
        """Setup fast training environment with proper data storage."""
        try:
            start_time = time.time()
            logger.info("Setting up Fast ML environment...")
            
            self.problem_type = problem_type
            self.target_column = target
            
            # Separate features and target
            X = data.drop(columns=[target])
            y = data[target]
            
            # Store original feature names
            self.original_feature_names = X.columns.tolist()
            self.feature_names = X.columns.tolist()
            
            # Train-test split
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, 
                stratify=y if problem_type == 'classification' else None
            )
            
            # Create fast preprocessor
            self.preprocessor = self._create_fast_preprocessor(X, preprocessing_config)
            
            # Apply preprocessing
            self.X_train_processed = self.preprocessor.fit_transform(self.X_train)
            self.X_test_processed = self.preprocessor.transform(self.X_test)
            
            # Store original target values for evaluation
            self.y_train_original = self.y_train.copy()
            self.y_test_original = self.y_test.copy()
            
            # Encode target for classification
            if problem_type == 'classification' and y.dtype == 'object':
                self.label_encoder = LabelEncoder()
                self.y_train = self.label_encoder.fit_transform(self.y_train)
                self.y_test = self.label_encoder.transform(self.y_test)
            
            # Store evaluation data structure
            self.evaluation_data = {
                'X_test_original': self.X_test.copy(),
                'X_test_processed': self.X_test_processed.copy(),
                'y_test_original': self.y_test_original.copy(),
                'y_test_encoded': self.y_test.copy(),
                'X_train_processed': self.X_train_processed.copy(),
                'y_train_encoded': self.y_train.copy(),
                'feature_names': self.original_feature_names,
                'preprocessor': self.preprocessor,
                'label_encoder': self.label_encoder,
                'problem_type': self.problem_type,
                'target_column': self.target_column
            }
            
            self.setup_complete = True
            setup_time = time.time() - start_time
            logger.info(f"Fast ML setup completed in {setup_time:.2f} seconds")
            return True
            
        except Exception as e:
            logger.error(f"Fast setup error: {e}")
            return False
    
    def _create_fast_preprocessor(self, X: pd.DataFrame, config: Dict[str, Any]):
        """Create optimized preprocessing pipeline."""
        # Identify column types
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Numeric pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical pipeline
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        return preprocessor
    
    def get_fast_models(self) -> Dict[str, Any]:
        """Get optimized fast models."""
        if self.problem_type == 'classification':
            models = {
                'rf_fast': RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),
                'lr': LogisticRegression(random_state=42, max_iter=1000),
                'nb': GaussianNB(),
                'knn_fast': KNeighborsClassifier(n_neighbors=5),
                'dt_fast': DecisionTreeClassifier(max_depth=10, random_state=42),
                'svm_fast': SVC(kernel='rbf', random_state=42, probability=True),
                'gb_fast': GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, random_state=42)
            }
            
            # Add fast gradient boosting if available
            if LIGHTGBM_AVAILABLE:
                models['lgb'] = LGBMClassifier(n_estimators=50, random_state=42, verbose=-1)
            if XGBOOST_AVAILABLE:
                models['xgb'] = XGBClassifier(n_estimators=50, random_state=42, eval_metric='logloss')
                
        else:  # regression
            models = {
                'rf_fast': RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),
                'lr': LinearRegression(),
                'ridge': Ridge(random_state=42),
                'lasso': Lasso(random_state=42),
                'knn_fast': KNeighborsRegressor(n_neighbors=5),
                'dt_fast': DecisionTreeRegressor(max_depth=10, random_state=42),
                'svm_fast': SVR(kernel='rbf'),
                'gb_fast': GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, random_state=42)
            }
            
            # Add fast gradient boosting if available
            if LIGHTGBM_AVAILABLE:
                models['lgb'] = LGBMRegressor(n_estimators=50, random_state=42, verbose=-1)
            if XGBOOST_AVAILABLE:
                models['xgb'] = XGBRegressor(n_estimators=50, random_state=42, eval_metric='rmse')
        
        return models
    
    def compare_models_fast(self, cv_folds: int = 3, timeout: int = 60) -> pd.DataFrame:
        """Ultra-fast model comparison with timeout."""
        if not self.setup_complete:
            raise ValueError("Environment not setup. Call setup_environment first.")
        
        start_time = time.time()
        models = self.get_fast_models()
        results = []
        
        logger.info(f"Comparing {len(models)} models with {cv_folds}-fold CV...")
        
        for name, model in models.items():
            # Check timeout
            if time.time() - start_time > timeout:
                logger.warning(f"Timeout reached at {timeout}s, stopping comparison")
                break
                
            try:
                model_start = time.time()
                
                # Fast cross-validation
                if self.problem_type == 'classification':
                    scores = cross_val_score(model, self.X_train_processed, self.y_train, 
                                           cv=cv_folds, scoring='accuracy', n_jobs=-1)
                    primary_metric = scores.mean()
                    std_metric = scores.std()
                else:
                    scores = cross_val_score(model, self.X_train_processed, self.y_train, 
                                           cv=cv_folds, scoring='r2', n_jobs=-1)
                    primary_metric = scores.mean()
                    std_metric = scores.std()
                
                model_time = time.time() - model_start
                
                results.append({
                    'Model': name,
                    'Score': primary_metric,
                    'Std': std_metric,
                    'Time (s)': model_time
                })
                
                logger.info(f"{name}: {primary_metric:.4f} (±{std_metric:.4f}) in {model_time:.2f}s")
                
            except Exception as e:
                logger.warning(f"Error with {name}: {e}")
                continue
        
        # Create results dataframe
        results_df = pd.DataFrame(results)
        if len(results_df) > 0:
            results_df = results_df.sort_values('Score', ascending=False).reset_index(drop=True)
        
        total_time = time.time() - start_time
        logger.info(f"Model comparison completed in {total_time:.2f} seconds")
        
        return results_df
    
    def train_best_model(self, comparison_results: pd.DataFrame) -> Any:
        """Train the best performing model."""
        if len(comparison_results) == 0:
            raise ValueError("No models to train")
        
        best_model_name = comparison_results.iloc[0]['Model']
        models = self.get_fast_models()
        best_model = models[best_model_name]
        
        start_time = time.time()
        logger.info(f"Training best model: {best_model_name}")
        
        # Train on full training set
        best_model.fit(self.X_train_processed, self.y_train)
        
        training_time = time.time() - start_time
        logger.info(f"Best model trained in {training_time:.2f} seconds")
        
        self.best_model = best_model
        self.trained_models[best_model_name] = best_model
        
        return best_model
    
    def train_single_model(self, model_name: str) -> Any:
        """Train a single specific model."""
        if not self.setup_complete:
            raise ValueError("Environment not setup. Call setup_environment first.")
        
        models = self.get_fast_models()
        if model_name not in models:
            raise ValueError(f"Model {model_name} not available")
        
        model = models[model_name]
        
        start_time = time.time()
        logger.info(f"Training {model_name}...")
        
        model.fit(self.X_train_processed, self.y_train)
        
        training_time = time.time() - start_time
        logger.info(f"{model_name} trained in {training_time:.2f} seconds")
        
        self.trained_models[model_name] = model
        return model
    
    def get_evaluation_predictions(self, model) -> Dict[str, Any]:
        """Get predictions and probabilities for evaluation."""
        try:
            # Get predictions
            y_pred = model.predict(self.X_test_processed)
            
            # Get probabilities if available
            y_proba = None
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(self.X_test_processed)
            
            # Decode predictions if classification with label encoding
            y_pred_decoded = y_pred.copy()
            y_test_decoded = self.y_test.copy()
            
            if self.problem_type == 'classification' and self.label_encoder is not None:
                y_pred_decoded = self.label_encoder.inverse_transform(y_pred)
                y_test_decoded = self.label_encoder.inverse_transform(self.y_test)
            
            return {
                'y_test': y_test_decoded,
                'y_pred': y_pred_decoded,
                'y_test_encoded': self.y_test,
                'y_pred_encoded': y_pred,
                'y_proba': y_proba,
                'X_test': self.X_test,
                'X_test_processed': self.X_test_processed,
                'feature_names': self.original_feature_names,
                'model_name': type(model).__name__
            }
            
        except Exception as e:
            logger.error(f"Error getting evaluation predictions: {e}")
            raise
    
    def evaluate_model(self, model) -> Dict[str, float]:
        """Fast model evaluation with comprehensive metrics."""
        try:
            predictions_data = self.get_evaluation_predictions(model)
            y_test = predictions_data['y_test_encoded']
            y_pred = predictions_data['y_pred_encoded']
            
            if self.problem_type == 'classification':
                # Get probabilities if available
                try:
                    y_proba = predictions_data['y_proba']
                    if y_proba is not None and len(np.unique(y_test)) == 2:
                        from sklearn.metrics import roc_auc_score
                        auc_score = roc_auc_score(y_test, y_proba[:, 1])
                    else:
                        auc_score = None
                except:
                    auc_score = None
                
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
                }
                
                if auc_score is not None:
                    metrics['auc'] = auc_score
                    
            else:
                metrics = {
                    'mae': mean_absolute_error(y_test, y_pred),
                    'mse': mean_squared_error(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'r2': r2_score(y_test, y_pred)
                }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            return {}
    
    def predict(self, model, data: pd.DataFrame) -> np.ndarray:
        """Make predictions with trained model."""
        try:
            # Preprocess the data
            processed_data = self.preprocessor.transform(data)
            predictions = model.predict(processed_data)
            
            # Decode predictions if classification with label encoding
            if (self.problem_type == 'classification' and 
                hasattr(self, 'label_encoder') and self.label_encoder is not None):
                predictions = self.label_encoder.inverse_transform(predictions)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    def predict_proba(self, model, data: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities (classification only)."""
        if self.problem_type != 'classification':
            raise ValueError("Probabilities only available for classification")
        
        try:
            processed_data = self.preprocessor.transform(data)
            probabilities = model.predict_proba(processed_data)
            return probabilities
            
        except Exception as e:
            logger.error(f"Probability prediction error: {e}")
            raise
    
    def get_feature_importance(self, model) -> Dict[str, float]:
        """Get feature importance if available."""
        try:
            if hasattr(model, 'feature_importances_'):
                # Get feature names after preprocessing
                feature_names = self._get_processed_feature_names()
                importance = model.feature_importances_
                return dict(zip(feature_names, importance))
            elif hasattr(model, 'coef_'):
                # For linear models
                feature_names = self._get_processed_feature_names()
                importance = np.abs(model.coef_).flatten()
                return dict(zip(feature_names, importance))
            else:
                return {}
                
        except Exception as e:
            logger.warning(f"Could not get feature importance: {e}")
            return {}
    
    def _get_processed_feature_names(self) -> List[str]:
        """Get feature names after preprocessing."""
        try:
            # Get feature names from the preprocessor
            if hasattr(self.preprocessor, 'get_feature_names_out'):
                feature_names = self.preprocessor.get_feature_names_out()
            else:
                # Fallback: use original feature names
                feature_names = self.original_feature_names
            
            return list(feature_names)
            
        except:
            # Last resort: generate generic names
            n_features = self.X_train_processed.shape[1] if hasattr(self, 'X_train_processed') else len(self.original_feature_names)
            return [f'feature_{i}' for i in range(n_features)]
    
    def get_available_models(self) -> Dict[str, str]:
        """Get available model names and descriptions."""
        if self.problem_type == 'classification':
            base_models = {
                'rf_fast': 'Random Forest (Fast)',
                'lr': 'Logistic Regression',
                'nb': 'Naive Bayes',
                'knn_fast': 'K-Nearest Neighbors (Fast)',
                'dt_fast': 'Decision Tree (Fast)',
                'svm_fast': 'SVM (Fast)',
                'gb_fast': 'Gradient Boosting (Fast)'
            }
        else:
            base_models = {
                'rf_fast': 'Random Forest (Fast)',
                'lr': 'Linear Regression',
                'ridge': 'Ridge Regression',
                'lasso': 'Lasso Regression',
                'knn_fast': 'K-Nearest Neighbors (Fast)',
                'dt_fast': 'Decision Tree (Fast)',
                'svm_fast': 'SVM (Fast)',
                'gb_fast': 'Gradient Boosting (Fast)'
            }
        
        # Add optional fast libraries
        if LIGHTGBM_AVAILABLE:
            base_models['lgb'] = 'LightGBM (Ultra Fast)'
        if XGBOOST_AVAILABLE:
            base_models['xgb'] = 'XGBoost (Fast)'
        
        return base_models