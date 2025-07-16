"""
Model Management Utilities
Handles model saving, loading, and management operations
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

def get_saved_models() -> List[str]:
    """Get list of saved models."""
    models_dir = Path("models")
    if models_dir.exists():
        return [f.stem for f in models_dir.glob("*.pkl")]
    return []

def save_model(model, model_name: str):
    """Save model to disk."""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    try:
        import pycaret.classification as pc
        import pycaret.regression as pr
        
        if st.session_state.problem_type == "classification":
            final_model = pc.finalize_model(model)
            pc.save_model(final_model, str(models_dir / model_name))
        else:
            final_model = pr.finalize_model(model)
            pr.save_model(final_model, str(models_dir / model_name))
            
        logger.info(f"Model saved: {model_name}")
            
    except Exception as e:
        logger.error(f"Model saving error: {e}")
        raise

def load_saved_model(model_name: str, model_trainer):
    """Load saved model."""
    models_dir = Path("models")
    model_path = models_dir / f"{model_name}.pkl"
    
    try:
        import pycaret.classification as pc
        import pycaret.regression as pr
        
        if st.session_state.problem_type == "classification":
            model = pc.load_model(str(model_path.with_suffix('')))
        else:
            model = pr.load_model(str(model_path.with_suffix('')))
        
        st.session_state.trained_model = model
        st.success(f"✅ Model {model_name} loaded successfully!")
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")

def get_model_metrics(model, problem_type: str) -> Dict[str, float]:
    """Get model performance metrics."""
    try:
        if problem_type == "classification":
            import pycaret.classification as pc
            metrics = pc.pull()
        else:
            import pycaret.regression as pr
            metrics = pr.pull()
        
        # Extract key metrics
        if problem_type == "classification":
            return {
                "Accuracy": metrics.loc[0, 'Accuracy'],
                "Precision": metrics.loc[0, 'Prec.'],
                "Recall": metrics.loc[0, 'Recall'],
                "F1-Score": metrics.loc[0, 'F1']
            }
        else:
            return {
                "MAE": metrics.loc[0, 'MAE'],
                "MSE": metrics.loc[0, 'MSE'],
                "RMSE": metrics.loc[0, 'RMSE'],
                "R²": metrics.loc[0, 'R2']
            }
    except:
        return {"Metric": 0.0}