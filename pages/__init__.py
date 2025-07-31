"""
Page modules for the ML Web Application
Updated to include the new advanced evaluation page
"""

# Import all page modules
from .upload_page import page_data_upload
from .exploration_page import page_data_exploration  
from .preprocessing_page import page_preprocessing
from .fast_training_page import page_fast_training
from .evaluation_page import page_model_evaluation_enhanced
from .prediction_page import page_predictions

# Legacy imports for compatibility
from .training_page import page_model_training

__all__ = [
    'page_data_upload',
    'page_data_exploration',
    'page_preprocessing', 
    'page_fast_training',
    'page_model_evaluation_enhanced',
    'page_predictions',
    'page_model_training'
]