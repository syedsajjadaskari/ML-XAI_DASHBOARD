"""
Updated Main Application Entry Point
Focuses on fast training and clean interface
"""

import streamlit as st
import yaml
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import custom modules with error handling
try:
    from src.data_handler import DataHandler
    from src.visualizer import Visualizer
    from src.predictor import Predictor
    
    # Import fast trainers
    from src.fast_model_trainer import FastModelTrainer
    from src.hybrid_trainer import HybridFastTrainer
    FAST_TRAINING_AVAILABLE = True

except ImportError as e:
    st.error(f"❌ Error importing core modules: {e}")
    st.stop()

# Import page modules with error handling
try:
    # Import updated pages
    from pages.upload_page import page_data_upload
    from pages.exploration_page import page_data_exploration
    from pages.preprocessing_page import page_preprocessing
    from pages.fast_training_page import page_fast_training_only
    from pages.evaluation_page import page_model_evaluation_enhanced
    from pages.prediction_page import page_predictions

except ImportError as e:
    st.error(f"❌ Error importing page modules: {e}")
    st.stop()

# Import utilities with error handling
try:
    from utils.config_loader import load_config
    from utils.session_manager import initialize_session_state
    from utils.navigation import show_progress_indicator, create_sidebar
    from utils.model_utils import get_saved_models, save_model, load_saved_model
except ImportError as e:
    st.error(f"❌ Error importing utility modules: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Modern ML Web Application",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main application function."""
    try:
        # Load configuration
        config = load_config()
        
        # Initialize session state
        initialize_session_state()
        
        # Initialize components
        data_handler = DataHandler(config)
        visualizer = Visualizer(config)
        predictor = Predictor(config)
        
        # Create sidebar navigation
        create_sidebar(None)  # No model trainer needed for navigation
        
        # Main content area
        st.title("🤖 Modern ML Web Application")
        st.markdown("*Build and deploy machine learning models in seconds*")
        
        # Progress indicator
        show_progress_indicator(st.session_state.current_step)
        
        # Route to appropriate page
        current_step = st.session_state.current_step
        
        if current_step == "upload":
            page_data_upload(data_handler)
            
        elif current_step == "explore":
            page_data_exploration(visualizer)
            
        elif current_step == "preprocess":
            page_preprocessing(data_handler)
            
        elif current_step == "train":
            # Only fast training available
            page_fast_training_only(data_handler)
                
        elif current_step == "evaluate":
            page_model_evaluation_enhanced(visualizer)
            
        elif current_step == "predict":
            page_predictions(predictor)
            
        else:
            st.error(f"Unknown step: {current_step}")
            
    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        logger.error(f"Application error: {e}")
        
        # Show debug info in expander
        with st.expander("🔧 Debug Information"):
            st.code(str(e))
            st.write("**Current session state:**")
            st.json({
                'current_step': st.session_state.get('current_step', 'unknown'),
                'has_data': st.session_state.get('data') is not None,
                'has_target': st.session_state.get('target_column') is not None,
                'has_model': st.session_state.get('trained_model') is not None
            })

if __name__ == "__main__":
    main()