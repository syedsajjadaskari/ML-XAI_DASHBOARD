"""
Modern PyCaret-Streamlit ML Web Application
Main Application Entry Point
Version: 2.0.0
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
    from src.model_trainer import ModelTrainer
    from src.visualizer import Visualizer
    from src.predictor import Predictor
    
    # Try to import fast trainers
    try:
        from src.fast_model_trainer import FastModelTrainer
        from src.hybrid_trainer import HybridFastTrainer
        FAST_TRAINING_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"Fast training modules not available: {e}")
        FastModelTrainer = None
        HybridFastTrainer = None
        FAST_TRAINING_AVAILABLE = False

except ImportError as e:
    st.error(f"❌ Error importing core modules: {e}")
    st.stop()

# Import page modules with error handling
try:
    from pages.upload_page import page_data_upload
    from pages.preprocessing_page import page_preprocessing
    from pages.training_page import page_model_training
    from pages.evaluation_page import page_model_evaluation
    from pages.prediction_page import page_predictions
    
    # Import the exploration page we just created
    from pages.exploration_page import page_data_exploration
    
    # Try to import fast training page
    try:
        from pages.fast_training_page import page_fast_training
        FAST_TRAINING_PAGE_AVAILABLE = True
    except ImportError:
        page_fast_training = None
        FAST_TRAINING_PAGE_AVAILABLE = False

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
        model_trainer = ModelTrainer(config)
        visualizer = Visualizer(config)
        predictor = Predictor(config)
        
        # Create sidebar navigation
        create_sidebar(model_trainer)
        
        # Main content area
        st.title(config.get('app', {}).get('title', 'Modern ML Web Application'))
        st.markdown("*Build, train, and deploy machine learning models with ease*")
        
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
            # Check if fast training is available and preferred
            training_method = st.sidebar.radio(
                "Training Method:",
                ["🎯 Standard Training", "⚡ Fast Training"] if FAST_TRAINING_AVAILABLE else ["🎯 Standard Training"],
                help="Choose between standard PyCaret training or lightning-fast alternatives"
            )
            
            if training_method.startswith("⚡") and FAST_TRAINING_PAGE_AVAILABLE:
                page_fast_training(data_handler)
            else:
                page_model_training(model_trainer, data_handler)
                
        elif current_step == "evaluate":
            page_model_evaluation(visualizer)
            
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