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

# Import custom modules
from src.data_handler import DataHandler
from src.model_trainer import ModelTrainer
from src.hybrid_trainer import HybridFastTrainer

from src.visualizer import Visualizer
from src.predictor import Predictor

# Import page modules
from pages.upload_page import page_data_upload
from pages.exploration_page import page_data_exploration
from pages.preprocessing_page import page_preprocessing
from pages.training_page import page_model_training
from pages.evaluation_page import page_model_evaluation
from pages.prediction_page import page_predictions

# Import utilities
from utils.config_loader import load_config
from utils.session_manager import initialize_session_state
from utils.navigation import show_progress_indicator, create_sidebar
from utils.model_utils import get_saved_models, save_model, load_saved_model

# Page configuration
st.set_page_config(
    page_title="Modern ML Web Application",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main application function."""
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
    st.title(config['app']['title'])
    st.markdown("*Build, train, and deploy machine learning models with ease*")
    
    # Progress indicator
    show_progress_indicator(st.session_state.current_step)
    
    # Route to appropriate page
    if st.session_state.current_step == "upload":
        page_data_upload(data_handler)
    elif st.session_state.current_step == "explore":
        page_data_exploration(visualizer)
    elif st.session_state.current_step == "preprocess":
        page_preprocessing(data_handler)
    elif st.session_state.current_step == "train":
        page_model_training(model_trainer, data_handler)
    elif st.session_state.current_step == "evaluate":
        page_model_evaluation(visualizer)
    elif st.session_state.current_step == "predict":
        page_predictions(predictor)

if __name__ == "__main__":
    main()