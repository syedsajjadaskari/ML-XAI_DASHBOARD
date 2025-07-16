"""
Configuration Loader Utility
Handles loading and default configuration management
"""

import yaml
import streamlit as st
import logging

logger = logging.getLogger(__name__)

@st.cache_data
def load_config():
    """Load application configuration."""
    try:
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
            if config is None:
                raise ValueError("Config file is empty")
            return config
    except (FileNotFoundError, yaml.YAMLError, ValueError) as e:
        st.warning(f"Could not load config.yaml: {e}. Using default configuration.")
        logger.warning(f"Config loading error: {e}")
        
        # Default configuration
        return {
            'app': {
                'title': 'Modern ML Web Application',
                'page_icon': '🤖',
                'layout': 'wide',
                'max_file_size': 200,
                'supported_formats': ['csv', 'xlsx', 'parquet']
            },
            'models': {
                'auto_save': True,
                'save_path': 'models/'
            },
            'visualization': {
                'theme': 'plotly_white',
                'default_plots': ['confusion_matrix', 'auc', 'feature']
            },
            'data': {
                'min_rows': 10,
                'min_columns': 2
            }
        }