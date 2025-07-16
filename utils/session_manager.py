"""
Session State Management Utility
Handles Streamlit session state initialization and management
"""

import streamlit as st

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    defaults = {
        'data': None,
        'target_column': None,
        'problem_type': None,
        'model': None,
        'model_results': None,
        'trained_model': None,
        'preprocessing_config': {},
        'columns_to_remove': [],
        'preview_data': None,
        'current_step': 'upload'
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def reset_session_state():
    """Reset session state to initial values."""
    keys_to_keep = ['current_step']
    for key in list(st.session_state.keys()):
        if key not in keys_to_keep:
            del st.session_state[key]
    initialize_session_state()

def get_session_info():
    """Get current session information."""
    return {
        'has_data': st.session_state.data is not None,
        'has_target': st.session_state.target_column is not None,
        'has_model': st.session_state.trained_model is not None,
        'current_step': st.session_state.current_step,
        'data_shape': st.session_state.data.shape if st.session_state.data is not None else None
    }