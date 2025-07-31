"""
Updated Session State Management Utility
Handles Streamlit session state initialization and management including XAI step
"""

import streamlit as st

def initialize_session_state():
    """Initialize Streamlit session state variables including XAI."""
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
        'current_step': 'upload',
        
        # XAI-specific session state
        'xai_analysis_cache': {},
        'xai_current_analysis': None,
        'xai_sample_size': 200,
        'xai_quick_start': None,
        'xai_preferred_method': 'auto',
        'xai_last_analysis_time': None,
        'xai_results': {},
        
        # Additional trainer info for XAI
        'fast_trainer': None,
        'fast_training_results': {},
        
        # Navigation
        'valid_steps': ['upload', 'explore', 'preprocess', 'train', 'evaluate', 'predict', 'xai']
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def reset_session_state():
    """Reset session state to initial values."""
    keys_to_keep = ['current_step', 'valid_steps']
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
        'data_shape': st.session_state.data.shape if st.session_state.data is not None else None,
        'xai_ready': _check_xai_readiness(),
        'valid_steps': st.session_state.get('valid_steps', [])
    }

def _check_xai_readiness():
    """Check if XAI analysis is ready."""
    return (
        st.session_state.data is not None and
        st.session_state.target_column is not None and
        st.session_state.trained_model is not None and
        st.session_state.get('fast_trainer') is not None
    )

def validate_step(step):
    """Validate if step is allowed."""
    valid_steps = st.session_state.get('valid_steps', ['upload', 'explore', 'preprocess', 'train', 'evaluate', 'predict', 'xai'])
    return step in valid_steps

def get_next_step():
    """Get the next logical step based on current state."""
    if st.session_state.data is None:
        return 'upload'
    elif st.session_state.target_column is None:
        return 'explore'
    elif st.session_state.trained_model is None:
        return 'train'
    elif st.session_state.current_step == 'train':
        return 'evaluate'
    elif st.session_state.current_step == 'evaluate':
        return 'predict'
    elif st.session_state.current_step == 'predict':
        return 'xai'
    else:
        return st.session_state.current_step

def can_access_step(step):
    """Check if user can access a specific step."""
    # Upload is always accessible
    if step == 'upload':
        return True
    
    # Need data for explore and beyond
    if step in ['explore', 'preprocess', 'train', 'evaluate', 'predict', 'xai']:
        if st.session_state.data is None:
            return False
    
    # Need target for train and beyond
    if step in ['train', 'evaluate', 'predict', 'xai']:
        if st.session_state.target_column is None:
            return False
    
    # Need trained model for evaluate, predict, xai
    if step in ['evaluate', 'predict', 'xai']:
        if st.session_state.trained_model is None:
            return False
    
    # XAI needs trainer data
    if step == 'xai':
        if st.session_state.get('fast_trainer') is None:
            return False
    
    return True

def update_step(new_step):
    """Safely update current step with validation."""
    if validate_step(new_step) and can_access_step(new_step):
        st.session_state.current_step = new_step
        return True
    else:
        st.error(f"Cannot access step: {new_step}")
        return False

def clear_xai_cache():
    """Clear XAI analysis cache."""
    st.session_state.xai_analysis_cache = {}
    st.session_state.xai_current_analysis = None
    st.session_state.xai_results = {}

def save_xai_analysis(analysis_type, results):
    """Save XAI analysis results to cache."""
    if 'xai_analysis_cache' not in st.session_state:
        st.session_state.xai_analysis_cache = {}
    
    st.session_state.xai_analysis_cache[analysis_type] = {
        'results': results,
        'timestamp': st.session_state.get('xai_last_analysis_time')
    }

def get_cached_xai_analysis(analysis_type):
    """Get cached XAI analysis results."""
    cache = st.session_state.get('xai_analysis_cache', {})
    return cache.get(analysis_type)

def get_pipeline_progress():
    """Get pipeline completion progress as percentage."""
    steps_completed = 0
    total_steps = 7  # Including XAI
    
    if st.session_state.data is not None:
        steps_completed += 1
    if st.session_state.target_column is not None:
        steps_completed += 1
    if st.session_state.get('preview_data') is not None:
        steps_completed += 1
    if st.session_state.trained_model is not None:
        steps_completed += 1
    if st.session_state.trained_model is not None:  # Evaluation available
        steps_completed += 1
    if st.session_state.trained_model is not None:  # Predictions available
        steps_completed += 1
    if _check_xai_readiness():  # XAI ready
        steps_completed += 1
    
    return (steps_completed / total_steps) * 100