"""
Navigation and UI Utilities
Handles navigation, progress indicators, and sidebar
"""

import streamlit as st
from utils.model_utils import get_saved_models, load_saved_model

def show_progress_indicator(current_step: str):
    """Show progress indicator."""
    steps = ["upload", "explore", "preprocess", "train", "evaluate", "predict"]
    step_names = ["Upload", "Explore", "Preprocess", "Train", "Evaluate", "Predict"]
    
    current_index = steps.index(current_step) if current_step in steps else 0
    
    cols = st.columns(len(steps))
    for i, (step, name) in enumerate(zip(steps, step_names)):
        with cols[i]:
            if i <= current_index:
                st.success(f"✅ {name}")
            else:
                st.info(f"⏳ {name}")

def create_sidebar(model_trainer):
    """Create sidebar navigation."""
    with st.sidebar:
        st.title("🤖 ML Pipeline")
        st.markdown("---")
        
        # Current status
        st.subheader("📊 Current Status")
        if st.session_state.data is not None:
            st.success(f"✅ Data loaded ({st.session_state.data.shape[0]} rows)")
        else:
            st.info("📁 No data loaded")
            
        if st.session_state.target_column:
            st.success(f"✅ Target: {st.session_state.target_column}")
            st.info(f"📈 Type: {st.session_state.problem_type}")
        else:
            st.warning("⚠️ No target selected")
            
        if st.session_state.trained_model is not None:
            st.success("✅ Model trained")
        else:
            st.info("🎯 No model trained")
        
        st.markdown("---")
        
        # Navigation steps
        steps = [
            ("📁", "Data Upload", "upload"),
            ("🔍", "Data Exploration", "explore"),
            ("⚙️", "Preprocessing", "preprocess"),
            ("🎯", "Model Training", "train"),
            ("📊", "Model Evaluation", "evaluate"),
            ("🔮", "Predictions", "predict")
        ]
        
        for icon, label, step_id in steps:
            # Disable steps if prerequisites not met
            disabled = False
            if step_id in ["explore", "preprocess", "train", "evaluate", "predict"] and st.session_state.data is None:
                disabled = True
            if step_id in ["train", "evaluate", "predict"] and st.session_state.target_column is None:
                disabled = True
            if step_id in ["evaluate", "predict"] and st.session_state.trained_model is None:
                disabled = True
                
            if st.button(f"{icon} {label}", key=f"nav_{step_id}", use_container_width=True, disabled=disabled):
                st.session_state.current_step = step_id
                st.rerun()
        
        st.markdown("---")
        
        # Model management
        st.subheader("📦 Model Management")
        saved_models = get_saved_models()
        if saved_models:
            selected_model = st.selectbox("Load Saved Model", ["None"] + saved_models)
            if selected_model != "None" and st.button("Load Model"):
                load_saved_model(selected_model, model_trainer)
        
        # App info
        st.markdown("---")
        st.info("""
        **Modern ML Web App v2.0**
        
        Built with:
        - Streamlit 1.39.0
        - PyCaret 3.3.2
        - Python 3.9+
        
        Features:
        - Automated ML pipeline
        - Interactive visualizations
        - Model comparison
        - Real-time predictions
        """)