"""
Navigation and UI Utilities
Handles navigation, progress indicators, and sidebar (Clean - No Page Names)
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
    """Create clean sidebar navigation without page names."""
    with st.sidebar:
        st.title("🤖 ML Pipeline")
        st.markdown("---")
        
        # Navigation steps only (no page name display)
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
                
            # Create navigation button
            if st.session_state.current_step == step_id:
                # Current step - show as highlighted but not clickable
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #00C851, #007E33);
                    color: white;
                    padding: 12px;
                    border-radius: 8px;
                    text-align: center;
                    font-weight: bold;
                    margin-bottom: 5px;
                    border: 2px solid #00C851;
                ">
                    👉 {icon} {label}
                </div>
                """, unsafe_allow_html=True)
            else:
                # Other steps - clickable buttons
                if st.button(f"{icon} {label}", 
                           key=f"nav_{step_id}", 
                           use_container_width=True, 
                           disabled=disabled,
                           type="secondary"):
                    st.session_state.current_step = step_id
                    st.rerun()
        
        st.markdown("---")
        
        # Model management (simplified)
        st.subheader("💾 Models")
        saved_models = get_saved_models()
        if saved_models:
            selected_model = st.selectbox("Load Model", [""] + saved_models, key="model_selector")
            if selected_model and st.button("Load", use_container_width=True):
                load_saved_model(selected_model, model_trainer)
        else:
            st.info("No saved models")
        
        # Quick info (minimal)
        st.markdown("---")
        st.markdown("**Status:**")
        if st.session_state.data is not None:
            st.write(f"📊 {st.session_state.data.shape[0]:,} rows")
        if st.session_state.target_column:
            st.write(f"🎯 {st.session_state.target_column}")
        if st.session_state.trained_model:
            st.write("✅ Model ready")
        
        # Hide streamlit page selector completely
        st.markdown("""
        <style>
        [data-testid="stSidebarNav"] {
            display: none;
        }
        
        [data-testid="stSidebarNavItems"] {
            display: none;
        }
        
        .css-1d391kg {
            display: none;
        }
        
        section[data-testid="stSidebar"] > div:first-child {
            padding-top: 0rem;
        }
        </style>
        """, unsafe_allow_html=True)