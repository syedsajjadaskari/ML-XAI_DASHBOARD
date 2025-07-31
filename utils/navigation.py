"""
Updated Navigation and UI Utilities with XAI Explainability Page
Handles navigation, progress indicators, and sidebar
"""

import streamlit as st
from utils.model_utils import get_saved_models, load_saved_model

def show_progress_indicator(current_step: str):
    """Show progress indicator with XAI step."""
    steps = ["upload", "explore", "preprocess", "train", "evaluate", "predict", "xai"]
    step_names = ["Upload", "Explore", "Preprocess", "Train", "Evaluate", "Predict", "XAI"]
    
    current_index = steps.index(current_step) if current_step in steps else 0
    
    cols = st.columns(len(steps))
    for i, (step, name) in enumerate(zip(steps, step_names)):
        with cols[i]:
            if i <= current_index:
                st.success(f"✅ {name}")
            else:
                st.info(f"⏳ {name}")

def create_sidebar(model_trainer):
    """Create enhanced sidebar navigation with XAI page."""
    with st.sidebar:
        st.title("🤖 ML Pipeline")
        st.markdown("---")
        
        # Navigation steps with XAI
        steps = [
            ("📁", "Data Upload", "upload"),
            ("🔍", "Data Exploration", "explore"),
            ("⚙️", "Preprocessing", "preprocess"),
            ("🎯", "Model Training", "train"),
            ("📊", "Model Evaluation", "evaluate"),
            ("🔮", "Predictions", "predict"),
            ("🧠", "XAI Analysis", "xai")
        ]
        
        for icon, label, step_id in steps:
            # Disable steps if prerequisites not met
            disabled = False
            if step_id in ["explore", "preprocess", "train", "evaluate", "predict", "xai"] and st.session_state.data is None:
                disabled = True
            if step_id in ["train", "evaluate", "predict", "xai"] and st.session_state.target_column is None:
                disabled = True
            if step_id in ["evaluate", "predict", "xai"] and st.session_state.trained_model is None:
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
                button_type = "secondary"
                if step_id == "xai" and not disabled:
                    button_type = "primary"  # Highlight XAI when available
                
                if st.button(f"{icon} {label}", 
                           key=f"nav_{step_id}", 
                           use_container_width=True, 
                           disabled=disabled,
                           type=button_type):
                    st.session_state.current_step = step_id
                    st.rerun()
        
        st.markdown("---")
        
        # XAI Quick Info
        if st.session_state.trained_model is not None:
            st.subheader("🧠 XAI Ready")
            
            # Show available XAI methods
            xai_methods = []
            xai_methods.append("✅ Permutation Importance")
            xai_methods.append("✅ Feature Analysis")
            xai_methods.append("✅ Model Behavior")
            
            try:
                import lime
                xai_methods.append("✅ LIME Explanations")
            except ImportError:
                xai_methods.append("❌ LIME (install needed)")
            
            try:
                from sklearn.inspection import partial_dependence
                xai_methods.append("✅ Partial Dependence")
            except ImportError:
                xai_methods.append("❌ Partial Dependence")
            
            with st.expander("🔍 Available XAI Methods"):
                for method in xai_methods:
                    st.write(f"  {method}")
            
            if st.button("🚀 Go to XAI Analysis", use_container_width=True, type="primary"):
                st.session_state.current_step = "xai"
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
        st.markdown("**📊 Status:**")
        if st.session_state.data is not None:
            st.write(f"📈 {st.session_state.data.shape[0]:,} rows")
        if st.session_state.target_column:
            st.write(f"🎯 {st.session_state.target_column}")
        if st.session_state.trained_model:
            model_type = type(st.session_state.trained_model).__name__
            st.write(f"🤖 {model_type}")
            st.write("✅ Model ready")
            
            # XAI compatibility indicator
            xai_compatible_models = [
                'RandomForest', 'XGB', 'LGB', 'CatBoost', 'LogisticRegression', 
                'LinearRegression', 'SVC', 'SVR', 'DecisionTree', 'GradientBoosting'
            ]
            
            if any(model in model_type for model in xai_compatible_models):
                st.success("🧠 XAI Compatible")
            else:
                st.info("🧠 XAI Available")
        
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
        
        /* Highlight XAI button when available */
        .stButton > button[data-testid="baseButton-primary"] {
            background: linear-gradient(135deg, #FF6B6B, #FF8E8E) !important;
            border: none !important;
            color: white !important;
            font-weight: bold !important;
        }
        </style>
        """, unsafe_allow_html=True)

def get_pipeline_status():
    """Get current pipeline status for XAI readiness."""
    status = {
        'data_uploaded': st.session_state.data is not None,
        'target_selected': st.session_state.target_column is not None,
        'model_trained': st.session_state.trained_model is not None,
        'xai_ready': False
    }
    
    # Check XAI readiness
    if (status['data_uploaded'] and status['target_selected'] and status['model_trained']):
        trainer = st.session_state.get('fast_trainer')
        if trainer and hasattr(trainer, 'X_train_processed'):
            status['xai_ready'] = True
    
    return status

def show_xai_readiness_banner():
    """Show XAI readiness banner on other pages."""
    status = get_pipeline_status()
    
    if status['xai_ready'] and st.session_state.current_step != 'xai':
        st.info("""
        🧠 **XAI Analysis Ready!** Your model is trained and ready for explainability analysis. 
        [Click here to explore model behavior →](/xai)
        """)

def create_xai_quick_access():
    """Create quick access to XAI features."""
    if st.session_state.trained_model is not None:
        with st.expander("🧠 Quick XAI Access"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🌍 Global Analysis", use_container_width=True):
                    st.session_state.current_step = "xai"
                    st.session_state.xai_quick_start = "global"
                    st.rerun()
            
            with col2:
                if st.button("🎯 Local Analysis", use_container_width=True):
                    st.session_state.current_step = "xai"
                    st.session_state.xai_quick_start = "local"
                    st.rerun()
            
            with col3:
                if st.button("📊 Feature Analysis", use_container_width=True):
                    st.session_state.current_step = "xai"
                    st.session_state.xai_quick_start = "features"
                    st.rerun()

def show_model_explainability_status():
    """Show model explainability compatibility status."""
    if st.session_state.trained_model is not None:
        model_type = type(st.session_state.trained_model).__name__
        
        # Define XAI compatibility for different model types
        xai_compatibility = {
            'RandomForestClassifier': {
                'built_in_importance': True,
                'shap_fast': True,
                'lime_compatible': True,
                'partial_dependence': True,
                'surrogate_quality': 'High'
            },
            'RandomForestRegressor': {
                'built_in_importance': True,
                'shap_fast': True,
                'lime_compatible': True,
                'partial_dependence': True,
                'surrogate_quality': 'High'
            },
            'XGBClassifier': {
                'built_in_importance': True,
                'shap_fast': True,
                'lime_compatible': True,
                'partial_dependence': True,
                'surrogate_quality': 'High'
            },
            'LGBMClassifier': {
                'built_in_importance': True,
                'shap_fast': True,
                'lime_compatible': True,
                'partial_dependence': True,
                'surrogate_quality': 'High'
            },
            'LogisticRegression': {
                'built_in_importance': False,
                'shap_fast': True,
                'lime_compatible': True,
                'partial_dependence': True,
                'surrogate_quality': 'Medium'
            },
            'LinearRegression': {
                'built_in_importance': False,
                'shap_fast': True,
                'lime_compatible': True,
                'partial_dependence': True,
                'surrogate_quality': 'Medium'
            }
        }
        
        # Get compatibility info
        compatibility = xai_compatibility.get(model_type, {
            'built_in_importance': False,
            'shap_fast': False,
            'lime_compatible': True,
            'partial_dependence': True,
            'surrogate_quality': 'Medium'
        })
        
        with st.expander(f"🔍 XAI Compatibility: {model_type}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Available Methods:**")
                if compatibility['built_in_importance']:
                    st.write("✅ Built-in Feature Importance")
                st.write("✅ Permutation Importance")
                st.write("✅ Feature Statistics")
                if compatibility['lime_compatible']:
                    st.write("✅ LIME Explanations")
                if compatibility['partial_dependence']:
                    st.write("✅ Partial Dependence")
            
            with col2:
                st.write("**Performance:**")
                if compatibility['shap_fast']:
                    st.write("⚡ Fast SHAP Available")
                st.write(f"📊 Surrogate Quality: {compatibility['surrogate_quality']}")
                st.write("🔄 Universal Methods: Yes")

def update_navigation_for_xai():
    """Update navigation system to include XAI page."""
    # This function updates the main app navigation
    # to include the XAI explainability page
    
    # Update session state to include XAI step
    if 'navigation_steps' not in st.session_state:
        st.session_state.navigation_steps = [
            "upload", "explore", "preprocess", "train", "evaluate", "predict", "xai"
        ]
    
    # Add XAI quick start option
    if 'xai_quick_start' not in st.session_state:
        st.session_state.xai_quick_start = None

def get_xai_recommendations():
    """Get XAI method recommendations based on model and data."""
    if st.session_state.trained_model is None:
        return []
    
    model_type = type(st.session_state.trained_model).__name__
    data_size = len(st.session_state.data) if st.session_state.data is not None else 0
    
    recommendations = []
    
    # Tree-based models
    if any(tree_type in model_type for tree_type in ['Forest', 'Tree', 'XGB', 'LGB', 'CatBoost']):
        recommendations.extend([
            "🌲 Start with built-in feature importance for quick insights",
            "⚡ Use TreeExplainer for fast SHAP analysis",
            "🎯 Try decision path analysis for individual predictions"
        ])
    
    # Linear models
    elif any(linear_type in model_type for linear_type in ['Linear', 'Logistic', 'Ridge', 'Lasso']):
        recommendations.extend([
            "📊 Analyze coefficients for feature importance",
            "⚡ Use LinearExplainer for fast SHAP analysis",
            "📈 Examine feature correlations and multicollinearity"
        ])
    
    # General recommendations based on data size
    if data_size < 1000:
        recommendations.append("🔍 Use LIME for detailed local explanations (small dataset)")
    elif data_size > 10000:
        recommendations.append("⚡ Start with permutation importance (large dataset)")
    
    # Problem type specific
    if st.session_state.problem_type == 'classification':
        recommendations.append("🎯 Analyze prediction confidence and class-specific performance")
    else:
        recommendations.append("📈 Focus on residual analysis and prediction intervals")
    
    return recommendations[:5]  # Return top 5 recommendations