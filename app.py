"""
Updated Main Application Entry Point with XAI Integration
Includes the new Universal XAI Explainability page
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
    from pages.fast_training_page import page_fast_training
    from pages.evaluation_page import page_model_evaluation_enhanced
    from pages.prediction_page import page_predictions
    
    # Import new XAI page
    from pages.xai_explainability_page import page_model_explainability
    XAI_PAGE_AVAILABLE = True

except ImportError as e:
    st.error(f"❌ Error importing page modules: {e}")
    XAI_PAGE_AVAILABLE = False
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
    page_title="Modern ML Web Application with XAI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main application function with XAI integration."""
    try:
        # Load configuration
        config = load_config()
        
        # Initialize session state
        initialize_session_state()
        
        # Initialize components
        data_handler = DataHandler(config)
        visualizer = Visualizer(config)
        predictor = Predictor(config)
        
        # Create sidebar navigation (now includes XAI)
        create_sidebar(None)
        
        # Main content area
        st.title("🤖 Modern ML Web Application with XAI")
        st.markdown("*Build, deploy, and explain machine learning models in seconds*")
        
        # Show XAI availability status
        _show_xai_status()
        
        # Progress indicator (now includes XAI step)
        show_progress_indicator(st.session_state.current_step)
        
        # Route to appropriate page
        current_step = st.session_state.current_step
        
        # Validate step
        from utils.session_manager import validate_step, can_access_step
        
        if not validate_step(current_step):
            st.error(f"Invalid step: {current_step}")
            st.session_state.current_step = "upload"
            st.rerun()
        
        if not can_access_step(current_step):
            st.warning(f"Cannot access {current_step} yet. Complete previous steps first.")
            # Redirect to appropriate step
            if st.session_state.data is None:
                st.session_state.current_step = "upload"
            elif st.session_state.target_column is None:
                st.session_state.current_step = "explore"
            elif st.session_state.trained_model is None:
                st.session_state.current_step = "train"
            st.rerun()
        
        # Route to pages
        if current_step == "upload":
            page_data_upload(data_handler)
            
        elif current_step == "explore":
            page_data_exploration(visualizer)
            
        elif current_step == "preprocess":
            page_preprocessing(data_handler)
            
        elif current_step == "train":
            page_fast_training(data_handler)
                
        elif current_step == "evaluate":
            page_model_evaluation_enhanced(visualizer)
            
        elif current_step == "predict":
            page_predictions(predictor)
            
        elif current_step == "xai":
            if XAI_PAGE_AVAILABLE:
                page_model_explainability()
            else:
                st.error("❌ XAI page not available. Check imports.")
                st.session_state.current_step = "predict"
                st.rerun()
            
        else:
            st.error(f"Unknown step: {current_step}")
            st.write("**Available steps:** upload, explore, preprocess, train, evaluate, predict, xai")
            st.session_state.current_step = "upload"
            if st.button("🔄 Reset to Upload"):
                st.rerun()
            
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
                'has_model': st.session_state.get('trained_model') is not None,
                'xai_available': XAI_PAGE_AVAILABLE
            })

def _show_xai_status():
    """Show XAI availability and compatibility status."""
    if not XAI_PAGE_AVAILABLE:
        st.warning("⚠️ XAI page not available. Check page imports.")
        return
    
    # Check XAI library availability
    xai_libraries = {}
    
    try:
        import lime
        xai_libraries['LIME'] = "✅ Available"
    except ImportError:
        xai_libraries['LIME'] = "❌ Install: pip install lime"
    
    try:
        from sklearn.inspection import permutation_importance
        xai_libraries['Sklearn Inspection'] = "✅ Available"
    except ImportError:
        xai_libraries['Sklearn Inspection'] = "❌ Update sklearn"
    
    try:
        import eli5
        xai_libraries['ELI5'] = "✅ Available"
    except ImportError:
        xai_libraries['ELI5'] = "❌ Install: pip install eli5"
    
    try:
        import shap
        xai_libraries['SHAP'] = "✅ Available"
    except ImportError:
        xai_libraries['SHAP'] = "❌ Install: pip install shap"
    
    # Show XAI readiness
    if st.session_state.trained_model is not None:
        st.success("🧠 **XAI Analysis Ready!** Your model is trained and ready for explainability analysis.")
        
        # Quick XAI preview
        with st.expander("🔍 XAI Quick Preview"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**📊 Available Analysis:**")
                st.write("• Global feature importance")
                st.write("• Local instance explanations")
                st.write("• Feature interaction analysis")
                st.write("• Model behavior patterns")
            
            with col2:
                st.write("**🤖 Model Compatibility:**")
                model_type = type(st.session_state.trained_model).__name__
                st.write(f"• Model: {model_type}")
                
                # Check compatibility
                universal_methods = ["Permutation Importance", "Feature Statistics", "Surrogate Models"]
                for method in universal_methods:
                    st.write(f"• ✅ {method}")
            
            with col3:
                st.write("**🔧 XAI Libraries:**")
                for lib, status in xai_libraries.items():
                    st.write(f"• {status.split()[0]} {lib}")
                
                # Quick start button
                if st.button("🚀 Start XAI Analysis", type="primary", use_container_width=True):
                    st.session_state.current_step = "xai"
                    st.rerun()
    
    elif st.session_state.data is not None and st.session_state.target_column is not None:
        st.info("🎯 **Almost Ready for XAI!** Train a model first, then explore explainability.")
    
    else:
        st.info("📊 **XAI Analysis Available** - Complete the ML pipeline to unlock model explainability features.")

def _initialize_xai_session_state():
    """Initialize XAI-specific session state variables."""
    xai_defaults = {
        'xai_analysis_cache': {},
        'xai_current_analysis': None,
        'xai_sample_size': 200,
        'xai_quick_start': None,
        'xai_preferred_method': 'auto'
    }
    
    for key, value in xai_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def _show_pipeline_overview():
    """Show enhanced pipeline overview with XAI."""
    with st.expander("🔍 Pipeline Overview"):
        steps_status = {
            "📁 Data Upload": st.session_state.data is not None,
            "🔍 Data Exploration": st.session_state.target_column is not None,
            "⚙️ Preprocessing": st.session_state.get('preview_data') is not None,
            "🎯 Model Training": st.session_state.trained_model is not None,
            "📊 Model Evaluation": st.session_state.trained_model is not None,
            "🔮 Predictions": st.session_state.trained_model is not None,
            "🧠 XAI Analysis": st.session_state.trained_model is not None and XAI_PAGE_AVAILABLE
        }
        
        cols = st.columns(len(steps_status))
        for i, (step, completed) in enumerate(steps_status.items()):
            with cols[i]:
                if completed:
                    st.success(f"✅ {step}")
                else:
                    st.info(f"⏳ {step}")
        
        # Show next recommended action
        if st.session_state.data is None:
            st.info("👆 **Next:** Upload your dataset to begin")
        elif st.session_state.target_column is None:
            st.info("👆 **Next:** Explore data and select target column")
        elif st.session_state.trained_model is None:
            st.info("👆 **Next:** Train a machine learning model")
        elif st.session_state.current_step != "xai":
            st.success("🎉 **Pipeline Complete!** Try XAI analysis to understand your model")

def _check_xai_prerequisites():
    """Check if XAI analysis can be performed."""
    prerequisites = {
        'data_available': st.session_state.data is not None,
        'target_selected': st.session_state.target_column is not None,
        'model_trained': st.session_state.trained_model is not None,
        'trainer_available': st.session_state.get('fast_trainer') is not None,
        'xai_page_available': XAI_PAGE_AVAILABLE
    }
    
    return all(prerequisites.values()), prerequisites

def _show_xai_recommendations():
    """Show XAI method recommendations based on current model."""
    if st.session_state.trained_model is None:
        return
    
    model_type = type(st.session_state.trained_model).__name__
    
    recommendations = []
    
    # Model-specific recommendations
    if 'Forest' in model_type or 'Tree' in model_type:
        recommendations.extend([
            "🌲 **Tree-based Model Detected**: Use built-in feature importance for quick insights",
            "⚡ **Fast Analysis**: Tree models work excellently with all XAI methods",
            "🎯 **Recommended**: Start with Global Analysis → Feature Importance"
        ])
    elif 'Linear' in model_type or 'Logistic' in model_type:
        recommendations.extend([
            "📊 **Linear Model Detected**: Coefficient analysis provides direct interpretability",
            "🔍 **Transparency**: Linear models are inherently interpretable",
            "🎯 **Recommended**: Focus on Feature Analysis → Coefficient interpretation"
        ])
    elif 'SVM' in model_type or 'SVC' in model_type:
        recommendations.extend([
            "🔧 **SVM Model Detected**: Use model-agnostic methods for best results",
            "🧠 **Complex Boundaries**: LIME explanations work well for local insights",
            "🎯 **Recommended**: Start with Local Analysis → LIME explanations"
        ])
    else:
        recommendations.extend([
            "🤖 **Universal Methods Available**: All XAI techniques compatible",
            "🔍 **Model-Agnostic**: Permutation importance works for any model",
            "🎯 **Recommended**: Start with Global Analysis → Permutation importance"
        ])
    
    # Data size recommendations
    data_size = len(st.session_state.data) if st.session_state.data is not None else 0
    if data_size < 1000:
        recommendations.append("📊 **Small Dataset**: Detailed analysis possible with all methods")
    elif data_size > 10000:
        recommendations.append("⚡ **Large Dataset**: Use sampling for faster analysis")
    
    if recommendations:
        with st.expander("💡 XAI Recommendations for Your Model"):
            for rec in recommendations:
                st.write(rec)

if __name__ == "__main__":
    # Initialize XAI session state
    _initialize_xai_session_state()
    
    # Run main application
    main()
    
    # Show additional information
    _show_pipeline_overview()
    _show_xai_recommendations()