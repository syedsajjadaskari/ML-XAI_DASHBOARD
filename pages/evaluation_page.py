"""
Model Evaluation Page
Handles model evaluation, metrics, and visualization
"""

import streamlit as st
import logging
from utils.model_utils import get_model_metrics

logger = logging.getLogger(__name__)

def page_model_evaluation(visualizer):
    """Model evaluation page."""
    # Check prerequisites
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first")
        return
    
    if st.session_state.target_column is None:
        st.warning("⚠️ Please select target column first")
        return
        
    if st.session_state.trained_model is None:
        st.warning("⚠️ Please train a model first")
        return
    
    st.header("📊 Model Evaluation")
    
    # Evaluation plots
    st.subheader("📈 Performance Plots")
    
    # Available plots based on problem type
    if st.session_state.problem_type == "classification":
        plot_options = [
            "confusion_matrix", "auc", "threshold", "pr", "class_report",
            "boundary", "roc", "lift", "calibration", "dimension"
        ]
    else:
        plot_options = [
            "residuals", "cooks", "rfe", "learning", "validation",
            "manifold", "feature", "parameter"
        ]
    
    # Plot selection
    col1, col2 = st.columns(2)
    with col1:
        selected_plots = st.multiselect(
            "Select evaluation plots",
            plot_options,
            default=plot_options[:3]
        )
    
    with col2:
        if st.button("Generate Plots"):
            try:
                for plot_type in selected_plots:
                    with st.spinner(f"Generating {plot_type} plot..."):
                        fig = visualizer.plot_model_evaluation(
                            st.session_state.trained_model,
                            plot_type
                        )
                        if fig is not None:
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning(f"Could not generate {plot_type} plot")
            except Exception as e:
                st.error(f"❌ Plot generation error: {str(e)}")
    
    # Model metrics
    st.subheader("📋 Performance Metrics")
    try:
        metrics = get_model_metrics(st.session_state.trained_model, st.session_state.problem_type)
        
        if metrics and len(metrics) > 1:  # More than just a placeholder
            cols = st.columns(min(4, len(metrics)))
            for i, (metric, value) in enumerate(metrics.items()):
                with cols[i % 4]:
                    st.metric(metric, f"{value:.4f}")
        else:
            st.info("📊 Train a model to see performance metrics")
                
    except Exception as e:
        st.warning(f"⚠️ Could not retrieve metrics: {str(e)}")
    
    # Feature importance
    st.subheader("🎯 Feature Importance")
    try:
        importance_fig = visualizer.plot_feature_importance_model(st.session_state.trained_model)
        if importance_fig is not None:
            st.plotly_chart(importance_fig, use_container_width=True)
        else:
            st.info("Feature importance plot not available for this model")
    except Exception as e:
        st.warning(f"⚠️ Feature importance not available: {str(e)}")
    
    # Model interpretation
    st.subheader("🔍 Model Interpretation")
    if st.button("Generate SHAP Analysis"):
        try:
            with st.spinner("Generating SHAP analysis..."):
                shap_fig = visualizer.plot_shap_analysis(st.session_state.trained_model)
                if shap_fig is not None:
                    st.plotly_chart(shap_fig, use_container_width=True)
                else:
                    st.info("SHAP analysis not available for this model")
        except Exception as e:
            st.error(f"❌ SHAP analysis error: {str(e)}")
    
    # Next step
    if st.button("🔮 Proceed to Predictions", type="primary", use_container_width=True):
        st.session_state.current_step = "predict"
        st.rerun()