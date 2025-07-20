"""
Enhanced Model Evaluation Page
Comprehensive model evaluation with best-in-class visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.inspection import permutation_importance
import seaborn as sns
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def page_model_evaluation_enhanced(visualizer):
    """Enhanced model evaluation page."""
    # Back navigation
    col_back, col_title = st.columns([1, 4])
    with col_back:
        if st.button("⬅️ Back to Training", use_container_width=True):
            st.session_state.current_step = "train"
            st.rerun()
    
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
    
    st.header("📊 Model Evaluation & Analysis")
    st.markdown("*Comprehensive model performance analysis with interactive visualizations*")
    
    # Get model and trainer
    model = st.session_state.trained_model
    trainer = st.session_state.get('fast_trainer')
    problem_type = st.session_state.problem_type
    
    # Performance metrics section
    _display_performance_metrics(model, trainer, problem_type)
    
    # Interactive visualizations
    _display_interactive_plots(model, trainer, problem_type)
    
    # Feature importance analysis
    _display_feature_analysis(model, trainer)
    
    # Model insights and recommendations
    _display_model_insights(model, trainer, problem_type)
    
    # Advanced analysis
    _display_advanced_analysis(model, trainer, problem_type)
    
    # Next step
    if st.button("🔮 Proceed to Predictions", type="primary", use_container_width=True):
        st.session_state.current_step = "predict"
        st.rerun()

def _display_performance_metrics(model, trainer, problem_type):
    """Display comprehensive performance metrics."""
    st.subheader("🎯 Performance Metrics")
    
    try:
        # Get predictions for evaluation
        if trainer and hasattr(trainer, 'X_test_processed') and hasattr(trainer, 'y_test'):
            y_test = trainer.y_test
            y_pred = model.predict(trainer.X_test_processed)
            
            if problem_type == 'classification':
                _show_classification_metrics(y_test, y_pred, model, trainer)
            else:
                _show_regression_metrics(y_test, y_pred)
        else:
            st.warning("⚠️ Test data not available for detailed metrics")
            
    except Exception as e:
        st.error(f"❌ Error calculating metrics: {str(e)}")
        logger.error(f"Metrics calculation error: {e}")

def _show_classification_metrics(y_test, y_pred, model, trainer):
    """Show classification metrics with beautiful cards."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # AUC for binary classification
    try:
        if hasattr(model, 'predict_proba') and len(np.unique(y_test)) == 2:
            y_proba = model.predict_proba(trainer.X_test_processed)[:, 1]
            auc_score = roc_auc_score(y_test, y_proba)
        else:
            auc_score = None
    except:
        auc_score = None
    
    # Display metrics in cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        _metric_card("🎯 Accuracy", accuracy, "Higher is better", get_performance_color(accuracy))
    with col2:
        _metric_card("🎯 Precision", precision, "Higher is better", get_performance_color(precision))
    with col3:
        _metric_card("🎯 Recall", recall, "Higher is better", get_performance_color(recall))
    with col4:
        _metric_card("🎯 F1-Score", f1, "Higher is better", get_performance_color(f1))
    
    if auc_score is not None:
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            _metric_card("📈 AUC Score", auc_score, "Higher is better", get_performance_color(auc_score))

def _show_regression_metrics(y_test, y_pred):
    """Show regression metrics with beautiful cards."""
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Display metrics in cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        _metric_card("📊 R² Score", r2, "Higher is better", get_performance_color(r2))
    with col2:
        _metric_card("📏 MAE", mae, "Lower is better", get_error_color(mae, np.std(y_test)))
    with col3:
        _metric_card("📏 RMSE", rmse, "Lower is better", get_error_color(rmse, np.std(y_test)))
    with col4:
        _metric_card("📏 MSE", mse, "Lower is better", get_error_color(mse, np.var(y_test)))

def _metric_card(title, value, description, color):
    """Create a beautiful metric card."""
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {color}22, {color}11);
        border: 2px solid {color}44;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    ">
        <h3 style="color: {color}; margin: 0; font-size: 16px;">{title}</h3>
        <h1 style="color: {color}; margin: 10px 0; font-size: 32px;">{value:.4f}</h1>
        <p style="color: #666; margin: 0; font-size: 12px;">{description}</p>
    </div>
    """, unsafe_allow_html=True)

def get_performance_color(score):
    """Get color based on performance score."""
    if score >= 0.9:
        return "#00C851"  # Green
    elif score >= 0.8:
        return "#ffbb33"  # Orange
    elif score >= 0.7:
        return "#ff6b6b"  # Red
    else:
        return "#6c757d"  # Gray

def get_error_color(error, baseline):
    """Get color based on error relative to baseline."""
    ratio = error / baseline if baseline > 0 else 1
    if ratio <= 0.1:
        return "#00C851"  # Green
    elif ratio <= 0.3:
        return "#ffbb33"  # Orange
    elif ratio <= 0.5:
        return "#ff6b6b"  # Red
    else:
        return "#6c757d"  # Gray

def _display_interactive_plots(model, trainer, problem_type):
    """Display interactive performance plots."""
    st.subheader("📈 Performance Visualizations")
    
    if not trainer or not hasattr(trainer, 'X_test_processed'):
        st.warning("⚠️ Test data not available for visualizations")
        return
    
    if problem_type == 'classification':
        _create_classification_plots(model, trainer)
    else:
        _create_regression_plots(model, trainer)

def _create_classification_plots(model, trainer):
    """Create comprehensive classification plots."""
    try:
        y_test = trainer.y_test
        y_pred = model.predict(trainer.X_test_processed)
        
        # Create tabs for different plot types
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Confusion Matrix", "📈 ROC Curve", "📊 Precision-Recall", "📋 Classification Report"])
        
        with tab1:
            _plot_confusion_matrix(y_test, y_pred)
        
        with tab2:
            if hasattr(model, 'predict_proba'):
                _plot_roc_curve(y_test, model, trainer)
            else:
                st.info("ROC curve requires probability predictions")
        
        with tab3:
            if hasattr(model, 'predict_proba'):
                _plot_precision_recall_curve(y_test, model, trainer)
            else:
                st.info("Precision-Recall curve requires probability predictions")
        
        with tab4:
            _plot_classification_report(y_test, y_pred)
            
    except Exception as e:
        st.error(f"❌ Error creating classification plots: {str(e)}")

def _create_regression_plots(model, trainer):
    """Create comprehensive regression plots."""
    try:
        y_test = trainer.y_test
        y_pred = model.predict(trainer.X_test_processed)
        
        # Create tabs for different plot types
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Predictions vs Actual", "📊 Residuals", "📈 Error Distribution", "📋 Performance Summary"])
        
        with tab1:
            _plot_predictions_vs_actual(y_test, y_pred)
        
        with tab2:
            _plot_residuals(y_test, y_pred)
        
        with tab3:
            _plot_error_distribution(y_test, y_pred)
        
        with tab4:
            _plot_regression_summary(y_test, y_pred)
            
    except Exception as e:
        st.error(f"❌ Error creating regression plots: {str(e)}")

def _plot_confusion_matrix(y_test, y_pred):
    """Plot interactive confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    
    # Create labels
    labels = sorted(np.unique(np.concatenate([y_test, y_pred])))
    
    # Create heatmap
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=labels,
        y=labels,
        color_continuous_scale='Blues',
        text_auto=True,
        title="Confusion Matrix"
    )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate accuracy per class
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    accuracy_df = pd.DataFrame({
        'Class': labels,
        'Accuracy': class_accuracy
    })
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Per-Class Accuracy:**")
        st.dataframe(accuracy_df, use_container_width=True)
    
    with col2:
        fig_acc = px.bar(
            accuracy_df,
            x='Class',
            y='Accuracy',
            title='Accuracy by Class',
            color='Accuracy',
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig_acc, use_container_width=True)

def _plot_roc_curve(y_test, model, trainer):
    """Plot ROC curve."""
    try:
        if len(np.unique(y_test)) == 2:  # Binary classification
            y_proba = model.predict_proba(trainer.X_test_processed)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc_score = auc(fpr, tpr)
            
            fig = go.Figure()
            
            # ROC curve
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'ROC Curve (AUC = {auc_score:.3f})',
                line=dict(color='darkblue', width=3)
            ))
            
            # Diagonal line
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(color='red', dash='dash', width=2)
            ))
            
            fig.update_layout(
                title='Receiver Operating Characteristic (ROC) Curve',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretation
            if auc_score >= 0.9:
                st.success(f"🌟 Excellent model performance! AUC = {auc_score:.3f}")
            elif auc_score >= 0.8:
                st.success(f"✅ Good model performance! AUC = {auc_score:.3f}")
            elif auc_score >= 0.7:
                st.warning(f"⚠️ Fair model performance. AUC = {auc_score:.3f}")
            else:
                st.error(f"❌ Poor model performance. AUC = {auc_score:.3f}")
                
        else:
            st.info("ROC curve is only available for binary classification")
            
    except Exception as e:
        st.error(f"Error plotting ROC curve: {str(e)}")

def _plot_precision_recall_curve(y_test, model, trainer):
    """Plot Precision-Recall curve."""
    try:
        if len(np.unique(y_test)) == 2:  # Binary classification
            y_proba = model.predict_proba(trainer.X_test_processed)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=recall, y=precision,
                mode='lines',
                name='Precision-Recall Curve',
                line=dict(color='green', width=3),
                fill='tonexty'
            ))
            
            fig.update_layout(
                title='Precision-Recall Curve',
                xaxis_title='Recall',
                yaxis_title='Precision',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Precision-Recall curve is only available for binary classification")
            
    except Exception as e:
        st.error(f"Error plotting Precision-Recall curve: {str(e)}")

def _plot_classification_report(y_test, y_pred):
    """Plot classification report as heatmap."""
    try:
        from sklearn.metrics import classification_report
        
        # Get classification report as dict
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        # Convert to DataFrame for better visualization
        df = pd.DataFrame(report).transpose()
        df = df.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
        
        # Create heatmap
        fig = px.imshow(
            df.values,
            labels=dict(x="Metrics", y="Classes", color="Score"),
            x=df.columns,
            y=df.index,
            color_continuous_scale='RdYlGn',
            text_auto='.3f',
            title="Classification Report Heatmap"
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show detailed report
        st.write("**Detailed Classification Report:**")
        st.text(classification_report(y_test, y_pred, zero_division=0))
        
    except Exception as e:
        st.error(f"Error plotting classification report: {str(e)}")

def _plot_predictions_vs_actual(y_test, y_pred):
    """Plot predictions vs actual values for regression."""
    # Create perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    
    fig = go.Figure()
    
    # Scatter plot
    fig.add_trace(go.Scatter(
        x=y_test, y=y_pred,
        mode='markers',
        name='Predictions',
        opacity=0.6,
        marker=dict(color='blue', size=8)
    ))
    
    # Perfect prediction line
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash', width=2)
    ))
    
    fig.update_layout(
        title='Predictions vs Actual Values',
        xaxis_title='Actual Values',
        yaxis_title='Predicted Values',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # R² score
    r2 = r2_score(y_test, y_pred)
    if r2 >= 0.9:
        st.success(f"🌟 Excellent fit! R² = {r2:.3f}")
    elif r2 >= 0.7:
        st.success(f"✅ Good fit! R² = {r2:.3f}")
    elif r2 >= 0.5:
        st.warning(f"⚠️ Fair fit. R² = {r2:.3f}")
    else:
        st.error(f"❌ Poor fit. R² = {r2:.3f}")

def _plot_residuals(y_test, y_pred):
    """Plot residuals analysis."""
    residuals = y_test - y_pred
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Residuals vs Predicted
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=y_pred, y=residuals,
            mode='markers',
            name='Residuals',
            opacity=0.6,
            marker=dict(color='purple', size=6)
        ))
        fig1.add_hline(y=0, line_dash="dash", line_color="red")
        fig1.update_layout(
            title='Residuals vs Predicted',
            xaxis_title='Predicted Values',
            yaxis_title='Residuals'
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Residuals histogram
        fig2 = px.histogram(
            x=residuals,
            nbins=30,
            title='Residuals Distribution',
            labels={'x': 'Residuals', 'y': 'Count'}
        )
        st.plotly_chart(fig2, use_container_width=True)

def _plot_error_distribution(y_test, y_pred):
    """Plot error distribution."""
    errors = np.abs(y_test - y_pred)
    
    fig = px.histogram(
        x=errors,
        nbins=30,
        title='Absolute Error Distribution',
        labels={'x': 'Absolute Error', 'y': 'Count'}
    )
    
    # Add mean error line
    mean_error = errors.mean()
    fig.add_vline(x=mean_error, line_dash="dash", line_color="red", 
                  annotation_text=f"Mean Error: {mean_error:.3f}")
    
    st.plotly_chart(fig, use_container_width=True)

def _plot_regression_summary(y_test, y_pred):
    """Plot regression performance summary."""
    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R²': r2_score(y_test, y_pred)
    }
    
    # Create metrics comparison
    fig = go.Figure()
    
    # Normalize metrics for comparison (except R²)
    normalized_metrics = {}
    baseline_std = np.std(y_test)
    
    for metric, value in metrics.items():
        if metric == 'R²':
            normalized_metrics[metric] = value
        else:
            normalized_metrics[metric] = value / baseline_std
    
    fig.add_trace(go.Bar(
        x=list(normalized_metrics.keys()),
        y=list(normalized_metrics.values()),
        marker_color=['green' if k == 'R²' else 'blue' for k in normalized_metrics.keys()]
    ))
    
    fig.update_layout(
        title='Regression Metrics Summary',
        yaxis_title='Normalized Value'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def _display_feature_analysis(model, trainer):
    """Display feature importance and analysis."""
    st.subheader("🔍 Feature Analysis")
    
    try:
        # Get feature importance
        importance_data = _get_feature_importance(model, trainer)
        
        if importance_data:
            col1, col2 = st.columns(2)
            
            with col1:
                # Feature importance plot
                importance_df = pd.DataFrame(list(importance_data.items()), 
                                           columns=['Feature', 'Importance'])
                importance_df = importance_df.sort_values('Importance', key=abs, ascending=False).head(15)
                
                fig = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Top 15 Feature Importance',
                    color='Importance',
                    color_continuous_scale='viridis'
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Feature importance table
                st.write("**Feature Importance Rankings:**")
                st.dataframe(importance_df, use_container_width=True)
                
                # Feature insights
                st.write("**Key Insights:**")
                top_feature = importance_df.iloc[0]
                st.write(f"🏆 **Most Important:** {top_feature['Feature']} ({top_feature['Importance']:.4f})")
                
                if len(importance_df) > 1:
                    second_feature = importance_df.iloc[1]
                    st.write(f"🥈 **Second Most:** {second_feature['Feature']} ({second_feature['Importance']:.4f})")
        else:
            st.info("Feature importance not available for this model type")
            
    except Exception as e:
        st.error(f"❌ Error in feature analysis: {str(e)}")

def _get_feature_importance(model, trainer):
    """Extract feature importance from model."""
    try:
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            feature_names = _get_feature_names(trainer)
            importance = model.feature_importances_
            return dict(zip(feature_names, importance))
        elif hasattr(model, 'coef_'):
            # Linear models
            feature_names = _get_feature_names(trainer)
            importance = np.abs(model.coef_).flatten()
            return dict(zip(feature_names, importance))
        else:
            return None
    except:
        return None

def _get_feature_names(trainer):
    """Get feature names from trainer."""
    try:
        if hasattr(trainer, 'feature_names'):
            return trainer.feature_names
        else:
            # Generate generic names
            n_features = trainer.X_train_processed.shape[1] if hasattr(trainer, 'X_train_processed') else 10
            return [f'Feature_{i}' for i in range(n_features)]
    except:
        return [f'Feature_{i}' for i in range(10)]

def _display_model_insights(model, trainer, problem_type):
    """Display model insights and recommendations."""
    st.subheader("💡 Model Insights & Recommendations")
    
    try:
        # Get performance summary
        if trainer and hasattr(trainer, 'X_test_processed'):
            y_test = trainer.y_test
            y_pred = model.predict(trainer.X_test_processed)
            
            insights = []
            recommendations = []
            
            if problem_type == 'classification':
                accuracy = accuracy_score(y_test, y_pred)
                
                if accuracy >= 0.95:
                    insights.append("🌟 **Excellent Performance**: Your model is performing exceptionally well!")
                    recommendations.append("✅ Model is ready for production use")
                elif accuracy >= 0.85:
                    insights.append("✅ **Good Performance**: Your model shows strong predictive ability")
                    recommendations.append("🔍 Consider feature engineering to push performance higher")
                elif accuracy >= 0.75:
                    insights.append("⚠️ **Fair Performance**: There's room for improvement")
                    recommendations.append("📊 Try different algorithms or ensemble methods")
                    recommendations.append("🔧 Review feature selection and data quality")
                else:
                    insights.append("❌ **Poor Performance**: Model needs significant improvement")
                    recommendations.append("🔄 Consider collecting more data")
                    recommendations.append("🧹 Review data cleaning and preprocessing steps")
                    recommendations.append("🎯 Verify target variable definition")
            
            else:  # regression
                r2 = r2_score(y_test, y_pred)
                
                if r2 >= 0.9:
                    insights.append("🌟 **Excellent Fit**: Model explains most of the variance!")
                    recommendations.append("✅ Model is ready for production use")
                elif r2 >= 0.7:
                    insights.append("✅ **Good Fit**: Model captures most important patterns")
                    recommendations.append("🔍 Fine-tune hyperparameters for better performance")
                elif r2 >= 0.5:
                    insights.append("⚠️ **Moderate Fit**: Model captures some patterns")
                    recommendations.append("📊 Try polynomial features or interaction terms")
                    recommendations.append("🔧 Consider different algorithms")
                else:
                    insights.append("❌ **Poor Fit**: Model struggles to predict the target")
                    recommendations.append("🔄 Collect more relevant features")
                    recommendations.append("🧹 Review outliers and data quality")
                    recommendations.append("🎯 Verify problem formulation")
            
            # Display insights
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🔍 Performance Insights:**")
                for insight in insights:
                    st.markdown(insight)
            
            with col2:
                st.write("**🚀 Recommendations:**")
                for rec in recommendations:
                    st.markdown(f"- {rec}")
        
    except Exception as e:
        st.warning(f"Could not generate insights: {str(e)}")

def _display_advanced_analysis(model, trainer, problem_type):
    """Display advanced analysis section."""
    st.subheader("🔬 Advanced Analysis")
    
    with st.expander("🔍 Click to explore advanced features"):
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🎯 Model Information:**")
            
            # Model type
            model_type = type(model).__name__
            st.write(f"- **Model Type:** {model_type}")
            
            # Training time
            if hasattr(st.session_state, 'fast_training_results'):
                training_time = st.session_state.fast_training_results.get('training_time', 0)
                st.write(f"- **Training Time:** {training_time:.2f} seconds")
            
            # Data info
            if trainer and hasattr(trainer, 'X_train_processed'):
                n_features = trainer.X_train_processed.shape[1]
                n_samples = trainer.X_train_processed.shape[0]
                st.write(f"- **Features Used:** {n_features}")
                st.write(f"- **Training Samples:** {n_samples:,}")
            
            # Model complexity
            if hasattr(model, 'n_estimators'):
                st.write(f"- **N Estimators:** {model.n_estimators}")
            elif hasattr(model, 'max_depth'):
                st.write(f"- **Max Depth:** {model.max_depth}")
        
        with col2:
            st.write("**🔧 Model Diagnostics:**")
            
            if trainer and hasattr(trainer, 'X_test_processed'):
                y_test = trainer.y_test
                y_pred = model.predict(trainer.X_test_processed)
                
                # Prediction confidence (for classification)
                if problem_type == 'classification' and hasattr(model, 'predict_proba'):
                    try:
                        y_proba = model.predict_proba(trainer.X_test_processed)
                        confidence = np.max(y_proba, axis=1)
                        avg_confidence = np.mean(confidence)
                        st.write(f"- **Avg Confidence:** {avg_confidence:.3f}")
                        
                        low_confidence = np.sum(confidence < 0.6) / len(confidence) * 100
                        st.write(f"- **Low Confidence Predictions:** {low_confidence:.1f}%")
                    except:
                        pass
                
                # Prediction spread (for regression)
                if problem_type == 'regression':
                    pred_std = np.std(y_pred)
                    actual_std = np.std(y_test)
                    st.write(f"- **Prediction Std:** {pred_std:.3f}")
                    st.write(f"- **Actual Std:** {actual_std:.3f}")
                    
                    # Bias
                    bias = np.mean(y_pred - y_test)
                    st.write(f"- **Model Bias:** {bias:.3f}")
        
        # Model comparison results
        if hasattr(st.session_state, 'fast_training_results'):
            results = st.session_state.fast_training_results
            if 'comparison_results' in results and len(results['comparison_results']) > 0:
                st.write("**🏆 Model Comparison Summary:**")
                comparison_df = results['comparison_results']
                
                # Show top 3 models
                top_models = comparison_df.head(3)
                for i, (_, row) in enumerate(top_models.iterrows()):
                    rank_emoji = ["🥇", "🥈", "🥉"][i]
                    st.write(f"{rank_emoji} **{row['Model']}:** {row['Score']:.4f} ({row.get('Time (s)', 0):.1f}s)")
        
        # Export model summary
        if st.button("📄 Generate Model Report"):
            _generate_model_report(model, trainer, problem_type)

def _generate_model_report(model, trainer, problem_type):
    """Generate comprehensive model report."""
    try:
        import json
        from datetime import datetime
        
        report = {
            "model_report": {
                "generated_at": datetime.now().isoformat(),
                "model_type": type(model).__name__,
                "problem_type": problem_type,
                "target_column": st.session_state.target_column
            }
        }
        
        # Add performance metrics
        if trainer and hasattr(trainer, 'X_test_processed'):
            y_test = trainer.y_test
            y_pred = model.predict(trainer.X_test_processed)
            
            if problem_type == 'classification':
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                
                report["performance_metrics"] = {
                    "accuracy": float(accuracy_score(y_test, y_pred)),
                    "precision": float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
                    "recall": float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
                    "f1_score": float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
                }
            else:
                report["performance_metrics"] = {
                    "r2_score": float(r2_score(y_test, y_pred)),
                    "mae": float(mean_absolute_error(y_test, y_pred)),
                    "mse": float(mean_squared_error(y_test, y_pred)),
                    "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred)))
                }
        
        # Add feature importance
        importance_data = _get_feature_importance(model, trainer)
        if importance_data:
            # Convert to serializable format
            report["feature_importance"] = {k: float(v) for k, v in importance_data.items()}
        
        # Add training info
        if hasattr(st.session_state, 'fast_training_results'):
            training_results = st.session_state.fast_training_results
            report["training_info"] = {
                "training_time_seconds": training_results.get('training_time', 0),
                "trainer_type": training_results.get('trainer_type', 'unknown'),
                "models_compared": training_results.get('models_compared', 1)
            }
        
        # Convert to JSON
        report_json = json.dumps(report, indent=2, default=str)
        
        # Download button
        st.download_button(
            label="📥 Download Model Report",
            data=report_json,
            file_name=f"model_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
        
        st.success("✅ Model report generated successfully!")
        
    except Exception as e:
        st.error(f"❌ Error generating report: {str(e)}")