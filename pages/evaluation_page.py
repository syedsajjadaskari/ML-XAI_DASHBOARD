"""
Enhanced Model Evaluation Page
Beautiful visualizations and comprehensive metrics
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
    mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, 
    recall_score, f1_score
)
import seaborn as sns
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def page_model_evaluation_enhanced(visualizer):
    """Enhanced model evaluation page with beautiful visualizations."""
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
    st.markdown("*Comprehensive model performance analysis with beautiful interactive visualizations*")
    
    # Get model and trainer
    model = st.session_state.trained_model
    trainer = st.session_state.get('fast_trainer')
    problem_type = st.session_state.problem_type
    
    if not trainer or not hasattr(trainer, 'evaluation_data'):
        st.error("❌ Training data not available. Please retrain the model.")
        return
    
    # Get evaluation predictions
    try:
        eval_data = trainer.get_evaluation_predictions(model)
    except Exception as e:
        st.error(f"❌ Error getting evaluation data: {str(e)}")
        return
    
    # Performance metrics section
    _display_beautiful_metrics(eval_data, problem_type)
    
    # Interactive visualizations
    _display_stunning_visualizations(eval_data, problem_type)
    
    # Feature importance analysis
    _display_feature_analysis(model, trainer, eval_data)
    
    # Model insights and recommendations
    _display_model_insights(eval_data, problem_type)
    
    # Advanced analysis
    _display_advanced_analysis(eval_data, problem_type)
    
    # Next step
    if st.button("🔮 Proceed to Predictions", type="primary", use_container_width=True):
        st.session_state.current_step = "predict"
        st.rerun()

def _display_beautiful_metrics(eval_data, problem_type):
    """Display beautiful performance metrics with enhanced styling."""
    st.subheader("🎯 Performance Metrics")
    
    y_test = eval_data['y_test_encoded']
    y_pred = eval_data['y_pred_encoded']
    y_proba = eval_data.get('y_proba')
    
    if problem_type == 'classification':
        _show_classification_metrics_beautiful(y_test, y_pred, y_proba)
    else:
        _show_regression_metrics_beautiful(y_test, y_pred)

def _show_classification_metrics_beautiful(y_test, y_pred, y_proba):
    """Show beautiful classification metrics with enhanced cards."""
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # AUC for binary classification
    auc_score = None
    if y_proba is not None and len(np.unique(y_test)) == 2:
        try:
            from sklearn.metrics import roc_auc_score
            auc_score = roc_auc_score(y_test, y_proba[:, 1])
        except:
            auc_score = None
    
    # Create beautiful metric cards
    if auc_score is not None:
        col1, col2, col3, col4, col5 = st.columns(5)
        with col5:
            _create_beautiful_metric_card("📈 AUC Score", auc_score, "Area Under ROC Curve", get_performance_color(auc_score))
    else:
        col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        _create_beautiful_metric_card("🎯 Accuracy", accuracy, "Overall Correctness", get_performance_color(accuracy))
    with col2:
        _create_beautiful_metric_card("🎯 Precision", precision, "Positive Predictive Value", get_performance_color(precision))
    with col3:
        _create_beautiful_metric_card("🎯 Recall", recall, "True Positive Rate", get_performance_color(recall))
    with col4:
        _create_beautiful_metric_card("🎯 F1-Score", f1, "Harmonic Mean of Precision & Recall", get_performance_color(f1))

def _show_regression_metrics_beautiful(y_test, y_pred):
    """Show beautiful regression metrics with enhanced cards."""
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate relative metrics
    mean_actual = np.mean(y_test)
    mape = np.mean(np.abs((y_test - y_pred) / np.where(y_test != 0, y_test, 1))) * 100
    
    # Display metrics in beautiful cards
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        _create_beautiful_metric_card("📊 R² Score", r2, "Coefficient of Determination", get_performance_color(max(0, r2)))
    with col2:
        _create_beautiful_metric_card("📏 MAE", mae, "Mean Absolute Error", get_error_color(mae, mean_actual))
    with col3:
        _create_beautiful_metric_card("📏 RMSE", rmse, "Root Mean Square Error", get_error_color(rmse, mean_actual))
    with col4:
        _create_beautiful_metric_card("📏 MSE", mse, "Mean Square Error", get_error_color(mse, mean_actual**2))
    with col5:
        _create_beautiful_metric_card("📊 MAPE", mape, "Mean Absolute Percentage Error", get_error_color(mape, 20))

def _create_beautiful_metric_card(title, value, description, color):
    """Create a beautiful metric card with enhanced styling."""
    # Determine icon based on metric type
    icon = "📈" if "Score" in title or "AUC" in title else "📊"
    
    # Format value
    if abs(value) < 0.01:
        formatted_value = f"{value:.6f}"
    elif abs(value) < 1:
        formatted_value = f"{value:.4f}"
    elif abs(value) < 100:
        formatted_value = f"{value:.2f}"
    else:
        formatted_value = f"{value:,.0f}"
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {color}15, {color}05);
        border: 2px solid {color}66;
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
        margin: 10px 0;
        position: relative;
        overflow: hidden;
    ">
        <div style="
            position: absolute;
            top: -50%;
            right: -50%;
            width: 100px;
            height: 100px;
            background: linear-gradient(45deg, {color}22, transparent);
            border-radius: 50%;
        "></div>
        <div style="position: relative; z-index: 1;">
            <div style="font-size: 24px; margin-bottom: 10px;">{icon}</div>
            <h3 style="color: {color}; margin: 0; font-size: 14px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">{title}</h3>
            <h1 style="color: {color}; margin: 15px 0; font-size: 36px; font-weight: 700; text-shadow: 0 2px 4px rgba(0,0,0,0.1);">{formatted_value}</h1>
            <p style="color: #666; margin: 0; font-size: 11px; opacity: 0.8;">{description}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def get_performance_color(score):
    """Get color based on performance score."""
    if score >= 0.9:
        return "#00C851"  # Excellent - Green
    elif score >= 0.8:
        return "#ffbb33"  # Good - Orange
    elif score >= 0.7:
        return "#ff6b6b"  # Fair - Red
    else:
        return "#6c757d"  # Poor - Gray

def get_error_color(error, baseline):
    """Get color based on error relative to baseline."""
    ratio = error / baseline if baseline > 0 else 1
    if ratio <= 0.1:
        return "#00C851"  # Excellent - Green
    elif ratio <= 0.3:
        return "#ffbb33"  # Good - Orange
    elif ratio <= 0.5:
        return "#ff6b6b"  # Fair - Red
    else:
        return "#6c757d"  # Poor - Gray

def _display_stunning_visualizations(eval_data, problem_type):
    """Display stunning interactive visualizations."""
    st.subheader("📈 Performance Visualizations")
    
    if problem_type == 'classification':
        _create_classification_visualizations(eval_data)
    else:
        _create_regression_visualizations(eval_data)

def _create_classification_visualizations(eval_data):
    """Create beautiful classification visualizations."""
    y_test = eval_data['y_test_encoded']
    y_pred = eval_data['y_pred_encoded']
    y_proba = eval_data.get('y_proba')
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Confusion Matrix", 
        "📈 ROC Curve", 
        "📊 Precision-Recall", 
        "📋 Classification Report",
        "🎨 Performance Dashboard"
    ])
    
    with tab1:
        _plot_beautiful_confusion_matrix(y_test, y_pred)
    
    with tab2:
        if y_proba is not None:
            _plot_beautiful_roc_curve(y_test, y_proba)
        else:
            st.info("ROC curve requires probability predictions")
    
    with tab3:
        if y_proba is not None:
            _plot_beautiful_precision_recall_curve(y_test, y_proba)
        else:
            st.info("Precision-Recall curve requires probability predictions")
    
    with tab4:
        _plot_beautiful_classification_report(y_test, y_pred)
    
    with tab5:
        _create_classification_dashboard(y_test, y_pred, y_proba)

def _create_regression_visualizations(eval_data):
    """Create beautiful regression visualizations."""
    y_test = eval_data['y_test_encoded']
    y_pred = eval_data['y_pred_encoded']
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Predictions vs Actual", 
        "📊 Residuals Analysis", 
        "📈 Error Distribution", 
        "📋 Performance Summary",
        "🎨 Regression Dashboard"
    ])
    
    with tab1:
        _plot_beautiful_predictions_vs_actual(y_test, y_pred)
    
    with tab2:
        _plot_beautiful_residuals(y_test, y_pred)
    
    with tab3:
        _plot_beautiful_error_distribution(y_test, y_pred)
    
    with tab4:
        _plot_beautiful_regression_summary(y_test, y_pred)
    
    with tab5:
        _create_regression_dashboard(y_test, y_pred)

def _plot_beautiful_confusion_matrix(y_test, y_pred):
    """Plot beautiful interactive confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    labels = sorted(np.unique(np.concatenate([y_test, y_pred])))
    
    # Create enhanced heatmap
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=[f"Class {label}" for label in labels],
        y=[f"Class {label}" for label in labels],
        color_continuous_scale='Viridis',
        text_auto=True,
        title="🎯 Confusion Matrix"
    )
    
    # Enhanced styling
    fig.update_layout(
        height=500,
        title_font_size=20,
        title_x=0.5,
        font=dict(size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    fig.update_traces(
        textfont_size=14,
        textfont_color='white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate and display per-class metrics
    col1, col2 = st.columns(2)
    
    with col1:
        # Per-class accuracy
        class_accuracy = cm.diagonal() / cm.sum(axis=1)
        accuracy_df = pd.DataFrame({
            'Class': labels,
            'Accuracy': class_accuracy,
            'Support': cm.sum(axis=1)
        })
        
        st.markdown("**📊 Per-Class Performance:**")
        st.dataframe(
            accuracy_df.style.format({'Accuracy': '{:.1%}', 'Support': '{:,}'})
            .background_gradient(subset=['Accuracy'], cmap='Greens'),
            use_container_width=True
        )
    
    with col2:
        # Create accuracy bar chart
        fig_acc = px.bar(
            accuracy_df,
            x='Class',
            y='Accuracy',
            title='🎯 Accuracy by Class',
            color='Accuracy',
            color_continuous_scale='viridis',
            text='Accuracy'
        )
        
        fig_acc.update_traces(texttemplate='%{text:.1%}', textposition='auto')
        fig_acc.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_acc, use_container_width=True)

def _plot_beautiful_roc_curve(y_test, y_proba):
    """Plot beautiful ROC curve."""
    if len(np.unique(y_test)) == 2:  # Binary classification
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
        auc_score = auc(fpr, tpr)
        
        fig = go.Figure()
        
        # ROC curve with fill
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {auc_score:.3f})',
            line=dict(color='#2E86AB', width=4),
            fill='tonexty',
            fillcolor='rgba(46, 134, 171, 0.2)'
        ))
        
        # Diagonal reference line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='#E63946', dash='dash', width=2),
            showlegend=True
        ))
        
        # Enhanced styling
        fig.update_layout(
            title={
                'text': '📈 Receiver Operating Characteristic (ROC) Curve',
                'x': 0.5,
                'font': {'size': 20}
            },
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(
                yanchor="bottom",
                y=0.01,
                xanchor="right",
                x=0.99
            )
        )
        
        # Add grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance interpretation
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if auc_score >= 0.9:
                st.success(f"🌟 **Excellent!** AUC = {auc_score:.3f}")
            elif auc_score >= 0.8:
                st.success(f"✅ **Good!** AUC = {auc_score:.3f}")
            elif auc_score >= 0.7:
                st.warning(f"⚠️ **Fair** AUC = {auc_score:.3f}")
            else:
                st.error(f"❌ **Poor** AUC = {auc_score:.3f}")
        
        with col2:
            st.metric("AUC Score", f"{auc_score:.3f}", delta=f"{auc_score - 0.5:.3f} vs Random")
        
        with col3:
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = _[optimal_idx] if len(_) > optimal_idx else 0.5
            st.metric("Optimal Threshold", f"{optimal_threshold:.3f}")
            
    else:
        st.info("📊 ROC curve is only available for binary classification")

def _plot_beautiful_precision_recall_curve(y_test, y_proba):
    """Plot beautiful Precision-Recall curve."""
    if len(np.unique(y_test)) == 2:  # Binary classification
        precision, recall, thresholds = precision_recall_curve(y_test, y_proba[:, 1])
        
        fig = go.Figure()
        
        # Precision-Recall curve with fill
        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            mode='lines',
            name='Precision-Recall Curve',
            line=dict(color='#F77F00', width=4),
            fill='tonexty',
            fillcolor='rgba(247, 127, 0, 0.2)'
        ))
        
        # Baseline (random classifier)
        baseline = np.sum(y_test) / len(y_test)
        fig.add_hline(
            y=baseline, 
            line_dash="dash", 
            line_color="#E63946",
            annotation_text=f"Random Baseline ({baseline:.3f})"
        )
        
        fig.update_layout(
            title={
                'text': '📊 Precision-Recall Curve',
                'x': 0.5,
                'font': {'size': 20}
            },
            xaxis_title='Recall',
            yaxis_title='Precision',
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        # Add grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate Average Precision
        from sklearn.metrics import average_precision_score
        avg_precision = average_precision_score(y_test, y_proba[:, 1])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Precision", f"{avg_precision:.3f}")
        with col2:
            st.metric("Baseline Precision", f"{baseline:.3f}")
        with col3:
            st.metric("Improvement", f"{avg_precision - baseline:.3f}")
            
    else:
        st.info("📊 Precision-Recall curve is only available for binary classification")

def _plot_beautiful_classification_report(y_test, y_pred):
    """Plot beautiful classification report."""
    try:
        from sklearn.metrics import classification_report
        
        # Get classification report as dict
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        # Convert to DataFrame
        df = pd.DataFrame(report).transpose()
        
        # Remove summary rows for heatmap
        metrics_df = df.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
        
        # Create heatmap
        fig = px.imshow(
            metrics_df.values,
            labels=dict(x="Metrics", y="Classes", color="Score"),
            x=metrics_df.columns,
            y=metrics_df.index,
            color_continuous_scale='RdYlGn',
            text_auto='.3f',
            title="📋 Classification Report Heatmap"
        )
        
        fig.update_layout(
            height=400,
            title_font_size=20,
            title_x=0.5
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show detailed report
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Detailed Classification Report:**")
            st.dataframe(
                df.style.format('{:.3f}').background_gradient(cmap='RdYlGn'),
                use_container_width=True
            )
        
        with col2:
            # Summary metrics
            st.markdown("**📈 Summary Metrics:**")
            if 'weighted avg' in report:
                weighted_avg = report['weighted avg']
                summary_metrics = {
                    'Weighted Precision': weighted_avg['precision'],
                    'Weighted Recall': weighted_avg['recall'],
                    'Weighted F1-Score': weighted_avg['f1-score'],
                    'Accuracy': report['accuracy']
                }
                
                for metric, value in summary_metrics.items():
                    st.metric(metric, f"{value:.3f}")
        
    except Exception as e:
        st.error(f"Error creating classification report: {str(e)}")

def _plot_beautiful_predictions_vs_actual(y_test, y_pred):
    """Plot beautiful predictions vs actual for regression."""
    # Calculate R² score
    r2 = r2_score(y_test, y_pred)
    
    # Create perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    
    fig = go.Figure()
    
    # Scatter plot with color based on error
    errors = np.abs(y_test - y_pred)
    
    fig.add_trace(go.Scatter(
        x=y_test, 
        y=y_pred,
        mode='markers',
        name='Predictions',
        marker=dict(
            color=errors,
            colorscale='Viridis',
            size=8,
            opacity=0.7,
            colorbar=dict(title="Absolute Error")
        ),
        hovertemplate='<b>Actual:</b> %{x:.3f}<br><b>Predicted:</b> %{y:.3f}<br><b>Error:</b> %{marker.color:.3f}<extra></extra>'
    ))
    
    # Perfect prediction line
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], 
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash', width=3),
        hovertemplate='Perfect Prediction Line<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': f'🎯 Predictions vs Actual Values (R² = {r2:.3f})',
            'x': 0.5,
            'font': {'size': 20}
        },
        xaxis_title='Actual Values',
        yaxis_title='Predicted Values',
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance interpretation
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if r2 >= 0.9:
            st.success(f"🌟 **Excellent fit!** R² = {r2:.3f}")
        elif r2 >= 0.7:
            st.success(f"✅ **Good fit!** R² = {r2:.3f}")
        elif r2 >= 0.5:
            st.warning(f"⚠️ **Fair fit** R² = {r2:.3f}")
        else:
            st.error(f"❌ **Poor fit** R² = {r2:.3f}")
    
    with col2:
        mae = mean_absolute_error(y_test, y_pred)
        st.metric("Mean Absolute Error", f"{mae:.3f}")
    
    with col3:
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        st.metric("Root Mean Square Error", f"{rmse:.3f}")

def _plot_beautiful_residuals(y_test, y_pred):
    """Plot beautiful residuals analysis."""
    residuals = y_test - y_pred
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Residuals vs Predicted', 'Residuals Distribution', 'Q-Q Plot', 'Residuals vs Fitted'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Residuals vs Predicted
    fig.add_trace(
        go.Scatter(
            x=y_pred, y=residuals,
            mode='markers',
            name='Residuals',
            marker=dict(color='blue', size=6, opacity=0.6)
        ),
        row=1, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
    
    # Residuals histogram
    fig.add_trace(
        go.Histogram(
            x=residuals,
            nbinsx=30,
            name='Distribution',
            marker_color='lightblue',
            opacity=0.7
        ),
        row=1, col=2
    )
    
    # Q-Q plot (simplified)
    from scipy import stats
    theoretical_quantiles, sample_quantiles = stats.probplot(residuals, dist="norm")
    fig.add_trace(
        go.Scatter(
            x=theoretical_quantiles[0],
            y=theoretical_quantiles[1],
            mode='markers',
            name='Q-Q Plot',
            marker=dict(color='green', size=4)
        ),
        row=2, col=1
    )
    
    # Add Q-Q reference line
    fig.add_trace(
        go.Scatter(
            x=theoretical_quantiles[0],
            y=theoretical_quantiles[0] * sample_quantiles[0] + sample_quantiles[1],
            mode='lines',
            name='Q-Q Reference',
            line=dict(color='red', dash='dash')
        ),
        row=2, col=1
    )
    
    # Residuals vs Index (fitted values)
    fig.add_trace(
        go.Scatter(
            x=list(range(len(residuals))),
            y=residuals,
            mode='markers',
            name='Index Plot',
            marker=dict(color='purple', size=4, opacity=0.6)
        ),
        row=2, col=2
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=2)
    
    fig.update_layout(
        title_text="📊 Residuals Analysis",
        height=700,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Residuals statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean Residual", f"{np.mean(residuals):.6f}")
    with col2:
        st.metric("Std Residual", f"{np.std(residuals):.3f}")
    with col3:
        # Durbin-Watson test statistic (simplified)
        dw = np.sum(np.diff(residuals)**2) / np.sum(residuals**2)
        st.metric("Durbin-Watson", f"{dw:.3f}")
    with col4:
        # Percentage of residuals within 1 std
        within_1_std = np.sum(np.abs(residuals) <= np.std(residuals)) / len(residuals) * 100
        st.metric("Within 1σ", f"{within_1_std:.1f}%")

def _display_feature_analysis(model, trainer, eval_data):
    """Display enhanced feature importance analysis."""
    st.subheader("🔍 Feature Analysis")
    
    try:
        # Get feature importance
        importance_data = trainer.get_feature_importance(model)
        
        if importance_data and len(importance_data) > 0:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Create beautiful feature importance plot
                importance_df = pd.DataFrame(
                    list(importance_data.items()), 
                    columns=['Feature', 'Importance']
                )
                importance_df = importance_df.sort_values('Importance', key=abs, ascending=False).head(15)
                
                # Create horizontal bar plot with gradient colors
                fig = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='🏆 Top 15 Feature Importance',
                    color='Importance',
                    color_continuous_scale='viridis',
                    text='Importance'
                )
                
                fig.update_traces(texttemplate='%{text:.4f}', textposition='auto')
                fig.update_layout(
                    height=500,
                    title_font_size=18,
                    title_x=0.5,
                    yaxis={'categoryorder':'total ascending'}
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Feature importance table with styling
                st.markdown("**📊 Feature Rankings:**")
                importance_df['Rank'] = range(1, len(importance_df) + 1)
                display_df = importance_df[['Rank', 'Feature', 'Importance']].copy()
                
                st.dataframe(
                    display_df.style.format({'Importance': '{:.4f}'})
                    .background_gradient(subset=['Importance'], cmap='viridis'),
                    use_container_width=True,
                    height=400
                )
                
                # Feature insights
                st.markdown("**💡 Key Insights:**")
                top_feature = importance_df.iloc[0]
                st.success(f"🏆 **Most Important:** {top_feature['Feature'][:20]}...")
                
                if len(importance_df) > 1:
                    second_feature = importance_df.iloc[1]
                    st.info(f"🥈 **Second:** {second_feature['Feature'][:20]}...")
                
                # Feature importance distribution
                total_importance = importance_df['Importance'].sum()
                top_5_importance = importance_df.head(5)['Importance'].sum()
                top_5_percentage = (top_5_importance / total_importance) * 100
                
                st.metric("Top 5 Features", f"{top_5_percentage:.1f}%", "of total importance")
                
        else:
            st.info("ℹ️ Feature importance not available for this model type")
            
    except Exception as e:
        st.error(f"❌ Error in feature analysis: {str(e)}")

def _display_model_insights(eval_data, problem_type):
    """Display beautiful model insights and recommendations."""
    st.subheader("💡 Model Insights & Recommendations")
    
    try:
        y_test = eval_data['y_test_encoded']
        y_pred = eval_data['y_pred_encoded']
        
        insights = []
        recommendations = []
        
        if problem_type == 'classification':
            accuracy = accuracy_score(y_test, y_pred)
            
            if accuracy >= 0.95:
                insights.append("🌟 **Outstanding Performance**: Your model achieves exceptional accuracy!")
                recommendations.append("✅ Model is production-ready")
                recommendations.append("🚀 Consider A/B testing in production")
            elif accuracy >= 0.85:
                insights.append("✅ **Strong Performance**: Your model shows excellent predictive ability")
                recommendations.append("🔍 Fine-tune hyperparameters for marginal gains")
                recommendations.append("📊 Monitor performance on new data")
            elif accuracy >= 0.75:
                insights.append("⚠️ **Moderate Performance**: There's room for improvement")
                recommendations.append("🎯 Try ensemble methods")
                recommendations.append("🧹 Review feature engineering opportunities")
                recommendations.append("📈 Consider collecting more training data")
            else:
                insights.append("❌ **Needs Improvement**: Model requires significant enhancement")
                recommendations.append("🔄 Try different algorithms")
                recommendations.append("🧹 Revisit data preprocessing")
                recommendations.append("🎯 Verify target variable definition")
                
        else:  # regression
            r2 = r2_score(y_test, y_pred)
            
            if r2 >= 0.9:
                insights.append("🌟 **Excellent Fit**: Model explains most variance in the data!")
                recommendations.append("✅ Model is ready for deployment")
                recommendations.append("📊 Monitor for data drift")
            elif r2 >= 0.7:
                insights.append("✅ **Good Fit**: Model captures important patterns well")
                recommendations.append("🔧 Fine-tune for better performance")
                recommendations.append("🔍 Analyze residuals for improvement opportunities")
            elif r2 >= 0.5:
                insights.append("⚠️ **Moderate Fit**: Model explains some variance")
                recommendations.append("🧮 Try polynomial or interaction features")
                recommendations.append("🎯 Consider ensemble methods")
                recommendations.append("📊 Review outliers and data quality")
            else:
                insights.append("❌ **Poor Fit**: Model struggles with predictions")
                recommendations.append("🔄 Try different algorithm families")
                recommendations.append("🧹 Extensive feature engineering needed")
                recommendations.append("📈 Collect more relevant features")
        
        # Display with beautiful styling
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔍 **Performance Analysis**")
            for insight in insights:
                st.markdown(f"""
                <div style="
                    padding: 15px;
                    margin: 10px 0;
                    background: linear-gradient(135deg, #e3f2fd, #f3e5f5);
                    border-left: 4px solid #2196f3;
                    border-radius: 8px;
                    font-size: 14px;
                ">
                    {insight}
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 🚀 **Action Items**")
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"""
                <div style="
                    padding: 12px;
                    margin: 8px 0;
                    background: linear-gradient(135deg, #f1f8e9, #e8f5e8);
                    border-left: 4px solid #4caf50;
                    border-radius: 6px;
                    font-size: 13px;
                ">
                    <strong>{i}.</strong> {rec}
                </div>
                """, unsafe_allow_html=True)
        
    except Exception as e:
        st.warning(f"Could not generate insights: {str(e)}")

def _display_advanced_analysis(eval_data, problem_type):
    """Display advanced analysis with model diagnostics."""
    st.subheader("🔬 Advanced Analysis")
    
    with st.expander("🔍 Click to explore advanced model diagnostics"):
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 **Model Information**")
            
            model_name = eval_data.get('model_name', 'Unknown')
            st.info(f"**Model Type:** {model_name}")
            
            # Training info from session
            if hasattr(st.session_state, 'fast_training_results'):
                results = st.session_state.fast_training_results
                training_time = results.get('training_time', 0)
                st.metric("Training Time", f"{training_time:.2f}s")
                
                models_compared = results.get('models_compared', 1)
                st.metric("Models Compared", models_compared)
                
                trainer_type = results.get('trainer_type', 'Unknown')
                st.info(f"**Trainer:** {trainer_type}")
            
            # Data info
            n_test_samples = len(eval_data['y_test_encoded'])
            st.metric("Test Samples", f"{n_test_samples:,}")
            
        with col2:
            st.markdown("### 🔧 **Model Diagnostics**")
            
            y_test = eval_data['y_test_encoded']
            y_pred = eval_data['y_pred_encoded']
            
            if problem_type == 'classification':
                # Classification diagnostics
                y_proba = eval_data.get('y_proba')
                if y_proba is not None:
                    confidence = np.max(y_proba, axis=1)
                    avg_confidence = np.mean(confidence)
                    st.metric("Avg Confidence", f"{avg_confidence:.3f}")
                    
                    low_confidence = np.sum(confidence < 0.6) / len(confidence) * 100
                    st.metric("Low Confidence", f"{low_confidence:.1f}%")
                
                # Class balance in test set
                unique, counts = np.unique(y_test, return_counts=True)
                class_balance = min(counts) / max(counts)
                st.metric("Class Balance", f"{class_balance:.3f}")
                
            else:
                # Regression diagnostics
                pred_std = np.std(y_pred)
                actual_std = np.std(y_test)
                st.metric("Prediction Std", f"{pred_std:.3f}")
                st.metric("Actual Std", f"{actual_std:.3f}")
                
                # Model bias
                bias = np.mean(y_pred - y_test)
                st.metric("Model Bias", f"{bias:.3f}")
                
                # Prediction range coverage
                pred_range = y_pred.max() - y_pred.min()
                actual_range = y_test.max() - y_test.min()
                coverage = pred_range / actual_range if actual_range > 0 else 0
                st.metric("Range Coverage", f"{coverage:.3f}")
        
        # Model comparison results if available
        if hasattr(st.session_state, 'fast_training_results'):
            results = st.session_state.fast_training_results
            if 'comparison_results' in results and len(results['comparison_results']) > 0:
                st.markdown("### 🏆 **Model Comparison Summary**")
                comparison_df = results['comparison_results']
                
                # Create beautiful comparison chart
                fig = px.bar(
                    comparison_df.head(8),
                    x='Model',
                    y='Score',
                    color='Score',
                    color_continuous_scale='viridis',
                    title='Model Comparison Results',
                    text='Score'
                )
                
                fig.update_traces(texttemplate='%{text:.3f}', textposition='auto')
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show top 3 models with medals
                st.markdown("**🏅 Top Performers:**")
                top_models = comparison_df.head(3)
                medals = ["🥇", "🥈", "🥉"]
                
                for i, (_, row) in enumerate(top_models.iterrows()):
                    medal = medals[i]
                    model_name = row['Model']
                    score = row['Score']
                    time_taken = row.get('Time (s)', 0)
                    
                    st.markdown(f"""
                    <div style="
                        padding: 10px;
                        margin: 5px 0;
                        background: linear-gradient(135deg, {'#FFD700' if i==0 else '#C0C0C0' if i==1 else '#CD7F32'}22, transparent);
                        border-radius: 8px;
                        border-left: 4px solid {'#FFD700' if i==0 else '#C0C0C0' if i==1 else '#CD7F32'};
                    ">
                        {medal} <strong>{model_name}</strong>: {score:.4f} ({time_taken:.1f}s)
                    </div>
                    """, unsafe_allow_html=True)
        
        # Export model summary
        if st.button("📄 Generate Comprehensive Model Report", use_container_width=True):
            _generate_beautiful_model_report(eval_data, problem_type)

def _create_classification_dashboard(y_test, y_pred, y_proba):
    """Create a comprehensive classification dashboard."""
    # Calculate various metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Create dashboard with multiple subplots
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=['Metrics Overview', 'Class Distribution', 'Prediction Confidence',
                       'Error Analysis', 'Model Calibration', 'Performance Summary'],
        specs=[[{"type": "bar"}, {"type": "pie"}, {"type": "histogram"}],
               [{"type": "bar"}, {"type": "scatter"}, {"type": "indicator"}]]
    )
    
    # Metrics overview
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [accuracy, precision, recall, f1]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    fig.add_trace(
        go.Bar(x=metrics, y=values, marker_color=colors, name='Metrics'),
        row=1, col=1
    )
    
    # Class distribution
    unique, counts = np.unique(y_test, return_counts=True)
    fig.add_trace(
        go.Pie(labels=[f'Class {u}' for u in unique], values=counts, name='Distribution'),
        row=1, col=2
    )
    
    # Prediction confidence (if available)
    if y_proba is not None:
        confidence = np.max(y_proba, axis=1)
        fig.add_trace(
            go.Histogram(x=confidence, nbinsx=20, name='Confidence'),
            row=1, col=3
        )
    
    # Error analysis
    errors = (y_test != y_pred).astype(int)
    error_by_class = [np.mean(errors[y_test == u]) for u in unique]
    fig.add_trace(
        go.Bar(x=[f'Class {u}' for u in unique], y=error_by_class, 
               marker_color='red', name='Error Rate'),
        row=2, col=1
    )
    
    # Model calibration (reliability diagram)
    if y_proba is not None and len(unique) == 2:
        from sklearn.calibration import calibration_curve
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_proba[:, 1], n_bins=10)
            fig.add_trace(
                go.Scatter(x=mean_predicted_value, y=fraction_of_positives, 
                          mode='markers+lines', name='Calibration'),
                row=2, col=2
            )
            # Perfect calibration line
            fig.add_trace(
                go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                          line=dict(dash='dash', color='red'), name='Perfect'),
                row=2, col=2
            )
        except:
            pass
    
    # Performance indicator
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=accuracy,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Overall Accuracy"},
            delta={'reference': 0.5},
            gauge={'axis': {'range': [None, 1]},
                   'bar': {'color': "darkblue"},
                   'steps': [{'range': [0, 0.5], 'color': "lightgray"},
                            {'range': [0.5, 0.8], 'color': "yellow"},
                            {'range': [0.8, 1], 'color': "green"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                                'thickness': 0.75, 'value': 0.9}}
        ),
        row=2, col=3
    )
    
    fig.update_layout(height=800, showlegend=False, title_text="🎨 Classification Performance Dashboard")
    st.plotly_chart(fig, use_container_width=True)

def _create_regression_dashboard(y_test, y_pred):
    """Create a comprehensive regression dashboard."""
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Create dashboard
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=['Metrics Overview', 'Predictions vs Actual', 'Residuals Distribution',
                       'Error vs Magnitude', 'Cumulative Error', 'R² Gauge'],
        specs=[[{"type": "bar"}, {"type": "scatter"}, {"type": "histogram"}],
               [{"type": "scatter"}, {"type": "scatter"}, {"type": "indicator"}]]
    )
    
    # Metrics overview
    metrics = ['R²', 'MAE', 'RMSE', 'MAPE']
    mape = np.mean(np.abs((y_test - y_pred) / np.where(y_test != 0, y_test, 1))) * 100
    values = [r2, mae/np.std(y_test), rmse/np.std(y_test), mape/100]  # Normalized
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    fig.add_trace(
        go.Bar(x=metrics, y=values, marker_color=colors, name='Metrics'),
        row=1, col=1
    )
    
    # Predictions vs Actual
    fig.add_trace(
        go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predictions',
                  marker=dict(color='blue', opacity=0.6)),
        row=1, col=2
    )
    # Perfect line
    min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    fig.add_trace(
        go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                  mode='lines', line=dict(color='red', dash='dash'), name='Perfect'),
        row=1, col=2
    )
    
    # Residuals distribution
    residuals = y_test - y_pred
    fig.add_trace(
        go.Histogram(x=residuals, nbinsx=30, name='Residuals'),
        row=1, col=3
    )
    
    # Error vs Magnitude
    abs_errors = np.abs(y_test - y_pred)
    fig.add_trace(
        go.Scatter(x=y_test, y=abs_errors, mode='markers', name='Error vs Actual',
                  marker=dict(color='orange', opacity=0.6)),
        row=2, col=1
    )
    
    # Cumulative error
    sorted_errors = np.sort(abs_errors)
    cumulative_pct = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
    fig.add_trace(
        go.Scatter(x=sorted_errors, y=cumulative_pct, mode='lines', name='Cumulative Error'),
        row=2, col=2
    )
    
    # R² gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=r2,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "R² Score"},
            delta={'reference': 0},
            gauge={'axis': {'range': [-1, 1]},
                   'bar': {'color': "darkblue"},
                   'steps': [{'range': [-1, 0], 'color': "red"},
                            {'range': [0, 0.5], 'color': "yellow"},
                            {'range': [0.5, 1], 'color': "green"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                                'thickness': 0.75, 'value': 0.8}}
        ),
        row=2, col=3
    )
    
    fig.update_layout(height=800, showlegend=False, title_text="🎨 Regression Performance Dashboard")
    st.plotly_chart(fig, use_container_width=True)

def _plot_beautiful_error_distribution(y_test, y_pred):
    """Plot beautiful error distribution analysis."""
    errors = np.abs(y_test - y_pred)
    relative_errors = errors / np.abs(y_test) * 100
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Absolute Error Distribution', 'Relative Error Distribution (%)',
                       'Error vs Actual Values', 'Cumulative Error Distribution'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Absolute error histogram
    fig.add_trace(
        go.Histogram(x=errors, nbinsx=30, name='Abs Error', marker_color='lightblue'),
        row=1, col=1
    )
    
    # Relative error histogram
    fig.add_trace(
        go.Histogram(x=relative_errors, nbinsx=30, name='Rel Error', marker_color='lightgreen'),
        row=1, col=2
    )
    
    # Error vs actual values
    fig.add_trace(
        go.Scatter(x=y_test, y=errors, mode='markers', name='Error vs Actual',
                  marker=dict(color='red', opacity=0.6, size=6)),
        row=2, col=1
    )
    
    # Cumulative error distribution
    sorted_errors = np.sort(errors)
    cumulative_pct = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
    fig.add_trace(
        go.Scatter(x=sorted_errors, y=cumulative_pct, mode='lines', 
                  name='Cumulative', line=dict(color='purple', width=3)),
        row=2, col=2
    )
    
    fig.update_layout(
        height=600,
        title_text="📈 Error Distribution Analysis",
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Error statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean Absolute Error", f"{np.mean(errors):.3f}")
    with col2:
        st.metric("Median Error", f"{np.median(errors):.3f}")
    with col3:
        percentile_95 = np.percentile(errors, 95)
        st.metric("95th Percentile", f"{percentile_95:.3f}")
    with col4:
        max_error = np.max(errors)
        st.metric("Maximum Error", f"{max_error:.3f}")

def _plot_beautiful_regression_summary(y_test, y_pred):
    """Plot beautiful regression performance summary."""
    # Calculate comprehensive metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Additional metrics
    mean_actual = np.mean(y_test)
    mape = np.mean(np.abs((y_test - y_pred) / np.where(y_test != 0, y_test, 1))) * 100
    max_error = np.max(np.abs(y_test - y_pred))
    
    # Create metrics comparison
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Primary Metrics', 'Error Metrics (Normalized)', 
                       'Model vs Baseline', 'Performance Radar'],
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "scatterpolar"}]]
    )
    
    # Primary metrics
    primary_metrics = ['R²', 'Explained Var', 'Correlation']
    from sklearn.metrics import explained_variance_score
    explained_var = explained_variance_score(y_test, y_pred)
    correlation = np.corrcoef(y_test, y_pred)[0, 1]
    primary_values = [r2, explained_var, correlation]
    
    fig.add_trace(
        go.Bar(x=primary_metrics, y=primary_values, 
               marker_color=['green', 'blue', 'purple'], name='Primary'),
        row=1, col=1
    )
    
    # Error metrics (normalized by standard deviation)
    error_metrics = ['MAE', 'RMSE', 'Max Error']
    std_actual = np.std(y_test)
    normalized_errors = [mae/std_actual, rmse/std_actual, max_error/std_actual]
    
    fig.add_trace(
        go.Bar(x=error_metrics, y=normalized_errors,
               marker_color=['red', 'orange', 'darkred'], name='Errors'),
        row=1, col=2
    )
    
    # Model vs baseline comparison
    baseline_pred = np.full_like(y_test, mean_actual)  # Mean baseline
    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
    
    comparison_metrics = ['MAE', 'RMSE']
    model_values = [mae, rmse]
    baseline_values = [baseline_mae, baseline_rmse]
    
    x_pos = np.arange(len(comparison_metrics))
    fig.add_trace(
        go.Bar(x=comparison_metrics, y=model_values, name='Model',
               marker_color='blue', offsetgroup=1),
        row=2, col=1
    )
    fig.add_trace(
        go.Bar(x=comparison_metrics, y=baseline_values, name='Baseline',
               marker_color='gray', offsetgroup=2),
        row=2, col=1
    )
    
    # Performance radar chart
    radar_metrics = ['R²', 'Low MAE', 'Low RMSE', 'High Corr', 'Low MAPE']
    # Normalize values to 0-1 scale
    radar_values = [
        max(0, r2),  # R²
        max(0, 1 - mae/std_actual),  # Inverted MAE
        max(0, 1 - rmse/std_actual),  # Inverted RMSE
        max(0, correlation),  # Correlation
        max(0, 1 - mape/100)  # Inverted MAPE
    ]
    
    fig.add_trace(
        go.Scatterpolar(
            r=radar_values,
            theta=radar_metrics,
            fill='toself',
            name='Performance',
            line_color='blue'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=700,
        title_text="📋 Comprehensive Regression Performance Summary",
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance summary table
    summary_data = {
        'Metric': ['R² Score', 'MAE', 'RMSE', 'MAPE (%)', 'Max Error', 'Mean Actual', 'Std Actual'],
        'Value': [r2, mae, rmse, mape, max_error, mean_actual, np.std(y_test)],
        'Interpretation': [
            'Higher is better (max 1.0)',
            'Lower is better',
            'Lower is better', 
            'Lower is better',
            'Lower is better',
            'Reference value',
            'Reference spread'
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(
        summary_df.style.format({'Value': '{:.4f}'}).background_gradient(subset=['Value']),
        use_container_width=True
    )

def _generate_beautiful_model_report(eval_data, problem_type):
    """Generate and download a comprehensive model report."""
    try:
        import json
        from datetime import datetime
        
        # Create comprehensive report
        report = {
            "model_evaluation_report": {
                "generated_at": datetime.now().isoformat(),
                "model_name": eval_data.get('model_name', 'Unknown'),
                "problem_type": problem_type,
                "target_column": st.session_state.get('target_column', 'Unknown')
            }
        }
        
        # Add performance metrics
        y_test = eval_data['y_test_encoded']
        y_pred = eval_data['y_pred_encoded']
        
        if problem_type == 'classification':
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            report["performance_metrics"] = {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "test_samples": int(len(y_test))
            }
            
            # Add AUC if available
            y_proba = eval_data.get('y_proba')
            if y_proba is not None and len(np.unique(y_test)) == 2:
                try:
                    from sklearn.metrics import roc_auc_score
                    auc_score = roc_auc_score(y_test, y_proba[:, 1])
                    report["performance_metrics"]["auc_score"] = float(auc_score)
                except:
                    pass
                    
        else:
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            report["performance_metrics"] = {
                "r2_score": float(r2),
                "mae": float(mae),
                "mse": float(mse),
                "rmse": float(rmse),
                "test_samples": int(len(y_test))
            }
        
        # Add training information
        if hasattr(st.session_state, 'fast_training_results'):
            training_results = st.session_state.fast_training_results
            report["training_info"] = {
                "training_time_seconds": training_results.get('training_time', 0),
                "trainer_type": training_results.get('trainer_type', 'unknown'),
                "models_compared": training_results.get('models_compared', 1)
            }
            
            # Add comparison results
            if 'comparison_results' in training_results:
                comparison_df = training_results['comparison_results']
                if len(comparison_df) > 0:
                    report["model_comparison"] = comparison_df.to_dict('records')
        
        # Add feature information
        feature_names = eval_data.get('feature_names', [])
        if feature_names:
            report["feature_info"] = {
                "total_features": len(feature_names),
                "feature_names": feature_names[:20]  # Limit to first 20
            }
        
        # Convert to formatted JSON
        report_json = json.dumps(report, indent=2, default=str)
        
        # Create download button
        st.download_button(
            label="📥 Download Comprehensive Report",
            data=report_json,
            file_name=f"model_evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
        
        st.success("✅ Comprehensive model report generated successfully!")
        
        # Show report preview
        with st.expander("👀 Report Preview"):
            st.json(report)
        
    except Exception as e:
        st.error(f"❌ Error generating report: {str(e)}")
        logger.error(f"Report generation error: {e}")