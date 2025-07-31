"""
Simple Essential XAI Page
Fast, lightweight model explainability with beautiful visualizations
Focus on essential insights without heavy computational overhead
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

# Only essential imports - no heavy libraries
try:
    from sklearn.inspection import permutation_importance
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)

def page_model_explainability():
    """Simple XAI explainability page with essential features."""
    # Back navigation
    col_back, col_title = st.columns([1, 4])
    with col_back:
        if st.button("⬅️ Back to Predictions", use_container_width=True):
            st.session_state.current_step = "predict"
            st.rerun()
    
    # Check prerequisites
    if not _check_basic_requirements():
        return
    
    st.header("🧠 Model Explainability")
    st.markdown("*Understand your model with essential insights and beautiful visualizations*")
    
    # Simple model info card
    _show_model_info()
    
    # Main XAI tabs - simplified to 3 essential ones
    tab1, tab2, tab3 = st.tabs(["🌍 Global Insights", "🎯 Feature Analysis", "📊 Quick Explanations"])
    
    with tab1:
        _handle_global_insights()
    
    with tab2:
        _handle_feature_analysis()
    
    with tab3:
        _handle_quick_explanations()

def _check_basic_requirements():
    """Check basic requirements for XAI."""
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first")
        return False
    
    if st.session_state.target_column is None:
        st.warning("⚠️ Please select target column first")
        return False
    
    if st.session_state.trained_model is None:
        st.warning("⚠️ Please train a model first")
        return False
    
    trainer = st.session_state.get('fast_trainer')
    if not trainer or not hasattr(trainer, 'X_test_processed'):
        st.warning("⚠️ Training data not available - please retrain model")
        return False
    
    return True

def _show_model_info():
    """Show simple model information card."""
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            model_type = type(st.session_state.trained_model).__name__
            st.metric("🤖 Model", model_type.replace('Classifier', '').replace('Regressor', ''))
        
        with col2:
            st.metric("📊 Type", st.session_state.problem_type.title())
        
        with col3:
            data_shape = st.session_state.data.shape
            st.metric("📋 Data", f"{data_shape[0]}×{data_shape[1]}")
        
        with col4:
            trainer = st.session_state.get('fast_trainer')
            if trainer and hasattr(trainer, 'X_test_processed'):
                test_size = len(trainer.X_test_processed)
                st.metric("🧪 Test", f"{test_size:,}")

def _handle_global_insights():
    """Handle global model insights - fast and simple."""
    st.subheader("🌍 Global Model Insights")
    st.markdown("*Overall model behavior and feature importance*")
    
    # Simple analysis button
    if st.button("🔍 Generate Global Insights", type="primary"):
        with st.spinner("Analyzing model... ⚡"):
            _generate_global_insights()

def _generate_global_insights():
    """Generate essential global insights quickly."""
    try:
        trainer = st.session_state.get('fast_trainer')
        model = st.session_state.trained_model
        
        # Use small sample for speed - 100 samples max
        sample_size = min(100, len(trainer.X_test_processed))
        X_sample = trainer.X_test_processed[:sample_size]
        y_sample = trainer.y_test[:sample_size]
        
        # 1. Quick Feature Importance
        st.write("**🏆 Feature Importance**")
        _quick_feature_importance(model, X_sample, y_sample)
        
        st.markdown("---")
        
        # 2. Model Performance Summary
        st.write("**📈 Performance Summary**")
        _quick_performance_summary(model, trainer)
        
        st.markdown("---")
        
        # 3. Top Insights
        st.write("**💡 Key Insights**")
        _generate_key_insights(model, trainer)
        
    except Exception as e:
        st.error(f"❌ Error generating insights: {str(e)}")

def _quick_feature_importance(model, X_sample, y_sample):
    """Fast feature importance calculation with beautiful visualization."""
    try:
        # Method 1: Built-in importance (fastest)
        importance_data = None
        method_used = "Unknown"
        
        if hasattr(model, 'feature_importances_'):
            importance_data = model.feature_importances_
            method_used = "Built-in Feature Importance"
        
        elif hasattr(model, 'coef_'):
            coef = model.coef_
            if len(coef.shape) > 1:
                coef = np.abs(coef).mean(axis=0)
            importance_data = np.abs(coef)
            method_used = "Coefficient Importance"
        
        # Method 2: Quick permutation importance (if needed)
        if importance_data is None and SKLEARN_AVAILABLE:
            perm_imp = permutation_importance(
                model, X_sample, y_sample, 
                n_repeats=3, random_state=42, n_jobs=-1
            )
            importance_data = perm_imp.importances_mean
            method_used = "Permutation Importance"
        
        if importance_data is not None:
            # Get feature names
            feature_names = _get_simple_feature_names(len(importance_data))
            
            # Create importance DataFrame
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance_data
            }).sort_values('Importance', ascending=False).head(10)
            
            # Beautiful horizontal bar chart
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = go.Figure()
                
                colors = px.colors.sequential.Viridis
                fig.add_trace(go.Bar(
                    x=importance_df['Importance'],
                    y=importance_df['Feature'],
                    orientation='h',
                    marker=dict(
                        color=importance_df['Importance'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Importance")
                    ),
                    text=importance_df['Importance'].round(3),
                    textposition='auto'
                ))
                
                fig.update_layout(
                    title=f"Top 10 Features ({method_used})",
                    xaxis_title="Importance Score",
                    yaxis_title="Features",
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("**📊 Summary:**")
                st.metric("Method", method_used.split()[0])
                st.metric("Top Feature", importance_df.iloc[0]['Feature'])
                st.metric("Max Importance", f"{importance_df['Importance'].max():.3f}")
                
                # Feature contribution
                total_importance = importance_df['Importance'].sum()
                top_3_contribution = importance_df.head(3)['Importance'].sum()
                contribution_pct = (top_3_contribution / total_importance) * 100 if total_importance > 0 else 0
                
                st.metric("Top 3 Contribution", f"{contribution_pct:.1f}%")
        
        else:
            st.info("Feature importance not available for this model type")
    
    except Exception as e:
        st.error(f"Feature importance error: {str(e)}")

def _quick_performance_summary(model, trainer):
    """Quick performance summary with key metrics."""
    try:
        X_test = trainer.X_test_processed
        y_test = trainer.y_test
        y_pred = model.predict(X_test)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.session_state.problem_type == 'classification':
                from sklearn.metrics import accuracy_score, f1_score
                
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                # Performance gauge
                fig = go.Figure()
                
                fig.add_trace(go.Indicator(
                    mode = "gauge+number+delta",
                    value = accuracy * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Accuracy %"},
                    delta = {'reference': 80},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "gray"},
                            {'range': [80, 100], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                st.metric("F1-Score", f"{f1:.3f}")
            
            else:
                from sklearn.metrics import r2_score, mean_absolute_error
                
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                # R² gauge
                fig = go.Figure()
                
                fig.add_trace(go.Indicator(
                    mode = "gauge+number",
                    value = r2,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "R² Score"},
                    gauge = {
                        'axis': {'range': [-1, 1]},
                        'bar': {'color': "darkgreen"},
                        'steps': [
                            {'range': [-1, 0], 'color': "lightcoral"},
                            {'range': [0, 0.5], 'color': "lightyellow"},
                            {'range': [0.5, 1], 'color': "lightgreen"}
                        ]
                    }
                ))
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                st.metric("MAE", f"{mae:.3f}")
        
        with col2:
            # Prediction distribution
            if st.session_state.problem_type == 'classification':
                # Class distribution
                unique_classes, counts = np.unique(y_pred, return_counts=True)
                
                fig = px.pie(
                    values=counts,
                    names=unique_classes,
                    title="Prediction Distribution"
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            
            else:
                # Prediction vs actual scatter
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=y_test,
                    y=y_pred,
                    mode='markers',
                    opacity=0.6,
                    name='Predictions',
                    marker=dict(size=6, color='blue')
                ))
                
                # Perfect prediction line
                min_val = min(y_test.min(), y_pred.min())
                max_val = max(y_test.max(), y_pred.max())
                fig.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(dash='dash', color='red')
                ))
                
                fig.update_layout(
                    title="Predictions vs Actual",
                    xaxis_title="Actual",
                    yaxis_title="Predicted",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Performance summary error: {str(e)}")

def _generate_key_insights(model, trainer):
    """Generate key insights and recommendations."""
    try:
        insights = []
        
        # Model-specific insights
        model_type = type(model).__name__
        
        if 'Forest' in model_type or 'Tree' in model_type:
            insights.append("🌲 **Tree-based model**: Naturally interpretable with built-in feature importance")
            insights.append("⚡ **Fast explanations**: This model type provides quick insights")
        
        elif 'Linear' in model_type or 'Logistic' in model_type:
            insights.append("📊 **Linear model**: Highly interpretable coefficients")
            insights.append("🎯 **Direct interpretation**: Each feature has a clear linear relationship")
        
        # Performance-based insights
        X_test = trainer.X_test_processed
        y_test = trainer.y_test
        y_pred = model.predict(X_test)
        
        if st.session_state.problem_type == 'classification':
            from sklearn.metrics import accuracy_score
            accuracy = accuracy_score(y_test, y_pred)
            
            if accuracy >= 0.9:
                insights.append("🎉 **Excellent performance**: Model is highly accurate")
            elif accuracy >= 0.8:
                insights.append("✅ **Good performance**: Model works well")
            elif accuracy >= 0.7:
                insights.append("⚠️ **Fair performance**: Consider improvements")
            else:
                insights.append("❌ **Poor performance**: Model needs significant work")
        
        else:
            from sklearn.metrics import r2_score
            r2 = r2_score(y_test, y_pred)
            
            if r2 >= 0.8:
                insights.append("🎉 **Excellent fit**: Model explains the data very well")
            elif r2 >= 0.6:
                insights.append("✅ **Good fit**: Model captures most patterns")
            elif r2 >= 0.4:
                insights.append("⚠️ **Fair fit**: Some patterns captured")
            else:
                insights.append("❌ **Poor fit**: Model struggles with the data")
        
        # Data insights
        data_size = len(st.session_state.data)
        if data_size < 1000:
            insights.append("📊 **Small dataset**: Results may vary with more data")
        elif data_size > 10000:
            insights.append("📈 **Large dataset**: Model has good data foundation")
        
        # Display insights
        for insight in insights:
            st.write(insight)
    
    except Exception as e:
        st.error(f"Insights generation error: {str(e)}")

def _handle_feature_analysis():
    """Simple feature analysis."""
    st.subheader("🎯 Feature Analysis")
    st.markdown("*Understanding individual features and their relationships*")
    
    if st.button("📊 Analyze Features", type="primary"):
        with st.spinner("Analyzing features... 📊"):
            _simple_feature_analysis()

def _simple_feature_analysis():
    """Simple, fast feature analysis."""
    try:
        trainer = st.session_state.get('fast_trainer')
        
        # Get feature data
        X_sample = trainer.X_test_processed[:100]  # Small sample for speed
        feature_names = _get_simple_feature_names(X_sample.shape[1])
        
        # Feature statistics
        st.write("**📈 Feature Statistics**")
        
        # Calculate simple statistics
        feature_stats = []
        for i, name in enumerate(feature_names[:10]):  # Top 10 only
            col_data = X_sample[:, i]
            
            feature_stats.append({
                'Feature': name,
                'Mean': np.mean(col_data),
                'Std': np.std(col_data),
                'Min': np.min(col_data),
                'Max': np.max(col_data),
                'Range': np.max(col_data) - np.min(col_data)
            })
        
        stats_df = pd.DataFrame(feature_stats)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Feature distribution plot
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=feature_names[:4],
                vertical_spacing=0.1
            )
            
            positions = [(1,1), (1,2), (2,1), (2,2)]
            
            for i, (row, col) in enumerate(positions):
                if i < len(feature_names) and i < X_sample.shape[1]:
                    data = X_sample[:, i]
                    
                    fig.add_trace(
                        go.Histogram(
                            x=data,
                            nbinsx=20,
                            name=feature_names[i],
                            showlegend=False,
                            opacity=0.7
                        ),
                        row=row, col=col
                    )
            
            fig.update_layout(
                title="Feature Distributions (Top 4)",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**📊 Statistics Summary:**")
            st.dataframe(stats_df.round(3), use_container_width=True)
            
            # Quick insights
            st.write("**💡 Quick Insights:**")
            
            # Find most variable feature
            most_variable = stats_df.loc[stats_df['Range'].idxmax()]
            st.write(f"📈 **Most Variable**: {most_variable['Feature']}")
            
            # Find most stable feature
            least_variable = stats_df.loc[stats_df['Range'].idxmin()]
            st.write(f"📊 **Most Stable**: {least_variable['Feature']}")
    
    except Exception as e:
        st.error(f"Feature analysis error: {str(e)}")

def _handle_quick_explanations():
    """Quick model explanations."""
    st.subheader("📊 Quick Model Explanations")
    st.markdown("*Fast insights about your model's behavior*")
    
    # Simple model explanation
    model = st.session_state.trained_model
    model_type = type(model).__name__
    
    # Model explanation
    st.write("**🤖 Your Model Explained:**")
    
    explanations = {
        'RandomForestClassifier': "🌲 **Random Forest**: Combines many decision trees to make predictions. Each tree votes, and the majority wins!",
        'RandomForestRegressor': "🌲 **Random Forest**: Combines many decision trees. Each tree predicts a number, and the average is your result!",
        'LogisticRegression': "📈 **Logistic Regression**: Finds the best line to separate your classes. Simple but powerful!",
        'LinearRegression': "📊 **Linear Regression**: Finds the best straight line through your data points.",
        'XGBClassifier': "⚡ **XGBoost**: Advanced tree method that learns from mistakes. Very accurate!",
        'XGBRegressor': "⚡ **XGBoost**: Advanced tree method that learns from mistakes to predict numbers.",
        'LGBMClassifier': "💡 **LightGBM**: Fast and efficient tree-based model. Great for large datasets!",
        'LGBMRegressor': "💡 **LightGBM**: Fast and efficient tree-based model for number predictions.",
        'SVC': "🎯 **Support Vector Machine**: Finds the best boundary to separate classes.",
        'SVR': "🎯 **Support Vector Regression**: Uses advanced math to fit complex patterns."
    }
    
    explanation = explanations.get(model_type, f"🤖 **{model_type}**: A machine learning model that learns patterns from your data!")
    st.write(explanation)
    
    # Simple prediction example
    if st.button("🔮 Show Prediction Example", type="secondary"):
        _show_prediction_example()

def _show_prediction_example():
    """Show a simple prediction example."""
    try:
        trainer = st.session_state.get('fast_trainer')
        model = st.session_state.trained_model
        
        # Get one test example
        X_test = trainer.X_test_processed
        y_test = trainer.y_test
        
        # Pick a random example
        idx = np.random.randint(0, len(X_test))
        example_X = X_test[idx:idx+1]
        example_y = y_test[idx]
        
        # Make prediction
        prediction = model.predict(example_X)[0]
        
        st.write("**🎯 Prediction Example:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🎲 Actual Value", f"{example_y}")
        
        with col2:
            st.metric("🔮 Predicted", f"{prediction:.3f}")
        
        with col3:
            error = abs(example_y - prediction)
            st.metric("❌ Error", f"{error:.3f}")
        
        # Show if it's a good or bad prediction
        if st.session_state.problem_type == 'classification':
            if example_y == prediction:
                st.success("✅ **Correct Prediction!** Model got this one right.")
            else:
                st.error("❌ **Incorrect Prediction** Model missed this one.")
        else:
            # For regression, check relative error
            relative_error = abs(error / example_y) if example_y != 0 else float('inf')
            if relative_error < 0.1:
                st.success("✅ **Great Prediction!** Very close to actual value.")
            elif relative_error < 0.2:
                st.success("✅ **Good Prediction** Reasonably close.")
            else:
                st.warning("⚠️ **Fair Prediction** Some error present.")
    
    except Exception as e:
        st.error(f"Prediction example error: {str(e)}")

def _get_simple_feature_names(n_features):
    """Get simple feature names."""
    try:
        # Try to get original feature names
        if st.session_state.data is not None and st.session_state.target_column:
            original_features = [col for col in st.session_state.data.columns 
                               if col != st.session_state.target_column]
            
            if len(original_features) >= n_features:
                return original_features[:n_features]
            else:
                # Extend with generic names if needed
                result = original_features.copy()
                for i in range(len(original_features), n_features):
                    result.append(f'Feature_{i}')
                return result
        
        # Fallback to generic names
        return [f'Feature_{i}' for i in range(n_features)]
    
    except:
        return [f'Feature_{i}' for i in range(n_features)]