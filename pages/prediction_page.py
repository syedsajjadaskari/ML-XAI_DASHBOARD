"""
Enhanced Prediction Page
Handles single and batch predictions with separate beautiful SHAP visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# SHAP imports with error handling
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

logger = logging.getLogger(__name__)

def page_predictions(predictor):
    """Enhanced predictions page with separate SHAP section."""
    # Back navigation
    col_back, col_title = st.columns([1, 4])
    with col_back:
        if st.button("⬅️ Back to Evaluation", use_container_width=True):
            st.session_state.current_step = "evaluate"
            st.rerun()
    
    if st.session_state.trained_model is None:
        st.warning("⚠️ Please train a model first")
        return
    
    st.header("🔮 Model Predictions & Analysis")
    st.markdown("*Make predictions and understand model behavior with SHAP*")
    
    # Show model info
    with st.expander("📋 Model Information"):
        model_type = type(st.session_state.trained_model).__name__
        st.write(f"**Model Type:** {model_type}")
        st.write(f"**Target Column:** {st.session_state.target_column}")
        st.write(f"**Problem Type:** {st.session_state.problem_type}")
        
        if hasattr(st.session_state, 'fast_training_results'):
            results = st.session_state.fast_training_results
            if 'training_time' in results:
                st.write(f"**Training Time:** {results['training_time']:.2f} seconds")
    
    # Main tabs for different sections
    tab1, tab2, tab3 = st.tabs(["🎯 Single Prediction", "📊 Batch Predictions", "🧠 SHAP Analysis"])
    
    with tab1:
        _handle_single_prediction(predictor)
    
    with tab2:
        _handle_batch_prediction(predictor)
    
    with tab3:
        _handle_shap_analysis()

def _handle_single_prediction(predictor):
    """Handle single record prediction."""
    st.subheader("🎯 Single Record Prediction")
    
    # Create input form
    with st.form("prediction_form"):
        input_data = {}
        
        # Get feature columns (exclude target)
        if hasattr(st.session_state, 'preview_data') and st.session_state.preview_data is not None:
            data_for_features = st.session_state.preview_data
        else:
            data_for_features = st.session_state.data
            
        feature_columns = [col for col in data_for_features.columns 
                         if col != st.session_state.target_column]
        
        if len(feature_columns) == 0:
            st.error("❌ No feature columns available for prediction")
            return
        
        st.write(f"**Enter values for {len(feature_columns)} features:**")
        
        # Dynamic input fields based on data types
        col1, col2 = st.columns(2)
        for i, column in enumerate(feature_columns):
            column_type = str(data_for_features[column].dtype)
            
            with col1 if i % 2 == 0 else col2:
                if 'int' in column_type or 'float' in column_type:
                    min_val = float(data_for_features[column].min())
                    max_val = float(data_for_features[column].max())
                    mean_val = float(data_for_features[column].mean())
                    
                    input_data[column] = st.number_input(
                        f"{column}",
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        key=f"input_{column}",
                        help=f"Range: {min_val:.2f} to {max_val:.2f}"
                    )
                else:
                    unique_values = data_for_features[column].unique()
                    # Handle NaN values
                    unique_values = [val for val in unique_values if pd.notna(val)]
                    if len(unique_values) > 0:
                        input_data[column] = st.selectbox(
                            f"{column}",
                            options=unique_values,
                            key=f"input_{column}",
                            help=f"{len(unique_values)} unique values"
                        )
                    else:
                        input_data[column] = st.text_input(
                            f"{column}",
                            value="",
                            key=f"input_{column}"
                        )
        
        submitted = st.form_submit_button("🚀 Make Prediction", type="primary")
        
        if submitted:
            try:
                # Make prediction
                input_df = pd.DataFrame([input_data])
                prediction = predictor.predict_single(
                    st.session_state.trained_model,
                    input_df
                )
                
                # Store for SHAP analysis
                st.session_state.prediction_input = input_df
                st.session_state.latest_prediction = prediction
                
                # Display result
                st.success("✅ Prediction completed!")
                
                col1, col2 = st.columns(2)
                with col1:
                    if isinstance(prediction, (int, float)):
                        if st.session_state.problem_type == "regression":
                            st.metric("🎯 Predicted Value", f"{prediction:.4f}")
                        else:
                            st.metric("🎯 Predicted Class", str(prediction))
                    else:
                        st.metric("🎯 Predicted Value", str(prediction))
                
                with col2:
                    if st.session_state.problem_type == "classification":
                        # Get prediction probability
                        try:
                            prob = predictor.predict_probability(
                                st.session_state.trained_model,
                                input_df
                            )
                            confidence_color = "🟢" if prob > 0.8 else "🟡" if prob > 0.6 else "🟠"
                            st.metric(f"{confidence_color} Confidence", f"{prob:.2%}")
                        except Exception as e:
                            st.info("Confidence not available")
                
                # Show input summary
                with st.expander("📋 Input Summary"):
                    input_summary = pd.DataFrame([input_data]).T
                    input_summary.columns = ['Value']
                    st.dataframe(input_summary, use_container_width=True)
            
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
                logger.error(f"Single prediction error: {e}")

def _handle_batch_prediction(predictor):
    """Handle batch predictions."""
    st.subheader("📊 Batch Predictions")
    
    # File upload for batch prediction
    batch_file = st.file_uploader(
        "Upload file for batch prediction",
        type=['csv', 'xlsx'],
        help="File should have the same columns as training data (except target)",
        key="batch_file"
    )
    
    if batch_file is not None:
        try:
            # Load batch data
            if batch_file.name.endswith('.csv'):
                batch_data = pd.read_csv(batch_file)
            else:
                batch_data = pd.read_excel(batch_file)
            
            st.write(f"**Batch data shape:** {batch_data.shape}")
            st.dataframe(batch_data.head(), use_container_width=True)
            
            # Validate columns
            if hasattr(st.session_state, 'preview_data') and st.session_state.preview_data is not None:
                expected_columns = [col for col in st.session_state.preview_data.columns 
                                  if col != st.session_state.target_column]
            else:
                expected_columns = [col for col in st.session_state.data.columns 
                                  if col != st.session_state.target_column]
            
            missing_columns = set(expected_columns) - set(batch_data.columns)
            extra_columns = set(batch_data.columns) - set(expected_columns)
            
            if missing_columns:
                st.warning(f"⚠️ Missing columns: {list(missing_columns)}")
            
            if extra_columns:
                st.info(f"ℹ️ Extra columns will be ignored: {list(extra_columns)}")
            
            # Make batch predictions
            if st.button("🚀 Generate Batch Predictions", type="primary"):
                with st.spinner("Making batch predictions..."):
                    try:
                        # Align columns with training data
                        aligned_data = batch_data.copy()
                        
                        # Add missing columns with default values
                        for col in missing_columns:
                            if col in st.session_state.data.columns:
                                # Use mean for numeric, mode for categorical
                                if pd.api.types.is_numeric_dtype(st.session_state.data[col]):
                                    default_val = st.session_state.data[col].mean()
                                else:
                                    mode_val = st.session_state.data[col].mode()
                                    default_val = mode_val[0] if len(mode_val) > 0 else "Unknown"
                                aligned_data[col] = default_val
                        
                        # Remove extra columns
                        aligned_data = aligned_data[[col for col in expected_columns if col in aligned_data.columns]]
                        
                        predictions = predictor.predict_batch(
                            st.session_state.trained_model,
                            aligned_data
                        )
                        
                        # Combine results
                        results_df = batch_data.copy()
                        results_df['Prediction'] = predictions
                        
                        # Add confidence for classification
                        if st.session_state.problem_type == "classification":
                            try:
                                confidence_scores = []
                                for idx in range(len(aligned_data)):
                                    row_data = aligned_data.iloc[idx:idx+1]
                                    prob = predictor.predict_probability(
                                        st.session_state.trained_model,
                                        row_data
                                    )
                                    confidence_scores.append(prob)
                                results_df['Confidence'] = confidence_scores
                            except Exception as e:
                                logger.warning(f"Could not compute confidence: {e}")
                        
                        st.success("✅ Batch predictions completed!")
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Store for potential SHAP analysis
                        st.session_state.batch_predictions = results_df
                        st.session_state.batch_aligned_data = aligned_data
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
                        # Show prediction summary
                        _show_prediction_summary(predictions)
                    
                    except Exception as e:
                        st.error(f"❌ Batch prediction error: {str(e)}")
                        logger.error(f"Batch prediction error: {e}")
        
        except Exception as e:
            st.error(f"❌ Error loading batch file: {str(e)}")
    
    else:
        _show_batch_prediction_template()

def _show_prediction_summary(predictions):
    """Show prediction summary with visualizations."""
    st.subheader("📈 Prediction Summary")
    
    if st.session_state.problem_type == "classification":
        pred_counts = pd.Series(predictions).value_counts()
        st.write("**Prediction Distribution:**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(pred_counts, use_container_width=True)
        with col2:
            fig = px.pie(
                values=pred_counts.values, 
                names=pred_counts.index, 
                title="Prediction Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("**Prediction Statistics:**")
        pred_stats = pd.Series(predictions).describe()
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(pred_stats, use_container_width=True)
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=predictions,
                nbinsx=30,
                name="Predictions",
                marker_color='lightblue',
                opacity=0.7
            ))
            fig.update_layout(
                title="Prediction Distribution",
                xaxis_title="Predicted Values",
                yaxis_title="Frequency"
            )
            st.plotly_chart(fig, use_container_width=True)

def _show_batch_prediction_template():
    """Show expected format and template for batch predictions."""
    st.info("👆 Upload a CSV or Excel file for batch predictions")
    
    # Show expected format
    if hasattr(st.session_state, 'preview_data') and st.session_state.preview_data is not None:
        sample_data = st.session_state.preview_data
    else:
        sample_data = st.session_state.data
        
    feature_columns = [col for col in sample_data.columns 
                     if col != st.session_state.target_column]
    
    st.subheader("📋 Expected File Format")
    st.write("Your file should contain the following columns:")
    
    format_df = pd.DataFrame({
        'Column Name': feature_columns,
        'Data Type': [str(sample_data[col].dtype) for col in feature_columns],
        'Sample Value': [str(sample_data[col].iloc[0]) if len(sample_data) > 0 else "N/A" for col in feature_columns]
    })
    
    st.dataframe(format_df, use_container_width=True)
    
    # Download template
    if st.button("📥 Download Template", use_container_width=True):
        template_df = pd.DataFrame(columns=feature_columns)
        # Add one sample row with example data
        if len(sample_data) > 0:
            sample_row = {}
            for col in feature_columns:
                if pd.api.types.is_numeric_dtype(sample_data[col]):
                    sample_row[col] = sample_data[col].mean()
                else:
                    mode_val = sample_data[col].mode()
                    sample_row[col] = mode_val[0] if len(mode_val) > 0 else "example"
            template_df = pd.concat([template_df, pd.DataFrame([sample_row])], ignore_index=True)
        
        csv_template = template_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV Template",
            data=csv_template,
            file_name="prediction_template.csv",
            mime="text/csv",
            use_container_width=True
        )

def _handle_shap_analysis():
    """Handle SHAP analysis with beautiful visualizations."""
    st.subheader("🧠 SHAP Analysis - Model Explainability")
    st.markdown("*Understand how your model makes decisions using SHAP (SHapley Additive exPlanations)*")
    
    if not SHAP_AVAILABLE:
        st.error("❌ SHAP library not available. Install with: `pip install shap`")
        return
    
    # Check if we have data for SHAP analysis
    trainer = st.session_state.get('fast_trainer')
    model = st.session_state.trained_model
    
    if not trainer or not hasattr(trainer, 'X_test_processed'):
        st.warning("⚠️ No test data available for SHAP analysis. Please retrain the model.")
        return
    
    # SHAP Analysis Options
    st.write("**Choose SHAP Analysis Type:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Global Analysis", "Single Prediction", "Feature Interaction", "Waterfall Plot"],
            help="Choose the type of SHAP analysis to perform"
        )
    
    with col2:
        sample_size = st.slider(
            "Sample Size for Analysis",
            min_value=10,
            max_value=min(500, len(trainer.X_test_processed)),
            value=min(100, len(trainer.X_test_processed)),
            help="Number of samples to use for SHAP analysis (more = slower but more accurate)"
        )
    
    if st.button("🔍 Generate SHAP Analysis", type="primary"):
        with st.spinner("Computing SHAP values... This may take a moment."):
            try:
                # Get sample data for SHAP
                X_sample = trainer.X_test_processed[:sample_size]
                
                # Initialize SHAP explainer
                explainer = _get_shap_explainer(model, trainer.X_train_processed[:100])  # Use small sample for background
                
                if explainer is None:
                    st.error("❌ Could not create SHAP explainer for this model type")
                    return
                
                # Compute SHAP values
                shap_values = explainer(X_sample)
                
                # Display different types of analysis
                if analysis_type == "Global Analysis":
                    _display_global_shap_analysis(shap_values, X_sample)
                elif analysis_type == "Single Prediction":
                    _display_single_prediction_shap(shap_values, X_sample)
                elif analysis_type == "Feature Interaction":
                    _display_feature_interaction_shap(shap_values, X_sample)
                elif analysis_type == "Waterfall Plot":
                    _display_waterfall_shap(shap_values, X_sample)
                
            except Exception as e:
                st.error(f"❌ SHAP analysis error: {str(e)}")
                logger.error(f"SHAP analysis error: {e}")
                
                # Show detailed error and suggestions
                with st.expander("🔧 Troubleshooting"):
                    st.write("**Possible solutions:**")
                    st.write("1. Try reducing the sample size")
                    st.write("2. Some model types may not support all SHAP explainers")
                    st.write("3. Ensure your model is properly trained")
                    st.code(str(e))

def _get_shap_explainer(model, X_background):
    """Get appropriate SHAP explainer for model type."""
    try:
        model_name = type(model).__name__.lower()
        
        # Try TreeExplainer for tree-based models
        if any(tree_type in model_name for tree_type in ['forest', 'tree', 'boost', 'lgbm', 'xgb']):
            try:
                return shap.TreeExplainer(model)
            except:
                pass
        
        # Try LinearExplainer for linear models
        if any(linear_type in model_name for linear_type in ['linear', 'logistic', 'ridge', 'lasso']):
            try:
                return shap.LinearExplainer(model, X_background)
            except:
                pass
        
        # Fallback to Explainer (works with most models but slower)
        try:
            return shap.Explainer(model, X_background)
        except:
            pass
        
        # Last resort: KernelExplainer (slowest but most compatible)
        try:
            return shap.KernelExplainer(model.predict, X_background[:50])  # Use even smaller background
        except:
            pass
        
        return None
        
    except Exception as e:
        logger.error(f"Error creating SHAP explainer: {e}")
        return None

def _display_global_shap_analysis(shap_values, X_sample):
    """Display global SHAP analysis."""
    st.subheader("🌍 Global Feature Importance")
    
    try:
        # Feature importance plot
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📊 Feature Importance (Mean |SHAP|)**")
            
            # Calculate mean absolute SHAP values
            if hasattr(shap_values, 'values'):
                shap_vals = shap_values.values
            else:
                shap_vals = shap_values
                
            if len(shap_vals.shape) == 3:  # Multi-class classification
                shap_vals = np.abs(shap_vals).mean(axis=2)
            
            feature_importance = np.abs(shap_vals).mean(axis=0)
            
            # Get feature names
            if hasattr(shap_values, 'feature_names') and shap_values.feature_names is not None:
                feature_names = shap_values.feature_names
            else:
                feature_names = [f'Feature_{i}' for i in range(len(feature_importance))]
            
            # Create importance DataFrame
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=True).tail(15)
            
            # Create horizontal bar plot
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=importance_df['Importance'],
                y=importance_df['Feature'],
                orientation='h',
                marker_color='lightblue',
                text=importance_df['Importance'].round(4),
                textposition='auto'
            ))
            
            fig.update_layout(
                title="Top 15 Most Important Features",
                xaxis_title="Mean |SHAP Value|",
                yaxis_title="Features",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**🎯 SHAP Summary Statistics**")
            
            # Summary statistics
            summary_stats = {
                'Total Features': len(feature_importance),
                'Max Importance': f"{feature_importance.max():.4f}",
                'Mean Importance': f"{feature_importance.mean():.4f}",
                'Most Important': feature_names[np.argmax(feature_importance)],
                'Samples Analyzed': len(X_sample)
            }
            
            for key, value in summary_stats.items():
                st.metric(key, value)
            
            # Feature importance distribution
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=feature_importance,
                nbinsx=20,
                marker_color='lightgreen',
                opacity=0.7
            ))
            
            fig_dist.update_layout(
                title="Feature Importance Distribution",
                xaxis_title="SHAP Importance",
                yaxis_title="Count",
                height=300
            )
            
            st.plotly_chart(fig_dist, use_container_width=True)
        
        # Top features detailed analysis
        st.write("**🔍 Top Features Detailed Analysis**")
        
        top_features_df = importance_df.tail(10).sort_values('Importance', ascending=False)
        
        # Add feature statistics
        feature_stats = []
        for feature in top_features_df['Feature']:
            if feature in X_sample.columns if hasattr(X_sample, 'columns') else False:
                idx = list(X_sample.columns).index(feature) if hasattr(X_sample, 'columns') else int(feature.split('_')[1])
                feature_data = X_sample.iloc[:, idx] if hasattr(X_sample, 'iloc') else X_sample[:, idx]
                
                stats = {
                    'Feature': feature,
                    'SHAP Importance': f"{top_features_df[top_features_df['Feature'] == feature]['Importance'].iloc[0]:.4f}",
                    'Mean Value': f"{np.mean(feature_data):.4f}",
                    'Std Value': f"{np.std(feature_data):.4f}",
                    'Min Value': f"{np.min(feature_data):.4f}",
                    'Max Value': f"{np.max(feature_data):.4f}"
                }
                feature_stats.append(stats)
        
        if feature_stats:
            feature_stats_df = pd.DataFrame(feature_stats)
            st.dataframe(feature_stats_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error in global SHAP analysis: {str(e)}")

def _display_single_prediction_shap(shap_values, X_sample):
    """Display SHAP analysis for a single prediction."""
    st.subheader("🎯 Single Prediction SHAP Analysis")
    
    # Select instance to analyze
    instance_idx = st.slider(
        "Select instance to analyze",
        0, len(X_sample) - 1, 0,
        help="Choose which prediction to analyze in detail"
    )
    
    try:
        # Get SHAP values for single instance
        if hasattr(shap_values, 'values'):
            single_shap = shap_values.values[instance_idx]
        else:
            single_shap = shap_values[instance_idx]
        
        # Handle multi-class classification
        if len(single_shap.shape) > 1:
            class_idx = 0  # Use first class for visualization
            single_shap = single_shap[:, class_idx]
            st.info(f"Showing SHAP values for class {class_idx}")
        
        # Get feature names
        if hasattr(shap_values, 'feature_names') and shap_values.feature_names is not None:
            feature_names = shap_values.feature_names
        else:
            feature_names = [f'Feature_{i}' for i in range(len(single_shap))]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🎯 Feature Contributions**")
            
            # Create feature contribution plot
            contrib_df = pd.DataFrame({
                'Feature': feature_names,
                'SHAP_Value': single_shap,
                'Abs_SHAP': np.abs(single_shap)
            }).sort_values('Abs_SHAP', ascending=True).tail(15)
            
            colors = ['red' if x < 0 else 'green' for x in contrib_df['SHAP_Value']]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=contrib_df['SHAP_Value'],
                y=contrib_df['Feature'],
                orientation='h',
                marker_color=colors,
                text=contrib_df['SHAP_Value'].round(4),
                textposition='auto'
            ))
            
            fig.update_layout(
                title=f"SHAP Values for Instance {instance_idx}",
                xaxis_title="SHAP Value",
                yaxis_title="Features",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**📊 Instance Details**")
            
            # Show feature values for this instance
            if hasattr(X_sample, 'iloc'):
                instance_values = X_sample.iloc[instance_idx]
            else:
                instance_values = X_sample[instance_idx]
            
            instance_df = pd.DataFrame({
                'Feature': feature_names,
                'Value': instance_values,
                'SHAP_Value': single_shap,
                'Impact': ['↑ Positive' if x > 0 else '↓ Negative' if x < 0 else '→ Neutral' for x in single_shap]
            })
            
            # Sort by absolute SHAP value
            instance_df['Abs_SHAP'] = np.abs(instance_df['SHAP_Value'])
            instance_df = instance_df.sort_values('Abs_SHAP', ascending=False).head(15)
            instance_df = instance_df.drop('Abs_SHAP', axis=1)
            
            st.dataframe(instance_df, use_container_width=True)
            
            # Summary metrics
            positive_impact = single_shap[single_shap > 0].sum()
            negative_impact = single_shap[single_shap < 0].sum()
            net_impact = positive_impact + negative_impact
            
            st.write("**Impact Summary:**")
            st.metric("Positive Impact", f"{positive_impact:.4f}")
            st.metric("Negative Impact", f"{negative_impact:.4f}")
            st.metric("Net Impact", f"{net_impact:.4f}")
            
    except Exception as e:
        st.error(f"Error in single prediction SHAP analysis: {str(e)}")

def _display_feature_interaction_shap(shap_values, X_sample):
    """Display SHAP feature interaction analysis."""
    st.subheader("🔗 Feature Interaction Analysis")
    
    try:
        # Get top features for interaction analysis
        if hasattr(shap_values, 'values'):
            shap_vals = shap_values.values
        else:
            shap_vals = shap_values
            
        if len(shap_vals.shape) == 3:  # Multi-class
            shap_vals = np.abs(shap_vals).mean(axis=2)
        
        feature_importance = np.abs(shap_vals).mean(axis=0)
        
        # Get feature names
        if hasattr(shap_values, 'feature_names') and shap_values.feature_names is not None:
            feature_names = shap_values.feature_names
        else:
            feature_names = [f'Feature_{i}' for i in range(len(feature_importance))]
        
        # Select top features for interaction
        top_indices = np.argsort(feature_importance)[-10:]  # Top 10 features
        top_features = [feature_names[i] for i in top_indices]
        
        col1, col2 = st.columns(2)
        
        with col1:
            feature1 = st.selectbox("Select First Feature", top_features, key="feat1")
            feature2 = st.selectbox("Select Second Feature", top_features, key="feat2")
        
        with col2:
            if feature1 != feature2:
                # Get indices
                feat1_idx = feature_names.index(feature1)
                feat2_idx = feature_names.index(feature2)
                
                # Get feature values
                if hasattr(X_sample, 'iloc'):
                    feat1_values = X_sample.iloc[:, feat1_idx]
                    feat2_values = X_sample.iloc[:, feat2_idx]
                else:
                    feat1_values = X_sample[:, feat1_idx]
                    feat2_values = X_sample[:, feat2_idx]
                
                # Get SHAP values for these features
                shap1_values = shap_vals[:, feat1_idx]
                shap2_values = shap_vals[:, feat2_idx]
                
                # Create interaction plot
                fig = go.Figure()
                
                # Color by SHAP value
                fig.add_trace(go.Scatter(
                    x=feat1_values,
                    y=feat2_values,
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=shap1_values + shap2_values,
                        colorscale='RdYlBu',
                        showscale=True,
                        colorbar=dict(title="Combined SHAP")
                    ),
                    text=[f"SHAP1: {s1:.3f}<br>SHAP2: {s2:.3f}" for s1, s2 in zip(shap1_values, shap2_values)],
                    hovertemplate=f"{feature1}: %{{x}}<br>{feature2}: %{{y}}<br>%{{text}}<extra></extra>"
                ))
                
                fig.update_layout(
                    title=f"Feature Interaction: {feature1} vs {feature2}",
                    xaxis_title=feature1,
                    yaxis_title=feature2,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Feature correlation with SHAP values
        st.write("**🔍 Feature-SHAP Correlations**")
        
        correlations = []
        for i, feat_name in enumerate(feature_names[:10]):  # Top 10 features
            if hasattr(X_sample, 'iloc'):
                feat_values = X_sample.iloc[:, i]
            else:
                feat_values = X_sample[:, i]
            
            shap_feat_values = shap_vals[:, i]
            correlation = np.corrcoef(feat_values, shap_feat_values)[0, 1]
            
            correlations.append({
                'Feature': feat_name,
                'Feature-SHAP Correlation': correlation,
                'Mean SHAP': np.mean(np.abs(shap_feat_values))
            })
        
        corr_df = pd.DataFrame(correlations).sort_values('Mean SHAP', ascending=False)
        st.dataframe(corr_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error in feature interaction analysis: {str(e)}")

def _display_waterfall_shap(shap_values, X_sample):
    """Display SHAP waterfall plot."""
    st.subheader("🌊 SHAP Waterfall Analysis")
    
    # Select instance for waterfall
    instance_idx = st.slider(
        "Select instance for waterfall plot",
        0, len(X_sample) - 1, 0,
        help="Choose which prediction to show as waterfall",
        key="waterfall_instance"
    )
    
    try:
        # Get SHAP values for single instance
        if hasattr(shap_values, 'values'):
            single_shap = shap_values.values[instance_idx]
            base_value = shap_values.base_values[instance_idx] if hasattr(shap_values, 'base_values') else 0
        else:
            single_shap = shap_values[instance_idx]
            base_value = 0
        
        # Handle multi-class classification
        if len(single_shap.shape) > 1:
            class_idx = st.selectbox("Select class", range(single_shap.shape[1]), key="waterfall_class")
            single_shap = single_shap[:, class_idx]
            if hasattr(base_value, '__len__') and len(base_value) > 1:
                base_value = base_value[class_idx]
        
        # Get feature names
        if hasattr(shap_values, 'feature_names') and shap_values.feature_names is not None:
            feature_names = shap_values.feature_names
        else:
            feature_names = [f'Feature_{i}' for i in range(len(single_shap))]
        
        # Create waterfall data
        waterfall_data = []
        
        # Base value
        waterfall_data.append({
            'Feature': 'Base Value',
            'SHAP_Value': base_value,
            'Cumulative': base_value,
            'Type': 'Base'
        })
        
        # Sort features by absolute SHAP value
        sorted_indices = np.argsort(np.abs(single_shap))[::-1][:15]  # Top 15 features
        
        cumulative = base_value
        for idx in sorted_indices:
            cumulative += single_shap[idx]
            waterfall_data.append({
                'Feature': feature_names[idx],
                'SHAP_Value': single_shap[idx],
                'Cumulative': cumulative,
                'Type': 'Positive' if single_shap[idx] > 0 else 'Negative'
            })
        
        # Final prediction
        final_prediction = cumulative
        waterfall_data.append({
            'Feature': 'Final Prediction',
            'SHAP_Value': 0,
            'Cumulative': final_prediction,
            'Type': 'Final'
        })
        
        waterfall_df = pd.DataFrame(waterfall_data)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create waterfall visualization
            fig = go.Figure()
            
            # Add bars for each contribution
            colors = []
            for _, row in waterfall_df.iterrows():
                if row['Type'] == 'Base':
                    colors.append('blue')
                elif row['Type'] == 'Positive':
                    colors.append('green')
                elif row['Type'] == 'Negative':
                    colors.append('red')
                else:
                    colors.append('orange')
            
            fig.add_trace(go.Bar(
                x=range(len(waterfall_df)),
                y=waterfall_df['Cumulative'],
                marker_color=colors,
                text=[f"{val:.3f}" for val in waterfall_df['Cumulative']],
                textposition='auto',
                hovertemplate="<b>%{customdata}</b><br>Value: %{y:.4f}<br>Contribution: %{text}<extra></extra>",
                customdata=waterfall_df['Feature']
            ))
            
            # Add connection lines
            for i in range(len(waterfall_df) - 1):
                fig.add_shape(
                    type="line",
                    x0=i + 0.4, y0=waterfall_df.iloc[i]['Cumulative'],
                    x1=i + 0.6, y1=waterfall_df.iloc[i]['Cumulative'],
                    line=dict(color="gray", width=1, dash="dash")
                )
            
            fig.update_layout(
                title=f"SHAP Waterfall Plot - Instance {instance_idx}",
                xaxis_title="Features",
                yaxis_title="Prediction Value",
                xaxis=dict(
                    tickmode='array',
                    tickvals=list(range(len(waterfall_df))),
                    ticktext=[feat[:15] + '...' if len(feat) > 15 else feat for feat in waterfall_df['Feature']],
                    tickangle=45
                ),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**📊 Waterfall Details**")
            
            # Show the waterfall breakdown
            display_df = waterfall_df.copy()
            display_df['SHAP_Value'] = display_df['SHAP_Value'].round(4)
            display_df['Cumulative'] = display_df['Cumulative'].round(4)
            
            st.dataframe(display_df[['Feature', 'SHAP_Value', 'Cumulative']], use_container_width=True)
            
            # Summary
            st.write("**Summary:**")
            st.metric("Base Value", f"{base_value:.4f}")
            st.metric("Final Prediction", f"{final_prediction:.4f}")
            st.metric("Total Change", f"{final_prediction - base_value:.4f}")
            
            # Top contributors
            contrib_df = waterfall_df[waterfall_df['Type'].isin(['Positive', 'Negative'])].copy()
            contrib_df['Abs_SHAP'] = np.abs(contrib_df['SHAP_Value'])
            top_contrib = contrib_df.nlargest(3, 'Abs_SHAP')
            
            st.write("**🏆 Top Contributors:**")
            for _, row in top_contrib.iterrows():
                direction = "↑" if row['SHAP_Value'] > 0 else "↓"
                st.write(f"{direction} **{row['Feature']}**: {row['SHAP_Value']:.4f}")
        
    except Exception as e:
        st.error(f"Error in waterfall analysis: {str(e)}")

# Additional SHAP utilities
def _create_shap_summary_plot(shap_values, X_sample):
    """Create a summary plot showing SHAP value distributions."""
    try:
        # This would typically use shap.summary_plot, but we'll create a custom version
        # Get feature importance
        if hasattr(shap_values, 'values'):
            shap_vals = shap_values.values
        else:
            shap_vals = shap_values
            
        if len(shap_vals.shape) == 3:  # Multi-class
            shap_vals = np.abs(shap_vals).mean(axis=2)
        
        feature_importance = np.abs(shap_vals).mean(axis=0)
        
        # Get feature names
        if hasattr(shap_values, 'feature_names') and shap_values.feature_names is not None:
            feature_names = shap_values.feature_names
        else:
            feature_names = [f'Feature_{i}' for i in range(len(feature_importance))]
        
        # Create violin plot for top features
        top_indices = np.argsort(feature_importance)[-10:]
        
        fig = go.Figure()
        
        for i, idx in enumerate(top_indices):
            fig.add_trace(go.Violin(
                y=shap_vals[:, idx],
                name=feature_names[idx],
                box_visible=True,
                meanline_visible=True
            ))
        
        fig.update_layout(
            title="SHAP Values Distribution for Top Features",
            yaxis_title="SHAP Value",
            height=600
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating SHAP summary plot: {e}")
        return None