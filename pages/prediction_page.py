"""
Prediction Page
Handles single and batch predictions with back navigation
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def page_predictions(predictor):
    """Predictions page with back navigation."""
    # Back navigation
    col_back, col_title = st.columns([1, 4])
    with col_back:
        if st.button("⬅️ Back to Evaluation", use_container_width=True):
            st.session_state.current_step = "evaluate"
            st.rerun()
    
    if st.session_state.trained_model is None:
        st.warning("⚠️ Please train a model first")
        return
    
    st.header("🔮 Model Predictions")
    st.markdown("*Make predictions with your trained model*")
    
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
    
    # Prediction modes
    tab1, tab2 = st.tabs(["🎯 Single Prediction", "📊 Batch Predictions"])
    
    with tab1:
        _handle_single_prediction(predictor)
    
    with tab2:
        _handle_batch_prediction(predictor)

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
        
        submitted = st.form_submit_button("🚀 Predict", type="primary")
        
        if submitted:
            try:
                # Make prediction
                input_df = pd.DataFrame([input_data])
                prediction = predictor.predict_single(
                    st.session_state.trained_model,
                    input_df
                )
                
                # Display result
                st.success("✅ Prediction completed!")
                
                col1, col2 = st.columns(2)
                with col1:
                    if isinstance(prediction, (int, float)):
                        if st.session_state.problem_type == "regression":
                            st.metric("Predicted Value", f"{prediction:.4f}")
                        else:
                            st.metric("Predicted Class", str(prediction))
                    else:
                        st.metric("Predicted Value", str(prediction))
                
                with col2:
                    if st.session_state.problem_type == "classification":
                        # Get prediction probability
                        try:
                            prob = predictor.predict_probability(
                                st.session_state.trained_model,
                                input_df
                            )
                            st.metric("Confidence", f"{prob:.2%}")
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
                        st.subheader("📈 Prediction Summary")
                        
                        if st.session_state.problem_type == "classification":
                            pred_counts = pd.Series(predictions).value_counts()
                            st.write("**Prediction Distribution:**")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.dataframe(pred_counts, use_container_width=True)
                            with col2:
                                import plotly.express as px
                                fig = px.pie(values=pred_counts.values, names=pred_counts.index, 
                                           title="Prediction Distribution")
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.write("**Prediction Statistics:**")
                            pred_stats = pd.Series(predictions).describe()
                            st.dataframe(pred_stats, use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"❌ Batch prediction error: {str(e)}")
                        logger.error(f"Batch prediction error: {e}")
        
        except Exception as e:
            st.error(f"❌ Error loading batch file: {str(e)}")
    
    else:
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