"""
Simplified Prediction Page - SHAP Removed
Handles single and batch predictions only
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

def page_predictions(predictor):
    """Simplified predictions page - Single and Batch only."""
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
    st.markdown("*Make single and batch predictions with your trained model*")
    
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
    
    # Main tabs - Only Single and Batch predictions
    tab1, tab2 = st.tabs(["🎯 Single Prediction", "📊 Batch Predictions"])
    
    with tab1:
        _handle_single_prediction(predictor)
    
    with tab2:
        _handle_batch_prediction(predictor)
    
    # Next step button
    st.markdown("---")
    if st.button("🧠 Proceed to XAI Analysis", type="primary", use_container_width=True):
        st.session_state.current_step = "xai"
        st.rerun()

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
                
                # Display result with enhanced styling
                st.markdown("---")
                st.markdown("### 🎯 Prediction Result")
                
                col1, col2 = st.columns(2)
                with col1:
                    # Main prediction result
                    if isinstance(prediction, (int, float)):
                        if st.session_state.problem_type == "regression":
                            st.markdown(f"""
                            <div style="
                                background: linear-gradient(135deg, #4CAF50, #45a049);
                                color: white;
                                padding: 20px;
                                border-radius: 10px;
                                text-align: center;
                                margin: 10px 0;
                            ">
                                <h2 style="margin: 0;">🎯 Predicted Value</h2>
                                <h1 style="margin: 10px 0; font-size: 2.5em;">{prediction:.4f}</h1>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div style="
                                background: linear-gradient(135deg, #2196F3, #1976D2);
                                color: white;
                                padding: 20px;
                                border-radius: 10px;
                                text-align: center;
                                margin: 10px 0;
                            ">
                                <h2 style="margin: 0;">🎯 Predicted Class</h2>
                                <h1 style="margin: 10px 0; font-size: 2.5em;">{str(prediction)}</h1>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, #FF9800, #F57C00);
                            color: white;
                            padding: 20px;
                            border-radius: 10px;
                            text-align: center;
                            margin: 10px 0;
                        ">
                            <h2 style="margin: 0;">🎯 Prediction</h2>
                            <h1 style="margin: 10px 0; font-size: 2.5em;">{str(prediction)}</h1>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    # Additional info
                    if st.session_state.problem_type == "classification":
                        # Get prediction probability
                        try:
                            prob = predictor.predict_probability(
                                st.session_state.trained_model,
                                input_df
                            )
                            confidence_color = "#4CAF50" if prob > 0.8 else "#FF9800" if prob > 0.6 else "#FF5722"
                            confidence_emoji = "🟢" if prob > 0.8 else "🟡" if prob > 0.6 else "🟠"
                            
                            st.markdown(f"""
                            <div style="
                                background: linear-gradient(135deg, {confidence_color}, {confidence_color}CC);
                                color: white;
                                padding: 20px;
                                border-radius: 10px;
                                text-align: center;
                                margin: 10px 0;
                            ">
                                <h2 style="margin: 0;">{confidence_emoji} Confidence</h2>
                                <h1 style="margin: 10px 0; font-size: 2.5em;">{prob:.1%}</h1>
                            </div>
                            """, unsafe_allow_html=True)
                        except Exception as e:
                            st.info("Confidence score not available")
                    else:
                        # For regression, show some statistics
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, #9C27B0, #7B1FA2);
                            color: white;
                            padding: 20px;
                            border-radius: 10px;
                            text-align: center;
                            margin: 10px 0;
                        ">
                            <h2 style="margin: 0;">📊 Model Info</h2>
                            <p style="margin: 5px 0;">Problem: Regression</p>
                            <p style="margin: 5px 0;">Features: {len(feature_columns)}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Show input summary
                st.markdown("### 📋 Input Summary")
                input_summary = pd.DataFrame([input_data]).T
                input_summary.columns = ['Value']
                
                # Style the input summary
                styled_summary = input_summary.style.apply(
                    lambda x: ['background-color: #f0f0f0; font-weight: bold' if i % 2 == 0 
                              else 'background-color: white' for i in range(len(x))], axis=0
                )
                st.dataframe(styled_summary, use_container_width=True)
            
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
                        
                        # Display results with enhanced styling
                        st.markdown("### 📊 Prediction Results")
                        
                        # Style the results
                        if st.session_state.problem_type == "classification" and 'Confidence' in results_df.columns:
                            # Color code confidence scores
                            def highlight_confidence(val):
                                if pd.isna(val):
                                    return ''
                                if val > 0.8:
                                    return 'background-color: #d4edda; color: #155724'  # Green
                                elif val > 0.6:
                                    return 'background-color: #fff3cd; color: #856404'  # Yellow
                                else:
                                    return 'background-color: #f8d7da; color: #721c24'  # Red
                            
                            styled_results = results_df.style.applymap(
                                highlight_confidence, subset=['Confidence']
                            ).format({'Confidence': '{:.1%}'})
                            
                            st.dataframe(styled_results, use_container_width=True)
                        else:
                            st.dataframe(results_df, use_container_width=True)
                        
                        # Show prediction summary
                        _show_prediction_summary(predictions)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    except Exception as e:
                        st.error(f"❌ Batch prediction error: {str(e)}")
                        logger.error(f"Batch prediction error: {e}")
        
        except Exception as e:
            st.error(f"❌ Error loading batch file: {str(e)}")
    
    else:
        _show_batch_prediction_template()

def _show_prediction_summary(predictions):
    """Show prediction summary with enhanced visualizations."""
    st.subheader("📈 Prediction Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Basic statistics
        st.markdown("#### 📊 Statistics")
        total_predictions = len(predictions)
        st.metric("Total Predictions", f"{total_predictions:,}")
        
        if st.session_state.problem_type == "classification":
            unique_predictions = len(np.unique(predictions))
            st.metric("Unique Classes", unique_predictions)
            
            # Most common prediction
            pred_counts = pd.Series(predictions).value_counts()
            most_common = pred_counts.index[0]
            most_common_count = pred_counts.iloc[0]
            st.metric("Most Common Class", f"{most_common} ({most_common_count})")
            
        else:
            pred_mean = np.mean(predictions)
            pred_std = np.std(predictions)
            st.metric("Mean Prediction", f"{pred_mean:.4f}")
            st.metric("Std Deviation", f"{pred_std:.4f}")
    
    with col2:
        # Visualization
        if st.session_state.problem_type == "classification":
            pred_counts = pd.Series(predictions).value_counts()
            
            fig = px.pie(
                values=pred_counts.values, 
                names=pred_counts.index, 
                title="Prediction Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
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
                yaxis_title="Frequency",
                height=400
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
    
    # Enhanced format display
    format_data = []
    for col in feature_columns:
        col_dtype = str(sample_data[col].dtype)
        sample_val = str(sample_data[col].iloc[0]) if len(sample_data) > 0 else "N/A"
        unique_count = sample_data[col].nunique()
        
        format_data.append({
            'Column Name': col,
            'Data Type': col_dtype,
            'Sample Value': sample_val[:20] + "..." if len(sample_val) > 20 else sample_val,
            'Unique Values': unique_count
        })
    
    format_df = pd.DataFrame(format_data)
    
    # Style the format table
    styled_format = format_df.style.apply(
        lambda x: ['background-color: #f8f9fa' if i % 2 == 0 else 'background-color: white' 
                  for i in range(len(x))], axis=0
    )
    st.dataframe(styled_format, use_container_width=True)
    
    # Download template
    st.markdown("#### 📥 Download Template")
    if st.button("📄 Download Template", use_container_width=True):
        template_df = pd.DataFrame(columns=feature_columns)
        # Add sample rows with example data
        if len(sample_data) > 0:
            sample_rows = []
            for i in range(min(3, len(sample_data))):  # Add 3 sample rows
                sample_row = {}
                for col in feature_columns:
                    if pd.api.types.is_numeric_dtype(sample_data[col]):
                        sample_row[col] = sample_data[col].iloc[i] if not pd.isna(sample_data[col].iloc[i]) else sample_data[col].mean()
                    else:
                        sample_row[col] = sample_data[col].iloc[i] if not pd.isna(sample_data[col].iloc[i]) else "example"
                sample_rows.append(sample_row)
            
            template_df = pd.concat([template_df, pd.DataFrame(sample_rows)], ignore_index=True)
        
        csv_template = template_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV Template",
            data=csv_template,
            file_name="prediction_template.csv",
            mime="text/csv",
            use_container_width=True
        )