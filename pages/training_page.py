"""
Model Training Page
Handles ML model training and comparison
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def page_model_training(model_trainer, data_handler):
    """Model training page."""
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first")
        return
    
    if st.session_state.target_column is None:
        st.warning("⚠️ Please select target column first")
        return
    
    st.header("🎯 Model Training")
    
    # Training configuration
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔧 Training Configuration")
        
        # Cross-validation folds
        cv_folds = st.slider("Cross-validation folds", 2, 10, 5)
        
        # Train-test split
        test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
        
        # Random state
        random_state = st.number_input("Random state", value=42, min_value=0)
    
    with col2:
        st.subheader("🎮 Model Selection")
        
        # Model comparison
        compare_models = st.checkbox("Compare all models", value=True, help="Compare multiple algorithms")
        
        # Specific models
        if not compare_models:
            available_models = model_trainer.get_available_models(st.session_state.problem_type)
            selected_models = st.multiselect(
                "Select models to train",
                available_models.keys(),
                default=list(available_models.keys())[:3]
            )
    
    # Start training
    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        try:
            # Use preprocessed data if available, otherwise use original
            if hasattr(st.session_state, 'preview_data') and st.session_state.preview_data is not None:
                training_data = st.session_state.preview_data.copy()
                st.info("ℹ️ Using preprocessed data for training")
            else:
                training_data = st.session_state.data.copy()
                
                # Apply basic preprocessing
                with st.spinner("Applying preprocessing..."):
                    training_data = data_handler.apply_preprocessing(
                        training_data,
                        st.session_state.target_column,
                        st.session_state.preprocessing_config
                    )
            
            # Validate data before training
            if st.session_state.target_column not in training_data.columns:
                st.error("❌ Target column was removed during preprocessing! Please adjust settings.")
                return
            
            if len(training_data.columns) < 2:
                st.error("❌ Need at least 2 columns (1 feature + 1 target) for training.")
                return
            
            # Additional data validation
            st.info("🔍 Validating data...")
            
            # Check data quality
            target_nulls = training_data[st.session_state.target_column].isnull().sum()
            if target_nulls > 0:
                st.warning(f"⚠️ Target column has {target_nulls} missing values")
            
            # Check for sufficient data
            if len(training_data) < 20:
                st.warning(f"⚠️ Small dataset ({len(training_data)} rows). Results may not be reliable.")
            
            # Check target distribution for classification
            if st.session_state.problem_type == 'classification':
                target_counts = training_data[st.session_state.target_column].value_counts()
                st.write("**Target Distribution:**")
                st.write(target_counts)
                
                min_class_count = target_counts.min()
                if min_class_count < 2:
                    st.error(f"❌ Some classes have less than 2 samples. Minimum: {min_class_count}")
                    return
            
            # Display final data info
            st.write(f"**Training Data Shape:** {training_data.shape}")
            st.write(f"**Features:** {len(training_data.columns) - 1}")
            st.write(f"**Target:** {st.session_state.target_column}")
            st.write(f"**Problem Type:** {st.session_state.problem_type}")
            
            with st.spinner("Setting up ML environment..."):
                # Setup PyCaret environment
                setup_success = model_trainer.setup_environment(
                    data=training_data,
                    target=st.session_state.target_column,
                    problem_type=st.session_state.problem_type,
                    preprocessing_config=st.session_state.preprocessing_config,
                    test_size=test_size,
                    cv_folds=cv_folds,
                    random_state=random_state
                )
                
                if not setup_success:
                    st.error("❌ Failed to setup ML environment.")
                    _show_debugging_info(training_data, model_trainer, test_size, cv_folds, random_state)
                    return
            
            st.success("✅ Environment setup complete!")
            
            if compare_models:
                with st.spinner("Comparing models... This may take a while."):
                    results = model_trainer.compare_models()
                    st.session_state.model_results = results
                
                st.success("✅ Model comparison complete!")
                
                # Display results
                st.subheader("📊 Model Comparison Results")
                st.dataframe(results, use_container_width=True)
                
                # Select best model
                best_model_name = results.index[0]
                st.info(f"🏆 Best model: {best_model_name}")
                
                # Train best model
                with st.spinner(f"Training {best_model_name}..."):
                    best_model = model_trainer.create_model(best_model_name)
                    st.session_state.trained_model = best_model
                
                st.success("✅ Best model trained successfully!")
            
            else:
                # Train selected models
                if 'selected_models' not in locals() or not selected_models:
                    st.error("❌ Please select at least one model to train.")
                    return
                
                trained_models = {}
                for model_name in selected_models:
                    with st.spinner(f"Training {model_name}..."):
                        model = model_trainer.create_model(model_name)
                        trained_models[model_name] = model
                
                st.session_state.trained_model = list(trained_models.values())[0]
                st.success("✅ Models trained successfully!")
            
            # Save model option
            if st.session_state.trained_model is not None:
                _show_model_save_options(model_trainer)
            
        except Exception as e:
            st.error(f"❌ Training error: {str(e)}")
            logger.error(f"Training error: {e}")
            
            # Show detailed error info
            with st.expander("🔧 Error Details"):
                st.code(str(e))
                st.write("**Common Solutions:**")
                st.write("1. Check data quality and format")
                st.write("2. Ensure target column is properly selected")
                st.write("3. Remove special characters from column names")
                st.write("4. Try with different preprocessing options")
    
    # Show current status
    if hasattr(st.session_state, 'preprocessing_config'):
        _show_configuration_summary()
    
    # Next step
    if st.session_state.trained_model is not None:
        if st.button("📊 Proceed to Model Evaluation", type="primary", use_container_width=True):
            st.session_state.current_step = "evaluate"
            st.rerun()

def _show_debugging_info(training_data, model_trainer, test_size, cv_folds, random_state):
    """Show debugging information when setup fails."""
    with st.expander("🔧 Debugging Information"):
        st.write("**Data Info:**")
        st.write(f"- Shape: {training_data.shape}")
        st.write(f"- Data types: {training_data.dtypes.value_counts().to_dict()}")
        st.write(f"- Missing values: {training_data.isnull().sum().sum()}")
        st.write(f"- Target column type: {training_data[st.session_state.target_column].dtype}")
        
        if st.session_state.problem_type == 'classification':
            st.write(f"- Unique target values: {training_data[st.session_state.target_column].nunique()}")
        
        st.write("**Possible Solutions:**")
        st.write("1. Check for special characters in column names")
        st.write("2. Ensure target column has valid values")
        st.write("3. Remove columns with all missing values")
        st.write("4. Try with minimal preprocessing options")
        
        # Option to try minimal setup
        if st.button("🔄 Try Minimal Setup"):
            with st.spinner("Trying minimal setup..."):
                minimal_config = {
                    'missing_strategy': 'drop',
                    'scaling_method': 'none',
                    'feature_selection': False,
                    'remove_outliers': False,
                    'balance_data': False,
                    'feature_engineering': False
                }
                
                setup_success = model_trainer.setup_environment(
                    data=training_data.dropna(),
                    target=st.session_state.target_column,
                    problem_type=st.session_state.problem_type,
                    preprocessing_config=minimal_config,
                    test_size=test_size,
                    cv_folds=3,
                    random_state=random_state
                )
                
                if setup_success:
                    st.success("✅ Minimal setup successful!")
                    st.rerun()
                else:
                    st.error("❌ Even minimal setup failed. Please check your data.")

def _show_model_save_options(model_trainer):
    """Show model saving options."""
    st.subheader("💾 Save Model")
    col1, col2 = st.columns(2)
    
    with col1:
        model_name = st.text_input(
            "Model name for saving", 
            value=f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    with col2:
        if st.button("💾 Save Model"):
            try:
                saved_path = model_trainer.save_model(st.session_state.trained_model, model_name)
                st.success(f"✅ Model saved as {model_name}")
            except Exception as e:
                st.error(f"❌ Error saving model: {str(e)}")

def _show_configuration_summary():
    """Show current configuration summary."""
    with st.expander("📋 Current Configuration"):
        st.write("**Data Info:**")
        st.write(f"- Shape: {st.session_state.data.shape}")
        st.write(f"- Target: {st.session_state.target_column}")
        st.write(f"- Problem Type: {st.session_state.problem_type}")
        
        st.write("**Preprocessing:**")
        config = st.session_state.preprocessing_config
        if config.get('columns_to_remove'):
            st.write(f"- Columns to remove: {len(config['columns_to_remove'])}")
        st.write(f"- Missing strategy: {config.get('missing_strategy', 'Not set')}")
        st.write(f"- Scaling: {config.get('scaling_method', 'Not set')}")