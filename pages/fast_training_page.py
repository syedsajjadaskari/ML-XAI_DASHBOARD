"""
Lightning Fast Training Page
Ultra-fast ML training alternatives to PyCaret
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import time
import logging

# Import fast trainers
from src.fast_model_trainer import FastModelTrainer
from src.hybrid_trainer import HybridFastTrainer

try:
    from src.flaml_trainer import FLAMLTrainer
    FLAML_AVAILABLE = True
except ImportError:
    FLAML_AVAILABLE = False

logger = logging.getLogger(__name__)

def page_fast_training(data_handler):
    """Lightning-fast model training page."""
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first")
        return
    
    if st.session_state.target_column is None:
        st.warning("⚠️ Please select target column first")
        return
    
    st.header("⚡ Lightning Fast Model Training")
    st.markdown("*Train ML models in seconds, not minutes!*")
    
    # Show speed comparison
    with st.expander("🏃‍♂️ Speed Comparison"):
        speed_data = {
            'Method': ['PyCaret', 'Fast Sklearn', 'FLAML', 'H2O AutoML', 'Hybrid'],
            'Typical Time': ['2-10 minutes', '10-30 seconds', '30-60 seconds', '1-3 minutes', '5-20 seconds'],
            'Best For': ['Complete Pipeline', 'Small Data', 'Medium Data', 'Large Data', 'Auto-Select'],
            'Models Tested': ['15+', '8-12', '10+', '20+', 'Adaptive']
        }
        st.dataframe(pd.DataFrame(speed_data), use_container_width=True)
    
    # Training method selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🚀 Training Configuration")
        
        # Method selection
        available_methods = {
            'hybrid': '🔥 Hybrid (Auto-Select Best Method)',
            'fast_sklearn': '⚡ Fast Scikit-learn (Always Available)',
        }
        
        if FLAML_AVAILABLE:
            available_methods['flaml'] = '🚀 FLAML AutoML (Microsoft)'
        
        training_method = st.selectbox(
            "Select training method:",
            options=list(available_methods.keys()),
            format_func=lambda x: available_methods[x],
            help="Hybrid automatically selects the best method for your data size"
        )
        
        # Time budget
        time_budget = st.slider(
            "Time budget (seconds):",
            min_value=10,
            max_value=300,
            value=30,
            help="Maximum time to spend training models"
        )
        
        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
            cv_folds = st.slider("Cross-validation folds", 2, 5, 3)
            random_state = st.number_input("Random state", value=42, min_value=0)
            
            # Quick mode
            quick_mode = st.checkbox(
                "Quick mode (even faster)", 
                value=True,
                help="Uses faster settings for ultra-quick results"
            )
    
    with col2:
        st.subheader("📊 Data Info")
        
        # Show data size and estimated training time
        data_size = len(st.session_state.data)
        n_features = len(st.session_state.data.columns) - 1
        
        st.metric("Dataset Size", f"{data_size:,} rows")
        st.metric("Features", f"{n_features} columns")
        st.metric("Problem Type", st.session_state.problem_type.title())
        
        # Estimated training time
        if data_size < 1000:
            est_time = "5-15 seconds"
            speed_emoji = "🟢"
        elif data_size < 10000:
            est_time = "15-45 seconds" 
            speed_emoji = "🟡"
        else:
            est_time = "30-120 seconds"
            speed_emoji = "🟠"
        
        st.info(f"{speed_emoji} Estimated time: {est_time}")
        
        # Show recommended method
        if training_method == 'hybrid':
            if data_size < 1000:
                recommended = "Fast Scikit-learn"
            elif data_size < 10000 and FLAML_AVAILABLE:
                recommended = "FLAML AutoML"
            else:
                recommended = "Fast Scikit-learn"
            
            st.success(f"🎯 Recommended: {recommended}")
    
    # Start training button
    if st.button("⚡ Start Lightning Training", type="primary", use_container_width=True):
        _run_fast_training(
            data_handler, training_method, time_budget, 
            test_size, cv_folds, random_state, quick_mode
        )
    
    # Show training results if available
    if hasattr(st.session_state, 'fast_training_results'):
        _display_training_results()
    
    # Model comparison section
    if st.session_state.get('trained_model') is not None:
        _show_model_actions()

def _run_fast_training(data_handler, method, time_budget, test_size, cv_folds, random_state, quick_mode):
    """Run the fast training process."""
    try:
        # Prepare data
        if hasattr(st.session_state, 'preview_data') and st.session_state.preview_data is not None:
            training_data = st.session_state.preview_data.copy()
            st.info("ℹ️ Using preprocessed data")
        else:
            training_data = st.session_state.data.copy()
            st.info("ℹ️ Using original data (basic preprocessing applied)")
            
            # Apply basic preprocessing
            with st.spinner("Applying quick preprocessing..."):
                training_data = data_handler.apply_preprocessing(
                    training_data,
                    st.session_state.target_column,
                    st.session_state.get('preprocessing_config', {})
                )
        
        # Validate data
        if st.session_state.target_column not in training_data.columns:
            st.error("❌ Target column missing from processed data!")
            return
        
        # Initialize trainer
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🔧 Initializing trainer...")
        progress_bar.progress(10)
        
        # Create trainer based on method
        if method == 'hybrid':
            trainer = HybridFastTrainer(st.session_state.get('config', {}))
        elif method == 'flaml' and FLAML_AVAILABLE:
            trainer = FLAMLTrainer(st.session_state.get('config', {}))
        else:
            trainer = FastModelTrainer(st.session_state.get('config', {}))
        
        # Setup environment
        status_text.text("⚡ Setting up environment...")
        progress_bar.progress(20)
        
        start_time = time.time()
        
        setup_success = trainer.setup_environment(
            data=training_data,
            target=st.session_state.target_column,
            problem_type=st.session_state.problem_type,
            preprocessing_config=st.session_state.get('preprocessing_config', {}),
            test_size=test_size,
            random_state=random_state
        )
        
        if not setup_success:
            st.error("❌ Failed to setup training environment")
            return
        
        progress_bar.progress(40)
        
        # Start training
        status_text.text("🚀 Training models...")
        
        if method == 'hybrid':
            results = trainer.train_lightning_fast(time_budget)
            
        elif method == 'flaml':
            results = trainer.train_ultra_fast(time_budget)
            
        else:  # fast_sklearn
            # Quick settings for fast training
            if quick_mode:
                cv_folds = min(cv_folds, 3)
                timeout = min(time_budget, 30)
            else:
                timeout = time_budget
            
            comparison_results = trainer.compare_models_fast(cv_folds=cv_folds, timeout=timeout)
            progress_bar.progress(80)
            
            if len(comparison_results) > 0:
                best_model = trainer.train_best_model(comparison_results)
                results = {
                    'best_model': best_model,
                    'comparison_results': comparison_results,
                    'trainer': trainer
                }
            else:
                st.warning("⚠️ No models completed, using single fast model")
                best_model = trainer.train_single_model('rf_fast')
                results = {
                    'best_model': best_model,
                    'trainer': trainer
                }
        
        progress_bar.progress(100)
        total_time = time.time() - start_time
        
        # Store results
        st.session_state.fast_training_results = results
        st.session_state.trained_model = results['best_model']
        st.session_state.fast_trainer = trainer
        st.session_state.training_time = total_time
        
        status_text.text("✅ Training completed!")
        
        # Success message
        st.success(f"⚡ Lightning training completed in {total_time:.1f} seconds!")
        
        # Show quick metrics
        if hasattr(trainer, 'evaluate_model'):
            metrics = trainer.evaluate_model()
            if metrics:
                metric_cols = st.columns(len(metrics))
                for i, (metric, value) in enumerate(metrics.items()):
                    with metric_cols[i]:
                        st.metric(metric.title(), f"{value:.4f}")
        
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Training error: {str(e)}")
        logger.error(f"Fast training error: {e}")

def _display_training_results():
    """Display training results."""
    results = st.session_state.fast_training_results
    
    st.subheader("📊 Training Results")
    
    # Training summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        training_time = st.session_state.get('training_time', 0)
        st.metric("Training Time", f"{training_time:.1f}s")
    
    with col2:
        if 'comparison_results' in results:
            n_models = len(results['comparison_results'])
            st.metric("Models Tested", n_models)
        else:
            st.metric("Method", results.get('trainer_type', 'AutoML'))
    
    with col3:
        if 'best_loss' in results:
            st.metric("Best Score", f"{-results['best_loss']:.4f}")
        elif 'comparison_results' in results and len(results['comparison_results']) > 0:
            best_score = results['comparison_results'].iloc[0]['Score']
            st.metric("Best Score", f"{best_score:.4f}")
    
    # Model comparison table
    if 'comparison_results' in results and len(results['comparison_results']) > 0:
        st.subheader("🏆 Model Comparison")
        
        comparison_df = results['comparison_results']
        
        # Add speed indicators
        comparison_df['Speed'] = comparison_df['Time (s)'].apply(
            lambda x: "🟢 Fast" if x < 5 else "🟡 Medium" if x < 15 else "🟠 Slow"
        )
        
        st.dataframe(comparison_df, use_container_width=True)
        
        # Best model info
        best_model_name = comparison_df.iloc[0]['Model']
        best_score = comparison_df.iloc[0]['Score']
        best_time = comparison_df.iloc[0]['Time (s)']
        
        st.info(f"🏆 Best Model: **{best_model_name}** (Score: {best_score:.4f}, Time: {best_time:.1f}s)")
    
    # FLAML specific results
    elif 'best_estimator' in results:
        st.subheader("🚀 FLAML AutoML Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Best Model:**", results.get('best_estimator', 'Unknown'))
            st.write("**Training Time:**", f"{results.get('training_time', 0):.1f}s")
        
        with col2:
            if 'best_config' in results:
                st.write("**Best Configuration:**")
                st.json(results['best_config'])

def _show_model_actions():
    """Show model action buttons."""
    st.subheader("🎯 Model Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📊 Evaluate Model", use_container_width=True):
            st.session_state.current_step = "evaluate"
            st.rerun()
    
    with col2:
        if st.button("🔮 Make Predictions", use_container_width=True):
            st.session_state.current_step = "predict"
            st.rerun()
    
    with col3:
        # Save model
        model_name = st.text_input(
            "Model name:",
            value=f"fast_model_{datetime.now().strftime('%H%M%S')}",
            key="fast_model_name"
        )
        
        if st.button("💾 Save Model", use_container_width=True):
            try:
                # Simple model saving using joblib
                import joblib
                from pathlib import Path
                
                models_dir = Path("models")
                models_dir.mkdir(exist_ok=True)
                
                model_path = models_dir / f"{model_name}.joblib"
                
                # Save model and trainer info
                save_data = {
                    'model': st.session_state.trained_model,
                    'trainer': st.session_state.get('fast_trainer'),
                    'preprocessing_config': st.session_state.get('preprocessing_config', {}),
                    'problem_type': st.session_state.problem_type,
                    'target_column': st.session_state.target_column,
                    'training_time': st.session_state.get('training_time', 0)
                }
                
                joblib.dump(save_data, model_path)
                st.success(f"✅ Model saved as {model_name}")
                
            except Exception as e:
                st.error(f"❌ Error saving model: {str(e)}")
    
    with col4:
        if st.button("🔄 Train New Model", use_container_width=True):
            # Clear current results
            if 'fast_training_results' in st.session_state:
                del st.session_state.fast_training_results
            st.rerun()

# Add this to the main training page selection
def show_training_method_selector():
    """Show training method selector."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🚀 Training Method")
    
    method = st.sidebar.radio(
        "Choose training approach:",
        ["⚡ Lightning Fast", "🔧 PyCaret (Original)"],
        help="Lightning Fast: 10-30 seconds, PyCaret: 2-10 minutes"
    )
    
    return "fast" if method.startswith("⚡") else "pycaret"