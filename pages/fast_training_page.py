"""
Fast Training Only Page - Updated with proper data passing to evaluation
Focuses on fast model training with comparison option
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
    """Fast training only page with model comparison and proper data storage."""
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first")
        return
    
    st.header("🎯 Fast Model Training")
    st.markdown("*Train machine learning models in seconds with comprehensive evaluation data*")
    
    # Back navigation
    col_back, col_title = st.columns([1, 4])
    with col_back:
        if st.button("⬅️ Back to Preprocessing", use_container_width=True):
            st.session_state.current_step = "preprocess"
            st.rerun()
    
    # Target column selection (if not already selected)
    if st.session_state.target_column is None:
        st.warning("⚠️ Please select your target column first")
        _show_target_selection(data_handler)
        return
    
    # Target column selected, show training options
    st.success(f"🎯 **Target:** {st.session_state.target_column} ({st.session_state.problem_type})")
    
    # Option to change target
    with st.expander("🔄 Change Target Column"):
        new_target = st.selectbox(
            "Select different target:",
            options=st.session_state.data.columns.tolist(),
            index=st.session_state.data.columns.tolist().index(st.session_state.target_column),
            key="change_target"
        )
        
        if st.button("Update Target Column"):
            st.session_state.target_column = new_target
            st.session_state.problem_type = data_handler.detect_problem_type(st.session_state.data, new_target)
            st.success(f"✅ Target updated to: {new_target}")
            st.rerun()
    
    # Training configuration
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("⚙️ Training Configuration")
        
        # Model comparison option
        compare_all_models = st.checkbox(
            "🏆 Compare All Models", 
            value=True,
            help="Train and compare multiple ML algorithms to find the best one"
        )
        
        # Training method
        if compare_all_models:
            training_method = st.selectbox(
                "Training method:",
                ["auto", "fast_sklearn", "flaml"] if FLAML_AVAILABLE else ["auto", "fast_sklearn"],
                format_func=lambda x: {
                    "auto": "🔥 Auto-Select (Recommended)",
                    "fast_sklearn": "⚡ Fast Scikit-learn", 
                    "flaml": "🚀 FLAML AutoML"
                }.get(x, x),
                help="Auto-select chooses the best method for your data size"
            )
        else:
            # Single model selection
            available_models = [
                "rf_fast", "lr", "nb", "knn_fast", "dt_fast", "gb_fast"
            ]
            if FLAML_AVAILABLE:
                available_models.extend(["lgb", "xgb"])
            
            single_model = st.selectbox(
                "Select single model:",
                available_models,
                format_func=lambda x: {
                    "rf_fast": "🌲 Random Forest (Fast)",
                    "lr": "📈 Logistic/Linear Regression", 
                    "nb": "🎯 Naive Bayes",
                    "knn_fast": "👥 K-Nearest Neighbors",
                    "dt_fast": "🌳 Decision Tree",
                    "gb_fast": "🚀 Gradient Boosting",
                    "lgb": "⚡ LightGBM",
                    "xgb": "🔥 XGBoost"
                }.get(x, x)
            )
        
        # Advanced settings
        with st.expander("⚙️ Advanced Settings"):
            col_a, col_b = st.columns(2)
            with col_a:
                time_budget = st.slider("Time budget (seconds):", 10, 300, 60)
                test_size = st.slider("Test set size:", 0.1, 0.4, 0.2, 0.05)
            with col_b:
                cv_folds = st.slider("Cross-validation folds:", 2, 5, 3)
                random_state = st.number_input("Random state:", value=42, min_value=0)
    
    with col2:
        st.subheader("📊 Data Info")
        
        data_size = len(st.session_state.data)
        n_features = len(st.session_state.data.columns) - 1
        
        st.metric("Dataset Size", f"{data_size:,} rows")
        st.metric("Features", f"{n_features} columns")
        st.metric("Target", st.session_state.target_column)
        st.metric("Problem Type", st.session_state.problem_type.title())
        
        # Estimated time
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
    
    # Training button
    if st.button("🚀 Start Fast Training", type="primary", use_container_width=True):
        if compare_all_models:
            _run_model_comparison(
                data_handler, training_method, time_budget, 
                test_size, cv_folds, random_state
            )
        else:
            _run_single_model_training(
                data_handler, single_model, test_size, cv_folds, random_state
            )
    
    # Show training results
    if hasattr(st.session_state, 'fast_training_results'):
        _display_comprehensive_results()
    
    # Model actions
    if st.session_state.get('trained_model') is not None:
        _show_model_actions()

def _show_target_selection(data_handler):
    """Show target column selection interface."""
    with st.container():
        st.subheader("🎯 Target Column Selection")
        st.info("Select the column you want to predict (target variable)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            target_column = st.selectbox(
                "Choose target column:",
                options=[""] + st.session_state.data.columns.tolist(),
                help="Select the column you want to predict",
                key="target_selection_train"
            )
            
            if target_column and target_column != "":
                st.session_state.target_column = target_column
                
                # Auto-detect problem type
                problem_type = data_handler.detect_problem_type(st.session_state.data, target_column)
                st.session_state.problem_type = problem_type
                
                st.success(f"✅ Target column set: {target_column}")
                st.rerun()
        
        with col2:
            if target_column and target_column != "":
                # Show target column info
                target_info = st.session_state.data[target_column]
                
                st.write("**Target Column Info:**")
                st.write(f"- Type: {target_info.dtype}")
                st.write(f"- Unique values: {target_info.nunique()}")
                st.write(f"- Missing values: {target_info.isnull().sum()}")
                
                # Auto-detect and allow manual override
                problem_type = data_handler.detect_problem_type(st.session_state.data, target_column)
                
                selected_problem_type = st.selectbox(
                    "Problem type:",
                    ["classification", "regression"],
                    index=0 if problem_type == "classification" else 1,
                    help="Classification for categories, Regression for continuous values"
                )
                
                if selected_problem_type != problem_type:
                    st.session_state.problem_type = selected_problem_type
                
                if problem_type == 'classification':
                    unique_vals = target_info.unique()
                    if len(unique_vals) <= 10:
                        st.write(f"- Classes: {list(unique_vals)}")
                    else:
                        st.write(f"- Classes: {len(unique_vals)} unique classes")
                else:
                    st.write(f"- Range: {target_info.min():.2f} to {target_info.max():.2f}")
            else:
                st.info("""
                **Choose your target column:**
                - For **Classification**: Categories, Yes/No, etc.
                - For **Regression**: Continuous numbers, prices, etc.
                """)

def _run_model_comparison(data_handler, method, time_budget, test_size, cv_folds, random_state):
    """Run comprehensive model comparison with proper data storage."""
    try:
        # Prepare data
        training_data = _prepare_training_data(data_handler)
        if training_data is None:
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🔧 Initializing trainer...")
        progress_bar.progress(10)
        
        # Initialize trainer based on method
        if method == "auto":
            trainer = HybridFastTrainer(st.session_state.get('config', {}))
        elif method == "flaml" and FLAML_AVAILABLE:
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
        status_text.text("🏆 Comparing models...")
        
        # Run model comparison
        if method == "auto":
            results = trainer.train_lightning_fast(time_budget)
            comparison_results = results.get('comparison_results', pd.DataFrame())
        elif method == "flaml":
            results = trainer.train_ultra_fast(time_budget)
            comparison_results = _create_flaml_comparison(results)
        else:
            comparison_results = trainer.compare_models_fast(cv_folds=cv_folds, timeout=time_budget)
            if len(comparison_results) > 0:
                best_model = trainer.train_best_model(comparison_results)
                results = {
                    'best_model': best_model,
                    'comparison_results': comparison_results,
                    'trainer': trainer
                }
            else:
                st.warning("⚠️ No models completed, training single fast model")
                best_model = trainer.train_single_model('rf_fast')
                results = {'best_model': best_model, 'trainer': trainer}
                comparison_results = pd.DataFrame()
        
        progress_bar.progress(100)
        total_time = time.time() - start_time
        
        # Store comprehensive results with trainer
        st.session_state.fast_training_results = {
            **results,
            'comparison_results': comparison_results,
            'training_time': total_time,
            'trainer_type': method,
            'models_compared': len(comparison_results) if len(comparison_results) > 0 else 1
        }
        st.session_state.trained_model = results['best_model']
        st.session_state.fast_trainer = trainer  # IMPORTANT: Store trainer for evaluation
        
        status_text.text("✅ Training completed!")
        st.success(f"🏆 Model comparison completed in {total_time:.1f} seconds!")
        
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Training error: {str(e)}")
        logger.error(f"Model comparison error: {e}")
        
        # Show debugging info
        with st.expander("🔧 Debug Information"):
            st.code(str(e))
            st.write("**Data Shape:**", training_data.shape if 'training_data' in locals() else "Unknown")
            st.write("**Target Column:**", st.session_state.target_column)
            st.write("**Problem Type:**", st.session_state.problem_type)

def _run_single_model_training(data_handler, model_name, test_size, cv_folds, random_state):
    """Run single model training with proper data storage."""
    try:
        training_data = _prepare_training_data(data_handler)
        if training_data is None:
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Initialize fast trainer
        trainer = FastModelTrainer(st.session_state.get('config', {}))
        
        status_text.text("⚡ Setting up environment...")
        progress_bar.progress(30)
        
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
        
        progress_bar.progress(60)
        status_text.text(f"🎯 Training {model_name}...")
        
        # Train single model
        model = trainer.train_single_model(model_name)
        
        progress_bar.progress(100)
        total_time = time.time() - start_time
        
        # Store results with trainer
        st.session_state.fast_training_results = {
            'best_model': model,
            'trainer': trainer,
            'training_time': total_time,
            'trainer_type': 'single_model',
            'model_name': model_name
        }
        st.session_state.trained_model = model
        st.session_state.fast_trainer = trainer  # IMPORTANT: Store trainer for evaluation
        
        status_text.text("✅ Training completed!")
        st.success(f"🎯 {model_name} trained in {total_time:.1f} seconds!")
        
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Training error: {str(e)}")
        logger.error(f"Single model training error: {e}")

def _prepare_training_data(data_handler):
    """Prepare data for training with validation."""
    try:
        if hasattr(st.session_state, 'preview_data') and st.session_state.preview_data is not None:
            training_data = st.session_state.preview_data.copy()
            st.info("ℹ️ Using preprocessed data")
        else:
            training_data = st.session_state.data.copy()
            st.info("ℹ️ Using original data with basic preprocessing")
            
            # Apply basic preprocessing
            with st.spinner("Applying quick preprocessing..."):
                training_data = data_handler.apply_preprocessing(
                    training_data,
                    st.session_state.target_column,
                    st.session_state.get('preprocessing_config', {})
                )
        
        # Validate data
        if st.session_state.target_column not in training_data.columns:
            st.error("❌ Target column missing from data!")
            return None
        
        if len(training_data.columns) < 2:
            st.error("❌ Need at least 2 columns for training!")
            return None
        
        # Check for sufficient data
        if len(training_data) < 10:
            st.error("❌ Need at least 10 rows for training!")
            return None
        
        # Additional validation
        target_nulls = training_data[st.session_state.target_column].isnull().sum()
        if target_nulls > len(training_data) * 0.5:
            st.error(f"❌ Target column has too many missing values: {target_nulls}")
            return None
        
        return training_data
        
    except Exception as e:
        st.error(f"❌ Data preparation error: {str(e)}")
        logger.error(f"Data preparation error: {e}")
        return None

def _create_flaml_comparison(flaml_results):
    """Create comparison results from FLAML output."""
    try:
        return pd.DataFrame([{
            'Model': flaml_results.get('best_estimator', 'FLAML_AutoML'),
            'Score': -flaml_results.get('best_loss', 0),
            'Time (s)': flaml_results.get('training_time', 0),
            'Method': 'FLAML'
        }])
    except:
        return pd.DataFrame()

def _display_comprehensive_results():
    """Display comprehensive training results with enhanced visuals."""
    results = st.session_state.fast_training_results
    
    st.subheader("🏆 Training Results")
    
    # Summary metrics with beautiful cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        training_time = results.get('training_time', 0)
        _create_result_card("⏱️ Training Time", f"{training_time:.1f}s", "Total time elapsed", "#4CAF50")
    
    with col2:
        models_compared = results.get('models_compared', 1)
        _create_result_card("🔬 Models Tested", str(models_compared), "Algorithms evaluated", "#2196F3")
    
    with col3:
        trainer_type = results.get('trainer_type', 'unknown')
        _create_result_card("🛠️ Method", trainer_type.replace('_', ' ').title(), "Training approach", "#FF9800")
    
    with col4:
        # Best score
        if 'comparison_results' in results and len(results['comparison_results']) > 0:
            best_score = results['comparison_results'].iloc[0]['Score']
            _create_result_card("🏆 Best Score", f"{best_score:.4f}", "Top performance", "#9C27B0")
        elif 'best_loss' in results:
            _create_result_card("🏆 Best Score", f"{-results['best_loss']:.4f}", "Top performance", "#9C27B0")
        else:
            _create_result_card("✅ Status", "Complete", "Training finished", "#4CAF50")
    
    # Model comparison table with styling
    if 'comparison_results' in results and len(results['comparison_results']) > 0:
        st.subheader("📊 Model Comparison")
        
        comparison_df = results['comparison_results'].copy()
        
        # Add performance indicators
        if 'Time (s)' in comparison_df.columns:
            comparison_df['Speed'] = comparison_df['Time (s)'].apply(
                lambda x: "🟢 Fast" if x < 5 else "🟡 Medium" if x < 15 else "🟠 Slow"
            )
        
        # Add rank column
        comparison_df['Rank'] = range(1, len(comparison_df) + 1)
        comparison_df = comparison_df[['Rank', 'Model', 'Score', 'Std', 'Time (s)', 'Speed']]
        
        # Display with styling
        st.dataframe(
            comparison_df.style.format({'Score': '{:.4f}', 'Std': '{:.4f}', 'Time (s)': '{:.1f}'})
            .background_gradient(subset=['Score'], cmap='Greens')
            .highlight_max(subset=['Score'], color='lightgreen'),
            use_container_width=True
        )
        
        # Highlight best model
        if len(comparison_df) > 0:
            best_model_name = comparison_df.iloc[0]['Model']
            best_score = comparison_df.iloc[0]['Score']
            best_time = comparison_df.iloc[0].get('Time (s)', 0)
            
            st.success(f"🏆 **Winner:** {best_model_name} | Score: {best_score:.4f} | Time: {best_time:.1f}s")
    
    # Performance visualization
    if 'comparison_results' in results and len(results['comparison_results']) > 0:
        _create_performance_charts(results['comparison_results'])

def _create_result_card(title, value, description, color):
    """Create a beautiful result card."""
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {color}15, {color}05);
        border: 2px solid {color}66;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    ">
        <h4 style="color: {color}; margin: 0; font-size: 14px; font-weight: 600;">{title}</h4>
        <h2 style="color: {color}; margin: 10px 0; font-size: 24px; font-weight: 700;">{value}</h2>
        <p style="color: #666; margin: 0; font-size: 11px; opacity: 0.8;">{description}</p>
    </div>
    """, unsafe_allow_html=True)

def _create_performance_charts(comparison_df):
    """Create beautiful performance visualization charts."""
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        if len(comparison_df) < 2:
            return
        
        st.subheader("📈 Performance Visualization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Score comparison with enhanced styling
            fig1 = px.bar(
                comparison_df.head(10),
                x='Model',
                y='Score',
                title='🏆 Model Performance Comparison',
                color='Score',
                color_continuous_scale='viridis',
                text='Score'
            )
            fig1.update_traces(texttemplate='%{text:.3f}', textposition='auto')
            fig1.update_xaxes(tickangle=45)
            fig1.update_layout(
                height=400,
                title_font_size=16,
                showlegend=False
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Speed vs Performance scatter plot
            if 'Time (s)' in comparison_df.columns:
                fig2 = px.scatter(
                    comparison_df.head(10),
                    x='Time (s)',
                    y='Score',
                    size='Score',
                    color='Score',
                    hover_name='Model',
                    title='⚡ Speed vs Performance',
                    color_continuous_scale='plasma'
                )
                fig2.update_layout(
                    height=400,
                    title_font_size=16
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                # Alternative: Model ranking
                fig2 = px.pie(
                    comparison_df.head(5),
                    values='Score',
                    names='Model',
                    title='🥧 Top 5 Models Distribution'
                )
                fig2.update_layout(height=400, title_font_size=16)
                st.plotly_chart(fig2, use_container_width=True)
    
    except Exception as e:
        logger.warning(f"Could not create performance charts: {e}")

def _show_model_actions():
    """Show enhanced model action buttons."""
    st.subheader("🎯 Next Steps")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📊 Evaluate Model", use_container_width=True, type="primary"):
            st.session_state.current_step = "evaluate"
            st.rerun()
    
    with col2:
        if st.button("🔮 Make Predictions", use_container_width=True):
            st.session_state.current_step = "predict"
            st.rerun()
    
    with col3:
        # Save model with enhanced interface
        model_name = st.text_input(
            "Model name:",
            value=f"fast_model_{datetime.now().strftime('%H%M%S')}",
            key="fast_model_name"
        )
        
        if st.button("💾 Save Model", use_container_width=True):
            try:
                import joblib
                from pathlib import Path
                
                models_dir = Path("models")
                models_dir.mkdir(exist_ok=True)
                
                model_path = models_dir / f"{model_name}.joblib"
                
                # Save comprehensive model data
                save_data = {
                    'model': st.session_state.trained_model,
                    'trainer': st.session_state.get('fast_trainer'),
                    'preprocessing_config': st.session_state.get('preprocessing_config', {}),
                    'problem_type': st.session_state.problem_type,
                    'target_column': st.session_state.target_column,
                    'training_results': st.session_state.get('fast_training_results', {}),
                    'feature_columns': [col for col in st.session_state.data.columns 
                                      if col != st.session_state.target_column],
                    'data_shape': st.session_state.data.shape,
                    'saved_at': datetime.now().isoformat()
                }
                
                joblib.dump(save_data, model_path)
                st.success(f"✅ Model saved as {model_name}")
                
                # Show save confirmation
                st.info(f"📁 Saved to: {model_path}")
                
            except Exception as e:
                st.error(f"❌ Error saving model: {str(e)}")
                logger.error(f"Model saving error: {e}")
    
    with col4:
        if st.button("🔄 Train New Model", use_container_width=True):
            # Clear current results with confirmation
            if st.session_state.get('trained_model') is not None:
                # Add a confirmation step
                if st.button("⚠️ Confirm Reset", use_container_width=True, type="secondary"):
                    _reset_training_session()
                    st.rerun()
            else:
                _reset_training_session()
                st.rerun()

def _reset_training_session():
    """Reset training session safely."""
    keys_to_clear = [
        'fast_training_results', 'trained_model', 'fast_trainer'
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    
    st.success("✅ Training session reset!")