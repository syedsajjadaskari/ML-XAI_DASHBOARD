"""
Data Upload Page
Handles file upload and data preview with reset functionality
"""

import streamlit as st
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def page_data_upload(data_handler):
    """Data upload page with reset functionality."""
    st.header("📁 Data Upload")
    
    # Reset button at the top
    col_reset, col_spacer = st.columns([1, 4])
    with col_reset:
        if st.button("🔄 Reset All Data", use_container_width=True, type="secondary"):
            # Clear all session state data
            keys_to_clear = ['data', 'target_column', 'problem_type', 'preview_data', 'preprocessing_config', 
                           'trained_model', 'fast_trainer', 'fast_training_results', 'columns_to_remove']
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("✅ All data cleared!")
            st.rerun()
    
    # Show current data status if exists
    if st.session_state.data is not None:
        st.info(f"📊 Current data: {st.session_state.data.shape[0]:,} rows, {st.session_state.data.shape[1]} columns")
        
        # Quick preview of current data
        with st.expander("👀 Current Data Preview"):
            st.dataframe(st.session_state.data.head(), use_container_width=True)
    
    # File upload section
    st.subheader("📂 Upload New File")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'xls', 'parquet'],
        help="Supported formats: CSV, Excel, Parquet (Max 200MB)"
    )
    
    if uploaded_file is not None:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        
        # Show file info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("File Size", f"{file_size_mb:.1f} MB")
        with col2:
            storage_method = "Temporary Storage (GCS)" if file_size_mb > 50 else "Memory"
            st.metric("Storage Method", storage_method)
        with col3:
            estimated_time = "2-5 minutes" if file_size_mb > 50 else "< 1 minute"
            st.metric("Est. Processing Time", estimated_time)
        
        # Warning for large files
        if file_size_mb > 100:
            st.warning(f"⚠️ Large file detected ({file_size_mb:.1f}MB). Processing may take 3-5 minutes.")
        
        # Progress tracking
        if st.button("🚀 Process File", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Upload phase
                if file_size_mb > 50:
                    status_text.text("📤 Uploading to temporary storage...")
                    progress_bar.progress(25)
                
                # Processing phase
                status_text.text("🔄 Processing data...")
                progress_bar.progress(50)
                
                data = data_handler.load_data(uploaded_file)
                progress_bar.progress(75)
                
                # Validation phase
                status_text.text("✅ Validating data...")
                st.session_state.data = data
                progress_bar.progress(100)
                
                status_text.text("🎉 File processed successfully!")
                
                # Show storage info
                if file_size_mb > 50:
                    st.info(f"📊 File stored in temporary storage. Will be automatically cleaned up in 7 days.")
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                progress_bar.empty()
                status_text.empty()
    
    # Sample data section
    elif st.session_state.data is None:  # Only show if no data loaded
        st.markdown("---")
        st.subheader("🎲 Try Sample Data")
        st.info("👆 Upload your dataset or try sample data below")
        
        sample_datasets = {
            "Titanic (Classification)": {
                "name": "titanic", 
                "target": "Survived",
                "description": "Predict passenger survival on Titanic"
            },
            "Boston Housing (Regression)": {
                "name": "boston", 
                "target": "medv",
                "description": "Predict house prices in Boston"
            },
            "Diabetes (Regression)": {
                "name": "diabetes", 
                "target": "target",
                "description": "Predict diabetes progression"
            },
            "Wine Quality (Classification)": {
                "name": "wine", 
                "target": "quality",
                "description": "Predict wine quality scores"
            }
        }
        
        # Sample dataset selection with descriptions
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_sample = st.selectbox(
                "Choose sample dataset:", 
                list(sample_datasets.keys()),
                help="Select a sample dataset to explore the application"
            )
            
            # Show description
            if selected_sample:
                description = sample_datasets[selected_sample]["description"]
                st.info(f"📖 {description}")
        
        with col2:
            if st.button("Load Sample Data", type="primary", use_container_width=True):
                try:
                    dataset_info = sample_datasets[selected_sample]
                    dataset_name = dataset_info["name"]
                    target_col = dataset_info["target"]
                    
                    with st.spinner(f"Loading {selected_sample}..."):
                        data = data_handler.load_sample_data(dataset_name)
                        st.session_state.data = data
                        
                        # Auto-set target column for sample data
                        if target_col in data.columns:
                            st.session_state.target_column = target_col
                            st.session_state.problem_type = data_handler.detect_problem_type(data, target_col)
                    
                    st.success(f"✅ {selected_sample} loaded successfully!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error loading sample data: {str(e)}")
                    logger.error(f"Sample data loading error: {e}")
    
    # Show data information and navigation if data exists
    if st.session_state.data is not None:
        _show_data_info_and_preview(st.session_state.data)
        
        # Navigation button
        st.markdown("---")
        if st.button("🔍 Proceed to Data Exploration", type="primary", use_container_width=True):
            st.session_state.current_step = "explore"
            st.rerun()

def _show_data_info_and_preview(data):
    """Show data information and preview."""
    st.markdown("---")
    st.subheader("📊 Dataset Information")
    
    # Basic metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", f"{data.shape[0]:,}")
    with col2:
        st.metric("Columns", f"{data.shape[1]:,}")
    with col3:
        missing_pct = (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100
        st.metric("Missing %", f"{missing_pct:.1f}%")
    with col4:
        memory_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
        st.metric("Memory", f"{memory_mb:.1f} MB")
    
    # Data preview
    st.subheader("📋 Data Preview")
    st.dataframe(data.head(10), use_container_width=True)
    
    # Column information
    st.subheader("📋 Column Details")
    try:
        from src.data_handler import DataHandler
        data_handler = DataHandler({})
        col_info = data_handler.get_column_info(data)
        st.dataframe(col_info, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not generate column info: {e}")
        
        # Fallback: simple column info
        col_info_simple = pd.DataFrame({
            'Column': data.columns,
            'Type': data.dtypes.astype(str),
            'Non-Null': data.count(),
            'Null': data.isnull().sum(),
            'Unique': data.nunique()
        })
        st.dataframe(col_info_simple, use_container_width=True)
    
    # Target column suggestion if not set
    if st.session_state.target_column is None:
        st.info("💡 **Next Step:** Go to Data Exploration to select your target column (what you want to predict)")
    else:
        st.success(f"🎯 **Target Column:** {st.session_state.target_column} ({st.session_state.problem_type})")