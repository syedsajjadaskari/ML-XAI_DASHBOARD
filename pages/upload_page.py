"""
Data Upload Page
Handles file upload, data preview, and target selection
"""

import streamlit as st
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def page_data_upload(data_handler):
    """Data upload page."""
    st.header("📁 Data Upload")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'xls', 'parquet'],
        help="Supported formats: CSV, Excel, Parquet (Max 200MB)"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            with st.spinner("Loading data..."):
                data = data_handler.load_data(uploaded_file)
                st.session_state.data = data
            
            st.success(f"✅ Data loaded successfully! Shape: {data.shape}")
            
            # Display basic info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", f"{data.shape[0]:,}")
            with col2:
                st.metric("Columns", f"{data.shape[1]:,}")
            with col3:
                missing_pct = (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100
                st.metric("Missing %", f"{missing_pct:.1f}%")
            
            # Data preview
            st.subheader("📋 Data Preview")
            st.dataframe(data.head(100), use_container_width=True)
            
            # Column information
            st.subheader("📊 Column Information")
            col_info = data_handler.get_column_info(data)
            st.dataframe(col_info, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            logger.error(f"Data loading error: {e}")
            return
    
    # Target selection - Show this section if we have data
    if st.session_state.data is not None:
        st.subheader("🎯 Target Column Selection")
        st.info("Select the column you want to predict (target variable)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            target_column = st.selectbox(
                "Select target column",
                options=[""] + st.session_state.data.columns.tolist(),
                index=0 if st.session_state.target_column is None else st.session_state.data.columns.tolist().index(st.session_state.target_column) + 1,
                help="Choose the column you want to predict",
                key="target_selection"
            )
            
            if target_column and target_column != "":
                st.session_state.target_column = target_column
                
                # Auto-detect problem type
                problem_type = data_handler.detect_problem_type(st.session_state.data, target_column)
                st.session_state.problem_type = problem_type
                
                st.success(f"✅ Target column set: {target_column}")
        
        with col2:
            if st.session_state.target_column:
                # Show problem type
                problem_type_options = ['classification', 'regression']
                current_index = problem_type_options.index(st.session_state.problem_type) if st.session_state.problem_type in problem_type_options else 0
                
                selected_problem_type = st.selectbox(
                    "Problem type",
                    options=problem_type_options,
                    index=current_index,
                    help="Classification for categories, Regression for continuous values",
                    key="problem_type_selection"
                )
                st.session_state.problem_type = selected_problem_type
                
                # Show target column info
                target_info = st.session_state.data[st.session_state.target_column]
                st.write("**Target Column Info:**")
                st.write(f"- Type: {target_info.dtype}")
                st.write(f"- Unique values: {target_info.nunique()}")
                st.write(f"- Missing values: {target_info.isnull().sum()}")
                
                if st.session_state.problem_type == 'classification':
                    st.write(f"- Classes: {list(target_info.unique())[:10]}")  # Show first 10
                else:
                    st.write(f"- Range: {target_info.min():.2f} to {target_info.max():.2f}")
        
        # Next step button - only show if target is selected
        if st.session_state.target_column:
            if st.button("🔍 Proceed to Data Exploration", type="primary", use_container_width=True):
                st.session_state.current_step = "explore"
                st.rerun()
        else:
            st.warning("⚠️ Please select a target column to proceed")
                
    else:
        # Sample data option
        st.info("👆 Upload your dataset to get started")
        
        st.subheader("🎲 Or Try Sample Data")
        sample_datasets = {
            "Titanic (Classification)": ("titanic", "Survived"),
            "Boston Housing (Regression)": ("boston", "medv"),
            "Diabetes (Regression)": ("diabetes", "target"),
            "Wine Quality (Classification)": ("wine", "quality")
        }
        
        selected_sample = st.selectbox("Choose sample dataset", list(sample_datasets.keys()))
        
        if st.button("Load Sample Data"):
            try:
                dataset_name, target_col = sample_datasets[selected_sample]
                data = data_handler.load_sample_data(dataset_name)
                st.session_state.data = data
                
                # Auto-set target column for sample data
                if target_col in data.columns:
                    st.session_state.target_column = target_col
                    st.session_state.problem_type = data_handler.detect_problem_type(data, target_col)
                
                st.success("✅ Sample data loaded!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error loading sample data: {str(e)}")