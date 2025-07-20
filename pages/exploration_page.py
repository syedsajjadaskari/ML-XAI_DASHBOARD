"""
Data Exploration Page
Handles data exploration and visualization with fixed download
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import json

logger = logging.getLogger(__name__)

def page_data_exploration(visualizer):
    """Data exploration page with target selection."""
    # Back navigation
    col_back, col_title = st.columns([1, 4])
    with col_back:
        if st.button("⬅️ Back to Upload", use_container_width=True):
            st.session_state.current_step = "upload"
            st.rerun()
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first")
        return
    
    st.header("🔍 Data Exploration")
    st.markdown("*Explore your dataset and select target variable*")
    
    data = st.session_state.data
    
    # Target selection section (moved here)
    if st.session_state.target_column is None:
        st.subheader("🎯 Target Column Selection")
        st.info("First, select the column you want to predict (target variable)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            target_column = st.selectbox(
                "Select target column",
                options=[""] + data.columns.tolist(),
                help="Choose the column you want to predict",
                key="target_selection"
            )
            
            if target_column and target_column != "":
                st.session_state.target_column = target_column
                
                # Auto-detect problem type
                from src.data_handler import DataHandler
                data_handler = DataHandler({})
                problem_type = data_handler.detect_problem_type(data, target_column)
                st.session_state.problem_type = problem_type
                
                st.success(f"✅ Target column set: {target_column}")
                st.rerun()
        
        with col2:
            st.info("""
            **Choose your target column:**
            - For **Classification**: Categories, Yes/No, etc.
            - For **Regression**: Continuous numbers, prices, etc.
            """)
        
        return  # Don't show exploration until target is selected
    
    # Dataset overview
    st.subheader("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", f"{len(data):,}")
    with col2:
        st.metric("Total Columns", f"{len(data.columns):,}")
    with col3:
        missing_pct = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
        st.metric("Missing Data", f"{missing_pct:.1f}%")
    with col4:
        memory_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
        st.metric("Memory Usage", f"{memory_mb:.1f} MB")
    
    # Data types breakdown
    st.subheader("📋 Data Types")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        dtype_counts = data.dtypes.value_counts()
        fig_dtype = px.pie(
            values=dtype_counts.values,
            names=dtype_counts.index.astype(str),
            title="Data Types Distribution"
        )
        st.plotly_chart(fig_dtype, use_container_width=True)
    
    with col2:
        # Column information table
        col_info = []
        for col in data.columns:
            col_info.append({
                'Column': col,
                'Type': str(data[col].dtype),
                'Non-Null': data[col].count(),
                'Null': data[col].isnull().sum(),
                'Unique': data[col].nunique(),
                'Memory (KB)': round(data[col].memory_usage(deep=True) / 1024, 2)
            })
        
        col_df = pd.DataFrame(col_info)
        st.dataframe(col_df, use_container_width=True, height=300)
    
    # Target variable analysis
    st.subheader(f"🎯 Target Variable Analysis: {st.session_state.target_column}")
    
    target_col = st.session_state.target_column
    problem_type = st.session_state.problem_type
    
    col1, col2 = st.columns(2)
    
    with col1:
        if problem_type == 'classification':
            # Target distribution for classification
            target_counts = data[target_col].value_counts()
            fig_target = px.bar(
                x=target_counts.index,
                y=target_counts.values,
                title=f"Target Distribution: {target_col}",
                labels={'x': target_col, 'y': 'Count'}
            )
            fig_target.update_traces(text=target_counts.values, textposition='auto')
            st.plotly_chart(fig_target, use_container_width=True)
        else:
            # Target distribution for regression
            fig_target = px.histogram(
                data, 
                x=target_col, 
                nbins=30,
                title=f"Target Distribution: {target_col}"
            )
            st.plotly_chart(fig_target, use_container_width=True)
    
    with col2:
        # Target statistics
        st.write("**Target Statistics:**")
        if problem_type == 'classification':
            stats = {
                'Unique Classes': data[target_col].nunique(),
                'Most Frequent': str(data[target_col].mode().iloc[0]) if len(data[target_col].mode()) > 0 else 'N/A',
                'Missing Values': data[target_col].isnull().sum(),
                'Missing %': f"{(data[target_col].isnull().sum() / len(data)) * 100:.2f}%"
            }
        else:
            stats = {
                'Mean': f"{data[target_col].mean():.4f}",
                'Std Dev': f"{data[target_col].std():.4f}",
                'Min': f"{data[target_col].min():.4f}",
                'Max': f"{data[target_col].max():.4f}",
                'Missing Values': data[target_col].isnull().sum()
            }
        
        for key, value in stats.items():
            st.write(f"- **{key}:** {value}")
        
        # Option to change problem type
        current_type = st.session_state.problem_type
        problem_type_options = ['classification', 'regression']
        current_index = problem_type_options.index(current_type)
        
        new_problem_type = st.selectbox(
            "Problem type",
            options=problem_type_options,
            index=current_index,
            help="Classification for categories, Regression for continuous values"
        )
        
        if new_problem_type != current_type:
            st.session_state.problem_type = new_problem_type
            st.rerun()
    
    # Missing data analysis
    missing_data = data.isnull().sum()
    if missing_data.sum() > 0:
        st.subheader("❓ Missing Data Analysis")
        
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing Count': missing_data.values,
            'Missing %': (missing_data.values / len(data)) * 100
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
        
        if len(missing_df) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                fig_missing = px.bar(
                    missing_df,
                    x='Column',
                    y='Missing %',
                    title='Missing Data by Column (%)',
                    color='Missing %',
                    color_continuous_scale='Reds'
                )
                fig_missing.update_xaxes(tickangle=45)
                st.plotly_chart(fig_missing, use_container_width=True)
            
            with col2:
                st.write("**Missing Data Details:**")
                st.dataframe(missing_df, use_container_width=True)
    else:
        st.success("🎉 No missing data found!")
    
    # Numeric features analysis
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if st.session_state.target_column in numeric_cols:
        numeric_cols.remove(st.session_state.target_column)
    
    if len(numeric_cols) > 0:
        st.subheader("📈 Numeric Features Analysis")
        
        # Select feature for detailed analysis
        selected_numeric = st.selectbox(
            "Select a numeric feature to analyze:",
            numeric_cols,
            key="numeric_analysis"
        )
        
        if selected_numeric:
            col1, col2 = st.columns(2)
            
            with col1:
                # Distribution plot
                fig_dist = px.histogram(
                    data,
                    x=selected_numeric,
                    nbins=30,
                    title=f"Distribution: {selected_numeric}"
                )
                st.plotly_chart(fig_dist, use_container_width=True)
            
            with col2:
                # Box plot
                fig_box = px.box(
                    data,
                    y=selected_numeric,
                    title=f"Box Plot: {selected_numeric}"
                )
                st.plotly_chart(fig_box, use_container_width=True)
            
            # Statistics
            st.write(f"**Statistics for {selected_numeric}:**")
            stats = data[selected_numeric].describe()
            stats_cols = st.columns(len(stats))
            for i, (stat_name, stat_value) in enumerate(stats.items()):
                with stats_cols[i]:
                    st.metric(stat_name.title(), f"{stat_value:.4f}")
    
    # Categorical features analysis
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    if st.session_state.target_column in categorical_cols:
        categorical_cols.remove(st.session_state.target_column)
    
    if len(categorical_cols) > 0:
        st.subheader("🏷️ Categorical Features Analysis")
        
        selected_categorical = st.selectbox(
            "Select a categorical feature to analyze:",
            categorical_cols,
            key="categorical_analysis"
        )
        
        if selected_categorical:
            col1, col2 = st.columns(2)
            
            with col1:
                # Count plot
                value_counts = data[selected_categorical].value_counts().head(20)  # Limit to top 20
                fig_count = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    title=f"Value Counts: {selected_categorical}",
                    labels={'x': selected_categorical, 'y': 'Count'}
                )
                fig_count.update_xaxes(tickangle=45)
                st.plotly_chart(fig_count, use_container_width=True)
            
            with col2:
                # Statistics
                st.write(f"**Statistics for {selected_categorical}:**")
                unique_count = data[selected_categorical].nunique()
                most_frequent = str(data[selected_categorical].mode().iloc[0]) if len(data[selected_categorical].mode()) > 0 else 'N/A'
                missing_count = data[selected_categorical].isnull().sum()
                
                st.metric("Unique Values", unique_count)
                st.metric("Most Frequent", str(most_frequent)[:20] + "..." if len(str(most_frequent)) > 20 else str(most_frequent))
                st.metric("Missing Values", missing_count)
                
                if unique_count <= 20:
                    st.write("**All unique values:**")
                    for val, count in data[selected_categorical].value_counts().items():
                        st.write(f"- {val}: {count}")
    
    # Correlation analysis
    if len(numeric_cols) > 1:
        st.subheader("🔗 Correlation Analysis")
        
        # Include target if numeric
        corr_cols = numeric_cols.copy()
        if (st.session_state.target_column and 
            st.session_state.target_column in data.select_dtypes(include=[np.number]).columns):
            corr_cols.append(st.session_state.target_column)
        
        correlation_matrix = data[corr_cols].corr()
        
        # Correlation heatmap
        fig_corr = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect="auto",
            title="Feature Correlation Heatmap",
            color_continuous_scale='RdBu',
            zmin=-1,
            zmax=1
        )
        fig_corr.update_layout(height=600)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Highest correlations with target
        if (st.session_state.target_column and 
            st.session_state.target_column in correlation_matrix.columns):
            
            target_corr = correlation_matrix[st.session_state.target_column].abs().sort_values(ascending=False)
            target_corr = target_corr[target_corr.index != st.session_state.target_column]
            
            if len(target_corr) > 0:
                st.write(f"**Features most correlated with {st.session_state.target_column}:**")
                top_corr = target_corr.head(10)
                
                fig_target_corr = px.bar(
                    x=top_corr.values,
                    y=top_corr.index,
                    orientation='h',
                    title=f"Correlation with {st.session_state.target_column}",
                    labels={'x': 'Absolute Correlation', 'y': 'Features'}
                )
                st.plotly_chart(fig_target_corr, use_container_width=True)
    
    # Data quality summary
    st.subheader("✅ Data Quality Summary")
    
    quality_issues = []
    
    # Check for duplicates
    duplicate_count = data.duplicated().sum()
    if duplicate_count > 0:
        quality_issues.append(f"🔄 {duplicate_count:,} duplicate rows found")
    
    # Check for constant columns
    constant_cols = []
    for col in data.columns:
        if data[col].nunique() <= 1:
            constant_cols.append(col)
    
    if constant_cols:
        quality_issues.append(f"⚠️ Constant columns: {', '.join(constant_cols)}")
    
    # Check for high cardinality categorical columns
    high_cardinality = []
    for col in categorical_cols:
        unique_ratio = data[col].nunique() / len(data)
        if unique_ratio > 0.9:
            high_cardinality.append(col)
    
    if high_cardinality:
        quality_issues.append(f"📈 High cardinality columns: {', '.join(high_cardinality)}")
    
    # Check for columns with many missing values
    high_missing = []
    for col in data.columns:
        missing_ratio = data[col].isnull().sum() / len(data)
        if missing_ratio > 0.5:
            high_missing.append(col)
    
    if high_missing:
        quality_issues.append(f"❓ Columns with >50% missing: {', '.join(high_missing)}")
    
    if quality_issues:
        st.warning("**Data Quality Issues Found:**")
        for issue in quality_issues:
            st.write(f"- {issue}")
    else:
        st.success("🎉 No major data quality issues detected!")
    
    # Data sample
    st.subheader("📄 Data Sample")
    sample_size = st.slider("Sample size to display:", 5, min(100, len(data)), 10)
    
    display_option = st.radio("Display option:", ["First rows", "Random sample", "Last rows"])
    
    if display_option == "First rows":
        sample_data = data.head(sample_size)
    elif display_option == "Random sample":
        sample_data = data.sample(n=min(sample_size, len(data)), random_state=42)
    else:
        sample_data = data.tail(sample_size)
    
    st.dataframe(sample_data, use_container_width=True)
    
    # Export option - FIXED
    if st.button("📥 Generate Data Summary Report", use_container_width=True):
        try:
            # Create comprehensive summary report with proper type conversion
            summary_report = {
                'Dataset Overview': {
                    'Total Rows': int(len(data)),
                    'Total Columns': int(len(data.columns)),
                    'Missing Data Percentage': float(missing_pct),
                    'Memory Usage (MB)': float(memory_mb),
                    'Target Column': str(st.session_state.target_column) if st.session_state.target_column else "Not selected",
                    'Problem Type': str(st.session_state.problem_type) if st.session_state.problem_type else "Not determined"
                },
                'Data Types': {str(k): int(v) for k, v in dtype_counts.to_dict().items()},
                'Numeric Features': int(len(numeric_cols)),
                'Categorical Features': int(len(categorical_cols))
            }
            
            # Convert column details safely
            if col_info:
                summary_report['Column Details'] = []
                for item in col_info:
                    safe_item = {}
                    for key, value in item.items():
                        if pd.isna(value):
                            safe_item[str(key)] = None
                        elif isinstance(value, (np.integer, np.floating)):
                            safe_item[str(key)] = float(value)
                        else:
                            safe_item[str(key)] = str(value)
                    summary_report['Column Details'].append(safe_item)
            
            # Add missing data analysis safely
            if 'missing_df' in locals() and len(missing_df) > 0:
                summary_report['Missing Data Analysis'] = []
                for _, row in missing_df.iterrows():
                    safe_row = {
                        'Column': str(row['Column']),
                        'Missing Count': int(row['Missing Count']),
                        'Missing Percentage': float(row['Missing %'])
                    }
                    summary_report['Missing Data Analysis'].append(safe_row)
            else:
                summary_report['Missing Data Analysis'] = "No missing data"
            
            # Add quality issues safely
            summary_report['Quality Issues'] = [str(issue) for issue in quality_issues] if quality_issues else ["No issues detected"]
            
            # Convert to JSON string with safe serialization
            def safe_json_serializer(obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif pd.isna(obj):
                    return None
                else:
                    return str(obj)
            
            report_json = json.dumps(summary_report, indent=2, default=safe_json_serializer)
            
            # Create download button
            st.download_button(
                label="📥 Download JSON Report",
                data=report_json,
                file_name=f"data_exploration_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
            
            st.success("✅ Report generated successfully!")
            
        except Exception as e:
            st.error(f"❌ Error generating report: {str(e)}")
            logger.error(f"Report generation error: {e}")
            
            # Show detailed error for debugging
            with st.expander("🔧 Error Details"):
                st.code(str(e))
                st.write("**Data types in report:**")
                try:
                    st.write(f"dtype_counts type: {type(dtype_counts)}")
                    st.write(f"dtype_counts values: {dtype_counts}")
                except:
                    st.write("Could not display dtype_counts")
    
    # Next step
    if st.button("⚙️ Proceed to Data Preprocessing", type="primary", use_container_width=True):
        st.session_state.current_step = "preprocess"
        st.rerun()