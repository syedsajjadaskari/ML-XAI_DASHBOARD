"""
Advanced Interactive Preprocessing Page
Handles data preprocessing with real-time preview
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def page_preprocessing(data_handler):
    """Advanced interactive preprocessing configuration page."""
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first")
        return
    
    st.header("⚙️ Advanced Data Preprocessing")
    st.markdown("Configure preprocessing options and see real-time transformations")
    
    # Initialize preview data
    if 'preview_data' not in st.session_state:
        st.session_state.preview_data = st.session_state.data.copy()
    
    # Create tabs for different preprocessing categories
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🗑️ Column Management", 
        "❓ Missing Values", 
        "🏷️ Categorical Encoding", 
        "📊 Feature Engineering",
        "🔍 Final Preview"
    ])
    
    with tab1:
        _handle_column_management()
    
    with tab2:
        _handle_missing_values()
    
    with tab3:
        _handle_categorical_encoding()
    
    with tab4:
        _handle_feature_engineering()
    
    with tab5:
        _handle_final_preview(data_handler)

def _handle_column_management():
    """Handle column removal and management."""
    st.subheader("🗑️ Column Management")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("**Remove Unnecessary Columns**")
        available_columns = [col for col in st.session_state.data.columns 
                           if col != st.session_state.target_column]
        
        # Quick selection buttons
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("🆔 Remove ID-like columns", help="Remove columns that look like IDs"):
                id_columns = [col for col in available_columns 
                             if any(keyword in col.lower() for keyword in ['id', 'index', 'key', 'uid'])]
                st.session_state.columns_to_remove = list(set(st.session_state.get('columns_to_remove', []) + id_columns))
        
        with col_b:
            if st.button("📛 Remove high-missing columns", help="Remove columns with >70% missing values"):
                high_missing = []
                for col in available_columns:
                    missing_pct = st.session_state.data[col].isnull().sum() / len(st.session_state.data)
                    if missing_pct > 0.7:
                        high_missing.append(col)
                st.session_state.columns_to_remove = list(set(st.session_state.get('columns_to_remove', []) + high_missing))
        
        with col_c:
            if st.button("🔄 Reset selection"):
                st.session_state.columns_to_remove = []
        
        # Manual column selection
        columns_to_remove = st.multiselect(
            "Select columns to remove:",
            options=available_columns,
            default=st.session_state.get('columns_to_remove', []),
            help="Choose columns that are not useful for prediction"
        )
        st.session_state.columns_to_remove = columns_to_remove
    
    with col2:
        st.write("**Column Statistics**")
        col_stats = []
        for col in st.session_state.data.columns:
            if col != st.session_state.target_column:
                missing_pct = st.session_state.data[col].isnull().sum() / len(st.session_state.data) * 100
                unique_count = st.session_state.data[col].nunique()
                data_type = str(st.session_state.data[col].dtype)
                
                status = "✅ Keep"
                if col in columns_to_remove:
                    status = "❌ Remove"
                elif missing_pct > 70:
                    status = "⚠️ High Missing"
                elif unique_count == 1:
                    status = "⚠️ Constant"
                
                col_stats.append({
                    'Column': col,
                    'Type': data_type,
                    'Missing %': f"{missing_pct:.1f}%",
                    'Unique': unique_count,
                    'Status': status
                })
        
        stats_df = pd.DataFrame(col_stats)
        st.dataframe(stats_df, use_container_width=True, height=300)
    
    # Update preview data
    if columns_to_remove:
        st.session_state.preview_data = st.session_state.data.drop(columns=columns_to_remove, errors='ignore')
        st.success(f"✅ Preview updated: Removed {len(columns_to_remove)} columns")
    else:
        st.session_state.preview_data = st.session_state.data.copy()

def _handle_missing_values():
    """Handle missing values processing."""
    st.subheader("❓ Missing Values Handling")
    
    missing_analysis = st.session_state.preview_data.isnull().sum()
    missing_analysis = missing_analysis[missing_analysis > 0].sort_values(ascending=False)
    
    if len(missing_analysis) > 0:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**Missing Values Analysis**")
            
            # Missing values by column
            missing_df = pd.DataFrame({
                'Column': missing_analysis.index,
                'Missing Count': missing_analysis.values,
                'Missing %': (missing_analysis.values / len(st.session_state.preview_data) * 100)
            })
            
            fig_missing = px.bar(
                missing_df, 
                x='Column', 
                y='Missing %',
                title='Missing Values by Column',
                color='Missing %',
                color_continuous_scale='Reds'
            )
            fig_missing.update_xaxes(tickangle=45)
            st.plotly_chart(fig_missing, use_container_width=True)
            
            st.dataframe(missing_df, use_container_width=True)
        
        with col2:
            st.write("**Missing Value Strategy**")
            
            global_strategy = st.selectbox(
                "Global missing value strategy:",
                ["Custom per column", "drop", "mean", "median", "mode", "constant"],
                help="Apply same strategy to all columns or customize per column"
            )
            
            column_strategies = {}
            
            if global_strategy == "Custom per column":
                st.write("**Custom Strategy per Column:**")
                for col in missing_analysis.index:
                    col_type = str(st.session_state.preview_data[col].dtype)
                    
                    if 'int' in col_type or 'float' in col_type:
                        options = ["mean", "median", "constant", "drop"]
                        default_idx = 0
                    else:
                        options = ["mode", "constant", "drop"]
                        default_idx = 0
                    
                    strategy = st.selectbox(
                        f"{col} ({missing_analysis[col]} missing):",
                        options,
                        index=default_idx,
                        key=f"strategy_{col}"
                    )
                    column_strategies[col] = strategy
                    
                    if strategy == "constant":
                        const_value = st.text_input(
                            f"Constant value for {col}:",
                            value="0" if 'int' in col_type or 'float' in col_type else "Unknown",
                            key=f"const_{col}"
                        )
                        column_strategies[f"{col}_constant"] = const_value
            else:
                for col in missing_analysis.index:
                    column_strategies[col] = global_strategy
                    if global_strategy == "constant":
                        col_type = str(st.session_state.preview_data[col].dtype)
                        const_value = st.text_input(
                            f"Constant value for {col}:",
                            value="0" if 'int' in col_type or 'float' in col_type else "Unknown",
                            key=f"const_global_{col}"
                        )
                        column_strategies[f"{col}_constant"] = const_value
            
            # Apply missing value handling
            if st.button("🔄 Apply Missing Value Handling", key="apply_missing"):
                try:
                    temp_data = st.session_state.preview_data.copy()
                    
                    for col in missing_analysis.index:
                        strategy = column_strategies.get(col, 'drop')
                        
                        if strategy == "drop":
                            temp_data = temp_data.dropna(subset=[col])
                        elif strategy == "mean":
                            temp_data[col] = temp_data[col].fillna(temp_data[col].mean())
                        elif strategy == "median":
                            temp_data[col] = temp_data[col].fillna(temp_data[col].median())
                        elif strategy == "mode":
                            mode_val = temp_data[col].mode()
                            if len(mode_val) > 0:
                                temp_data[col] = temp_data[col].fillna(mode_val[0])
                        elif strategy == "constant":
                            const_val = column_strategies.get(f"{col}_constant", "Unknown")
                            temp_data[col] = temp_data[col].fillna(const_val)
                    
                    st.session_state.preview_data = temp_data
                    st.session_state.missing_strategies = column_strategies
                    st.success("✅ Missing values handled!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error handling missing values: {str(e)}")
    else:
        st.success("🎉 No missing values found in the current dataset!")

def _handle_categorical_encoding():
    """Handle categorical encoding."""
    st.subheader("🏷️ Categorical Encoding")
    
    categorical_cols = st.session_state.preview_data.select_dtypes(include=['object', 'category']).columns
    categorical_cols = [col for col in categorical_cols if col != st.session_state.target_column]
    
    if len(categorical_cols) > 0:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**Categorical Columns Analysis**")
            
            cat_analysis = []
            for col in categorical_cols:
                unique_count = st.session_state.preview_data[col].nunique()
                memory_usage = st.session_state.preview_data[col].memory_usage(deep=True) / 1024
                top_values = st.session_state.preview_data[col].value_counts().head(3).to_dict()
                
                cat_analysis.append({
                    'Column': col,
                    'Unique Values': unique_count,
                    'Memory (KB)': f"{memory_usage:.1f}",
                    'Top Values': str(top_values)
                })
            
            cat_df = pd.DataFrame(cat_analysis)
            st.dataframe(cat_df, use_container_width=True)
            
            encoding_method = st.selectbox(
                "Select encoding method:",
                ["onehot", "ordinal", "target", "binary", "custom"],
                help="Choose how to encode categorical variables"
            )
        
        with col2:
            st.write("**Encoding Preview**")
            
            if st.button("🔄 Apply Categorical Encoding", key="apply_encoding"):
                try:
                    temp_data = st.session_state.preview_data.copy()
                    
                    for col in categorical_cols:
                        unique_count = temp_data[col].nunique()
                        
                        if encoding_method == "onehot":
                            if unique_count > 20:
                                top_categories = temp_data[col].value_counts().head(15).index
                                temp_data[col] = temp_data[col].apply(
                                    lambda x: x if x in top_categories else 'Other'
                                )
                            
                            encoded = pd.get_dummies(temp_data[col], prefix=col, drop_first=True)
                            temp_data = pd.concat([temp_data.drop(columns=[col]), encoded], axis=1)
                            
                        elif encoding_method == "ordinal":
                            from sklearn.preprocessing import LabelEncoder
                            le = LabelEncoder()
                            temp_data[col] = le.fit_transform(temp_data[col].astype(str))
                    
                    st.session_state.preview_data = temp_data
                    st.session_state.encoding_method = encoding_method
                    st.success("✅ Categorical encoding applied!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error applying encoding: {str(e)}")
    else:
        st.info("ℹ️ No categorical columns found in the current dataset")

def _handle_feature_engineering():
    """Handle feature engineering and scaling."""
    st.subheader("📊 Feature Engineering & Scaling")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Feature Scaling**")
        
        numeric_cols = st.session_state.preview_data.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != st.session_state.target_column]
        
        if len(numeric_cols) > 0:
            scaling_method = st.selectbox(
                "Select scaling method:",
                ["none", "standard", "minmax", "robust"],
                help="Choose how to scale numeric features"
            )
            
            if scaling_method != "none":
                scale_columns = st.multiselect(
                    "Select columns to scale:",
                    options=numeric_cols,
                    default=numeric_cols,
                    help="Choose which numeric columns to scale"
                )
            
            if scaling_method != "none" and st.button("📊 Apply Scaling", key="apply_scaling"):
                try:
                    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
                    
                    temp_data = st.session_state.preview_data.copy()
                    
                    if scaling_method == "standard":
                        scaler = StandardScaler()
                    elif scaling_method == "minmax":
                        scaler = MinMaxScaler()
                    elif scaling_method == "robust":
                        scaler = RobustScaler()
                    
                    if 'scale_columns' in locals() and scale_columns:
                        temp_data[scale_columns] = scaler.fit_transform(temp_data[scale_columns])
                    
                    st.session_state.preview_data = temp_data
                    st.session_state.scaling_method = scaling_method
                    st.success("✅ Scaling applied!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error applying scaling: {str(e)}")
        else:
            st.info("ℹ️ No numeric columns available for scaling")
    
    with col2:
        st.write("**Advanced Feature Engineering**")
        
        feature_engineering = st.checkbox("Enable feature engineering", help="Create additional features")
        
        if feature_engineering:
            engineering_options = st.multiselect(
                "Select feature engineering techniques:",
                [
                    "polynomial_features",
                    "statistical_features",
                    "binning"
                ],
                help="Choose which feature engineering techniques to apply"
            )
            
            if st.button("🚀 Apply Feature Engineering", key="apply_engineering"):
                try:
                    temp_data = st.session_state.preview_data.copy()
                    
                    if "statistical_features" in engineering_options:
                        for col in numeric_cols[:3]:  # Limit to first 3 columns
                            temp_data[f"{col}_squared"] = temp_data[col] ** 2
                            temp_data[f"{col}_log"] = np.log1p(np.abs(temp_data[col]))
                    
                    st.session_state.preview_data = temp_data
                    st.session_state.feature_engineering = engineering_options
                    st.success("✅ Feature engineering applied!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error in feature engineering: {str(e)}")

def _handle_final_preview(data_handler):
    """Handle final preview and configuration saving."""
    st.subheader("🔍 Final Data Preview & Summary")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("**Transformed Dataset Preview**")
        
        # Show shape comparison
        original_shape = st.session_state.data.shape
        current_shape = st.session_state.preview_data.shape
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Original Rows", f"{original_shape[0]:,}")
        with col_b:
            st.metric("Current Rows", f"{current_shape[0]:,}", delta=current_shape[0] - original_shape[0])
        with col_c:
            st.metric("Current Columns", f"{current_shape[1]:,}", delta=current_shape[1] - original_shape[1])
        
        # Show data preview
        st.dataframe(st.session_state.preview_data.head(100), use_container_width=True)
        
        # Data quality summary
        st.write("**Data Quality Summary**")
        quality_checks = {
            "Missing Values": st.session_state.preview_data.isnull().sum().sum(),
            "Duplicate Rows": st.session_state.preview_data.duplicated().sum(),
            "Numeric Columns": len(st.session_state.preview_data.select_dtypes(include=[np.number]).columns),
            "Object Columns": len(st.session_state.preview_data.select_dtypes(include=['object']).columns),
            "Memory Usage (MB)": st.session_state.preview_data.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        quality_df = pd.DataFrame(list(quality_checks.items()), columns=['Metric', 'Value'])
        st.dataframe(quality_df, use_container_width=True)
    
    with col2:
        st.write("**Transformation Summary**")
        
        transformations = []
        
        if st.session_state.get('columns_to_remove'):
            transformations.append(f"✅ Removed {len(st.session_state.columns_to_remove)} columns")
        
        if st.session_state.get('missing_strategies'):
            transformations.append("✅ Applied missing value handling")
        
        if st.session_state.get('encoding_method'):
            transformations.append(f"✅ Applied {st.session_state.encoding_method} encoding")
        
        if st.session_state.get('scaling_method', 'none') != 'none':
            transformations.append(f"✅ Applied {st.session_state.scaling_method} scaling")
        
        if st.session_state.get('feature_engineering'):
            transformations.append("✅ Applied feature engineering")
        
        if transformations:
            for transform in transformations:
                st.write(transform)
        else:
            st.info("No transformations applied yet")
        
        # Save preprocessing configuration
        final_config = {
            'columns_to_remove': st.session_state.get('columns_to_remove', []),
            'missing_strategies': st.session_state.get('missing_strategies', {}),
            'encoding_method': st.session_state.get('encoding_method', 'onehot'),
            'scaling_method': st.session_state.get('scaling_method', 'none'),
            'feature_engineering': st.session_state.get('feature_engineering', [])
        }
        
        st.session_state.preprocessing_config = final_config
        
        # Download processed data
        if st.button("💾 Save Processed Data", key="save_processed"):
            csv = st.session_state.preview_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Processed CSV",
                data=csv,
                file_name=f"processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # Control buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Reset All Changes", key="reset_all"):
            st.session_state.preview_data = st.session_state.data.copy()
            st.session_state.preprocessing_config = {}
            st.success("✅ Reset to original data")
            st.rerun()
    
    with col2:
        if st.button("↩️ Revert Last Change", key="revert_last"):
            st.session_state.preview_data = st.session_state.data.copy()
            st.info("ℹ️ Reverted to original data")
            st.rerun()
    
    with col3:
        if st.button("🎯 Proceed to Model Training", type="primary", use_container_width=True):
            # Final validation
            if st.session_state.target_column not in st.session_state.preview_data.columns:
                st.error("❌ Target column was removed during preprocessing!")
            elif st.session_state.preview_data.empty:
                st.error("❌ No data remaining after preprocessing!")
            elif len(st.session_state.preview_data.columns) < 2:
                st.error("❌ Need at least 2 columns for training!")
            else:
                st.session_state.current_step = "train"
                st.rerun()