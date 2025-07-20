"""
Clean and Tidy Preprocessing Page
Handles data preprocessing with streamlined interface (Fixed NoneType errors)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def page_preprocessing(data_handler):
    """Clean and tidy preprocessing page."""
    # Back navigation
    col_back, col_title = st.columns([1, 4])
    with col_back:
        if st.button("⬅️ Back to Exploration", use_container_width=True):
            st.session_state.current_step = "explore"
            st.rerun()
    
    if st.session_state.get('data') is None:
        st.warning("⚠️ Please upload data first")
        return
    
    st.header("⚙️ Data Preprocessing")
    st.markdown("*Prepare your data for machine learning*")
    
    # Initialize preview data safely
    if st.session_state.get('preview_data') is None:
        st.session_state.preview_data = st.session_state.data.copy()
    
    # Quick overview
    data = st.session_state.data
    current_data = st.session_state.preview_data
    
    # Ensure we have valid data
    if data is None or current_data is None:
        st.error("❌ Data not available. Please go back and upload data.")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Original Rows", f"{len(data):,}")
    with col2:
        st.metric("Current Rows", f"{len(current_data):,}", 
                 delta=len(current_data) - len(data))
    with col3:
        st.metric("Original Columns", f"{len(data.columns)}")
    with col4:
        st.metric("Current Columns", f"{len(current_data.columns)}", 
                 delta=len(current_data.columns) - len(data.columns))
    
    # Preprocessing options in clean tabs
    tab1, tab2, tab3 = st.tabs(["🗑️ Remove Columns", "❓ Handle Missing Data", "🔧 Final Steps"])
    
    with tab1:
        _handle_column_removal()
    
    with tab2:
        _handle_missing_data_clean()
    
    with tab3:
        _handle_final_preprocessing(data_handler)

def _handle_column_removal():
    """Clean column removal interface."""
    st.subheader("🗑️ Column Management")
    
    # Check if we have data and target column
    if st.session_state.get('data') is None:
        st.error("❌ No data available")
        return
    
    target_column = st.session_state.get('target_column')
    available_columns = [col for col in st.session_state.data.columns 
                        if col != target_column]
    
    if len(available_columns) == 0:
        st.warning("⚠️ No columns available for removal")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Quick removal options
        st.write("**Quick Actions:**")
        quick_col1, quick_col2, quick_col3 = st.columns(3)
        
        with quick_col1:
            if st.button("🆔 Remove ID Columns", help="Remove ID-like columns"):
                id_columns = [col for col in available_columns 
                             if any(keyword in col.lower() for keyword in ['id', 'index', 'key', 'uid'])]
                current_removals = st.session_state.get('columns_to_remove', [])
                st.session_state.columns_to_remove = list(set(current_removals + id_columns))
                st.rerun()
        
        with quick_col2:
            if st.button("🔄 Remove High Missing", help="Remove columns with >70% missing"):
                high_missing = []
                for col in available_columns:
                    if col in st.session_state.data.columns:
                        missing_pct = st.session_state.data[col].isnull().sum() / len(st.session_state.data)
                        if missing_pct > 0.7:
                            high_missing.append(col)
                current_removals = st.session_state.get('columns_to_remove', [])
                st.session_state.columns_to_remove = list(set(current_removals + high_missing))
                st.rerun()
        
        with quick_col3:
            if st.button("🔄 Reset Selection", help="Clear all selections"):
                st.session_state.columns_to_remove = []
                st.rerun()
        
        st.write("**Manual Selection:**")
        current_removals = st.session_state.get('columns_to_remove', [])
        columns_to_remove = st.multiselect(
            "Select columns to remove:",
            options=available_columns,
            default=current_removals,
            help="Choose columns that won't help with prediction"
        )
        st.session_state.columns_to_remove = columns_to_remove
        
        # Apply removal
        if st.button("✂️ Apply Column Removal", type="primary"):
            if columns_to_remove:
                try:
                    st.session_state.preview_data = st.session_state.data.drop(columns=columns_to_remove, errors='ignore')
                    st.success(f"✅ Removed {len(columns_to_remove)} columns")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error removing columns: {str(e)}")
    
    with col2:
        st.write("**Column Status:**")
        
        # Show status for first 10 columns to avoid overwhelming UI
        columns_to_show = available_columns[:10]
        for col in columns_to_show:
            try:
                if col in st.session_state.data.columns:
                    missing_pct = st.session_state.data[col].isnull().sum() / len(st.session_state.data) * 100
                    unique_count = st.session_state.data[col].nunique()
                    
                    if col in st.session_state.get('columns_to_remove', []):
                        st.error(f"❌ {col}")
                    elif missing_pct > 70:
                        st.warning(f"⚠️ {col} ({missing_pct:.0f}% missing)")
                    elif unique_count == 1:
                        st.warning(f"⚠️ {col} (constant)")
                    else:
                        st.success(f"✅ {col}")
            except Exception as e:
                st.info(f"ℹ️ {col} (check failed)")
        
        if len(available_columns) > 10:
            st.info(f"... and {len(available_columns) - 10} more columns")

def _handle_missing_data_clean():
    """Clean missing data handling interface."""
    st.subheader("❓ Missing Data Strategy")
    
    if st.session_state.get('preview_data') is None:
        st.error("❌ No data available for processing")
        return
    
    data = st.session_state.preview_data
    missing_analysis = data.isnull().sum()
    missing_analysis = missing_analysis[missing_analysis > 0].sort_values(ascending=False)
    
    if len(missing_analysis) == 0:
        st.success("🎉 No missing data found!")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("**Missing Data Overview:**")
        try:
            missing_df = pd.DataFrame({
                'Column': missing_analysis.index,
                'Missing': missing_analysis.values,
                'Percentage': (missing_analysis.values / len(data) * 100).round(1)
            })
            st.dataframe(missing_df, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error displaying missing data: {str(e)}")
            return
    
    with col2:
        st.write("**Choose Strategy:**")
        
        # Global strategy
        strategy = st.selectbox(
            "How to handle missing values?",
            [
                "Drop rows with missing values",
                "Fill with mean (numeric) / mode (categorical)",
                "Fill with median (numeric) / mode (categorical)",
                "Fill with constant value"
            ],
            help="Applied to all columns with missing data"
        )
        
        constant_value = "0"
        if "constant" in strategy:
            constant_value = st.text_input(
                "Constant value:",
                value="0",
                help="Value to fill missing data"
            )
        
        # Apply strategy
        if st.button("🔧 Apply Missing Data Strategy", type="primary"):
            temp_data = data.copy()
            
            try:
                if "Drop rows" in strategy:
                    temp_data = temp_data.dropna()
                elif "mean" in strategy:
                    for col in missing_analysis.index:
                        if temp_data[col].dtype in ['int64', 'float64']:
                            temp_data[col] = temp_data[col].fillna(temp_data[col].mean())
                        else:
                            mode_val = temp_data[col].mode()
                            if len(mode_val) > 0:
                                temp_data[col] = temp_data[col].fillna(mode_val[0])
                elif "median" in strategy:
                    for col in missing_analysis.index:
                        if temp_data[col].dtype in ['int64', 'float64']:
                            temp_data[col] = temp_data[col].fillna(temp_data[col].median())
                        else:
                            mode_val = temp_data[col].mode()
                            if len(mode_val) > 0:
                                temp_data[col] = temp_data[col].fillna(mode_val[0])
                elif "constant" in strategy:
                    for col in missing_analysis.index:
                        temp_data[col] = temp_data[col].fillna(constant_value)
                
                st.session_state.preview_data = temp_data
                st.session_state.missing_strategy = strategy
                st.success("✅ Missing data handled!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

def _handle_final_preprocessing(data_handler):
    """Final preprocessing steps and summary."""
    st.subheader("🔧 Final Preprocessing")
    
    if st.session_state.get('preview_data') is None:
        st.error("❌ No data available for processing")
        return
    
    data = st.session_state.preview_data
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Categorical Encoding:**")
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        target_column = st.session_state.get('target_column')
        categorical_cols = [col for col in categorical_cols if col != target_column]
        
        if len(categorical_cols) > 0:
            encoding_method = st.selectbox(
                "Encoding method:",
                ["one-hot", "label"],
                help="How to convert text to numbers"
            )
            
            if st.button("🏷️ Apply Encoding"):
                temp_data = data.copy()
                
                try:
                    if encoding_method == "one-hot":
                        # Limit categories for one-hot encoding
                        for col in categorical_cols:
                            if temp_data[col].nunique() > 10:
                                # Keep only top 10 categories
                                top_cats = temp_data[col].value_counts().head(10).index
                                temp_data[col] = temp_data[col].apply(
                                    lambda x: x if x in top_cats else 'Other'
                                )
                        
                        # Apply one-hot encoding
                        temp_data = pd.get_dummies(temp_data, columns=categorical_cols, prefix=categorical_cols)
                    
                    elif encoding_method == "label":
                        from sklearn.preprocessing import LabelEncoder
                        for col in categorical_cols:
                            le = LabelEncoder()
                            temp_data[col] = le.fit_transform(temp_data[col].astype(str))
                    
                    st.session_state.preview_data = temp_data
                    st.session_state.encoding_method = encoding_method
                    st.success("✅ Encoding applied!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Encoding error: {str(e)}")
        else:
            st.info("No categorical columns found")
        
        st.write("**Feature Scaling:**")
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != target_column]
        
        if len(numeric_cols) > 0:
            scaling_method = st.selectbox(
                "Scaling method:",
                ["none", "standard", "min-max"],
                help="Normalize numeric features"
            )
            
            if scaling_method != "none" and st.button("📊 Apply Scaling"):
                temp_data = data.copy()
                
                try:
                    from sklearn.preprocessing import StandardScaler, MinMaxScaler
                    
                    if scaling_method == "standard":
                        scaler = StandardScaler()
                    else:
                        scaler = MinMaxScaler()
                    
                    temp_data[numeric_cols] = scaler.fit_transform(temp_data[numeric_cols])
                    
                    st.session_state.preview_data = temp_data
                    st.session_state.scaling_method = scaling_method
                    st.success("✅ Scaling applied!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Scaling error: {str(e)}")
        else:
            st.info("No numeric columns found")
    
    with col2:
        st.write("**Data Summary:**")
        current_data = st.session_state.preview_data
        
        # Show changes
        changes = []
        if st.session_state.get('columns_to_remove'):
            changes.append(f"✅ Removed {len(st.session_state.columns_to_remove)} columns")
        if st.session_state.get('missing_strategy'):
            changes.append("✅ Handled missing data")
        if st.session_state.get('encoding_method'):
            changes.append(f"✅ Applied {st.session_state.encoding_method} encoding")
        if st.session_state.get('scaling_method', 'none') != 'none':
            changes.append(f"✅ Applied {st.session_state.scaling_method} scaling")
        
        if changes:
            for change in changes:
                st.write(change)
        else:
            st.info("No preprocessing applied yet")
        
        # Data quality check
        st.write("**Quality Check:**")
        target_column = st.session_state.get('target_column')
        
        if target_column is None:
            st.warning("⚠️ No target column selected!")
        elif target_column not in current_data.columns:
            st.error("❌ Target column missing!")
        elif current_data.empty:
            st.error("❌ No data remaining!")
        elif len(current_data.columns) < 2:
            st.error("❌ Need at least 2 columns!")
        else:
            st.success("✅ Data ready for training")
        
        # Preview sample
        st.write("**Data Preview:**")
        try:
            st.dataframe(current_data.head(3), use_container_width=True)
        except Exception as e:
            st.error(f"❌ Preview error: {str(e)}")
    
    # Save configuration
    final_config = {
        'columns_to_remove': st.session_state.get('columns_to_remove', []),
        'missing_strategy': st.session_state.get('missing_strategy', 'none'),
        'encoding_method': st.session_state.get('encoding_method', 'none'),
        'scaling_method': st.session_state.get('scaling_method', 'none')
    }
    st.session_state.preprocessing_config = final_config
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Reset All", help="Start over"):
            try:
                st.session_state.preview_data = st.session_state.data.copy()
                st.session_state.preprocessing_config = {}
                keys_to_clear = ['columns_to_remove', 'missing_strategy', 'encoding_method', 'scaling_method']
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                st.success("✅ Reset complete!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Reset error: {str(e)}")
    
    with col2:
        if st.button("💾 Download Processed Data"):
            try:
                csv = st.session_state.preview_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"❌ Download error: {str(e)}")
    
    with col3:
        # Validation check before proceeding
        current_data = st.session_state.preview_data
        target_column = st.session_state.get('target_column')
        
        can_proceed = (
            target_column is not None and
            target_column in current_data.columns and
            not current_data.empty and
            len(current_data.columns) >= 2
        )
        
        if st.button("🎯 Proceed to Training", type="primary", disabled=not can_proceed):
            if can_proceed:
                st.session_state.current_step = "train"
                st.rerun()
            else:
                st.error("❌ Please fix data issues before proceeding")

            if st.button("🔄 Reset Selection", help="Clear all selections"):
                st.session_state.columns_to_remove = []
                st.rerun()
        
        st.write("**Manual Selection:**")
        columns_to_remove = st.multiselect(
            "Select columns to remove:",
            options=available_columns,
            default=st.session_state.get('columns_to_remove', []),
            help="Choose columns that won't help with prediction"
        )
        st.session_state.columns_to_remove = columns_to_remove
        
        # Apply removal
        if st.button("✂️ Apply Column Removal", type="primary"):
            if columns_to_remove:
                st.session_state.preview_data = st.session_state.data.drop(columns=columns_to_remove, errors='ignore')
                st.success(f"✅ Removed {len(columns_to_remove)} columns")
                st.rerun()
    
    with col2:
        st.write("**Column Status:**")
        for col in available_columns[:10]:  # Show first 10
            missing_pct = st.session_state.data[col].isnull().sum() / len(st.session_state.data) * 100
            unique_count = st.session_state.data[col].nunique()
            
            if col in st.session_state.get('columns_to_remove', []):
                st.error(f"❌ {col}")
            elif missing_pct > 70:
                st.warning(f"⚠️ {col} ({missing_pct:.0f}% missing)")
            elif unique_count == 1:
                st.warning(f"⚠️ {col} (constant)")
            else:
                st.success(f"✅ {col}")
        
        if len(available_columns) > 10:
            st.info(f"... and {len(available_columns) - 10} more columns")

def _handle_missing_data_clean():
    """Clean missing data handling interface."""
    st.subheader("❓ Missing Data Strategy")
    
    data = st.session_state.preview_data
    missing_analysis = data.isnull().sum()
    missing_analysis = missing_analysis[missing_analysis > 0].sort_values(ascending=False)
    
    if len(missing_analysis) == 0:
        st.success("🎉 No missing data found!")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("**Missing Data Overview:**")
        missing_df = pd.DataFrame({
            'Column': missing_analysis.index,
            'Missing': missing_analysis.values,
            'Percentage': (missing_analysis.values / len(data) * 100).round(1)
        })
        st.dataframe(missing_df, use_container_width=True)
    
    with col2:
        st.write("**Choose Strategy:**")
        
        # Global strategy
        strategy = st.selectbox(
            "How to handle missing values?",
            [
                "Drop rows with missing values",
                "Fill with mean (numeric) / mode (categorical)",
                "Fill with median (numeric) / mode (categorical)",
                "Fill with constant value"
            ],
            help="Applied to all columns with missing data"
        )
        
        if "constant" in strategy:
            constant_value = st.text_input(
                "Constant value:",
                value="0",
                help="Value to fill missing data"
            )
        
        # Apply strategy
        if st.button("🔧 Apply Missing Data Strategy", type="primary"):
            temp_data = data.copy()
            
            try:
                if "Drop rows" in strategy:
                    temp_data = temp_data.dropna()
                elif "mean" in strategy:
                    for col in missing_analysis.index:
                        if temp_data[col].dtype in ['int64', 'float64']:
                            temp_data[col] = temp_data[col].fillna(temp_data[col].mean())
                        else:
                            mode_val = temp_data[col].mode()
                            if len(mode_val) > 0:
                                temp_data[col] = temp_data[col].fillna(mode_val[0])
                elif "median" in strategy:
                    for col in missing_analysis.index:
                        if temp_data[col].dtype in ['int64', 'float64']:
                            temp_data[col] = temp_data[col].fillna(temp_data[col].median())
                        else:
                            mode_val = temp_data[col].mode()
                            if len(mode_val) > 0:
                                temp_data[col] = temp_data[col].fillna(mode_val[0])
                elif "constant" in strategy:
                    for col in missing_analysis.index:
                        temp_data[col] = temp_data[col].fillna(constant_value)
                
                st.session_state.preview_data = temp_data
                st.session_state.missing_strategy = strategy
                st.success("✅ Missing data handled!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

def _handle_final_preprocessing(data_handler):
    """Final preprocessing steps and summary."""
    st.subheader("🔧 Final Preprocessing")
    
    data = st.session_state.preview_data
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Categorical Encoding:**")
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        categorical_cols = [col for col in categorical_cols if col != st.session_state.target_column]
        
        if len(categorical_cols) > 0:
            encoding_method = st.selectbox(
                "Encoding method:",
                ["one-hot", "label"],
                help="How to convert text to numbers"
            )
            
            if st.button("🏷️ Apply Encoding"):
                temp_data = data.copy()
                
                try:
                    if encoding_method == "one-hot":
                        # Limit categories for one-hot encoding
                        for col in categorical_cols:
                            if temp_data[col].nunique() > 10:
                                # Keep only top 10 categories
                                top_cats = temp_data[col].value_counts().head(10).index
                                temp_data[col] = temp_data[col].apply(
                                    lambda x: x if x in top_cats else 'Other'
                                )
                        
                        # Apply one-hot encoding
                        temp_data = pd.get_dummies(temp_data, columns=categorical_cols, prefix=categorical_cols)
                    
                    elif encoding_method == "label":
                        from sklearn.preprocessing import LabelEncoder
                        for col in categorical_cols:
                            le = LabelEncoder()
                            temp_data[col] = le.fit_transform(temp_data[col].astype(str))
                    
                    st.session_state.preview_data = temp_data
                    st.session_state.encoding_method = encoding_method
                    st.success("✅ Encoding applied!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Encoding error: {str(e)}")
        else:
            st.info("No categorical columns found")
        
        st.write("**Feature Scaling:**")
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != st.session_state.target_column]
        
        if len(numeric_cols) > 0:
            scaling_method = st.selectbox(
                "Scaling method:",
                ["none", "standard", "min-max"],
                help="Normalize numeric features"
            )
            
            if scaling_method != "none" and st.button("📊 Apply Scaling"):
                temp_data = data.copy()
                
                try:
                    from sklearn.preprocessing import StandardScaler, MinMaxScaler
                    
                    if scaling_method == "standard":
                        scaler = StandardScaler()
                    else:
                        scaler = MinMaxScaler()
                    
                    temp_data[numeric_cols] = scaler.fit_transform(temp_data[numeric_cols])
                    
                    st.session_state.preview_data = temp_data
                    st.session_state.scaling_method = scaling_method
                    st.success("✅ Scaling applied!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Scaling error: {str(e)}")
        else:
            st.info("No numeric columns found")
    
    with col2:
        st.write("**Data Summary:**")
        current_data = st.session_state.preview_data
        
        # Show changes
        changes = []
        if st.session_state.get('columns_to_remove'):
            changes.append(f"✅ Removed {len(st.session_state.columns_to_remove)} columns")
        if st.session_state.get('missing_strategy'):
            changes.append("✅ Handled missing data")
        if st.session_state.get('encoding_method'):
            changes.append(f"✅ Applied {st.session_state.encoding_method} encoding")
        if st.session_state.get('scaling_method', 'none') != 'none':
            changes.append(f"✅ Applied {st.session_state.scaling_method} scaling")
        
        if changes:
            for change in changes:
                st.write(change)
        else:
            st.info("No preprocessing applied yet")
        
        # Data quality check
        st.write("**Quality Check:**")
        if st.session_state.target_column not in current_data.columns:
            st.error("❌ Target column missing!")
        elif current_data.empty:
            st.error("❌ No data remaining!")
        elif len(current_data.columns) < 2:
            st.error("❌ Need at least 2 columns!")
        else:
            st.success("✅ Data ready for training")
        
        # Preview sample
        st.write("**Data Preview:**")
        st.dataframe(current_data.head(3), use_container_width=True)
    
    # Save configuration
    final_config = {
        'columns_to_remove': st.session_state.get('columns_to_remove', []),
        'missing_strategy': st.session_state.get('missing_strategy', 'none'),
        'encoding_method': st.session_state.get('encoding_method', 'none'),
        'scaling_method': st.session_state.get('scaling_method', 'none')
    }
    st.session_state.preprocessing_config = final_config
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Reset All", help="Start over"):
            st.session_state.preview_data = st.session_state.data.copy()
            st.session_state.preprocessing_config = {}
            if 'columns_to_remove' in st.session_state:
                del st.session_state.columns_to_remove
            st.rerun()
    
    with col2:
        if st.button("💾 Download Processed Data"):
            csv = st.session_state.preview_data.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col3:
        # Validation check before proceeding
        current_data = st.session_state.preview_data
        can_proceed = (
            st.session_state.target_column in current_data.columns and
            not current_data.empty and
            len(current_data.columns) >= 2
        )
        
        if st.button("🎯 Proceed to Training", type="primary", disabled=not can_proceed):
            if can_proceed:
                st.session_state.current_step = "train"
                st.rerun()
            else:
                st.error("❌ Please fix data issues before proceeding")