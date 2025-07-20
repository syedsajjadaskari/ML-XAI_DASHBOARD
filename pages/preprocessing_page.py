import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, chi2
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

def page_preprocessing(data_handler):
    """Advanced preprocessing page with comprehensive features."""
    # Back navigation
    col_back, col_title = st.columns([1, 4])
    with col_back:
        if st.button("⬅️ Back to Exploration", use_container_width=True):
            st.session_state.current_step = "explore"
            st.rerun()
    
    if st.session_state.get('data') is None:
        st.warning("⚠️ Please upload data first")
        return
    
    st.header("⚙️ Advanced Data Preprocessing")
    st.markdown("*Comprehensive data preparation with advanced features and real-time visualization*")
    
    # Initialize preview data safely
    if st.session_state.get('preview_data') is None:
        st.session_state.preview_data = st.session_state.data.copy()
    
    # Initialize current tab if not exists
    if 'preprocessing_tab' not in st.session_state:
        st.session_state.preprocessing_tab = 0
    
    # Data overview dashboard
    _show_preprocessing_dashboard()
    
    # Tab navigation with forward/back buttons
    tab_names = ["🗑️ Data Cleaning", "❓ Missing Values", "🏷️ Feature Encoding", 
                 "📊 Feature Scaling", "🔍 Feature Engineering", "✅ Final Review"]
    
    # Navigation buttons
    col_prev, col_current, col_next = st.columns([1, 2, 1])
    
    with col_prev:
        if st.button("⬅️ Previous Step", disabled=st.session_state.preprocessing_tab == 0):
            st.session_state.preprocessing_tab = max(0, st.session_state.preprocessing_tab - 1)
            st.rerun()
    
    with col_current:
        st.markdown(f"**Current Step: {tab_names[st.session_state.preprocessing_tab]}**")
    
    with col_next:
        if st.button("Next Step ➡️", disabled=st.session_state.preprocessing_tab == len(tab_names) - 1):
            st.session_state.preprocessing_tab = min(len(tab_names) - 1, st.session_state.preprocessing_tab + 1)
            st.rerun()
    
    # Progress indicator
    progress = (st.session_state.preprocessing_tab + 1) / len(tab_names)
    st.progress(progress)
    
    # Show current tab content
    if st.session_state.preprocessing_tab == 0:
        _handle_advanced_data_cleaning()
    elif st.session_state.preprocessing_tab == 1:
        _handle_advanced_missing_values()
    elif st.session_state.preprocessing_tab == 2:
        _handle_advanced_encoding()
    elif st.session_state.preprocessing_tab == 3:
        _handle_advanced_scaling()
    elif st.session_state.preprocessing_tab == 4:
        _handle_feature_engineering()
    elif st.session_state.preprocessing_tab == 5:
        _handle_final_review_and_export(data_handler)
    
    # Skip/Jump to specific tab
    st.markdown("---")
    st.write("**🎯 Quick Navigation:**")
    
    cols = st.columns(len(tab_names))
    for i, (col, tab_name) in enumerate(zip(cols, tab_names)):
        with col:
            if st.button(tab_name.split()[1] if len(tab_name.split()) > 1 else tab_name, 
                        key=f"jump_tab_{i}",
                        use_container_width=True,
                        type="primary" if i == st.session_state.preprocessing_tab else "secondary"):
                st.session_state.preprocessing_tab = i
                st.rerun()

def _show_preprocessing_dashboard():
    """Show comprehensive preprocessing dashboard."""
    st.subheader("📊 Preprocessing Dashboard")
    
    data = st.session_state.data
    current_data = st.session_state.preview_data
    
    # Main metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("Original Rows", f"{len(data):,}")
    with col2:
        current_rows = len(current_data) if current_data is not None else 0
        st.metric("Current Rows", f"{current_rows:,}", 
                 delta=current_rows - len(data))
    with col3:
        st.metric("Original Columns", f"{len(data.columns)}")
    with col4:
        current_cols = len(current_data.columns) if current_data is not None else 0
        st.metric("Current Columns", f"{current_cols}", 
                 delta=current_cols - len(data.columns))
    with col5:
        missing_pct = (current_data.isnull().sum().sum() / (len(current_data) * len(current_data.columns))) * 100 if current_data is not None else 0
        st.metric("Missing Data", f"{missing_pct:.1f}%")
    with col6:
        memory_mb = current_data.memory_usage(deep=True).sum() / 1024 / 1024 if current_data is not None else 0
        st.metric("Memory Usage", f"{memory_mb:.1f} MB")
    
    # Transformation history
    transformations = []
    if st.session_state.get('columns_removed'):
        transformations.append(f"✅ Removed {len(st.session_state.columns_removed)} columns")
    if st.session_state.get('outliers_removed'):
        transformations.append(f"✅ Removed {st.session_state.outliers_removed} outliers")
    if st.session_state.get('missing_strategy_applied'):
        transformations.append("✅ Applied missing value strategy")
    if st.session_state.get('encoding_applied'):
        transformations.append("✅ Applied feature encoding")
    if st.session_state.get('scaling_applied'):
        transformations.append("✅ Applied feature scaling")
    if st.session_state.get('features_engineered'):
        transformations.append("✅ Applied feature engineering")
    
    if transformations:
        st.success("**Applied Transformations:**")
        for transform in transformations:
            st.write(f"  {transform}")
    else:
        st.info("No transformations applied yet")

def _handle_advanced_data_cleaning():
    """Advanced data cleaning with visualizations."""
    st.subheader("🗑️ Advanced Data Cleaning")
    
    if st.session_state.get('preview_data') is None:
        st.error("❌ No data available")
        return
    
    data = st.session_state.preview_data
    target_column = st.session_state.get('target_column')
    
    # Column management
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("**🔍 Column Analysis & Removal**")
        
        # Analyze columns
        column_analysis = []
        for col in data.columns:
            if col != target_column:
                missing_pct = data[col].isnull().sum() / len(data) * 100
                unique_ratio = data[col].nunique() / len(data)
                constant = data[col].nunique() <= 1
                
                status = "Keep"
                reason = "Good quality"
                
                if constant:
                    status = "Remove"
                    reason = "Constant values"
                elif missing_pct > 90:
                    status = "Remove"
                    reason = f"{missing_pct:.1f}% missing"
                elif unique_ratio > 0.95 and data[col].dtype == 'object':
                    status = "Remove"
                    reason = "High cardinality"
                elif missing_pct > 70:
                    status = "Caution"
                    reason = f"{missing_pct:.1f}% missing"
                
                column_analysis.append({
                    'Column': col,
                    'Type': str(data[col].dtype),
                    'Missing %': f"{missing_pct:.1f}%",
                    'Unique %': f"{unique_ratio * 100:.1f}%",
                    'Status': status,
                    'Reason': reason
                })
        
        analysis_df = pd.DataFrame(column_analysis)
        
        # Interactive column selection
        columns_to_remove = st.multiselect(
            "Select columns to remove:",
            options=analysis_df['Column'].tolist(),
            default=analysis_df[analysis_df['Status'] == 'Remove']['Column'].tolist(),
            help="Columns marked for removal based on quality analysis"
        )
        
        # Quick action buttons
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("🤖 Auto-Remove Low Quality"):
                auto_remove = analysis_df[analysis_df['Status'] == 'Remove']['Column'].tolist()
                st.session_state.columns_to_remove = auto_remove
                st.rerun()
        
        with col_b:
            if st.button("🆔 Remove ID-like Columns"):
                id_cols = [col for col in data.columns if any(keyword in col.lower() 
                          for keyword in ['id', 'index', 'key', 'uid', 'url'])]
                st.session_state.columns_to_remove = list(set(st.session_state.get('columns_to_remove', []) + id_cols))
                st.rerun()
        
        with col_c:
            if st.button("🔄 Reset Selection"):
                st.session_state.columns_to_remove = []
                st.rerun()
        
        # Apply column removal
        if st.button("✂️ Apply Column Removal", type="primary") and columns_to_remove:
            try:
                st.session_state.preview_data = data.drop(columns=columns_to_remove, errors='ignore')
                st.session_state.columns_removed = columns_to_remove
                st.success(f"✅ Removed {len(columns_to_remove)} columns")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    with col2:
        st.write("**📊 Column Quality Overview**")
        
        # Display analysis table
        if len(analysis_df) > 0:
            # Color coding
            def color_status(val):
                if val == 'Remove':
                    return 'background-color: #ffebee'
                elif val == 'Caution':
                    return 'background-color: #fff3e0'
                else:
                    return 'background-color: #e8f5e8'
            
            styled_df = analysis_df.style.applymap(color_status, subset=['Status'])
            st.dataframe(styled_df, use_container_width=True, height=300)
        
        # Quality distribution chart
        if len(analysis_df) > 0:
            status_counts = analysis_df['Status'].value_counts()
            fig = px.pie(values=status_counts.values, names=status_counts.index,
                        title="Column Quality Distribution")
            st.plotly_chart(fig, use_container_width=True)
    
    # Outlier detection and removal
    st.markdown("---")
    st.write("**🎯 Outlier Detection & Removal**")
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != target_column]
    
    if len(numeric_cols) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            outlier_method = st.selectbox(
                "Outlier detection method:",
                ["IQR (Interquartile Range)", "Z-Score", "Isolation Forest"],
                help="Choose method for outlier detection"
            )
            
            threshold = st.slider("Sensitivity threshold:", 1.0, 3.0, 1.5, 0.1,
                                help="Lower = more sensitive to outliers")
            
            if st.button("🔍 Detect Outliers"):
                outliers_detected = _detect_outliers(data, numeric_cols, outlier_method, threshold)
                st.session_state.outliers_detected = outliers_detected
                
                if outliers_detected > 0:
                    st.warning(f"⚠️ Detected {outliers_detected} outlier rows")
                    
                    if st.button("🗑️ Remove Outliers"):
                        cleaned_data = _remove_outliers(data, numeric_cols, outlier_method, threshold)
                        st.session_state.preview_data = cleaned_data
                        st.session_state.outliers_removed = outliers_detected
                        st.success(f"✅ Removed {outliers_detected} outlier rows")
                        # Don't rerun to stay on same tab
                else:
                    st.success("✅ No outliers detected")
        
        with col2:
            # Outlier visualization
            if len(numeric_cols) > 0:
                selected_col = st.selectbox("Select column for outlier visualization:", numeric_cols)
                
                if selected_col:
                    fig = make_subplots(rows=2, cols=1, 
                                      subplot_titles=['Distribution', 'Box Plot'])
                    
                    # Histogram
                    fig.add_trace(go.Histogram(x=data[selected_col], name='Distribution'), row=1, col=1)
                    
                    # Box plot
                    fig.add_trace(go.Box(y=data[selected_col], name='Box Plot'), row=2, col=1)
                    
                    fig.update_layout(height=400, title=f"Outlier Analysis: {selected_col}")
                    st.plotly_chart(fig, use_container_width=True)

def _handle_advanced_missing_values():
    """Advanced missing value handling with multiple strategies."""
    st.subheader("❓ Advanced Missing Value Treatment")
    
    if st.session_state.get('preview_data') is None:
        st.error("❌ No data available")
        return
    
    data = st.session_state.preview_data
    missing_analysis = data.isnull().sum()
    missing_cols = missing_analysis[missing_analysis > 0]
    
    if len(missing_cols) == 0:
        st.success("🎉 No missing values found!")
        return
    
    # Missing data visualization
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("**📊 Missing Data Pattern Analysis**")
        
        # Missing data heatmap
        if len(missing_cols) > 0:
            missing_matrix = data[missing_cols.index].isnull().astype(int)
            
            fig = px.imshow(missing_matrix.T, 
                          labels=dict(x="Rows", y="Columns", color="Missing"),
                          title="Missing Data Pattern")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**📋 Missing Data Summary**")
        
        missing_df = pd.DataFrame({
            'Column': missing_cols.index,
            'Missing Count': missing_cols.values,
            'Missing %': (missing_cols.values / len(data) * 100).round(1)
        })
        
        st.dataframe(missing_df, use_container_width=True)
    
    # Advanced imputation strategies
    st.write("**🔧 Advanced Imputation Strategies**")
    
    # Strategy selection per column
    imputation_strategies = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Global Strategy:**")
        global_strategy = st.selectbox(
            "Choose global strategy:",
            ["Custom per column", "Drop rows", "Forward fill", "Backward fill", "Interpolate"],
            help="Apply same strategy to all columns or customize individually"
        )
        
        if global_strategy == "Custom per column":
            for col in missing_cols.index:
                col_type = data[col].dtype
                
                if pd.api.types.is_numeric_dtype(col_type):
                    options = ["Mean", "Median", "Mode", "KNN", "Forward fill", "Interpolate", "Constant"]
                else:
                    options = ["Mode", "Forward fill", "Constant", "Most frequent"]
                
                strategy = st.selectbox(f"Strategy for {col}:", options, key=f"strategy_{col}")
                imputation_strategies[col] = strategy
                
                if strategy == "Constant":
                    const_val = st.text_input(f"Constant value for {col}:", 
                                            value="0" if pd.api.types.is_numeric_dtype(col_type) else "Unknown",
                                            key=f"const_{col}")
                    imputation_strategies[f"{col}_constant"] = const_val
                elif strategy == "KNN":
                    k_neighbors = st.slider(f"K neighbors for {col}:", 1, 10, 5, key=f"knn_{col}")
                    imputation_strategies[f"{col}_k"] = k_neighbors
    
    with col2:
        st.write("**Imputation Preview:**")
        
        if st.button("🔍 Preview Imputation Effects"):
            preview_data = _apply_imputation_preview(data, imputation_strategies, global_strategy)
            
            if preview_data is not None:
                # Show before/after comparison
                before_missing = data.isnull().sum().sum()
                after_missing = preview_data.isnull().sum().sum()
                
                st.metric("Missing values before", before_missing)
                st.metric("Missing values after", after_missing, delta=after_missing - before_missing)
                
                # Show data distribution changes
                selected_col = st.selectbox("Select column to compare:", missing_cols.index)
                if selected_col and pd.api.types.is_numeric_dtype(data[selected_col]):
                    fig = make_subplots(rows=1, cols=2, subplot_titles=['Before', 'After'])
                    
                    fig.add_trace(go.Histogram(x=data[selected_col].dropna(), name='Original'), row=1, col=1)
                    fig.add_trace(go.Histogram(x=preview_data[selected_col], name='Imputed'), row=1, col=2)
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    # Apply imputation
    if st.button("🔧 Apply Missing Value Treatment", type="primary"):
        try:
            if global_strategy == "Custom per column":
                final_data = _apply_custom_imputation(data, imputation_strategies)
            else:
                final_data = _apply_global_imputation(data, global_strategy)
            
            if final_data is not None:
                st.session_state.preview_data = final_data
                st.session_state.missing_strategy_applied = True
                st.success("✅ Missing value treatment applied!")
                # Don't rerun to stay on same tab
        except Exception as e:
            st.error(f"❌ Error applying imputation: {str(e)}")

def _handle_advanced_encoding():
    """Advanced feature encoding with multiple techniques."""
    st.subheader("🏷️ Advanced Feature Encoding")
    
    if st.session_state.get('preview_data') is None:
        st.error("❌ No data available")
        return
    
    data = st.session_state.preview_data
    target_column = st.session_state.get('target_column')
    
    # Identify categorical columns
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    categorical_cols = [col for col in categorical_cols if col != target_column]
    
    if len(categorical_cols) == 0:
        st.info("ℹ️ No categorical columns found for encoding")
        return
    
    # Categorical column analysis
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("**📊 Categorical Column Analysis**")
        
        cat_analysis = []
        for col in categorical_cols:
            unique_count = data[col].nunique()
            memory_usage = data[col].memory_usage(deep=True) / 1024
            most_frequent = data[col].mode().iloc[0] if len(data[col].mode()) > 0 else 'N/A'
            
            # Recommend encoding method
            if unique_count <= 2:
                recommended = "Binary/Label"
            elif unique_count <= 10:
                recommended = "One-Hot"
            elif unique_count <= 50:
                recommended = "Target/Ordinal"
            else:
                recommended = "Frequency/Hash"
            
            cat_analysis.append({
                'Column': col,
                'Unique Values': unique_count,
                'Memory (KB)': f"{memory_usage:.1f}",
                'Most Frequent': str(most_frequent)[:20],
                'Recommended': recommended
            })
        
        cat_df = pd.DataFrame(cat_analysis)
        st.dataframe(cat_df, use_container_width=True)
    
    with col2:
        st.write("**🎨 Encoding Methods**")
        
        encoding_info = {
            "One-Hot": "Creates binary columns for each category",
            "Label": "Assigns integer labels to categories",
            "Target": "Uses target mean for each category",
            "Ordinal": "Assigns ordered integers to categories",
            "Binary": "Binary representation of categories",
            "Frequency": "Uses category frequency as encoding"
        }
        
        for method, description in encoding_info.items():
            st.write(f"**{method}:** {description}")
    
    # Encoding strategy selection
    st.write("**🔧 Encoding Configuration**")
    
    encoding_strategies = {}
    
    # Global or custom strategy
    encoding_approach = st.radio(
        "Encoding approach:",
        ["Global strategy", "Custom per column"],
        help="Apply same encoding to all columns or customize individually"
    )
    
    if encoding_approach == "Global strategy":
        global_encoding = st.selectbox(
            "Global encoding method:",
            ["One-Hot", "Label", "Target", "Frequency"],
            help="Applied to all categorical columns"
        )
        
        # Global parameters
        if global_encoding == "One-Hot":
            max_categories = st.slider("Max categories for One-Hot:", 5, 20, 10)
            handle_unknown = st.selectbox("Handle unknown categories:", ["ignore", "error"])
        
        elif global_encoding == "Target":
            smoothing = st.slider("Target encoding smoothing:", 0.0, 1.0, 0.1)
    
    else:
        # Custom strategy per column
        for col in categorical_cols:
            unique_count = data[col].nunique()
            
            # Smart default based on cardinality
            if unique_count <= 2:
                default_method = "Label"
            elif unique_count <= 10:
                default_method = "One-Hot"
            else:
                default_method = "Target"
            
            method = st.selectbox(
                f"Encoding for {col} ({unique_count} categories):",
                ["One-Hot", "Label", "Target", "Ordinal", "Binary", "Frequency"],
                index=["One-Hot", "Label", "Target", "Ordinal", "Binary", "Frequency"].index(default_method),
                key=f"encoding_{col}"
            )
            encoding_strategies[col] = method
    
    # Encoding preview and application
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 Preview Encoding Effects"):
            if encoding_approach == "Global strategy":
                preview_result = _preview_global_encoding(data, categorical_cols, global_encoding)
            else:
                preview_result = _preview_custom_encoding(data, encoding_strategies)
            
            if preview_result:
                original_shape = data.shape
                new_shape = preview_result.shape
                
                st.write("**Shape Change:**")
                st.write(f"Before: {original_shape}")
                st.write(f"After: {new_shape}")
                st.write(f"New columns: {new_shape[1] - original_shape[1]}")
    
    with col2:
        if st.button("🎨 Apply Encoding", type="primary"):
            try:
                if encoding_approach == "Global strategy":
                    encoded_data = _apply_global_encoding(data, categorical_cols, global_encoding)
                else:
                    encoded_data = _apply_custom_encoding(data, encoding_strategies)
                
                if encoded_data is not None:
                    st.session_state.preview_data = encoded_data
                    st.session_state.encoding_applied = True
                    st.success("✅ Feature encoding applied!")
                    # Don't rerun to stay on same tab
            except Exception as e:
                st.error(f"❌ Encoding error: {str(e)}")

def _handle_advanced_scaling():
    """Advanced feature scaling with multiple techniques."""
    st.subheader("📊 Advanced Feature Scaling")
    
    if st.session_state.get('preview_data') is None:
        st.error("❌ No data available")
        return
    
    data = st.session_state.preview_data
    target_column = st.session_state.get('target_column')
    
    # Identify numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != target_column]
    
    if len(numeric_cols) == 0:
        st.info("ℹ️ No numeric columns found for scaling")
        return
    
    # Numeric column analysis
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("**📊 Numeric Column Analysis**")
        
        numeric_analysis = []
        for col in numeric_cols:
            stats = data[col].describe()
            skewness = data[col].skew()
            
            # Recommend scaling method
            if abs(skewness) > 2:
                recommended = "Robust"
            elif stats['std'] > 1000 or stats['max'] - stats['min'] > 1000:
                recommended = "MinMax"
            else:
                recommended = "Standard"
            
            numeric_analysis.append({
                'Column': col,
                'Mean': f"{stats['mean']:.2f}",
                'Std': f"{stats['std']:.2f}",
                'Min': f"{stats['min']:.2f}",
                'Max': f"{stats['max']:.2f}",
                'Skewness': f"{skewness:.2f}",
                'Recommended': recommended
            })
        
        numeric_df = pd.DataFrame(numeric_analysis)
        st.dataframe(numeric_df, use_container_width=True)
        
        # Distribution visualization
        selected_cols = st.multiselect(
            "Select columns to visualize distributions:",
            numeric_cols,
            default=numeric_cols[:3]
        )
        
        if selected_cols and st.button("📊 Show Distributions"):
            fig = make_subplots(
                rows=len(selected_cols), cols=2,
                subplot_titles=[f'{col} - Original' for col in selected_cols] + 
                              [f'{col} - After Scaling' for col in selected_cols]
            )
            
            for i, col in enumerate(selected_cols):
                # Original distribution
                fig.add_trace(go.Histogram(x=data[col], name=f'{col} Original'), 
                            row=i+1, col=1)
                
                # Scaled distribution preview (using StandardScaler as example)
                scaled_data = StandardScaler().fit_transform(data[[col]])
                fig.add_trace(go.Histogram(x=scaled_data.flatten(), name=f'{col} Scaled'), 
                            row=i+1, col=2)
            
            fig.update_layout(height=300*len(selected_cols), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**🔧 Scaling Methods**")
        
        scaling_info = {
            "Standard": "Mean=0, Std=1 (Z-score normalization)",
            "MinMax": "Scale to range [0,1]",
            "Robust": "Uses median and IQR (outlier resistant)",
            "Quantile": "Maps to uniform distribution",
            "Power": "Box-Cox transformation",
            "Unit Vector": "Scale to unit norm"
        }
        
        for method, description in scaling_info.items():
            st.write(f"**{method}:** {description}")
    
    # Scaling configuration
    st.write("**⚙️ Scaling Configuration**")
    
    scaling_approach = st.radio(
        "Scaling approach:",
        ["Global strategy", "Custom per column"],
        help="Apply same scaling to all columns or customize individually"
    )
    
    scaling_strategies = {}
    
    if scaling_approach == "Global strategy":
        global_scaling = st.selectbox(
            "Global scaling method:",
            ["Standard", "MinMax", "Robust", "Quantile"],
            help="Applied to all numeric columns"
        )
        
        # Additional parameters
        if global_scaling == "MinMax":
            feature_range = st.slider("Feature range:", 0.0, 1.0, (0.0, 1.0))
        elif global_scaling == "Quantile":
            n_quantiles = st.slider("Number of quantiles:", 10, 1000, 100)
    
    else:
        # Custom strategy per column
        for col in numeric_cols:
            skewness = data[col].skew()
            
            # Smart default
            if abs(skewness) > 2:
                default_method = "Robust"
            else:
                default_method = "Standard"
            
            method = st.selectbox(
                f"Scaling for {col} (skew: {skewness:.2f}):",
                ["Standard", "MinMax", "Robust", "Quantile", "Power"],
                index=["Standard", "MinMax", "Robust", "Quantile", "Power"].index(default_method),
                key=f"scaling_{col}"
            )
            scaling_strategies[col] = method
    
    # Apply scaling
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 Preview Scaling Effects"):
            # Show before/after statistics
            if scaling_approach == "Global strategy":
                scaled_preview = _preview_global_scaling(data, numeric_cols, global_scaling)
            else:
                scaled_preview = _preview_custom_scaling(data, scaling_strategies)
            
            if scaled_preview is not None:
                st.write("**Scaling Preview:**")
                comparison_stats = []
                
                for col in numeric_cols[:3]:  # Show first 3 columns
                    original_stats = data[col].describe()
                    scaled_stats = scaled_preview[col].describe()
                    
                    comparison_stats.append({
                        'Column': col,
                        'Original Mean': f"{original_stats['mean']:.2f}",
                        'Scaled Mean': f"{scaled_stats['mean']:.2f}",
                        'Original Std': f"{original_stats['std']:.2f}",
                        'Scaled Std': f"{scaled_stats['std']:.2f}"
                    })
                
                comp_df = pd.DataFrame(comparison_stats)
                st.dataframe(comp_df, use_container_width=True)
    
    with col2:
        if st.button("📊 Apply Scaling", type="primary"):
            try:
                if scaling_approach == "Global strategy":
                    scaled_data = _apply_global_scaling(data, numeric_cols, global_scaling)
                else:
                    scaled_data = _apply_custom_scaling(data, scaling_strategies)
                
                if scaled_data is not None:
                    st.session_state.preview_data = scaled_data
                    st.session_state.scaling_applied = True
                    st.success("✅ Feature scaling applied!")
                    # Don't rerun to stay on same tab
            except Exception as e:
                st.error(f"❌ Scaling error: {str(e)}")

def _handle_feature_engineering():
    """Advanced feature engineering techniques."""
    st.subheader("🔍 Advanced Feature Engineering")
    
    if st.session_state.get('preview_data') is None:
        st.error("❌ No data available")
        return
    
    data = st.session_state.preview_data
    target_column = st.session_state.get('target_column')
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != target_column]
    
    # Feature engineering options
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**🏗️ Feature Creation**")
        
        # Polynomial features
        create_polynomial = st.checkbox("Create Polynomial Features", 
                                       help="Generate interaction and polynomial terms")
        if create_polynomial:
            poly_degree = st.slider("Polynomial degree:", 2, 4, 2)
            include_interaction = st.checkbox("Include interaction terms", value=True)
        
        # Mathematical transformations
        create_math_features = st.checkbox("Create Mathematical Transformations",
                                         help="Log, sqrt, square transformations")
        if create_math_features:
            math_operations = st.multiselect(
                "Select transformations:",
                ["Log", "Square Root", "Square", "Reciprocal"],
                default=["Log", "Square"]
            )
        
        # Statistical features
        create_statistical = st.checkbox("Create Statistical Features",
                                       help="Rolling statistics and aggregations")
        if create_statistical:
            window_size = st.slider("Rolling window size:", 3, 10, 5)
            stat_operations = st.multiselect(
                "Statistical operations:",
                ["Mean", "Std", "Min", "Max", "Median"],
                default=["Mean", "Std"]
            )
        
        # Binning features
        create_binning = st.checkbox("Create Binned Features",
                                   help="Convert continuous to categorical")
        if create_binning:
            binning_cols = st.multiselect("Columns to bin:", numeric_cols)
            n_bins = st.slider("Number of bins:", 3, 10, 5)
            binning_strategy = st.selectbox("Binning strategy:", 
                                          ["uniform", "quantile", "kmeans"])
    
    with col2:
        st.write("**🎯 Feature Selection**")
        
        # Feature selection methods
        apply_feature_selection = st.checkbox("Apply Feature Selection",
                                            help="Select most important features")
        if apply_feature_selection:
            selection_method = st.selectbox(
                "Selection method:",
                ["SelectKBest", "Variance Threshold", "Correlation Filter"],
                help="Method for selecting features"
            )
            
            if selection_method == "SelectKBest":
                k_features = st.slider("Number of features to select:", 5, min(50, len(data.columns)), 20)
            elif selection_method == "Variance Threshold":
                variance_threshold = st.slider("Variance threshold:", 0.0, 1.0, 0.1)
            elif selection_method == "Correlation Filter":
                corr_threshold = st.slider("Correlation threshold:", 0.5, 1.0, 0.95)
        
        # PCA
        apply_pca = st.checkbox("Apply PCA (Dimensionality Reduction)",
                              help="Reduce feature dimensions using PCA")
        if apply_pca:
            n_components = st.slider("Number of components:", 2, min(20, len(numeric_cols)), 10)
            explained_variance = st.slider("Min explained variance:", 0.8, 0.99, 0.95)
    
    # Feature engineering preview and application
    if st.button("🔍 Preview Feature Engineering Effects"):
        try:
            engineered_data = data.copy()
            
            # Apply selected feature engineering
            if create_polynomial and len(numeric_cols) > 0:
                engineered_data = _create_polynomial_features(engineered_data, numeric_cols[:5], poly_degree)
            
            if create_math_features and len(numeric_cols) > 0:
                engineered_data = _create_math_features(engineered_data, numeric_cols, math_operations)
            
            if create_binning and binning_cols:
                engineered_data = _create_binned_features(engineered_data, binning_cols, n_bins, binning_strategy)
            
            # Show preview
            original_shape = data.shape
            new_shape = engineered_data.shape
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original Features", original_shape[1])
            with col2:
                st.metric("Engineered Features", new_shape[1], delta=new_shape[1] - original_shape[1])
            with col3:
                memory_change = (engineered_data.memory_usage(deep=True).sum() - data.memory_usage(deep=True).sum()) / 1024 / 1024
                st.metric("Memory Change (MB)", f"{memory_change:.1f}")
                
        except Exception as e:
            st.error(f"❌ Preview error: {str(e)}")
    
    # Apply feature engineering
    if st.button("🏗️ Apply Feature Engineering", type="primary"):
        try:
            engineered_data = data.copy()
            
            # Apply all selected feature engineering techniques
            if create_polynomial and len(numeric_cols) > 0:
                engineered_data = _create_polynomial_features(engineered_data, numeric_cols[:5], poly_degree)
            
            if create_math_features and len(numeric_cols) > 0:
                engineered_data = _create_math_features(engineered_data, numeric_cols, math_operations)
            
            if create_statistical and len(numeric_cols) > 0:
                engineered_data = _create_statistical_features(engineered_data, numeric_cols, window_size, stat_operations)
            
            if create_binning and binning_cols:
                engineered_data = _create_binned_features(engineered_data, binning_cols, n_bins, binning_strategy)
            
            # Apply feature selection
            if apply_feature_selection:
                engineered_data = _apply_feature_selection(engineered_data, target_column, selection_method, 
                                                         k_features if selection_method == "SelectKBest" else None)
            
            # Apply PCA
            if apply_pca and len(numeric_cols) > 0:
                engineered_data = _apply_pca_transformation(engineered_data, target_column, n_components, explained_variance)
            
            if engineered_data is not None:
                st.session_state.preview_data = engineered_data
                st.session_state.features_engineered = True
                st.success("✅ Feature engineering applied!")
                # Don't rerun to stay on same tab
                
        except Exception as e:
            st.error(f"❌ Feature engineering error: {str(e)}")

def _handle_final_review_and_export(data_handler):
    """Final review and export of preprocessed data."""
    st.subheader("✅ Final Review & Export")
    
    if st.session_state.get('preview_data') is None:
        st.error("❌ No data available")
        return
    
    original_data = st.session_state.data
    final_data = st.session_state.preview_data
    target_column = st.session_state.get('target_column')
    
    # Comprehensive comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**📊 Transformation Summary**")
        
        comparison_metrics = {
            'Metric': ['Rows', 'Columns', 'Memory (MB)', 'Missing Values', 'Numeric Columns', 'Categorical Columns'],
            'Original': [
                f"{len(original_data):,}",
                f"{len(original_data.columns)}",
                f"{original_data.memory_usage(deep=True).sum() / 1024 / 1024:.1f}",
                f"{original_data.isnull().sum().sum():,}",
                f"{len(original_data.select_dtypes(include=[np.number]).columns)}",
                f"{len(original_data.select_dtypes(include=['object', 'category']).columns)}"
            ],
            'Final': [
                f"{len(final_data):,}",
                f"{len(final_data.columns)}",
                f"{final_data.memory_usage(deep=True).sum() / 1024 / 1024:.1f}",
                f"{final_data.isnull().sum().sum():,}",
                f"{len(final_data.select_dtypes(include=[np.number]).columns)}",
                f"{len(final_data.select_dtypes(include=['object', 'category']).columns)}"
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_metrics)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Data quality assessment
        st.write("**🎯 Data Quality Assessment**")
        
        quality_checks = []
        
        # Check data integrity
        if target_column and target_column in final_data.columns:
            quality_checks.append("✅ Target column preserved")
        else:
            quality_checks.append("❌ Target column missing")
        
        if len(final_data) > 0:
            quality_checks.append("✅ Data rows retained")
        else:
            quality_checks.append("❌ No data rows remaining")
        
        if final_data.isnull().sum().sum() == 0:
            quality_checks.append("✅ No missing values")
        else:
            quality_checks.append(f"⚠️ {final_data.isnull().sum().sum()} missing values remain")
        
        if len(final_data.columns) >= 2:
            quality_checks.append("✅ Sufficient features for modeling")
        else:
            quality_checks.append("❌ Insufficient features")
        
        for check in quality_checks:
            st.write(check)
    
    with col2:
        st.write("**📋 Applied Transformations**")
        
        applied_transformations = []
        
        if st.session_state.get('columns_removed'):
            applied_transformations.append(f"🗑️ Removed {len(st.session_state.columns_removed)} columns")
        
        if st.session_state.get('outliers_removed'):
            applied_transformations.append(f"🎯 Removed {st.session_state.outliers_removed} outliers")
        
        if st.session_state.get('missing_strategy_applied'):
            applied_transformations.append("❓ Applied missing value treatment")
        
        if st.session_state.get('encoding_applied'):
            applied_transformations.append("🏷️ Applied feature encoding")
        
        if st.session_state.get('scaling_applied'):
            applied_transformations.append("📊 Applied feature scaling")
        
        if st.session_state.get('features_engineered'):
            applied_transformations.append("🔍 Applied feature engineering")
        
        if applied_transformations:
            for transform in applied_transformations:
                st.write(transform)
        else:
            st.info("No transformations applied")
        
        # Configuration summary
        st.write("**⚙️ Configuration**")
        config = {
            'columns_removed': st.session_state.get('columns_removed', []),
            'outliers_removed': st.session_state.get('outliers_removed', 0),
            'missing_strategy': st.session_state.get('missing_strategy_applied', False),
            'encoding_applied': st.session_state.get('encoding_applied', False),
            'scaling_applied': st.session_state.get('scaling_applied', False),
            'features_engineered': st.session_state.get('features_engineered', False)
        }
        st.session_state.preprocessing_config = config
    
    # Data preview and download
    st.write("**👀 Final Data Preview**")
    
    preview_size = st.slider("Preview rows:", 5, min(100, len(final_data)), 10)
    st.dataframe(final_data.head(preview_size), use_container_width=True)
    
    # Export options
    st.write("**💾 Export Options**")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📥 Download Processed Data"):
            csv = final_data.to_csv(index=False)
            st.download_button(
                label="📄 Download CSV",
                data=csv,
                file_name=f"processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col2:
        if st.button("📊 Download Excel"):
            from io import BytesIO
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                final_data.to_excel(writer, sheet_name='Processed_Data', index=False)
                comparison_df.to_excel(writer, sheet_name='Summary', index=False)
            
            st.download_button(
                label="📈 Download Excel",
                data=buffer.getvalue(),
                file_name=f"processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
    
    with col3:
        if st.button("🔄 Reset All Preprocessing"):
            # Reset to original data
            st.session_state.preview_data = original_data.copy()
            
            # Clear all preprocessing flags
            reset_keys = ['columns_removed', 'outliers_removed', 'missing_strategy_applied', 
                         'encoding_applied', 'scaling_applied', 'features_engineered']
            for key in reset_keys:
                if key in st.session_state:
                    del st.session_state[key]
            
            st.success("✅ Reset to original data!")
            st.rerun()
    
    with col4:
        # Validation and proceed
        can_proceed = (
            target_column and target_column in final_data.columns and
            not final_data.empty and
            len(final_data.columns) >= 2
        )
        
        if st.button("🎯 Proceed to Training", type="primary", disabled=not can_proceed):
            if can_proceed:
                st.session_state.current_step = "train"
                st.rerun()
            else:
                st.error("❌ Data validation failed. Please fix issues before proceeding.")

# Helper functions for preprocessing operations

def _detect_outliers(data, numeric_cols, method, threshold):
    """Detect outliers using specified method."""
    outlier_count = 0
    
    if method == "IQR (Interquartile Range)":
        for col in numeric_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
            outlier_count += len(outliers)
    
    elif method == "Z-Score":
        from scipy import stats
        for col in numeric_cols:
            z_scores = np.abs(stats.zscore(data[col].dropna()))
            outliers = data[z_scores > threshold]
            outlier_count += len(outliers)
    
    elif method == "Isolation Forest":
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outliers = iso_forest.fit_predict(data[numeric_cols].fillna(data[numeric_cols].mean()))
        outlier_count = sum(outliers == -1)
    
    return outlier_count

def _remove_outliers(data, numeric_cols, method, threshold):
    """Remove outliers using specified method."""
    clean_data = data.copy()
    
    if method == "IQR (Interquartile Range)":
        for col in numeric_cols:
            Q1 = clean_data[col].quantile(0.25)
            Q3 = clean_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            clean_data = clean_data[(clean_data[col] >= lower_bound) & (clean_data[col] <= upper_bound)]
    
    elif method == "Z-Score":
        from scipy import stats
        for col in numeric_cols:
            z_scores = np.abs(stats.zscore(clean_data[col].dropna()))
            clean_data = clean_data[z_scores <= threshold]
    
    elif method == "Isolation Forest":
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outlier_pred = iso_forest.fit_predict(clean_data[numeric_cols].fillna(clean_data[numeric_cols].mean()))
        clean_data = clean_data[outlier_pred == 1]
    
    return clean_data

def _apply_imputation_preview(data, strategies, global_strategy):
    """Preview imputation effects."""
    preview_data = data.copy()
    
    try:
        if global_strategy == "Drop rows":
            preview_data = preview_data.dropna()
        elif global_strategy == "Forward fill":
            preview_data = preview_data.fillna(method='ffill')
        elif global_strategy == "Backward fill":
            preview_data = preview_data.fillna(method='bfill')
        elif global_strategy == "Interpolate":
            numeric_cols = preview_data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                preview_data[col] = preview_data[col].interpolate()
        
        return preview_data
    except:
        return None

def _apply_custom_imputation(data, strategies):
    """Apply custom imputation strategies."""
    imputed_data = data.copy()
    
    for col, strategy in strategies.items():
        if col.endswith('_constant') or col.endswith('_k'):
            continue
            
        if strategy == "Mean":
            imputed_data[col] = imputed_data[col].fillna(imputed_data[col].mean())
        elif strategy == "Median":
            imputed_data[col] = imputed_data[col].fillna(imputed_data[col].median())
        elif strategy == "Mode":
            mode_val = imputed_data[col].mode()
            if len(mode_val) > 0:
                imputed_data[col] = imputed_data[col].fillna(mode_val[0])
        elif strategy == "Constant":
            const_val = strategies.get(f"{col}_constant", 0)
            imputed_data[col] = imputed_data[col].fillna(const_val)
        elif strategy == "KNN":
            k = strategies.get(f"{col}_k", 5)
            imputer = KNNImputer(n_neighbors=k)
            imputed_data[[col]] = imputer.fit_transform(imputed_data[[col]])
    
    return imputed_data

def _apply_global_imputation(data, strategy):
    """Apply global imputation strategy."""
    imputed_data = data.copy()
    
    if strategy == "Drop rows":
        imputed_data = imputed_data.dropna()
    elif strategy == "Forward fill":
        imputed_data = imputed_data.fillna(method='ffill')
    elif strategy == "Backward fill":
        imputed_data = imputed_data.fillna(method='bfill')
    elif strategy == "Interpolate":
        numeric_cols = imputed_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            imputed_data[col] = imputed_data[col].interpolate()
    
    return imputed_data

def _preview_global_encoding(data, categorical_cols, encoding_method):
    """Preview global encoding effects."""
    preview_data = data.copy()
    
    if encoding_method == "One-Hot":
        preview_data = pd.get_dummies(preview_data, columns=categorical_cols[:3])  # Preview first 3
    elif encoding_method == "Label":
        for col in categorical_cols[:3]:
            le = LabelEncoder()
            preview_data[col] = le.fit_transform(preview_data[col].astype(str))
    
    return preview_data

def _apply_global_encoding(data, categorical_cols, encoding_method):
    """Apply global encoding method."""
    encoded_data = data.copy()
    
    if encoding_method == "One-Hot":
        # Limit categories to prevent memory explosion
        for col in categorical_cols:
            if encoded_data[col].nunique() > 20:
                top_cats = encoded_data[col].value_counts().head(15).index
                encoded_data[col] = encoded_data[col].apply(lambda x: x if x in top_cats else 'Other')
        
        encoded_data = pd.get_dummies(encoded_data, columns=categorical_cols, prefix=categorical_cols)
    
    elif encoding_method == "Label":
        for col in categorical_cols:
            le = LabelEncoder()
            encoded_data[col] = le.fit_transform(encoded_data[col].astype(str))
    
    elif encoding_method == "Frequency":
        for col in categorical_cols:
            freq_map = encoded_data[col].value_counts().to_dict()
            encoded_data[col] = encoded_data[col].map(freq_map)
    
    return encoded_data

def _apply_custom_encoding(data, strategies):
    """Apply custom encoding strategies."""
    encoded_data = data.copy()
    
    for col, method in strategies.items():
        if method == "One-Hot":
            # Limit categories
            if encoded_data[col].nunique() > 15:
                top_cats = encoded_data[col].value_counts().head(10).index
                encoded_data[col] = encoded_data[col].apply(lambda x: x if x in top_cats else 'Other')
            
            dummies = pd.get_dummies(encoded_data[col], prefix=col)
            encoded_data = pd.concat([encoded_data.drop(columns=[col]), dummies], axis=1)
        
        elif method == "Label":
            le = LabelEncoder()
            encoded_data[col] = le.fit_transform(encoded_data[col].astype(str))
        
        elif method == "Frequency":
            freq_map = encoded_data[col].value_counts().to_dict()
            encoded_data[col] = encoded_data[col].map(freq_map)
    
    return encoded_data

def _preview_global_scaling(data, numeric_cols, scaling_method):
    """Preview global scaling effects."""
    preview_data = data.copy()
    
    if scaling_method == "Standard":
        scaler = StandardScaler()
    elif scaling_method == "MinMax":
        scaler = MinMaxScaler()
    elif scaling_method == "Robust":
        scaler = RobustScaler()
    else:
        return preview_data
    
    preview_data[numeric_cols] = scaler.fit_transform(preview_data[numeric_cols])
    return preview_data

def _apply_global_scaling(data, numeric_cols, scaling_method):
    """Apply global scaling method."""
    scaled_data = data.copy()
    
    if scaling_method == "Standard":
        scaler = StandardScaler()
    elif scaling_method == "MinMax":
        scaler = MinMaxScaler()
    elif scaling_method == "Robust":
        scaler = RobustScaler()
    else:
        return scaled_data
    
    scaled_data[numeric_cols] = scaler.fit_transform(scaled_data[numeric_cols])
    return scaled_data

def _preview_custom_scaling(data, strategies):
    """Preview custom scaling effects for different columns."""
    preview_data = data.copy()
    
    try:
        for col, method in strategies.items():
            if col not in preview_data.columns:
                continue
                
            if method == "Standard":
                scaler = StandardScaler()
                preview_data[[col]] = scaler.fit_transform(preview_data[[col]])
            elif method == "MinMax":
                scaler = MinMaxScaler()
                preview_data[[col]] = scaler.fit_transform(preview_data[[col]])
            elif method == "Robust":
                scaler = RobustScaler()
                preview_data[[col]] = scaler.fit_transform(preview_data[[col]])
            elif method == "Quantile":
                from sklearn.preprocessing import QuantileTransformer
                scaler = QuantileTransformer(n_quantiles=100, random_state=42)
                preview_data[[col]] = scaler.fit_transform(preview_data[[col]])
            elif method == "Power":
                from sklearn.preprocessing import PowerTransformer
                try:
                    scaler = PowerTransformer(method='box-cox', standardize=True)
                    # Box-Cox requires positive values
                    if preview_data[col].min() > 0:
                        preview_data[[col]] = scaler.fit_transform(preview_data[[col]])
                    else:
                        # Use Yeo-Johnson for non-positive values
                        scaler = PowerTransformer(method='yeo-johnson', standardize=True)
                        preview_data[[col]] = scaler.fit_transform(preview_data[[col]])
                except:
                    # Fallback to standard scaling if power transform fails
                    scaler = StandardScaler()
                    preview_data[[col]] = scaler.fit_transform(preview_data[[col]])
        
        return preview_data
        
    except Exception as e:
        logger.error(f"Preview custom scaling error: {e}")
        return None

def _preview_custom_encoding(data, strategies):
    """Preview custom encoding effects for different columns."""
    preview_data = data.copy()
    
    try:
        for col, method in strategies.items():
            if col not in preview_data.columns:
                continue
                
            if method == "One-Hot":
                # Limit categories to prevent memory explosion in preview
                if preview_data[col].nunique() > 10:
                    top_cats = preview_data[col].value_counts().head(5).index
                    preview_data[col] = preview_data[col].apply(lambda x: x if x in top_cats else 'Other')
                
                dummies = pd.get_dummies(preview_data[col], prefix=f'{col}_preview')
                preview_data = pd.concat([preview_data.drop(columns=[col]), dummies], axis=1)
            
            elif method == "Label":
                le = LabelEncoder()
                preview_data[col] = le.fit_transform(preview_data[col].astype(str))
            
            elif method == "Target":
                # Simplified target encoding for preview
                if st.session_state.get('target_column') in preview_data.columns:
                    target_col = st.session_state.target_column
                    target_mean = preview_data.groupby(col)[target_col].mean()
                    preview_data[col] = preview_data[col].map(target_mean)
            
            elif method == "Frequency":
                freq_map = preview_data[col].value_counts().to_dict()
                preview_data[col] = preview_data[col].map(freq_map)
            
            elif method == "Binary":
                # Simple binary encoding
                le = LabelEncoder()
                encoded = le.fit_transform(preview_data[col].astype(str))
                # Convert to binary representation
                n_bits = int(np.ceil(np.log2(len(le.classes_))))
                for i in range(n_bits):
                    preview_data[f'{col}_binary_{i}'] = (encoded >> i) & 1
                preview_data = preview_data.drop(columns=[col])

    except:
        logger.error(f"Apply custom scaling error: {e}")
        return data  # Return original data if scaling fails
        
def _apply_custom_scaling(data, strategies):
    """Apply custom scaling strategies to different columns."""
    scaled_data = data.copy()
    
    try:
        for col, method in strategies.items():
            if col not in scaled_data.columns:
                continue
                
            if method == "Standard":
                scaler = StandardScaler()
                scaled_data[[col]] = scaler.fit_transform(scaled_data[[col]])
            elif method == "MinMax":
                scaler = MinMaxScaler()
                scaled_data[[col]] = scaler.fit_transform(scaled_data[[col]])
            elif method == "Robust":
                scaler = RobustScaler()
                scaled_data[[col]] = scaler.fit_transform(scaled_data[[col]])
            elif method == "Quantile":
                from sklearn.preprocessing import QuantileTransformer
                scaler = QuantileTransformer(n_quantiles=100, random_state=42)
                scaled_data[[col]] = scaler.fit_transform(scaled_data[[col]])
            elif method == "Power":
                from sklearn.preprocessing import PowerTransformer
                try:
                    scaler = PowerTransformer(method='box-cox', standardize=True)
                    # Box-Cox requires positive values
                    if scaled_data[col].min() > 0:
                        scaled_data[[col]] = scaler.fit_transform(scaled_data[[col]])
                    else:
                        # Use Yeo-Johnson for non-positive values
                        scaler = PowerTransformer(method='yeo-johnson', standardize=True)
                        scaled_data[[col]] = scaler.fit_transform(scaled_data[[col]])
                except Exception as e:
                    logger.warning(f"Power transform failed for {col}, using Standard scaling: {e}")
                    # Fallback to standard scaling if power transform fails
                    scaler = StandardScaler()
                    scaled_data[[col]] = scaler.fit_transform(scaled_data[[col]])
        
        return scaled_data
        
    except Exception as e:
        logger.error(f"Apply custom scaling error: {e}")
        return data  # Return original data if scaling fails

def _create_polynomial_features(data, numeric_cols, degree):
    """Create polynomial features."""
    from sklearn.preprocessing import PolynomialFeatures
    
    poly_data = data.copy()
    
    # Limit to prevent memory explosion
    selected_cols = numeric_cols[:3]
    
    poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
    poly_features = poly.fit_transform(poly_data[selected_cols])
    
    # Create feature names
    feature_names = poly.get_feature_names_out(selected_cols)
    
    # Add new features
    for i, name in enumerate(feature_names):
        if name not in selected_cols:  # Skip original features
            poly_data[f'poly_{name}'] = poly_features[:, i]
    
    return poly_data

def _create_math_features(data, numeric_cols, operations):
    """Create mathematical transformation features."""
    math_data = data.copy()
    
    for col in numeric_cols[:5]:  # Limit to first 5 columns
        if "Log" in operations:
            # Handle negative values
            min_val = math_data[col].min()
            if min_val <= 0:
                math_data[f'{col}_log'] = np.log1p(math_data[col] - min_val + 1)
            else:
                math_data[f'{col}_log'] = np.log(math_data[col])
        
        if "Square Root" in operations:
            # Handle negative values
            math_data[f'{col}_sqrt'] = np.sqrt(np.abs(math_data[col]))
        
        if "Square" in operations:
            math_data[f'{col}_square'] = math_data[col] ** 2
        
        if "Reciprocal" in operations:
            # Avoid division by zero
            math_data[f'{col}_reciprocal'] = 1 / (math_data[col] + 1e-8)
    
    return math_data

def _create_statistical_features(data, numeric_cols, window_size, operations):
    """Create statistical features."""
    stat_data = data.copy()
    
    for col in numeric_cols[:3]:  # Limit to first 3 columns
        if "Mean" in operations:
            stat_data[f'{col}_rolling_mean'] = stat_data[col].rolling(window=window_size).mean()
        
        if "Std" in operations:
            stat_data[f'{col}_rolling_std'] = stat_data[col].rolling(window=window_size).std()
        
        if "Min" in operations:
            stat_data[f'{col}_rolling_min'] = stat_data[col].rolling(window=window_size).min()
        
        if "Max" in operations:
            stat_data[f'{col}_rolling_max'] = stat_data[col].rolling(window=window_size).max()
    
    return stat_data.fillna(stat_data.mean())

def _create_binned_features(data, binning_cols, n_bins, strategy):
    """Create binned features."""
    binned_data = data.copy()
    
    for col in binning_cols:
        if strategy == "uniform":
            binned_data[f'{col}_binned'] = pd.cut(binned_data[col], bins=n_bins, labels=False)
        elif strategy == "quantile":
            binned_data[f'{col}_binned'] = pd.qcut(binned_data[col], q=n_bins, labels=False, duplicates='drop')
        elif strategy == "kmeans":
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_bins, random_state=42)
            binned_data[f'{col}_binned'] = kmeans.fit_predict(binned_data[[col]])
    
    return binned_data

def _apply_feature_selection(data, target_column, method, k_features=None):
    """Apply feature selection."""
    if target_column not in data.columns:
        return data
    
    feature_data = data.copy()
    X = feature_data.drop(columns=[target_column])
    y = feature_data[target_column]
    
    if method == "SelectKBest":
        if st.session_state.get('problem_type') == 'classification':
            selector = SelectKBest(score_func=f_classif, k=min(k_features, len(X.columns)))
        else:
            selector = SelectKBest(score_func=f_regression, k=min(k_features, len(X.columns)))
        
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        result_data = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        result_data[target_column] = y
        
        return result_data
    
    return data

def _apply_pca_transformation(data, target_column, n_components, explained_variance):
    """Apply PCA transformation."""
    if target_column not in data.columns:
        return data
    
    pca_data = data.copy()
    numeric_cols = pca_data.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != target_column]
    
    if len(numeric_cols) == 0:
        return data
    
    # Apply PCA
    pca = PCA(n_components=min(n_components, len(numeric_cols)))
    pca_features = pca.fit_transform(pca_data[numeric_cols])
    
    # Check explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components_needed = np.argmax(cumulative_variance >= explained_variance) + 1
    
    # Create PCA dataframe
    pca_cols = [f'PCA_{i+1}' for i in range(min(n_components_needed, pca_features.shape[1]))]
    pca_df = pd.DataFrame(pca_features[:, :len(pca_cols)], columns=pca_cols, index=pca_data.index)
    
    # Keep non-numeric columns and target
    non_numeric_cols = pca_data.select_dtypes(exclude=[np.number]).columns
    result_data = pd.concat([pca_df, pca_data[non_numeric_cols], pca_data[[target_column]]], axis=1)
    
    return result_data
    
"""
Advanced Preprocessing Page
Comprehensive data preprocessing with advanced features and visualizations
"""