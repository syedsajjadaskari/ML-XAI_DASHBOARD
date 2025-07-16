"""
Visualizer Module
Handles all plotting and visualization functionality
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from typing import Dict, Any, Optional, List
import logging
import streamlit as st

logger = logging.getLogger(__name__)

class Visualizer:
    """Handles all visualization operations."""
    
    def __init__(self, config: Dict[str, Any]):
        # Handle None config gracefully
        if config is None:
            config = {}
            
        self.config = config
        visualization_config = config.get('visualization', {})
        self.theme = visualization_config.get('theme', 'plotly_white')
        
        # Set default plotly template
        self.template = {
            'plotly_white': 'plotly_white',
            'plotly_dark': 'plotly_dark',
            'ggplot2': 'ggplot2',
            'seaborn': 'seaborn',
            'simple_white': 'simple_white'
        }.get(self.theme, 'plotly_white')
    
    def plot_distribution(self, data: pd.DataFrame, column: str) -> go.Figure:
        """Plot distribution of a numeric column."""
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Histogram', 'Box Plot', 'Violin Plot', 'Q-Q Plot'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Histogram
            fig.add_trace(
                go.Histogram(
                    x=data[column],
                    name='Distribution',
                    nbinsx=30,
                    opacity=0.7,
                    marker_color='lightblue'
                ),
                row=1, col=1
            )
            
            # Box plot
            fig.add_trace(
                go.Box(
                    y=data[column],
                    name='Box Plot',
                    marker_color='lightgreen'
                ),
                row=1, col=2
            )
            
            # Violin plot
            fig.add_trace(
                go.Violin(
                    y=data[column],
                    name='Violin Plot',
                    marker_color='lightcoral'
                ),
                row=2, col=1
            )
            
            # Q-Q plot (approximation using sorted values)
            sorted_data = np.sort(data[column].dropna())
            n = len(sorted_data)
            theoretical_quantiles = np.linspace(0, 1, n)
            
            fig.add_trace(
                go.Scatter(
                    x=theoretical_quantiles,
                    y=sorted_data,
                    mode='markers',
                    name='Q-Q Plot',
                    marker_color='purple'
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title=f'Distribution Analysis: {column}',
                template=self.template,
                height=600,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Data overview plot error: {e}")
            return self._create_error_plot(f"Error creating data overview: {str(e)}")(f"Distribution plot error: {e}")
            return self._create_error_plot(f"Error creating distribution plot: {str(e)}")
    
    def plot_countplot(self, data: pd.DataFrame, column: str) -> go.Figure:
        """Plot count plot for categorical column."""
        try:
            value_counts = data[column].value_counts()
            
            # Limit to top 20 categories if more exist
            if len(value_counts) > 20:
                value_counts = value_counts.head(20)
                title_suffix = " (Top 20)"
            else:
                title_suffix = ""
            
            fig = go.Figure()
            
            # Bar plot
            fig.add_trace(
                go.Bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    marker_color='lightblue',
                    text=value_counts.values,
                    textposition='auto'
                )
            )
            
            fig.update_layout(
                title=f'Count Plot: {column}{title_suffix}',
                xaxis_title=column,
                yaxis_title='Count',
                template=self.template,
                height=400
            )
            
            # Rotate x-axis labels if too many categories
            if len(value_counts) > 10:
                fig.update_xaxes(tickangle=45)
            
            return fig
            
        except Exception as e:
            logger.error(f"Count plot error: {e}")
            return self._create_error_plot(f"Error creating count plot: {str(e)}")
    
    def plot_correlation_heatmap(self, data: pd.DataFrame) -> go.Figure:
        """Plot correlation heatmap."""
        try:
            correlation_matrix = data.corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(correlation_matrix.values, 2),
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title='Correlation Heatmap',
                template=self.template,
                height=600,
                width=600
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Correlation heatmap error: {e}")
            return self._create_error_plot(f"Error creating correlation heatmap: {str(e)}")
    
    def plot_missing_data(self, missing_df: pd.DataFrame) -> go.Figure:
        """Plot missing data analysis."""
        try:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Missing Count', 'Missing Percentage'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Missing count bar plot
            fig.add_trace(
                go.Bar(
                    x=missing_df['Column'],
                    y=missing_df['Missing Count'],
                    name='Missing Count',
                    marker_color='red',
                    opacity=0.7
                ),
                row=1, col=1
            )
            
            # Missing percentage bar plot
            fig.add_trace(
                go.Bar(
                    x=missing_df['Column'],
                    y=missing_df['Missing %'],
                    name='Missing %',
                    marker_color='orange',
                    opacity=0.7
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                title='Missing Data Analysis',
                template=self.template,
                height=400,
                showlegend=False
            )
            
            # Rotate x-axis labels
            fig.update_xaxes(tickangle=45)
            
            return fig
            
        except Exception as e:
            logger.error(f"Missing data plot error: {e}")
            return self._create_error_plot(f"Error creating missing data plot: {str(e)}")
    
    def plot_feature_importance(self, importance_data: pd.Series) -> go.Figure:
        """Plot feature importance."""
        try:
            # Sort by absolute values
            importance_data = importance_data.sort_values(key=abs, ascending=True)
            
            # Take top 20 features
            if len(importance_data) > 20:
                importance_data = importance_data.tail(20)
            
            colors = ['red' if x < 0 else 'blue' for x in importance_data.values]
            
            fig = go.Figure()
            
            fig.add_trace(
                go.Bar(
                    x=importance_data.values,
                    y=importance_data.index,
                    orientation='h',
                    marker_color=colors,
                    opacity=0.7
                )
            )
            
            fig.update_layout(
                title='Feature Importance',
                xaxis_title='Importance',
                yaxis_title='Features',
                template=self.template,
                height=600
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Feature importance plot error: {e}")
            return self._create_error_plot(f"Error creating feature importance plot: {str(e)}")
    
    def plot_model_evaluation(self, model, plot_type: str):
        """Plot model evaluation using PyCaret."""
        try:
            # Import appropriate PyCaret module
            import pycaret.classification as pc
            import pycaret.regression as pr
            
            # Determine which module to use based on session state
            if hasattr(st.session_state, 'problem_type'):
                if st.session_state.problem_type == 'classification':
                    plot = pc.plot_model(model, plot=plot_type, display_format='streamlit')
                else:
                    plot = pr.plot_model(model, plot=plot_type, display_format='streamlit')
            else:
                # Default to classification
                plot = pc.plot_model(model, plot=plot_type, display_format='streamlit')
            
            return plot
            
        except Exception as e:
            logger.error(f"Model evaluation plot error: {e}")
            return self._create_error_plot(f"Error creating model evaluation plot: {str(e)}")
    
    def plot_feature_importance_model(self, model):
        """Plot feature importance from trained model."""
        try:
            # Import appropriate PyCaret module
            import pycaret.classification as pc
            import pycaret.regression as pr
            
            # Determine which module to use
            if hasattr(st.session_state, 'problem_type'):
                if st.session_state.problem_type == 'classification':
                    plot = pc.plot_model(model, plot='feature', display_format='streamlit')
                else:
                    plot = pr.plot_model(model, plot='feature', display_format='streamlit')
            else:
                plot = pc.plot_model(model, plot='feature', display_format='streamlit')
            
            return plot
            
        except Exception as e:
            logger.error(f"Model feature importance plot error: {e}")
            return self._create_error_plot(f"Error creating feature importance plot: {str(e)}")
    
    def plot_shap_analysis(self, model):
        """Plot SHAP analysis."""
        try:
            # Import appropriate PyCaret module
            import pycaret.classification as pc
            import pycaret.regression as pr
            
            # Determine which module to use
            if hasattr(st.session_state, 'problem_type'):
                if st.session_state.problem_type == 'classification':
                    plot = pc.interpret_model(model, plot='summary', display_format='streamlit')
                else:
                    plot = pr.interpret_model(model, plot='summary', display_format='streamlit')
            else:
                plot = pc.interpret_model(model, plot='summary', display_format='streamlit')
            
            return plot
            
        except Exception as e:
            logger.error(f"SHAP analysis error: {e}")
            return self._create_error_plot(f"Error creating SHAP analysis: {str(e)}")
    
    def plot_target_distribution(self, data: pd.DataFrame, target_column: str, problem_type: str) -> go.Figure:
        """Plot target variable distribution."""
        try:
            fig = go.Figure()
            
            if problem_type == 'classification':
                # Count plot for classification
                value_counts = data[target_column].value_counts()
                
                fig.add_trace(
                    go.Bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        marker_color='lightgreen',
                        text=value_counts.values,
                        textposition='auto'
                    )
                )
                
                fig.update_layout(
                    title=f'Target Distribution: {target_column}',
                    xaxis_title=target_column,
                    yaxis_title='Count'
                )
            
            else:
                # Histogram for regression
                fig.add_trace(
                    go.Histogram(
                        x=data[target_column],
                        nbinsx=30,
                        marker_color='lightcoral',
                        opacity=0.7
                    )
                )
                
                fig.update_layout(
                    title=f'Target Distribution: {target_column}',
                    xaxis_title=target_column,
                    yaxis_title='Frequency'
                )
            
            fig.update_layout(
                template=self.template,
                height=400
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Target distribution plot error: {e}")
            return self._create_error_plot(f"Error creating target distribution plot: {str(e)}")
    
    def plot_pairwise_relationships(self, data: pd.DataFrame, target_column: str) -> go.Figure:
        """Plot pairwise relationships between numeric features."""
        try:
            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            
            # Remove target column and limit to 5 features for performance
            numeric_columns = [col for col in numeric_columns if col != target_column][:5]
            
            if len(numeric_columns) < 2:
                return self._create_error_plot("Need at least 2 numeric features for pairwise plot")
            
            # Create scatter plot matrix
            fig = ff.create_scatterplotmatrix(
                data[numeric_columns],
                diag='histogram',
                height=600,
                width=600
            )
            
            fig.update_layout(
                title='Pairwise Feature Relationships',
                template=self.template
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Pairwise relationships plot error: {e}")
            return self._create_error_plot(f"Error creating pairwise relationships plot: {str(e)}")
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame) -> go.Figure:
        """Plot model comparison results."""
        try:
            # Select key metrics for visualization
            if 'Accuracy' in comparison_df.columns:
                # Classification metrics
                metrics = ['Accuracy', 'Prec.', 'Recall', 'F1']
            else:
                # Regression metrics
                metrics = ['MAE', 'MSE', 'RMSE', 'R2']
            
            # Filter available metrics
            available_metrics = [m for m in metrics if m in comparison_df.columns]
            
            if not available_metrics:
                return self._create_error_plot("No suitable metrics found for comparison")
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=available_metrics[:4]
            )
            
            positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
            
            for i, metric in enumerate(available_metrics[:4]):
                row, col = positions[i]
                
                fig.add_trace(
                    go.Bar(
                        x=comparison_df.index[:10],  # Top 10 models
                        y=comparison_df[metric][:10],
                        name=metric,
                        showlegend=False
                    ),
                    row=row, col=col
                )
            
            fig.update_layout(
                title='Model Comparison Results',
                template=self.template,
                height=600
            )
            
            # Rotate x-axis labels
            fig.update_xaxes(tickangle=45)
            
            return fig
            
        except Exception as e:
            logger.error(f"Model comparison plot error: {e}")
            return self._create_error_plot(f"Error creating model comparison plot: {str(e)}")
    
    def _create_error_plot(self, error_message: str) -> go.Figure:
        """Create an error plot when visualization fails."""
        fig = go.Figure()
        
        fig.add_annotation(
            text=error_message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="red")
        )
        
        fig.update_layout(
            title="Visualization Error",
            template=self.template,
            height=400
        )
        
        return fig
    
    def create_dashboard_summary(self, data: pd.DataFrame, target_column: str) -> Dict[str, go.Figure]:
        """Create a comprehensive dashboard summary."""
        try:
            dashboard = {}
            
            # Data overview
            dashboard['data_overview'] = self._create_data_overview_plot(data)
            
            # Target distribution
            problem_type = 'classification' if data[target_column].dtype == 'object' else 'regression'
            dashboard['target_dist'] = self.plot_target_distribution(data, target_column, problem_type)
            
            # Missing data
            missing_data = data.isnull().sum()
            if missing_data.sum() > 0:
                missing_df = pd.DataFrame({
                    'Column': missing_data.index,
                    'Missing Count': missing_data.values,
                    'Missing %': (missing_data.values / len(data)) * 100
                })
                dashboard['missing_data'] = self.plot_missing_data(missing_df)
            
            # Correlation for numeric features
            numeric_data = data.select_dtypes(include=[np.number])
            if len(numeric_data.columns) > 1:
                dashboard['correlation'] = self.plot_correlation_heatmap(numeric_data)
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Dashboard creation error: {e}")
            return {'error': self._create_error_plot(f"Error creating dashboard: {str(e)}")}
    
    def _create_data_overview_plot(self, data: pd.DataFrame) -> go.Figure:
        """Create data overview visualization."""
        try:
            # Data types summary
            dtype_counts = data.dtypes.value_counts()
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Data Types', 'Column Statistics'),
                specs=[[{"type": "pie"}, {"type": "bar"}]]
            )
            
            # Pie chart for data types
            fig.add_trace(
                go.Pie(
                    labels=dtype_counts.index.astype(str),
                    values=dtype_counts.values,
                    name="Data Types"
                ),
                row=1, col=1
            )
            
            # Bar chart for basic statistics
            stats = {
                'Total Rows': len(data),
                'Total Columns': len(data.columns),
                'Missing Values': data.isnull().sum().sum(),
                'Numeric Columns': len(data.select_dtypes(include=[np.number]).columns),
                'Categorical Columns': len(data.select_dtypes(include=['object', 'category']).columns)
            }
            
            fig.add_trace(
                go.Bar(
                    x=list(stats.keys()),
                    y=list(stats.values()),
                    name="Statistics"
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                title='Data Overview',
                template=self.template,
                height=400
            )
            
            return fig
            
        except Exception as e:
            logger.error