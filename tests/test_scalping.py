import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.8;
    }
    
    .profit-positive {
        color: #00ff88;
        font-weight: bold;
    }
    
    .profit-negative {
        color: #ff4444;
        font-weight: bold;
    }
    
    .sidebar-header {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 1rem;
        border-radius: 5px;
        text-align: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load data from the Excel file"""
    try:
        # Load all sheets
        deals_df = pd.read_excel('Scalping.xlsx', sheet_name='Deals')
        orders_df = pd.read_excel('Scalping.xlsx', sheet_name='Orders')
        live_orders_df = pd.read_excel('Scalping.xlsx', sheet_name='Live Orders')
        
        # Clean and process data
        # Convert time columns to datetime
        if 'Time' in deals_df.columns:
            deals_df['Time'] = pd.to_datetime(deals_df['Time'])
        if 'Open Time' in orders_df.columns:
            orders_df['Open Time'] = pd.to_datetime(orders_df['Open Time'])
        if 'Open Time' in live_orders_df.columns:
            live_orders_df['Open Time'] = pd.to_datetime(live_orders_df['Open Time'])
            
        # Clean numeric columns
        numeric_columns = ['Profit', 'Balance', 'Volume', 'Price', 'Commission', 'Fee', 'Swap']
        for col in numeric_columns:
            if col in deals_df.columns:
                deals_df[col] = pd.to_numeric(deals_df[col], errors='coerce').fillna(0)
        
        return deals_df, orders_df, live_orders_df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please make sure 'Scalping.xlsx' file is in the same directory as this script.")
        return None, None, None

def calculate_key_metrics(deals_df):
    """Calculate key trading metrics"""
    if deals_df is None or deals_df.empty:
        return {}
    
    # Filter completed deals (in/out pairs)
    completed_deals = deals_df[deals_df['Direction'] == 'out'].copy()
    
    # Basic metrics
    total_deals = len(completed_deals)
    total_profit = deals_df['Profit'].sum()
    winning_deals = completed_deals[completed_deals['Profit'] > 0]
    losing_deals = completed_deals[completed_deals['Profit'] < 0]
    
    win_rate = (len(winning_deals) / total_deals * 100) if total_deals > 0 else 0
    avg_win = winning_deals['Profit'].mean() if len(winning_deals) > 0 else 0
    avg_loss = losing_deals['Profit'].mean() if len(losing_deals) > 0 else 0
    profit_factor = (winning_deals['Profit'].sum() / abs(losing_deals['Profit'].sum())) if len(losing_deals) > 0 and losing_deals['Profit'].sum() != 0 else 0
    
    # Risk metrics
    max_profit = deals_df['Profit'].max()
    max_loss = deals_df['Profit'].min()
    
    # Calculate balance progression
    balance_progression = deals_df['Balance'].tolist()
    max_balance = max(balance_progression) if balance_progression else 0
    min_balance = min(balance_progression) if balance_progression else 0
    
    return {
        'total_deals': total_deals,
        'total_profit': total_profit,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'max_profit': max_profit,
        'max_loss': max_loss,
        'max_balance': max_balance,
        'min_balance': min_balance,
        'winning_deals': len(winning_deals),
        'losing_deals': len(losing_deals)
    }

def create_balance_chart(deals_df):
    """Create balance progression chart"""
    if deals_df is None or deals_df.empty:
        return None
    
    fig = go.Figure()
    
    # Balance line
    fig.add_trace(go.Scatter(
        x=deals_df['Time'],
        y=deals_df['Balance'],
        mode='lines',
        name='Balance',
        line=dict(color='#00ff88', width=3),
        hovertemplate='<b>Time:</b> %{x}<br><b>Balance:</b> $%{y:,.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Account Balance Progression",
        xaxis_title="Time",
        yaxis_title="Balance ($)",
        template="plotly_dark",
        height=400,
        hovermode='x unified'
    )
    
    return fig

def create_profit_distribution(deals_df):
    """Create profit distribution chart"""
    if deals_df is None or deals_df.empty:
        return None
    
    completed_deals = deals_df[deals_df['Direction'] == 'out'].copy()
    
    fig = go.Figure()
    
    # Profit histogram
    fig.add_trace(go.Histogram(
        x=completed_deals['Profit'],
        nbinsx=30,
        name='Profit Distribution',
        marker_color='rgba(100, 200, 255, 0.7)',
        hovertemplate='<b>Profit Range:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Profit/Loss Distribution",
        xaxis_title="Profit/Loss ($)",
        yaxis_title="Frequency",
        template="plotly_dark",
        height=400
    )
    
    return fig

def create_symbol_performance(deals_df):
    """Create symbol performance chart"""
    if deals_df is None or deals_df.empty:
        return None
    
    symbol_profit = deals_df.groupby('Symbol')['Profit'].sum().sort_values(ascending=False)
    
    colors = ['#00ff88' if x >= 0 else '#ff4444' for x in symbol_profit.values]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=symbol_profit.index,
        y=symbol_profit.values,
        marker_color=colors,
        name='Symbol P&L',
        hovertemplate='<b>Symbol:</b> %{x}<br><b>Total P&L:</b> $%{y:,.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Performance by Trading Symbol",
        xaxis_title="Symbol",
        yaxis_title="Total Profit/Loss ($)",
        template="plotly_dark",
        height=400,
        xaxis_tickangle=-45
    )
    
    return fig

def create_daily_performance(deals_df):
    """Create daily performance chart"""
    if deals_df is None or deals_df.empty:
        return None
    
    deals_df['Date'] = deals_df['Time'].dt.date
    daily_profit = deals_df.groupby('Date')['Profit'].sum()
    
    colors = ['#00ff88' if x >= 0 else '#ff4444' for x in daily_profit.values]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=daily_profit.index,
        y=daily_profit.values,
        marker_color=colors,
        name='Daily P&L',
        hovertemplate='<b>Date:</b> %{x}<br><b>Daily P&L:</b> $%{y:,.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Daily Performance",
        xaxis_title="Date",
        yaxis_title="Daily Profit/Loss ($)",
        template="plotly_dark",
        height=400
    )
    
    return fig

def create_order_status_chart(orders_df):
    """Create order status distribution chart"""
    if orders_df is None or orders_df.empty:
        return None
    
    status_counts = orders_df['State'].value_counts()
    
    fig = go.Figure()
    
    fig.add_trace(go.Pie(
        labels=status_counts.index,
        values=status_counts.values,
        hole=0.4,
        hovertemplate='<b>Status:</b> %{label}<br><b>Count:</b> %{value}<br><b>Percent:</b> %{percent}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Order Status Distribution",
        template="plotly_dark",
        height=400
    )
    
    return fig

def main():
    # Header
    st.markdown('<div class="main-header">📈 Trading Performance Dashboard</div>', unsafe_allow_html=True)
    
    # Load data
    deals_df, orders_df, live_orders_df = load_data()
    
    if deals_df is None:
        st.stop()
    
    # Sidebar
    st.sidebar.markdown('<div class="sidebar-header"><h2>📊 Dashboard Controls</h2></div>', unsafe_allow_html=True)
    
    # Date filter
    if 'Time' in deals_df.columns and not deals_df.empty:
        min_date = deals_df['Time'].min().date()
        max_date = deals_df['Time'].max().date()
        
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            deals_df_filtered = deals_df[
                (deals_df['Time'].dt.date >= date_range[0]) & 
                (deals_df['Time'].dt.date <= date_range[1])
            ]
        else:
            deals_df_filtered = deals_df
    else:
        deals_df_filtered = deals_df
    
    # Symbol filter
    if 'Symbol' in deals_df.columns:
        symbols = st.sidebar.multiselect(
            "Select Symbols",
            options=deals_df['Symbol'].unique(),
            default=deals_df['Symbol'].unique()
        )
        deals_df_filtered = deals_df_filtered[deals_df_filtered['Symbol'].isin(symbols)]
    
    # Calculate metrics
    metrics = calculate_key_metrics(deals_df_filtered)
    
    # Key Metrics Row
    st.markdown("## 📊 Key Performance Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        profit_color = "profit-positive" if metrics.get('total_profit', 0) >= 0 else "profit-negative"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total P&L</div>
            <div class="metric-value {profit_color}">${metrics.get('total_profit', 0):,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Win Rate</div>
            <div class="metric-value">{metrics.get('win_rate', 0):.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Trades</div>
            <div class="metric-value">{metrics.get('total_deals', 0)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Profit Factor</div>
            <div class="metric-value">{metrics.get('profit_factor', 0):.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Max Balance</div>
            <div class="metric-value">${metrics.get('max_balance', 0):,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Charts Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        balance_chart = create_balance_chart(deals_df_filtered)
        if balance_chart:
            st.plotly_chart(balance_chart, use_container_width=True)
    
    with col2:
        profit_dist_chart = create_profit_distribution(deals_df_filtered)
        if profit_dist_chart:
            st.plotly_chart(profit_dist_chart, use_container_width=True)
    
    # Charts Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        symbol_perf_chart = create_symbol_performance(deals_df_filtered)
        if symbol_perf_chart:
            st.plotly_chart(symbol_perf_chart, use_container_width=True)
    
    with col2:
        daily_perf_chart = create_daily_performance(deals_df_filtered)
        if daily_perf_chart:
            st.plotly_chart(daily_perf_chart, use_container_width=True)
    
    # Order Status Chart
    if orders_df is not None and not orders_df.empty:
        st.markdown("## 📋 Order Analysis")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            order_status_chart = create_order_status_chart(orders_df)
            if order_status_chart:
                st.plotly_chart(order_status_chart, use_container_width=True)
        
        with col2:
            st.markdown("### 📈 Additional Metrics")
            
            # Additional metrics
            avg_win = metrics.get('avg_win', 0)
            avg_loss = metrics.get('avg_loss', 0)
            max_profit = metrics.get('max_profit', 0)
            max_loss = metrics.get('max_loss', 0)
            
            st.metric("Average Win", f"${avg_win:.2f}")
            st.metric("Average Loss", f"${avg_loss:.2f}")
            st.metric("Best Trade", f"${max_profit:.2f}")
            st.metric("Worst Trade", f"${max_loss:.2f}")
    
    # Live Orders Table
    if live_orders_df is not None and not live_orders_df.empty:
        st.markdown("## 🔴 Live Orders")
        st.dataframe(
            live_orders_df,
            use_container_width=True,
            hide_index=True
        )
    
    # Recent Deals Table
    if deals_df_filtered is not None and not deals_df_filtered.empty:
        st.markdown("## 📊 Recent Deals")
        recent_deals = deals_df_filtered.tail(20).sort_values('Time', ascending=False)
        
        # Format the dataframe for better display
        display_df = recent_deals[['Time', 'Symbol', 'Type', 'Direction', 'Volume', 'Price', 'Profit', 'Balance']].copy()
        display_df['Profit'] = display_df['Profit'].apply(lambda x: f"${x:.2f}")
        display_df['Balance'] = display_df['Balance'].apply(lambda x: f"${x:.2f}")
        display_df['Price'] = display_df['Price'].apply(lambda x: f"{x:.5f}")
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
    
    # Footer
    st.markdown("---")
    st.markdown("**Dashboard created with Streamlit** | Data refreshed automatically")

if __name__ == "__main__":
    main()