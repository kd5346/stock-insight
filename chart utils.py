import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional

def create_price_chart(data: pd.DataFrame, symbol: str, chart_type: str = "candlestick") -> go.Figure:
    """
    Create an interactive price chart with candlestick or line options.
    
    Args:
        data (pd.DataFrame): Historical stock data
        symbol (str): Stock symbol for chart title
        chart_type (str): Type of chart ('candlestick' or 'line')
    
    Returns:
        go.Figure: Plotly figure object
    """
    # Create subplot with secondary y-axis for volume
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(f'{symbol} Stock Price', 'Volume'),
        row_width=[0.7, 0.3]
    )
    
    # Price chart (candlestick)
    if chart_type == "candlestick":
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name="Price",
                increasing_line_color='#00d4aa',
                decreasing_line_color='#ff6692',
                increasing_fillcolor='rgba(0, 212, 170, 0.3)',
                decreasing_fillcolor='rgba(255, 102, 146, 0.3)'
            ),
            row=1, col=1
        )
    else:
        # Line chart
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='#00d4aa', width=2),
                hovertemplate='<b>Date:</b> %{x}<br><b>Price:</b> $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # Add moving averages
    if len(data) >= 20:
        ma_20 = data['Close'].rolling(window=20).mean()
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=ma_20,
                mode='lines',
                name='MA20',
                line=dict(color='#ffd700', width=1, dash='dash'),
                hovertemplate='<b>20-day MA:</b> $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    if len(data) >= 50:
        ma_50 = data['Close'].rolling(window=50).mean()
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=ma_50,
                mode='lines',
                name='MA50',
                line=dict(color='#ff9500', width=1, dash='dash'),
                hovertemplate='<b>50-day MA:</b> $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # Volume bars
    colors = ['#00d4aa' if close >= open else '#ff6692' 
              for close, open in zip(data['Close'], data['Open'])]
    
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.7,
            hovertemplate='<b>Date:</b> %{x}<br><b>Volume:</b> %{y:,}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        template='plotly_dark',
        height=700,
        showlegend=True,
        hovermode='x unified',
        margin=dict(l=50, r=50, t=50, b=50),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#fafafa'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(0,0,0,0.5)'
        )
    )
    
    # Update axes
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        showline=True,
        linewidth=1,
        linecolor='rgba(128,128,128,0.3)'
    )
    
    fig.update_yaxes(
        title_text="Price ($)",
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        showline=True,
        linewidth=1,
        linecolor='rgba(128,128,128,0.3)',
        row=1, col=1
    )
    
    fig.update_yaxes(
        title_text="Volume",
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        row=2, col=1
    )
    
    return fig

def create_volume_chart(data: pd.DataFrame, symbol: str) -> go.Figure:
    """
    Create a standalone volume chart with moving average.
    
    Args:
        data (pd.DataFrame): Historical stock data
        symbol (str): Stock symbol for chart title
    
    Returns:
        go.Figure: Plotly figure object
    """
    # Calculate volume moving average
    volume_ma = data['Volume'].rolling(window=20).mean()
    
    # Create colors based on price movement
    colors = ['#00d4aa' if close >= open else '#ff6692' 
              for close, open in zip(data['Close'], data['Open'])]
    
    fig = go.Figure()
    
    # Volume bars
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.8,
            hovertemplate='<b>Date:</b> %{x}<br><b>Volume:</b> %{y:,}<extra></extra>'
        )
    )
    
    # Volume moving average
    if len(data) >= 20:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=volume_ma,
                mode='lines',
                name='Volume MA20',
                line=dict(color='#ffd700', width=2),
                hovertemplate='<b>Volume MA20:</b> %{y:,.0f}<extra></extra>'
            )
        )
    
    # Update layout
    fig.update_layout(
        title=f'{symbol} Trading Volume',
        template='plotly_dark',
        height=400,
        showlegend=True,
        hovermode='x unified',
        margin=dict(l=50, r=50, t=50, b=50),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#fafafa'),
        xaxis_title="Date",
        yaxis_title="Volume",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(0,0,0,0.5)'
        )
    )
    
    # Update axes
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        showline=True,
        linewidth=1,
        linecolor='rgba(128,128,128,0.3)'
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        showline=True,
        linewidth=1,
        linecolor='rgba(128,128,128,0.3)'
    )
    
    return fig

def create_returns_chart(data: pd.DataFrame, symbol: str) -> go.Figure:
    """
    Create a daily returns chart.
    
    Args:
        data (pd.DataFrame): Historical stock data
        symbol (str): Stock symbol for chart title
    
    Returns:
        go.Figure: Plotly figure object
    """
    # Calculate daily returns
    daily_returns = data['Close'].pct_change() * 100
    daily_returns = daily_returns.dropna()
    
    # Create colors based on positive/negative returns
    colors = ['#00d4aa' if ret >= 0 else '#ff6692' for ret in daily_returns]
    
    fig = go.Figure()
    
    # Returns bar chart
    fig.add_trace(
        go.Bar(
            x=daily_returns.index,
            y=daily_returns,
            name='Daily Returns',
            marker_color=colors,
            opacity=0.8,
            hovertemplate='<b>Date:</b> %{x}<br><b>Return:</b> %{y:.2f}%<extra></extra>'
        )
    )
    
    # Add zero line
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="rgba(128,128,128,0.5)",
        line_width=1
    )
    
    # Update layout
    fig.update_layout(
        title=f'{symbol} Daily Returns (%)',
        template='plotly_dark',
        height=400,
        showlegend=False,
        hovermode='x',
        margin=dict(l=50, r=50, t=50, b=50),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#fafafa'),
        xaxis_title="Date",
        yaxis_title="Daily Return (%)"
    )
    
    # Update axes
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        showline=True,
        linewidth=1,
        linecolor='rgba(128,128,128,0.3)'
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        showline=True,
        linewidth=1,
        linecolor='rgba(128,128,128,0.3)'
    )
    
    return fig

def create_correlation_heatmap(symbols: list, period: str = "1y") -> Optional[go.Figure]:
    """
    Create a correlation heatmap for multiple stocks.
    
    Args:
        symbols (list): List of stock symbols
        period (str): Time period for correlation analysis
    
    Returns:
        go.Figure: Plotly figure object or None if error
    """
    try:
        import yfinance as yf
        
        # Fetch data for all symbols
        data = {}
        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            if not hist.empty:
                data[symbol] = hist['Close']
        
        if len(data) < 2:
            return None
        
        # Create DataFrame and calculate correlation
        df = pd.DataFrame(data)
        df = df.dropna()
        correlation_matrix = df.corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(correlation_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 12},
            hovertemplate='<b>%{x} vs %{y}</b><br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Stock Price Correlation Matrix',
            template='plotly_dark',
            height=500,
            font=dict(color='#fafafa'),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
        
    except Exception as e:
        return None
