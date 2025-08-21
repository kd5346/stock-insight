import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
from utils.data_fetcher import StockDataFetcher
from utils.chart_utils import create_price_chart, create_volume_chart

# Page configuration
st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme enhancement
st.markdown("""
<style>
    .stMetric {
        background-color: #262730;
        border: 1px solid #464646;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .main-header {
        color: #00d4aa;
        text-align: center;
        padding: 1rem 0;
        font-size: 2.5rem;
        font-weight: bold;
    }
    
    .stock-info {
        background-color: #1e2130;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #00d4aa;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">📈 Stock Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'stock_data' not in st.session_state:
        st.session_state.stock_data = None
    if 'stock_info' not in st.session_state:
        st.session_state.stock_info = None
    if 'current_symbol' not in st.session_state:
        st.session_state.current_symbol = ""

    # Sidebar for input controls
    with st.sidebar:
        st.header("🔍 Stock Search")
        
        # Stock symbol input
        stock_symbol = st.text_input(
            "Enter Stock Symbol:",
            value="AAPL",
            placeholder="e.g., AAPL, GOOGL, TSLA",
            help="Enter a valid stock ticker symbol"
        ).upper()
        
        # Time period selection
        time_period = st.selectbox(
            "Select Time Period:",
            options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
            index=3,
            help="Choose the time range for historical data"
        )
        
        # Data interval
        interval = st.selectbox(
            "Data Interval:",
            options=["1d", "5d", "1wk", "1mo"],
            index=0,
            help="Choose the data frequency"
        )
        
        # Search button
        search_button = st.button("🔍 Analyze Stock", type="primary", use_container_width=True)
        
        # Clear button
        if st.button("🗑️ Clear Data", use_container_width=True):
            st.session_state.stock_data = None
            st.session_state.stock_info = None
            st.session_state.current_symbol = ""
            st.rerun()

    # Main content area
    if search_button or (st.session_state.current_symbol == stock_symbol and st.session_state.stock_data is not None):
        if search_button:
            with st.spinner(f"Fetching data for {stock_symbol}..."):
                try:
                    # Initialize data fetcher
                    fetcher = StockDataFetcher(stock_symbol)
                    
                    # Fetch stock data
                    stock_data = fetcher.get_historical_data(period=time_period, interval=interval)
                    stock_info = fetcher.get_stock_info()
                    
                    if stock_data is not None and not stock_data.empty:
                        st.session_state.stock_data = stock_data
                        st.session_state.stock_info = stock_info
                        st.session_state.current_symbol = stock_symbol
                        st.success(f"Successfully loaded data for {stock_symbol}")
                    else:
                        st.error(f"No data found for symbol '{stock_symbol}'. Please check the symbol and try again.")
                        return
                        
                except Exception as e:
                    st.error(f"Error fetching data for '{stock_symbol}': {str(e)}")
                    st.info("Please verify the stock symbol is correct and try again.")
                    return
        
        # Display data if available
        if st.session_state.stock_data is not None and st.session_state.stock_info is not None:
            stock_data = st.session_state.stock_data
            stock_info = st.session_state.stock_info
            current_symbol = st.session_state.current_symbol
            
            # Stock information header
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown(f"""
                <div class="stock-info">
                    <h2>{stock_info.get('longName', current_symbol)} ({current_symbol})</h2>
                    <p><strong>Sector:</strong> {stock_info.get('sector', 'N/A')} | 
                    <strong>Industry:</strong> {stock_info.get('industry', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Key metrics
            st.subheader("📊 Key Financial Metrics")
            
            # Get latest price and calculate change
            latest_price = stock_data['Close'].iloc[-1]
            previous_price = stock_data['Close'].iloc[-2] if len(stock_data) > 1 else latest_price
            price_change = latest_price - previous_price
            price_change_pct = (price_change / previous_price) * 100 if previous_price != 0 else 0
            
            # Display metrics in columns
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric(
                    "Current Price",
                    f"${latest_price:.2f}",
                    delta=f"{price_change:+.2f} ({price_change_pct:+.2f}%)"
                )
            
            with col2:
                market_cap = stock_info.get('marketCap', 0)
                if market_cap > 1e12:
                    market_cap_str = f"${market_cap/1e12:.2f}T"
                elif market_cap > 1e9:
                    market_cap_str = f"${market_cap/1e9:.2f}B"
                elif market_cap > 1e6:
                    market_cap_str = f"${market_cap/1e6:.2f}M"
                else:
                    market_cap_str = f"${market_cap:,.0f}"
                
                st.metric("Market Cap", market_cap_str)
            
            with col3:
                avg_volume = stock_data['Volume'].mean()
                if avg_volume > 1e6:
                    volume_str = f"{avg_volume/1e6:.1f}M"
                elif avg_volume > 1e3:
                    volume_str = f"{avg_volume/1e3:.1f}K"
                else:
                    volume_str = f"{avg_volume:.0f}"
                st.metric("Avg Volume", volume_str)
            
            with col4:
                pe_ratio = stock_info.get('trailingPE', 'N/A')
                if isinstance(pe_ratio, (int, float)):
                    st.metric("P/E Ratio", f"{pe_ratio:.2f}")
                else:
                    st.metric("P/E Ratio", "N/A")
            
            with col5:
                dividend_yield = stock_info.get('dividendYield', 0)
                if dividend_yield and dividend_yield > 0:
                    st.metric("Dividend Yield", f"{dividend_yield*100:.2f}%")
                else:
                    st.metric("Dividend Yield", "N/A")
            
            # Charts section
            st.subheader("📈 Price & Volume Charts")
            
            # Create price chart
            price_fig = create_price_chart(stock_data, current_symbol)
            st.plotly_chart(price_fig, use_container_width=True)
            
            # Create volume chart
            volume_fig = create_volume_chart(stock_data, current_symbol)
            st.plotly_chart(volume_fig, use_container_width=True)
            
            # Data table section
            st.subheader("📋 Historical Data")
            
            # Prepare data for display
            display_data = stock_data.copy()
            display_data.index = display_data.index.strftime('%Y-%m-%d')
            display_data = display_data.round(2)
            
            # Add some calculated columns
            display_data['Daily Return (%)'] = ((display_data['Close'] - display_data['Close'].shift(1)) / display_data['Close'].shift(1) * 100).round(2)
            display_data['Price Range'] = (display_data['High'] - display_data['Low']).round(2)
            
            # Reorder columns for better display
            column_order = ['Open', 'High', 'Low', 'Close', 'Volume', 'Daily Return (%)', 'Price Range']
            if 'Adj Close' in display_data.columns:
                column_order.insert(-2, 'Adj Close')
            
            display_data = display_data[column_order]
            
            # Show data table
            st.dataframe(
                display_data.sort_index(ascending=False),
                use_container_width=True,
                height=400
            )
            
            # Download section
            st.subheader("💾 Export Data")
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                # Prepare CSV data
                csv_data = display_data.sort_index(ascending=False)
                csv_string = csv_data.to_csv()
                
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_string,
                    file_name=f"{current_symbol}_stock_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    type="primary"
                )
            
            with col2:
                st.info(f"💡 Download includes {len(display_data)} days of historical data for {current_symbol}")
            
            # Additional information
            st.subheader("ℹ️ Company Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Company:** {stock_info.get('longName', 'N/A')}")
                st.write(f"**Symbol:** {current_symbol}")
                st.write(f"**Exchange:** {stock_info.get('exchange', 'N/A')}")
                st.write(f"**Currency:** {stock_info.get('currency', 'N/A')}")
                st.write(f"**Country:** {stock_info.get('country', 'N/A')}")
            
            with col2:
                st.write(f"**52 Week High:** ${stock_info.get('fiftyTwoWeekHigh', 'N/A')}")
                st.write(f"**52 Week Low:** ${stock_info.get('fiftyTwoWeekLow', 'N/A')}")
                st.write(f"**Beta:** {stock_info.get('beta', 'N/A')}")
                st.write(f"**Employees:** {stock_info.get('fullTimeEmployees', 'N/A'):,}" if stock_info.get('fullTimeEmployees') else "**Employees:** N/A")
                
                # Website link if available
                website = stock_info.get('website')
                if website:
                    st.markdown(f"**Website:** [{website}]({website})")
    
    else:
        # Welcome message
        st.markdown("""
        ### Welcome to the Stock Analysis Dashboard! 🎯
        
        **Get started by:**
        1. 🔍 Enter a stock symbol in the sidebar (e.g., AAPL, GOOGL, TSLA)
        2. ⏰ Choose your preferred time period and data interval
        3. 🔍 Click "Analyze Stock" to fetch real-time data
        
        **Features:**
        - 📊 Real-time financial data from Yahoo Finance
        - 📈 Interactive price and volume charts
        - 📋 Comprehensive data tables
        - 💾 CSV export functionality
        - 🌙 Professional dark theme interface
        
        **Popular Stocks to Try:**
        - **Tech:** AAPL, GOOGL, MSFT, TSLA, NVDA
        - **Finance:** JPM, BAC, GS, WFC
        - **Healthcare:** JNJ, PFE, UNH, MRK
        - **Consumer:** AMZN, WMT, HD, MCD
        """)

if __name__ == "__main__":
    main()
