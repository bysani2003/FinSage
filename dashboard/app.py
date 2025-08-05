import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
from datetime import datetime, timedelta
import time

# Page configuration
st.set_page_config(
    page_title="Stock Predictor Dashboard",
    page_icon="📈",
    layout="wide"
)

# Backend URL
# BACKEND_URL = "http://backend:5000"  # For Docker
BACKEND_URL = "http://localhost:5000"  # For local development

def get_popular_stocks():
    """Get list of popular stock symbols"""
    try:
        response = requests.get(f"{BACKEND_URL}/api/stocks")
        if response.status_code == 200:
            data = response.json()
            return {
                "all_symbols": data.get("symbols", []),
                "indian_stocks": data.get("indian_stocks", []),
                "global_stocks": data.get("global_stocks", []),
                "indices": data.get("indices", [])
            }
        return {
            "all_symbols": ["RELIANCE.NS", "TCS.NS", "AAPL", "GOOGL"],
            "indian_stocks": ["RELIANCE.NS", "TCS.NS"],
            "global_stocks": ["AAPL", "GOOGL"],
            "indices": ["^NSEI", "^BSESN"]
        }
    except:
        return {
            "all_symbols": ["RELIANCE.NS", "TCS.NS", "AAPL", "GOOGL"],
            "indian_stocks": ["RELIANCE.NS", "TCS.NS"],
            "global_stocks": ["AAPL", "GOOGL"],
            "indices": ["^NSEI", "^BSESN"]
        }

def get_stock_data(symbol, days=365):
    """Get historical stock data"""
    try:
        response = requests.get(f"{BACKEND_URL}/api/stock/{symbol}/data?days={days}")
        if response.status_code == 200:
            return response.json()["data"]
        return None
    except:
        return None

def get_currency_symbol(stock_symbol):
    """Get appropriate currency symbol for stock"""
    # Indian stocks and indices
    if (stock_symbol.endswith('.NS') or 
        stock_symbol.endswith('.BO') or 
        stock_symbol == '^NSEI' or  # Nifty 50
        stock_symbol == '^BSESN'):  # BSE Sensex
        return "₹"  # Indian Rupees
    else:
        return "$"  # US Dollars for others

def format_price(price, stock_symbol):
    """Format price with correct currency"""
    currency = get_currency_symbol(stock_symbol)
    return f"{currency}{price:,.2f}"

def predict_stock_price(symbol):
    """Get stock price prediction"""
    try:
        response = requests.post(f"{BACKEND_URL}/api/stock/{symbol}/predict")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def get_sentiment_data(symbol):
    """Get sentiment analysis data"""
    try:
        response = requests.get(f"{BACKEND_URL}/api/stock/{symbol}/sentiment")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def get_accuracy_metrics(symbol):
    """Get prediction accuracy metrics"""
    try:
        response = requests.get(f"{BACKEND_URL}/api/stock/{symbol}/accuracy")
        if response.status_code == 200:
            return response.json()["accuracy_metrics"]
        return None
    except:
        return None

def train_model(symbol):
    """Train model for symbol"""
    try:
        response = requests.post(f"{BACKEND_URL}/api/train/{symbol}")
        return response.status_code == 200, response.json()
    except:
        return False, {"error": "Failed to connect to backend"}

def plot_stock_chart(data, symbol):
    """Create stock price chart"""
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df['date'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name=symbol
    ))
    
    fig.update_layout(
        title=f"{symbol} Stock Price",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        template="plotly_white",
        height=500
    )
    
    return fig

def plot_volume_chart(data):
    """Create volume chart"""
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    
    fig = px.bar(df, x='date', y='volume', title="Trading Volume")
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Volume",
        template="plotly_white",
        height=300
    )
    
    return fig

def plot_sentiment_chart(sentiment_data):
    """Create sentiment chart"""
    if not sentiment_data or "historical_sentiment" not in sentiment_data:
        return None
    
    df = pd.DataFrame(sentiment_data["historical_sentiment"])
    if df.empty:
        return None
    
    df['date'] = pd.to_datetime(df['date'])
    
    fig = px.line(df, x='date', y='sentiment_score', title="Sentiment Analysis")
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Sentiment Score",
        template="plotly_white",
        height=300
    )
    
    return fig

# Main app
def main():
    st.title("📈 Indian Stock Market Predictor")
    st.markdown("🇮🇳 AI-powered stock price prediction and analysis for NSE/BSE stocks")
    
    # Market Overview Header
    display_market_overview()
    
    # Navigation
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Analysis", "📈 Market", "💼 Portfolio", "🔔 Alerts", "⚙️ Settings"])
    
    with tab1:
        stock_analysis_tab()
    
    with tab2:
        market_overview_tab()
    
    with tab3:
        portfolio_tab()
    
    with tab4:
        alerts_tab()
    
    with tab5:
        settings_tab()

def display_market_overview():
    """Display market indices at the top"""
    try:
        response = requests.get(f"{BACKEND_URL}/api/market/indices")
        if response.status_code == 200:
            indices = response.json().get("indices", {})
            
            if indices:
                st.markdown("### 📊 Market Overview")
                cols = st.columns(len(indices))
                
                for i, (symbol, data) in enumerate(indices.items()):
                    with cols[i]:
                        change_color = "🟢" if data.get("change", 0) >= 0 else "🔴"
                        currency = get_currency_symbol(symbol)
                        st.metric(
                            data["name"],
                            f"{currency}{data['current_price']:,.2f}",
                            f"{change_color} {data['change']:+.2f} ({data['change_percent']:+.2f}%)"
                        )
                st.markdown("---")
    except:
        pass  # Silently fail if indices can't be loaded

def stock_analysis_tab():
    """Stock analysis and prediction tab"""
    # Sidebar
    st.sidebar.header("Stock Selection")
    
    # Get popular stocks
    stocks_data = get_popular_stocks()
    
    # Market selection
    market_type = st.sidebar.radio("Select Market", ["🇮🇳 Indian Stocks (NSE/BSE)", "🌍 Global Stocks"])
    
    if market_type == "🇮🇳 Indian Stocks (NSE/BSE)":
        available_stocks = stocks_data["indian_stocks"]
        st.sidebar.markdown("*NSE stocks end with .NS*")
    else:
        available_stocks = stocks_data["global_stocks"]
    
    # Stock search functionality
    st.sidebar.markdown("### 🔍 Stock Search")
    search_query = st.sidebar.text_input("Search for any Indian stock", placeholder="e.g., Reliance, TCS, HDFC")
    
    symbol = None
    
    if search_query and len(search_query) >= 2:
        # Local stock database for search
        indian_stocks_db = {
            'RELIANCE.NS': 'Reliance Industries Limited',
            'TCS.NS': 'Tata Consultancy Services Limited',
            'INFY.NS': 'Infosys Limited',
            'HINDUNILVR.NS': 'Hindustan Unilever Limited',
            'ICICIBANK.NS': 'ICICI Bank Limited',
            'SBIN.NS': 'State Bank of India',
            'BHARTIARTL.NS': 'Bharti Airtel Limited',
            'ITC.NS': 'ITC Limited',
            'KOTAKBANK.NS': 'Kotak Mahindra Bank Limited',
            'LT.NS': 'Larsen & Toubro Limited',
            'ASIANPAINT.NS': 'Asian Paints Limited',
            'MARUTI.NS': 'Maruti Suzuki India Limited',
            'HDFCBANK.NS': 'HDFC Bank Limited',
            'WIPRO.NS': 'Wipro Limited',
            'ONGC.NS': 'Oil and Natural Gas Corporation Limited',
            'BAJFINANCE.NS': 'Bajaj Finance Limited',
            'HCLTECH.NS': 'HCL Technologies Limited',
            'TECHM.NS': 'Tech Mahindra Limited',
            'SUNPHARMA.NS': 'Sun Pharmaceutical Industries Limited',
            'TITAN.NS': 'Titan Company Limited',
            'ULTRACEMCO.NS': 'UltraTech Cement Limited',
            'POWERGRID.NS': 'Power Grid Corporation of India Limited',
            'NTPC.NS': 'NTPC Limited',
            'NESTLEIND.NS': 'Nestle India Limited',
            'DRREDDY.NS': 'Dr. Reddy\'s Laboratories Limited',
            'AXISBANK.NS': 'Axis Bank Limited',
            'BAJAJ-AUTO.NS': 'Bajaj Auto Limited',
            'BPCL.NS': 'Bharat Petroleum Corporation Limited',
            'CIPLA.NS': 'Cipla Limited',
            'COALINDIA.NS': 'Coal India Limited',
            'TATAMOTORS.NS': 'Tata Motors Limited',
            'TATASTEEL.NS': 'Tata Steel Limited',
            'ADANIPORTS.NS': 'Adani Ports and Special Economic Zone Limited',
            'APOLLOHOSP.NS': 'Apollo Hospitals Enterprise Limited',
            'BRITANNIA.NS': 'Britannia Industries Limited'
        }
        
        # Search locally first
        query_upper = search_query.upper()
        local_matches = []
        
        for symbol, name in indian_stocks_db.items():
            if (query_upper in symbol.replace('.NS', '') or 
                query_upper in name.upper()):
                local_matches.append({
                    'symbol': symbol,
                    'name': name,
                    'exchange': 'NSE'
                })
        
        if local_matches:
            st.sidebar.markdown("**Search Results:**")
            selected_result = st.sidebar.selectbox(
                "Select from search results:",
                options=local_matches,
                format_func=lambda x: f"{x['symbol'].replace('.NS', '')} - {x['name']}"
            )
            if selected_result:
                symbol = selected_result['symbol']
        else:
            # Try API search as backup
            try:
                response = requests.get(f"{BACKEND_URL}/api/search/stocks?q={search_query}")
                if response.status_code == 200:
                    api_results = response.json().get("stocks", [])
                    if api_results:
                        st.sidebar.markdown("**Search Results:**")
                        selected_result = st.sidebar.selectbox(
                            "Select from search results:",
                            options=api_results,
                            format_func=lambda x: f"{x['symbol'].replace('.NS', '')} - {x['name']}"
                        )
                        if selected_result:
                            symbol = selected_result['symbol']
                    else:
                        st.sidebar.warning("No stocks found. Try: Reliance, TCS, HDFC, Infosys")
                else:
                    st.sidebar.warning("No stocks found. Try: Reliance, TCS, HDFC, Infosys")
            except:
                st.sidebar.warning("Search not available. Try: Reliance, TCS, HDFC, Infosys")
    
    # Fallback to dropdown selection
    if not symbol:
        st.sidebar.markdown("### 📊 Popular Stocks")
        symbol = st.sidebar.selectbox("Select Stock Symbol", available_stocks)
    
    # Display stock info
    if symbol:
        display_stock_info(symbol)
    
    # Time period selection
    time_periods = {
        "1 Month": 30,
        "3 Months": 90,
        "6 Months": 180,
        "1 Year": 365,
        "2 Years": 730
    }
    selected_period = st.sidebar.selectbox("Time Period", list(time_periods.keys()), index=3)
    days = time_periods[selected_period]
    
    # Custom symbol input
    custom_symbol = st.sidebar.text_input("Or enter custom symbol:")
    if custom_symbol:
        symbol = custom_symbol.upper()
    
    # Model training
    st.sidebar.header("Model Management")
    if st.sidebar.button("Train Model"):
        with st.spinner(f"Training model for {symbol}..."):
            success, result = train_model(symbol)
            if success:
                st.sidebar.success("Model trained successfully!")
            else:
                st.sidebar.error(f"Training failed: {result.get('error', 'Unknown error')}")
    
    # Main content
    col1, col2, col3, col4 = st.columns(4)
    
    # Get stock data
    with st.spinner("Loading data..."):
        data = get_stock_data(symbol, days)
        
        if data is None:
            st.error(f"Could not load data for {symbol}. Please check if the symbol is valid.")
            return
        
        # Current metrics
        latest = data[-1] if data else None
        if latest:
            with col1:
                st.metric("Current Price", format_price(latest['close'], symbol))
            
            with col2:
                change = latest['close'] - latest['open']
                change_pct = (change / latest['open']) * 100
                currency = get_currency_symbol(symbol)
                st.metric("Daily Change", f"{currency}{change:+.2f}", f"{change_pct:+.2f}%")
            
            with col3:
                st.metric("Volume", f"{latest['volume']:,}")
            
            with col4:
                high_52w = max([d['high'] for d in data])
                low_52w = min([d['low'] for d in data])
                st.metric("52W High/Low", format_price(high_52w, symbol), format_price(low_52w, symbol))
    
    # Stock price chart
    st.subheader(f"{symbol} Price Chart")
    if data:
        chart = plot_stock_chart(data, symbol)
        st.plotly_chart(chart, use_container_width=True)
    
    # Two columns for additional charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Volume chart
        st.subheader("Trading Volume")
        if data:
            volume_chart = plot_volume_chart(data)
            st.plotly_chart(volume_chart, use_container_width=True)
    
    with col2:
        # Sentiment analysis
        st.subheader("Sentiment Analysis")
        sentiment_data = get_sentiment_data(symbol)
        if sentiment_data:
            current_sentiment = sentiment_data.get("current_sentiment", 0)
            sentiment_color = "green" if current_sentiment > 0 else "red" if current_sentiment < 0 else "gray"
            st.markdown(f"**Current Sentiment:** <span style='color: {sentiment_color}'>{current_sentiment:.3f}</span>", unsafe_allow_html=True)
            
            # Show sentiment interpretation
            if current_sentiment > 0.2:
                st.success("📈 Positive sentiment detected")
            elif current_sentiment < -0.2:
                st.error("📉 Negative sentiment detected")
            else:
                st.info("😐 Neutral sentiment")
            
            # Show recent news headlines
            if "news_headlines" in sentiment_data:
                st.markdown("**Recent Headlines:**")
                for headline in sentiment_data["news_headlines"][:3]:
                    st.markdown(f"• {headline}")
            
            sentiment_chart = plot_sentiment_chart(sentiment_data)
            if sentiment_chart:
                st.plotly_chart(sentiment_chart, use_container_width=True)
        else:
            st.info("Loading sentiment data...")
    
    # Prediction section
    st.subheader("Price Prediction")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Get Prediction", type="primary"):
            with st.spinner("Generating prediction..."):
                prediction = predict_stock_price(symbol)
                if prediction:
                    predicted_price = prediction["predicted_price"]
                    current_price = latest['close'] if latest else 0
                    change = predicted_price - current_price
                    change_pct = (change / current_price) * 100 if current_price > 0 else 0
                    
                    st.success(f"**Predicted Price:** {format_price(predicted_price, symbol)}")
                    currency = get_currency_symbol(symbol)
                    st.info(f"**Expected Change:** {currency}{change:+.2f} ({change_pct:+.2f}%)")
                else:
                    st.error("Could not generate prediction. Try training the model first.")
    
    with col2:
        # Model accuracy
        accuracy = get_accuracy_metrics(symbol)
        if accuracy:
            currency = get_currency_symbol(symbol)
            st.metric("Model Accuracy (MAE)", f"{currency}{accuracy['mae']:.2f}")
            st.metric("Mean Abs % Error", f"{accuracy['mape']:.1f}%")
            st.metric("Predictions Made", accuracy['samples'])
        else:
            st.info("No accuracy data available")
    
    with col3:
        # Additional stats
        if data and len(data) > 1:
            prices = [d['close'] for d in data]
            volatility = pd.Series(prices).pct_change().std() * 100
            st.metric("Volatility", f"{volatility:.2f}%")
            
            # Price trend
            recent_trend = (prices[-1] - prices[-30]) / prices[-30] * 100 if len(prices) >= 30 else 0
            st.metric("30-Day Trend", f"{recent_trend:.2f}%")
    
    # Footer
    st.markdown("---")
    st.markdown("*Predictions are for educational purposes only. Not financial advice.*")

def portfolio_tab():
    """Portfolio management tab"""
    st.header("Portfolio Management")
    
    # Get user portfolios
    user_id = st.session_state.get('user_id', 'demo_user')
    
    try:
        response = requests.get(f"{BACKEND_URL}/api/portfolios?user_id={user_id}")
        portfolios = response.json().get("portfolios", []) if response.status_code == 200 else []
    except:
        portfolios = []
    
    # Portfolio selection or creation
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if portfolios:
            portfolio_names = [f"{p['name']} (ID: {p['id']})" for p in portfolios]
            selected_portfolio = st.selectbox("Select Portfolio", portfolio_names)
            if selected_portfolio:
                portfolio_id = int(selected_portfolio.split("ID: ")[1].split(")")[0])
        else:
            st.info("No portfolios found. Create one below.")
            portfolio_id = None
    
    with col2:
        if st.button("Create Portfolio"):
            with st.form("create_portfolio"):
                name = st.text_input("Portfolio Name")
                description = st.text_area("Description")
                submitted = st.form_submit_button("Create")
                
                if submitted and name:
                    try:
                        response = requests.post(f"{BACKEND_URL}/api/portfolios", json={
                            "user_id": user_id,
                            "name": name,
                            "description": description
                        })
                        if response.status_code == 200:
                            st.success("Portfolio created successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to create portfolio")
                    except:
                        st.error("Failed to connect to backend")
    
    # Portfolio details
    if portfolio_id:
        try:
            response = requests.get(f"{BACKEND_URL}/api/portfolios/{portfolio_id}")
            if response.status_code == 200:
                portfolio_data = response.json()
                
                # Portfolio summary
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Value", f"${portfolio_data['total_value']:.2f}")
                with col2:
                    st.metric("Total Cost", f"${portfolio_data['total_cost']:.2f}")
                with col3:
                    gain_loss = portfolio_data['total_gain_loss']
                    st.metric("Total Gain/Loss", f"${gain_loss:.2f}", f"{portfolio_data['total_gain_loss_percent']:.2f}%")
                with col4:
                    st.metric("Holdings", portfolio_data['holdings_count'])
                
                # Holdings table
                st.subheader("Holdings")
                holdings = portfolio_data.get('holdings', [])
                
                if holdings:
                    df = pd.DataFrame(holdings)
                    df['Gain/Loss %'] = df['gain_loss_percent'].round(2)
                    df['Market Value'] = df['market_value'].round(2)
                    df['Gain/Loss $'] = df['gain_loss'].round(2)
                    
                    st.dataframe(df[['symbol', 'quantity', 'purchase_price', 'current_price', 'Market Value', 'Gain/Loss $', 'Gain/Loss %']])
                
                # Add new holding
                st.subheader("Add New Holding")
                with st.form("add_holding"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        symbol = st.text_input("Symbol")
                    with col2:
                        quantity = st.number_input("Quantity", min_value=0.01, step=0.01)
                    with col3:
                        purchase_price = st.number_input("Purchase Price", min_value=0.01, step=0.01)
                    
                    submitted = st.form_submit_button("Add Holding")
                    
                    if submitted and all([symbol, quantity, purchase_price]):
                        try:
                            response = requests.post(f"{BACKEND_URL}/api/portfolios/{portfolio_id}/holdings", json={
                                "symbol": symbol.upper(),
                                "quantity": quantity,
                                "purchase_price": purchase_price
                            })
                            if response.status_code == 200:
                                st.success("Holding added successfully!")
                                st.rerun()
                            else:
                                st.error("Failed to add holding")
                        except:
                            st.error("Failed to connect to backend")
                
        except:
            st.error("Failed to load portfolio data")

def alerts_tab():
    """Price alerts management tab"""
    st.header("Price Alerts")
    
    user_id = st.session_state.get('user_id', 'demo_user')
    
    # Alert summary
    try:
        response = requests.get(f"{BACKEND_URL}/api/alerts/summary?user_id={user_id}")
        if response.status_code == 200:
            summary = response.json()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Alerts", summary['total_alerts'])
            with col2:
                st.metric("Active Alerts", summary['active_alerts'])
            with col3:
                st.metric("Triggered Alerts", summary['triggered_alerts'])
    except:
        st.warning("Could not load alert summary")
    
    # Create new alert
    st.subheader("Create New Alert")
    with st.form("create_alert"):
        col1, col2, col3 = st.columns(3)
        with col1:
            symbol = st.text_input("Stock Symbol")
        with col2:
            target_price = st.number_input("Target Price", min_value=0.01, step=0.01)
        with col3:
            alert_type = st.selectbox("Alert Type", ["above", "below"])
        
        submitted = st.form_submit_button("Create Alert")
        
        if submitted and all([symbol, target_price]):
            try:
                response = requests.post(f"{BACKEND_URL}/api/alerts", json={
                    "user_id": user_id,
                    "symbol": symbol.upper(),
                    "target_price": target_price,
                    "alert_type": alert_type
                })
                if response.status_code == 200:
                    st.success("Alert created successfully!")
                    st.rerun()
                else:
                    st.error("Failed to create alert")
            except:
                st.error("Failed to connect to backend")
    
    # Active alerts
    st.subheader("Your Alerts")
    try:
        response = requests.get(f"{BACKEND_URL}/api/alerts?user_id={user_id}&active_only=false")
        if response.status_code == 200:
            alerts = response.json().get("alerts", [])
            
            if alerts:
                for alert in alerts:
                    currency = get_currency_symbol(alert['symbol'])
                    with st.expander(f"{alert['symbol']} - {currency}{alert['target_price']:.2f} ({alert['alert_type']})"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if alert['current_price']:
                                currency = get_currency_symbol(alert['symbol'])
                                st.write(f"**Current Price:** {currency}{alert['current_price']:.2f}")
                            else:
                                st.write("**Current Price:** Price not available")
                            st.write(f"**Status:** {'Active' if alert['is_active'] else 'Inactive'}")
                        with col2:
                            if alert['distance_to_target']:
                                currency = get_currency_symbol(alert['symbol'])
                                st.write(f"**Distance:** {currency}{alert['distance_to_target']['price_distance']:.2f}")
                                st.write(f"**Percentage:** {alert['distance_to_target']['percentage_distance']:.2f}%")
                        with col3:
                            if st.button(f"Delete", key=f"delete_{alert['id']}"):
                                try:
                                    response = requests.delete(f"{BACKEND_URL}/api/alerts/{alert['id']}?user_id={user_id}")
                                    if response.status_code == 200:
                                        st.success("Alert deleted!")
                                        st.rerun()
                                except:
                                    st.error("Failed to delete alert")
            else:
                st.info("No alerts created yet")
    except:
        st.error("Failed to load alerts")

def settings_tab():
    """Settings and configuration tab"""
    st.header("Settings")
    
    # User ID setting
    st.subheader("User Configuration")
    user_id = st.text_input("User ID", value=st.session_state.get('user_id', 'demo_user'))
    if st.button("Save User ID"):
        st.session_state['user_id'] = user_id
        st.success("User ID saved!")
    
    # Data management
    st.subheader("Data Management")
    if st.button("Update Stock Data"):
        with st.spinner("Updating stock data..."):
            try:
                response = requests.post(f"{BACKEND_URL}/api/update-data")
                if response.status_code == 200:
                    st.success("Stock data updated successfully!")
                else:
                    st.error("Failed to update stock data")
            except:
                st.error("Failed to connect to backend")
    
    if st.button("Check Price Alerts"):
        with st.spinner("Checking alerts..."):
            try:
                response = requests.post(f"{BACKEND_URL}/api/alerts/check")
                if response.status_code == 200:
                    result = response.json()
                    triggered_count = result['count']
                    if triggered_count > 0:
                        st.success(f"{triggered_count} alerts were triggered!")
                    else:
                        st.info("No alerts triggered")
                else:
                    st.error("Failed to check alerts")
            except:
                st.error("Failed to connect to backend")
    
    # Backend status
    st.subheader("System Status")
    try:
        response = requests.get(f"{BACKEND_URL}/health")
        if response.status_code == 200:
            st.success("✅ Backend is running")
        else:
            st.error("❌ Backend is not responding")
    except:
        st.error("❌ Cannot connect to backend")

def display_stock_info(symbol):
    """Display stock company information"""
    try:
        response = requests.get(f"{BACKEND_URL}/api/stock/{symbol}/info")
        if response.status_code == 200:
            info = response.json()
            
            st.sidebar.markdown("### 📊 Company Info")
            st.sidebar.markdown(f"**{info.get('company_name', 'N/A')}**")
            st.sidebar.markdown(f"**Sector:** {info.get('sector', 'N/A')}")
            st.sidebar.markdown(f"**Exchange:** {info.get('exchange', 'N/A')}")
            
            if info.get('market_cap', 0) > 0:
                market_cap = info['market_cap'] / 1e9  # Convert to billions
                currency = get_currency_symbol(symbol)
                st.sidebar.markdown(f"**Market Cap:** {currency}{market_cap:.2f}B")
            
            if info.get('pe_ratio', 0) > 0:
                st.sidebar.markdown(f"**P/E Ratio:** {info['pe_ratio']:.2f}")
                
    except:
        pass  # Silently fail if info can't be loaded

def market_overview_tab():
    """Market overview and sector analysis tab"""
    st.header("📈 Market Overview & Analysis")
    
    # Market indices
    try:
        response = requests.get(f"{BACKEND_URL}/api/market/indices")
        if response.status_code == 200:
            indices = response.json().get("indices", {})
            
            if indices:
                st.subheader("Market Indices Performance")
                
                # Create a chart for indices
                indices_df = pd.DataFrame([
                    {
                        "Index": data["name"],
                        "Price": data["current_price"],
                        "Change": data["change"],
                        "Change %": data["change_percent"]
                    }
                    for symbol, data in indices.items()
                ])
                
                # Display as metrics
                cols = st.columns(len(indices))
                for i, (symbol, data) in enumerate(indices.items()):
                    with cols[i]:
                        currency = get_currency_symbol(symbol)
                        st.metric(
                            data["name"],
                            f"{currency}{data['current_price']:,.2f}",
                            f"{data['change']:+.2f} ({data['change_percent']:+.2f}%)"
                        )
                
                # Bar chart of changes
                fig = px.bar(
                    indices_df, 
                    x="Index", 
                    y="Change %",
                    title="Market Indices Performance (%)",
                    color="Change %",
                    color_continuous_scale=["red", "white", "green"]
                )
                st.plotly_chart(fig, use_container_width=True)
    except:
        st.error("Could not load market indices data")
    
    # Top Indian Stocks Performance
    st.subheader("🇮🇳 Top Indian Stocks")
    
    indian_stocks = [
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HINDUNILVR.NS", "ICICIBANK.NS"
    ]
    
    stock_performance = []
    
    with st.spinner("Loading stock performance..."):
        for symbol in indian_stocks:
            try:
                data = get_stock_data(symbol, 30)  # Last 30 days
                if data and len(data) > 1:
                    latest_price = data[-1]["close"]
                    prev_price = data[-2]["close"]
                    change_pct = ((latest_price - prev_price) / prev_price) * 100
                    
                    stock_performance.append({
                        "Symbol": symbol.replace(".NS", ""),
                        "Price": latest_price,
                        "Change %": change_pct
                    })
            except:
                pass
    
    if stock_performance:
        performance_df = pd.DataFrame(stock_performance)
        
        # Display performance table
        st.dataframe(performance_df, use_container_width=True)
        
        # Performance chart
        fig = px.bar(
            performance_df,
            x="Symbol",
            y="Change %",
            title="Top Indian Stocks Performance (Daily %)",
            color="Change %",
            color_continuous_scale=["red", "white", "green"]
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Loading stock performance data...")
    
    # Sector Analysis
    st.subheader("📊 Sector Analysis")
    
    sectors = {
        "Banking": ["ICICIBANK.NS", "HDFCBANK.NS", "SBIN.NS", "KOTAKBANK.NS"],
        "IT": ["TCS.NS", "INFY.NS", "WIPRO.NS"],
        "Energy": ["RELIANCE.NS", "ONGC.NS"],
        "Auto": ["MARUTI.NS"],
        "FMCG": ["HINDUNILVR.NS", "ITC.NS"]
    }
    
    sector_performance = []
    
    for sector_name, stocks in sectors.items():
        sector_changes = []
        for stock in stocks:
            try:
                data = get_stock_data(stock, 7)  # Last week
                if data and len(data) > 1:
                    latest = data[-1]["close"]
                    week_ago = data[0]["close"]
                    change = ((latest - week_ago) / week_ago) * 100
                    sector_changes.append(change)
            except:
                pass
        
        if sector_changes:
            avg_change = sum(sector_changes) / len(sector_changes)
            sector_performance.append({
                "Sector": sector_name,
                "Weekly Change %": avg_change
            })
    
    if sector_performance:
        sector_df = pd.DataFrame(sector_performance)
        
        fig = px.bar(
            sector_df,
            x="Sector",
            y="Weekly Change %",
            title="Sector Performance (Weekly %)",
            color="Weekly Change %",
            color_continuous_scale=["red", "white", "green"]
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
