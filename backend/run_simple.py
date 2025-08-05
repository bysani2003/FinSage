#!/usr/bin/env python3
"""
Simple version using SQLite instead of PostgreSQL
Run this for quick local testing
"""

import os
import sqlite3
from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

app = Flask(__name__)
CORS(app)

# SQLite database path
DB_PATH = "stock_predictor.db"

def init_db():
    """Initialize SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stocks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            date DATE NOT NULL,
            open_price REAL,
            high_price REAL,
            low_price REAL,
            close_price REAL,
            volume INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, date)
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            prediction_date DATE NOT NULL,
            predicted_price REAL,
            model_version TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS price_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            target_price REAL NOT NULL,
            alert_type TEXT NOT NULL CHECK (alert_type IN ('above', 'below')),
            is_active BOOLEAN DEFAULT TRUE,
            triggered_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()

def fetch_and_save_stock_data(symbol):
    """Fetch stock data from Yahoo Finance and save to SQLite"""
    try:
        print(f"Fetching data for symbol: {symbol}")
        stock = yf.Ticker(symbol)
        
        # Try different periods if 1y fails
        periods = ["1y", "6mo", "3mo", "1mo"]
        data = None
        
        for period in periods:
            try:
                print(f"Trying period: {period}")
                data = stock.history(period=period)
                if not data.empty:
                    print(f"Successfully fetched {len(data)} records for {symbol}")
                    break
            except Exception as e:
                print(f"Failed to fetch {period} data for {symbol}: {e}")
                continue
        
        if data is None or data.empty:
            print(f"No data available for {symbol}")
            return None
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        saved_count = 0
        for date, row in data.iterrows():
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO stocks (symbol, date, open_price, high_price, low_price, close_price, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol,
                    date.date(),
                    float(row['Open']),
                    float(row['High']),
                    float(row['Low']),
                    float(row['Close']),
                    int(row['Volume'])
                ))
                saved_count += 1
            except Exception as e:
                print(f"Error saving row for {symbol} on {date}: {e}")
                continue
        
        conn.commit()
        conn.close()
        print(f"Saved {saved_count} records for {symbol}")
        return data
    
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

def get_stock_data_from_db(symbol, days=365):
    """Get stock data from SQLite"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT date, open_price, high_price, low_price, close_price, volume
            FROM stocks
            WHERE symbol = ? AND date >= date('now', '-{} days')
            ORDER BY date
        """.format(days), (symbol,))
        
        data = cursor.fetchall()
        if not data:
            return None
        
        df = pd.DataFrame(data, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        return df
    
    except Exception as e:
        print(f"Error getting data: {e}")
        return None
    
    finally:
        conn.close()

def create_technical_features(data):
    """Create technical indicators for ML model"""
    df = data.copy()
    
    # Moving averages
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Price changes and volatility
    df['Price_Change'] = df['Close'].pct_change()
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Volatility'] = df['Close'].rolling(window=10).std()
    
    # High-Low spread
    df['HL_Spread'] = (df['High'] - df['Low']) / df['Close']
    
    # Target variable (next day's closing price)
    df['Target'] = df['Close'].shift(-1)
    
    return df.dropna()

def train_and_predict_price(symbol):
    """Train ML model and predict next day price"""
    try:
        # Get sufficient data for training
        data = get_stock_data_from_db(symbol, 365)
        if data is None or len(data) < 50:
            # Fetch fresh data if not enough in DB
            fresh_data = fetch_and_save_stock_data(symbol)
            if fresh_data is not None:
                data = get_stock_data_from_db(symbol, 365)
        
        if data is None or len(data) < 50:
            return None, "Insufficient data for prediction (need at least 50 days)"
        
        # Create features
        df_features = create_technical_features(data)
        if len(df_features) < 30:
            return None, "Insufficient data after feature creation"
        
        # Prepare features and target
        feature_columns = ['Open', 'High', 'Low', 'Volume', 'MA_5', 'MA_10', 'MA_20', 
                          'RSI', 'Price_Change', 'Volume_Change', 'Volatility', 'HL_Spread']
        
        X = df_features[feature_columns].iloc[:-1]  # Exclude last row (no target)
        y = df_features['Target'].iloc[:-1]
        
        # Split data for training
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Train Random Forest model
        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test) if len(X_test) > 0 else 0
        
        # Make prediction using latest data
        latest_features = df_features[feature_columns].iloc[-1:].values
        prediction = model.predict(latest_features)[0]
        
        # Get current price for validation
        current_price = data['Close'].iloc[-1]
        
        # Save prediction to database
        save_prediction_to_db(symbol, prediction)
        
        return prediction, f"Model trained successfully. Train R²: {train_score:.3f}, Test R²: {test_score:.3f}"
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None, f"Prediction error: {str(e)}"

def save_prediction_to_db(symbol, predicted_price):
    """Save prediction to SQLite database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO predictions (symbol, prediction_date, predicted_price, model_version)
            VALUES (?, ?, ?, ?)
        """, (symbol, datetime.now().date(), predicted_price, "RandomForest_v1"))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error saving prediction: {e}")

# Routes
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route('/api/test/<symbol>', methods=['GET'])
def test_symbol(symbol):
    """Test if a symbol can be fetched from Yahoo Finance"""
    try:
        print(f"Testing symbol: {symbol}")
        stock = yf.Ticker(symbol)
        
        # Try to get basic info
        info = stock.info
        
        # Try to get recent data
        data = stock.history(period="5d")
        
        result = {
            "symbol": symbol,
            "valid": not data.empty,
            "data_points": len(data),
            "company_name": info.get('longName', 'Unknown'),
            "currency": info.get('currency', 'Unknown'),
            "exchange": info.get('exchange', 'Unknown')
        }
        
        if not data.empty:
            latest = data.iloc[-1]
            result["latest_price"] = float(latest['Close'])
            result["latest_date"] = data.index[-1].date().isoformat()
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            "symbol": symbol,
            "valid": False,
            "error": str(e)
        })

@app.route('/api/stocks', methods=['GET'])
def get_popular_stocks():
    """Get popular Indian and global stocks"""
    indian_stocks = [
        'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS',
        'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS', 'KOTAKBANK.NS', 'LT.NS',
        'ASIANPAINT.NS', 'MARUTI.NS', 'HDFCBANK.NS', 'WIPRO.NS', 'ONGC.NS'
    ]
    
    global_stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    
    return jsonify({
        "symbols": indian_stocks + global_stocks,
        "indian_stocks": indian_stocks,
        "global_stocks": global_stocks,
        "indices": ['^NSEI', '^BSESN', '^DJI', '^GSPC']  # Nifty 50, Sensex, Dow Jones, S&P 500
    })

@app.route('/api/stock/<symbol>/data', methods=['GET'])
def get_stock_data(symbol):
    try:
        days = request.args.get('days', 365, type=int)
        symbol = symbol.upper()
        
        # First try to get from database
        data = get_stock_data_from_db(symbol, days)
        
        if data is None or data.empty:
            print(f"No data in DB for {symbol}, fetching fresh data...")
            
            # Try to fetch fresh data
            fresh_data = fetch_and_save_stock_data(symbol)
            if fresh_data is not None:
                print(f"Fresh data fetched successfully for {symbol}")
                data = get_stock_data_from_db(symbol, days)
            else:
                print(f"Failed to fetch fresh data for {symbol}")
                
                # Generate mock data as fallback for demonstration
                print(f"Generating mock data for {symbol}")
                mock_data = generate_mock_data(symbol, days)
                return jsonify({"symbol": symbol, "data": mock_data, "source": "mock"})
        
        if data is None or data.empty:
            return jsonify({"error": f"No data available for {symbol}. This might be an invalid symbol or data source issue."}), 404
        
        result = []
        for date, row in data.iterrows():
            result.append({
                "date": date.isoformat(),
                "open": float(row['Open']),
                "high": float(row['High']),
                "low": float(row['Low']),
                "close": float(row['Close']),
                "volume": int(row['Volume'])
            })
        
        return jsonify({"symbol": symbol, "data": result, "source": "yahoo_finance"})
    
    except Exception as e:
        print(f"Error in get_stock_data: {e}")
        return jsonify({"error": str(e)}), 500

def generate_mock_data(symbol, days):
    """Generate mock stock data for demonstration when real data fails"""
    try:
        import random
        base_price = 2500 if symbol.endswith('.NS') else 150  # Base price in rupees or dollars
        
        mock_data = []
        current_price = base_price
        
        for i in range(min(days, 30)):  # Generate up to 30 days of mock data
            date = (datetime.now() - timedelta(days=i)).date()
            
            # Generate realistic price movements
            change_percent = random.uniform(-3, 3)  # -3% to +3% daily change
            current_price *= (1 + change_percent / 100)
            
            open_price = current_price * random.uniform(0.98, 1.02)
            high_price = max(open_price, current_price) * random.uniform(1.0, 1.03)
            low_price = min(open_price, current_price) * random.uniform(0.97, 1.0)
            volume = random.randint(100000, 5000000)
            
            mock_data.insert(0, {  # Insert at beginning for chronological order
                "date": date.isoformat(),
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "close": round(current_price, 2),
                "volume": volume
            })
        
        return mock_data
    
    except Exception as e:
        print(f"Error generating mock data: {e}")
        return []

@app.route('/api/stock/<symbol>/predict', methods=['POST'])
def predict_stock_price(symbol):
    try:
        prediction, message = train_and_predict_price(symbol.upper())
        
        if prediction is None:
            return jsonify({"error": message}), 400
        
        return jsonify({
            "symbol": symbol.upper(),
            "predicted_price": float(prediction),
            "prediction_date": (datetime.now().date()).isoformat(),
            "message": message
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/update-data', methods=['POST'])
def update_data():
    try:
        # Get symbols from request or use defaults
        request_data = request.get_json() if request.content_type == 'application/json' else {}
        symbols = request_data.get('symbols', [
            'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'ICICIBANK.NS', 'SBIN.NS',
            'BHARTIARTL.NS', 'ITC.NS', 'HDFCBANK.NS', 'AAPL', 'GOOGL'
        ])
        
        updated = []
        failed = []
        
        for symbol in symbols:
            print(f"Updating {symbol}...")
            try:
                data = fetch_and_save_stock_data(symbol)
                if data is not None:
                    updated.append(symbol)
                else:
                    failed.append(symbol)
            except Exception as e:
                print(f"Failed to update {symbol}: {e}")
                failed.append(symbol)
        
        return jsonify({
            "message": f"Updated data for {len(updated)} symbols",
            "symbols": updated,
            "failed": failed,
            "total_attempted": len(symbols)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/market/indices', methods=['GET'])
def get_market_indices():
    """Get market indices data"""
    try:
        indices = ['^NSEI', '^BSESN', '^DJI', '^GSPC']
        indices_data = {}
        
        for index in indices:
            try:
                data = fetch_and_save_stock_data(index)
                if data is not None:
                    latest = data.iloc[-1]
                    indices_data[index] = {
                        "name": get_index_name(index),
                        "current_price": float(latest['Close']),
                        "change": float(latest['Close'] - data.iloc[-2]['Close']) if len(data) > 1 else 0,
                        "change_percent": ((latest['Close'] - data.iloc[-2]['Close']) / data.iloc[-2]['Close'] * 100) if len(data) > 1 else 0
                    }
            except Exception as e:
                print(f"Error fetching {index}: {e}")
        
        return jsonify({"indices": indices_data})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def get_index_name(symbol):
    """Get human readable index names"""
    names = {
        '^NSEI': 'Nifty 50',
        '^BSESN': 'BSE Sensex',
        '^DJI': 'Dow Jones',
        '^GSPC': 'S&P 500'
    }
    return names.get(symbol, symbol)

@app.route('/api/stock/<symbol>/info', methods=['GET'])
def get_stock_info(symbol):
    """Get stock company information"""
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        
        # Extract relevant information
        stock_info = {
            "symbol": symbol.upper(),
            "company_name": info.get('longName', 'N/A'),
            "sector": info.get('sector', 'N/A'),
            "industry": info.get('industry', 'N/A'),
            "market_cap": info.get('marketCap', 0),
            "pe_ratio": info.get('trailingPE', 0),
            "dividend_yield": info.get('dividendYield', 0),
            "52_week_high": info.get('fiftyTwoWeekHigh', 0),
            "52_week_low": info.get('fiftyTwoWeekLow', 0),
            "currency": info.get('currency', 'USD'),
            "exchange": info.get('exchange', 'N/A')
        }
        
        return jsonify(stock_info)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/search/stocks', methods=['GET'])
def search_stocks():
    """Search for Indian stocks by name or symbol"""
    query = request.args.get('q', '').strip().upper()
    
    if not query or len(query) < 2:
        return jsonify({"stocks": []})
    
    # Predefined list of Indian stocks with company names
    indian_stocks = {
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
        'DIVISLAB.NS': 'Divi\'s Laboratories Limited',
        'EICHERMOT.NS': 'Eicher Motors Limited',
        'GRASIM.NS': 'Grasim Industries Limited',
        'HEROMOTOCO.NS': 'Hero MotoCorp Limited',
        'HINDALCO.NS': 'Hindalco Industries Limited',
        'HINDPETRO.NS': 'Hindustan Petroleum Corporation Limited',
        'INDUSINDBK.NS': 'IndusInd Bank Limited',
        'IOC.NS': 'Indian Oil Corporation Limited',
        'JSWSTEEL.NS': 'JSW Steel Limited',
        'M&M.NS': 'Mahindra & Mahindra Limited',
        'SHREECEM.NS': 'Shree Cement Limited',
        'TATACONSUM.NS': 'Tata Consumer Products Limited',
        'TATAMOTORS.NS': 'Tata Motors Limited',
        'TATASTEEL.NS': 'Tata Steel Limited',
        'UPL.NS': 'UPL Limited',
        'VEDL.NS': 'Vedanta Limited',
        'ADANIPORTS.NS': 'Adani Ports and Special Economic Zone Limited',
        'APOLLOHOSP.NS': 'Apollo Hospitals Enterprise Limited',
        'BRITANNIA.NS': 'Britannia Industries Limited',
        'GODREJCP.NS': 'Godrej Consumer Products Limited',
        'HDFC.NS': 'Housing Development Finance Corporation Limited',
        'HDFCLIFE.NS': 'HDFC Life Insurance Company Limited',
        'ICICIPRULI.NS': 'ICICI Prudential Life Insurance Company Limited',
        'PIDILITIND.NS': 'Pidilite Industries Limited',
        'SBILIFE.NS': 'SBI Life Insurance Company Limited',
        'ZEEL.NS': 'Zee Entertainment Enterprises Limited'
    }
    
    # Search by symbol or company name
    matches = []
    for symbol, name in indian_stocks.items():
        symbol_match = query in symbol.replace('.NS', '')
        name_match = query in name.upper()
        
        if symbol_match or name_match:
            matches.append({
                'symbol': symbol,
                'name': name,
                'exchange': 'NSE'
            })
    
    # If query looks like a stock symbol, also try adding .NS
    if query.replace('.NS', '') not in [s['symbol'].replace('.NS', '') for s in matches]:
        test_symbol = query if query.endswith('.NS') else query + '.NS'
        try:
            # Validate by trying to fetch data
            stock = yf.Ticker(test_symbol)
            info = stock.info
            if info and info.get('symbol'):
                matches.append({
                    'symbol': test_symbol,
                    'name': info.get('longName', test_symbol),
                    'exchange': 'NSE'
                })
        except:
            pass
    
    return jsonify({"stocks": matches[:10]})  # Limit to 10 results

@app.route('/api/validate/stock/<symbol>', methods=['GET'])
def validate_stock(symbol):
    """Validate if a stock symbol exists and can fetch data"""
    try:
        # Add .NS if not present for Indian stocks
        if not symbol.endswith('.NS') and '.' not in symbol:
            symbol = symbol + '.NS'
        
        stock = yf.Ticker(symbol)
        info = stock.info
        
        # Try to get recent data
        data = stock.history(period="5d")
        
        if data.empty or not info.get('symbol'):
            return jsonify({"valid": False, "error": "Stock not found or no data available"})
        
        return jsonify({
            "valid": True,
            "symbol": symbol,
            "name": info.get('longName', symbol),
            "current_price": float(data['Close'].iloc[-1]) if not data.empty else None,
            "currency": info.get('currency', 'INR'),
            "exchange": info.get('exchange', 'NSE')
        })
    
    except Exception as e:
        return jsonify({"valid": False, "error": str(e)})

# Simple alert checking without email
def check_triggered_alerts():
    """Check for triggered alerts (without email notifications)"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get all active alerts with current prices
        cursor.execute("""
            SELECT pa.id, pa.user_id, pa.symbol, pa.target_price, pa.alert_type,
                   s.close_price, s.date
            FROM price_alerts pa
            LEFT JOIN (
                SELECT symbol, close_price, date,
                       ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date DESC) as rn
                FROM stocks
            ) s ON pa.symbol = s.symbol AND s.rn = 1
            WHERE pa.is_active = 1
        """)
        
        alerts = cursor.fetchall()
        triggered_alerts = []
        
        for alert in alerts:
            alert_id, user_id, symbol, target_price, alert_type, current_price, price_date = alert
            
            if current_price is None:
                continue
            
            is_triggered = False
            if alert_type == "above" and current_price >= target_price:
                is_triggered = True
            elif alert_type == "below" and current_price <= target_price:
                is_triggered = True
            
            if is_triggered:
                # Mark alert as triggered
                cursor.execute("""
                    UPDATE price_alerts
                    SET is_active = 0, triggered_at = ?
                    WHERE id = ?
                """, (datetime.now(), alert_id))
                
                triggered_alerts.append({
                    "id": alert_id,
                    "user_id": user_id,
                    "symbol": symbol,
                    "target_price": target_price,
                    "current_price": current_price,
                    "alert_type": alert_type
                })
                
                print(f"Alert triggered for {symbol}: {alert_type} {target_price}")
        
        conn.commit()
        return triggered_alerts
        
    except Exception as e:
        print(f"Error checking alerts: {e}")
        return []
    finally:
        if 'conn' in locals():
            conn.close()

def generate_sentiment_analysis(symbol):
    """Generate sentiment analysis for a stock symbol"""
    try:
        # Remove .NS suffix for company name lookup
        company_symbol = symbol.replace('.NS', '').replace('.BO', '')
        
        # Company name mapping for better sentiment context
        company_names = {
            'RELIANCE': 'Reliance Industries',
            'TCS': 'Tata Consultancy Services',
            'INFY': 'Infosys',
            'HINDUNILVR': 'Hindustan Unilever',
            'ICICIBANK': 'ICICI Bank',
            'SBIN': 'State Bank of India',
            'BHARTIARTL': 'Bharti Airtel',
            'ITC': 'ITC Limited',
            'KOTAKBANK': 'Kotak Mahindra Bank',
            'LT': 'Larsen & Toubro',
            'ASIANPAINT': 'Asian Paints',
            'MARUTI': 'Maruti Suzuki',
            'HDFCBANK': 'HDFC Bank',
            'WIPRO': 'Wipro',
            'ONGC': 'ONGC',
            'BAJFINANCE': 'Bajaj Finance',
            'HCLTECH': 'HCL Technologies',
            'TECHM': 'Tech Mahindra',
            'SUNPHARMA': 'Sun Pharma',
            'TITAN': 'Titan Company'
        }
        
        company_name = company_names.get(company_symbol, company_symbol)
        
        # Simulate news headlines based on recent stock performance
        stock_data = get_stock_data_from_db(symbol, 7)  # Last 7 days
        
        if stock_data is not None and len(stock_data) > 1:
            latest_price = stock_data['Close'].iloc[-1]
            week_ago_price = stock_data['Close'].iloc[0]
            performance = ((latest_price - week_ago_price) / week_ago_price) * 100
            
            # Generate contextual news based on performance
            if performance > 5:
                news_headlines = [
                    f"{company_name} shows strong quarterly performance",
                    f"Analysts upgrade {company_name} price target",
                    f"{company_name} reports better than expected earnings",
                    f"Institutional investors increase stake in {company_name}",
                    f"{company_name} announces expansion plans"
                ]
                base_sentiment = 0.3 + (performance / 100)
            elif performance < -5:
                news_headlines = [
                    f"{company_name} faces market headwinds",
                    f"Concerns over {company_name} quarterly outlook",
                    f"{company_name} stock under pressure",
                    f"Market volatility affects {company_name}",
                    f"Analysts cautious on {company_name} near-term prospects"
                ]
                base_sentiment = -0.2 + (performance / 100)
            else:
                news_headlines = [
                    f"{company_name} maintains steady performance",
                    f"Mixed signals for {company_name} stock",
                    f"{company_name} trading in range",
                    f"Market consolidation affects {company_name}",
                    f"{company_name} awaits sector developments"
                ]
                base_sentiment = 0.1 + (performance / 200)
        else:
            # Default neutral sentiment
            news_headlines = [
                f"{company_name} in focus for investors",
                f"Market watches {company_name} developments",
                f"{company_name} trading steady",
                f"Sector outlook impacts {company_name}"
            ]
            base_sentiment = 0.0
        
        # Analyze sentiment of each headline
        sentiment_scores = []
        for headline in news_headlines:
            # Simple keyword-based sentiment analysis
            positive_words = ['strong', 'upgrade', 'better', 'expected', 'increase', 'expansion', 'growth', 'positive', 'good']
            negative_words = ['concerns', 'pressure', 'volatility', 'cautious', 'headwinds', 'under', 'weak', 'decline']
            
            headline_lower = headline.lower()
            pos_score = sum(1 for word in positive_words if word in headline_lower)
            neg_score = sum(1 for word in negative_words if word in headline_lower)
            
            # Calculate sentiment score (-1 to 1)
            if pos_score > neg_score:
                score = min(0.8, (pos_score - neg_score) * 0.3)
            elif neg_score > pos_score:
                score = max(-0.8, -(neg_score - pos_score) * 0.3)
            else:
                score = base_sentiment
            
            sentiment_scores.append(score)
        
        # Calculate average sentiment
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
        
        # Generate historical sentiment data (last 10 days)
        historical_sentiment = []
        for i in range(10):
            date = (datetime.now() - timedelta(days=i)).date()
            # Add some variation to sentiment
            daily_sentiment = avg_sentiment + np.random.normal(0, 0.1)
            daily_sentiment = max(-1, min(1, daily_sentiment))  # Clamp between -1 and 1
            
            historical_sentiment.append({
                "date": date.isoformat(),
                "sentiment_score": round(daily_sentiment, 3),
                "news_count": len(news_headlines) + np.random.randint(-2, 3)
            })
        
        # Reverse to get chronological order
        historical_sentiment.reverse()
        
        return {
            "symbol": symbol,
            "current_sentiment": round(avg_sentiment, 3),
            "historical_sentiment": historical_sentiment,
            "news_headlines": news_headlines[:5],  # Return top 5 headlines
            "analysis_method": "Performance-based sentiment with keyword analysis"
        }
        
    except Exception as e:
        print(f"Error generating sentiment for {symbol}: {e}")
        # Return neutral sentiment as fallback
        return {
            "symbol": symbol,
            "current_sentiment": 0.0,
            "historical_sentiment": [
                {
                    "date": (datetime.now() - timedelta(days=i)).date().isoformat(),
                    "sentiment_score": 0.0,
                    "news_count": 5
                } for i in range(10, 0, -1)
            ],
            "news_headlines": [f"No recent news available for {symbol}"],
            "analysis_method": "Fallback neutral sentiment"
        }

@app.route('/api/stock/<symbol>/sentiment', methods=['GET'])
def get_stock_sentiment(symbol):
    """Get sentiment analysis for a stock"""
    try:
        sentiment_data = generate_sentiment_analysis(symbol.upper())
        return jsonify(sentiment_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Alert management endpoints
@app.route('/api/portfolios', methods=['GET'])
def get_portfolios():
    return jsonify({"portfolios": []})

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    return jsonify({"alerts": []})

@app.route('/api/alerts/summary', methods=['GET'])
def get_alert_summary():
    return jsonify({"total_alerts": 0, "active_alerts": 0, "triggered_alerts": 0})

if __name__ == '__main__':
    print("Initializing SQLite database...")
    init_db()
    print("Starting simple stock predictor server...")
    print("Dashboard will be available at: http://localhost:8501")
    print("API available at: http://localhost:5000")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
