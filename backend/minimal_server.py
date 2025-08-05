#!/usr/bin/env python3
"""
Minimal server for testing - no ML dependencies
"""

from flask import Flask, jsonify
from flask_cors import CORS
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Mock data for testing
MOCK_STOCK_DATA = [
    {"date": "2024-01-01", "open": 190.0, "high": 195.0, "low": 188.0, "close": 193.0, "volume": 1000000},
    {"date": "2024-01-02", "open": 193.0, "high": 198.0, "low": 191.0, "close": 196.0, "volume": 1100000},
    {"date": "2024-01-03", "open": 196.0, "high": 200.0, "low": 194.0, "close": 199.0, "volume": 1200000},
]

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route('/api/stocks', methods=['GET'])
def get_popular_stocks():
    return jsonify({"symbols": ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']})

@app.route('/api/stock/<symbol>/data', methods=['GET'])
def get_stock_data(symbol):
    return jsonify({"symbol": symbol.upper(), "data": MOCK_STOCK_DATA})

@app.route('/api/stock/<symbol>/predict', methods=['POST'])
def predict_stock_price(symbol):
    return jsonify({
        "symbol": symbol.upper(),
        "predicted_price": 205.50,
        "prediction_date": datetime.now().date().isoformat(),
        "message": "Mock prediction successful"
    })

@app.route('/api/update-data', methods=['POST'])
def update_data():
    return jsonify({"message": "Mock data updated", "symbols": ["AAPL", "GOOGL"]})

@app.route('/api/portfolios', methods=['GET'])
def get_portfolios():
    return jsonify({"portfolios": []})

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    return jsonify({"alerts": []})

@app.route('/api/alerts/summary', methods=['GET'])
def get_alert_summary():
    return jsonify({"total_alerts": 0, "active_alerts": 0, "triggered_alerts": 0})

@app.route('/api/stock/<symbol>/sentiment', methods=['GET'])
def get_stock_sentiment(symbol):
    """Mock sentiment analysis"""
    import random
    sentiment_score = random.uniform(-0.5, 0.5)  # Random sentiment between -0.5 and 0.5
    
    historical_sentiment = []
    for i in range(10):  # Generate 10 days of mock data
        date = datetime.now().date()
        historical_sentiment.append({
            "date": (date).isoformat(),
            "sentiment_score": random.uniform(-0.3, 0.3),
            "news_count": random.randint(5, 15)
        })
    
    return jsonify({
        "symbol": symbol.upper(),
        "current_sentiment": sentiment_score,
        "historical_sentiment": historical_sentiment
    })

@app.route('/api/stock/<symbol>/accuracy', methods=['GET'])
def get_prediction_accuracy(symbol):
    """Mock accuracy metrics"""
    return jsonify({
        "symbol": symbol.upper(),
        "accuracy_metrics": {
            "mae": 2.45,
            "mape": 1.8,
            "samples": 25
        }
    })

@app.route('/api/train/<symbol>', methods=['POST'])
def train_model(symbol):
    """Mock model training"""
    import time
    time.sleep(2)  # Simulate training time
    
    return jsonify({
        "symbol": symbol.upper(),
        "status": "trained",
        "message": "Mock model trained successfully. MSE: 2.45, MAE: 1.8"
    })

@app.route('/api/portfolios', methods=['POST'])
def create_portfolio():
    """Mock portfolio creation"""
    return jsonify({"portfolio_id": 1, "message": "Portfolio created successfully"})

@app.route('/api/portfolios/<int:portfolio_id>', methods=['GET'])
def get_portfolio_summary(portfolio_id):
    """Mock portfolio summary"""
    return jsonify({
        "total_value": 10500.00,
        "total_cost": 10000.00,
        "total_gain_loss": 500.00,
        "total_gain_loss_percent": 5.0,
        "holdings_count": 3,
        "holdings": [
            {
                "id": 1,
                "symbol": "AAPL",
                "quantity": 10,
                "purchase_price": 150.00,
                "current_price": 175.00,
                "market_value": 1750.00,
                "cost_basis": 1500.00,
                "gain_loss": 250.00,
                "gain_loss_percent": 16.67
            },
            {
                "id": 2,
                "symbol": "GOOGL",
                "quantity": 5,
                "purchase_price": 1700.00,
                "current_price": 1750.00,
                "market_value": 8750.00,
                "cost_basis": 8500.00,
                "gain_loss": 250.00,
                "gain_loss_percent": 2.94
            }
        ]
    })

@app.route('/api/alerts', methods=['POST'])
def create_alert():
    """Mock alert creation"""
    return jsonify({"alert_id": 1, "message": "Alert created successfully"})

if __name__ == '__main__':
    print("=" * 50)
    print("MINIMAL STOCK PREDICTOR SERVER STARTING...")
    print("=" * 50)
    print("Backend API: http://localhost:5000")
    print("Health check: http://localhost:5000/health")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
