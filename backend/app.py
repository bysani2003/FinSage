from flask import Flask, request, jsonify
from flask_cors import CORS
from data_fetcher import DataFetcher
from model import StockPredictor
from sentiment import SentimentAnalyzer
from portfolio import PortfolioManager
from alerts import AlertManager
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Initialize components
data_fetcher = DataFetcher()
predictor = StockPredictor()
sentiment_analyzer = SentimentAnalyzer()
portfolio_manager = PortfolioManager()
alert_manager = AlertManager()

# Common stock symbols
POPULAR_SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route('/api/stocks', methods=['GET'])
def get_popular_stocks():
    """Get list of popular stock symbols"""
    return jsonify({"symbols": POPULAR_SYMBOLS})

@app.route('/api/stock/<symbol>/data', methods=['GET'])
def get_stock_data(symbol):
    """Get historical stock data"""
    try:
        days = request.args.get('days', 365, type=int)
        data = data_fetcher.get_stock_data_from_db(symbol.upper(), days)
        
        if data is None or data.empty:
            # Fetch fresh data if not in database
            fresh_data = data_fetcher.fetch_stock_data(symbol.upper())
            if fresh_data is not None:
                data_fetcher.save_stock_data(symbol.upper(), fresh_data)
                data = data_fetcher.get_stock_data_from_db(symbol.upper(), days)
        
        if data is None or data.empty:
            return jsonify({"error": "No data found for symbol"}), 404
        
        # Convert to JSON format
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
        
        return jsonify({"symbol": symbol.upper(), "data": result})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/stock/<symbol>/predict', methods=['POST'])
def predict_stock_price(symbol):
    """Predict stock price for next day"""
    try:
        prediction, message = predictor.predict_price(symbol.upper())
        
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

@app.route('/api/stock/<symbol>/sentiment', methods=['GET'])
def get_stock_sentiment(symbol):
    """Get sentiment analysis for stock"""
    try:
        # Update sentiment
        sentiment_score, news_count = sentiment_analyzer.get_news_sentiment(symbol.upper())
        sentiment_analyzer.save_sentiment_data(symbol.upper(), sentiment_score, news_count)
        
        # Get historical sentiment
        sentiment_data = sentiment_analyzer.get_sentiment_data(symbol.upper())
        
        result = []
        for date, score, count in sentiment_data:
            result.append({
                "date": date.isoformat(),
                "sentiment_score": float(score),
                "news_count": count
            })
        
        return jsonify({
            "symbol": symbol.upper(),
            "current_sentiment": sentiment_score,
            "historical_sentiment": result
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/stock/<symbol>/accuracy', methods=['GET'])
def get_prediction_accuracy(symbol):
    """Get prediction accuracy metrics"""
    try:
        days = request.args.get('days', 30, type=int)
        accuracy = predictor.get_prediction_accuracy(symbol.upper(), days)
        
        if accuracy is None:
            return jsonify({"error": "No accuracy data available"}), 404
        
        return jsonify({
            "symbol": symbol.upper(),
            "accuracy_metrics": accuracy
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/train/<symbol>', methods=['POST'])
def train_model(symbol):
    """Train prediction model for a symbol"""
    try:
        success, message = predictor.train_model(symbol.upper())
        
        if not success:
            return jsonify({"error": message}), 400
        
        return jsonify({
            "symbol": symbol.upper(),
            "status": "trained",
            "message": message
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/update-data', methods=['POST'])
def update_data():
    """Update stock data for all popular symbols"""
    try:
        symbols = request.json.get('symbols', POPULAR_SYMBOLS)
        
        # Update stock data
        data_fetcher.update_stock_data(symbols)
        
        # Update sentiment data
        sentiment_analyzer.update_sentiment_for_symbols(symbols)
        
        return jsonify({
            "message": f"Updated data for {len(symbols)} symbols",
            "symbols": symbols
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Portfolio Management Routes
@app.route('/api/portfolios', methods=['GET'])
def get_portfolios():
    """Get all portfolios for a user"""
    user_id = request.args.get('user_id', 'demo_user')
    portfolios = portfolio_manager.get_portfolios(user_id)
    return jsonify({"portfolios": portfolios})

@app.route('/api/portfolios', methods=['POST'])
def create_portfolio():
    """Create a new portfolio"""
    data = request.json
    user_id = data.get('user_id', 'demo_user')
    name = data.get('name')
    description = data.get('description', '')
    
    if not name:
        return jsonify({"error": "Portfolio name is required"}), 400
    
    portfolio_id = portfolio_manager.create_portfolio(user_id, name, description)
    if portfolio_id:
        return jsonify({"portfolio_id": portfolio_id, "message": "Portfolio created successfully"})
    else:
        return jsonify({"error": "Failed to create portfolio"}), 500

@app.route('/api/portfolios/<int:portfolio_id>', methods=['GET'])
def get_portfolio_summary(portfolio_id):
    """Get portfolio summary with holdings"""
    summary = portfolio_manager.get_portfolio_summary(portfolio_id)
    return jsonify(summary)

@app.route('/api/portfolios/<int:portfolio_id>/holdings', methods=['POST'])
def add_holding():
    """Add a holding to portfolio"""
    data = request.json
    portfolio_id = int(request.view_args['portfolio_id'])
    
    symbol = data.get('symbol')
    quantity = data.get('quantity')
    purchase_price = data.get('purchase_price')
    purchase_date = data.get('purchase_date')
    
    if not all([symbol, quantity, purchase_price]):
        return jsonify({"error": "Symbol, quantity, and purchase_price are required"}), 400
    
    holding_id = portfolio_manager.add_holding(
        portfolio_id, symbol, quantity, purchase_price, purchase_date
    )
    
    if holding_id:
        return jsonify({"holding_id": holding_id, "message": "Holding added successfully"})
    else:
        return jsonify({"error": "Failed to add holding"}), 500

@app.route('/api/holdings/<int:holding_id>', methods=['DELETE'])
def remove_holding(holding_id):
    """Remove a holding"""
    success = portfolio_manager.remove_holding(holding_id)
    if success:
        return jsonify({"message": "Holding removed successfully"})
    else:
        return jsonify({"error": "Failed to remove holding"}), 500

@app.route('/api/holdings/<int:holding_id>', methods=['PUT'])
def update_holding(holding_id):
    """Update a holding"""
    data = request.json
    quantity = data.get('quantity')
    purchase_price = data.get('purchase_price')
    
    success = portfolio_manager.update_holding(holding_id, quantity, purchase_price)
    if success:
        return jsonify({"message": "Holding updated successfully"})
    else:
        return jsonify({"error": "Failed to update holding"}), 500

@app.route('/api/portfolios/<int:portfolio_id>/performance', methods=['GET'])
def get_portfolio_performance(portfolio_id):
    """Get portfolio performance history"""
    days = request.args.get('days', 30, type=int)
    performance = portfolio_manager.get_portfolio_performance_history(portfolio_id, days)
    return jsonify({"performance": performance})

# Price Alerts Routes
@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """Get user alerts"""
    user_id = request.args.get('user_id', 'demo_user')
    active_only = request.args.get('active_only', 'true').lower() == 'true'
    alerts = alert_manager.get_user_alerts(user_id, active_only)
    return jsonify({"alerts": alerts})

@app.route('/api/alerts', methods=['POST'])
def create_alert():
    """Create a price alert"""
    data = request.json
    user_id = data.get('user_id', 'demo_user')
    symbol = data.get('symbol')
    target_price = data.get('target_price')
    alert_type = data.get('alert_type')
    
    if not all([symbol, target_price, alert_type]):
        return jsonify({"error": "Symbol, target_price, and alert_type are required"}), 400
    
    alert_id, message = alert_manager.create_alert(user_id, symbol, target_price, alert_type)
    
    if alert_id:
        return jsonify({"alert_id": alert_id, "message": message})
    else:
        return jsonify({"error": message}), 400

@app.route('/api/alerts/<int:alert_id>', methods=['DELETE'])
def delete_alert(alert_id):
    """Delete an alert"""
    user_id = request.args.get('user_id', 'demo_user')
    success = alert_manager.delete_alert(alert_id, user_id)
    
    if success:
        return jsonify({"message": "Alert deleted successfully"})
    else:
        return jsonify({"error": "Failed to delete alert"}), 500

@app.route('/api/alerts/<int:alert_id>/toggle', methods=['PUT'])
def toggle_alert(alert_id):
    """Toggle alert active status"""
    user_id = request.args.get('user_id', 'demo_user')
    data = request.json
    is_active = data.get('is_active') if data else None
    
    success = alert_manager.toggle_alert(alert_id, user_id, is_active)
    
    if success:
        return jsonify({"message": "Alert status updated"})
    else:
        return jsonify({"error": "Failed to update alert"}), 500

@app.route('/api/alerts/check', methods=['POST'])
def check_alerts():
    """Manually check for triggered alerts"""
    triggered = alert_manager.check_triggered_alerts()
    return jsonify({"triggered_alerts": triggered, "count": len(triggered)})

@app.route('/api/alerts/summary', methods=['GET'])
def get_alert_summary():
    """Get alert summary for user"""
    user_id = request.args.get('user_id', 'demo_user')
    summary = alert_manager.get_alert_summary(user_id)
    return jsonify(summary)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
