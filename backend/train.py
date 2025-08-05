"""
Training script for stock prediction models
Run this script to train models for popular stocks
"""

from data_fetcher import DataFetcher
from model import StockPredictor
from sentiment import SentimentAnalyzer
import time

def main():
    # Initialize components
    data_fetcher = DataFetcher()
    predictor = StockPredictor()
    sentiment_analyzer = SentimentAnalyzer()
    
    # Popular stock symbols
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']
    
    print("Starting data update and model training...")
    
    # Update stock data
    print("Fetching latest stock data...")
    data_fetcher.update_stock_data(symbols)
    
    # Update sentiment data
    print("Updating sentiment data...")
    sentiment_analyzer.update_sentiment_for_symbols(symbols)
    
    # Train models
    print("Training prediction models...")
    for symbol in symbols:
        print(f"Training model for {symbol}...")
        success, message = predictor.train_model(symbol)
        print(f"{symbol}: {message}")
        time.sleep(1)  # Small delay between training
    
    print("Training completed!")

if __name__ == "__main__":
    main()
