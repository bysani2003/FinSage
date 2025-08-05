import requests
from textblob import TextBlob
import psycopg2
from datetime import datetime
import os

class SentimentAnalyzer:
    def __init__(self):
        self.db_url = os.getenv('DATABASE_URL', 'postgresql://postgres:password@localhost:5432/stockdb')
    
    def get_connection(self):
        return psycopg2.connect(self.db_url)
    
    def analyze_text_sentiment(self, text):
        """Analyze sentiment of given text using TextBlob"""
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity  # Returns value between -1 and 1
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            return 0.0
    
    def get_news_sentiment(self, symbol):
        """Fetch news and analyze sentiment for a stock symbol"""
        # This is a simplified version - in production, you'd use news APIs
        # like NewsAPI, Alpha Vantage, or similar services
        
        # Mock news data for demonstration
        mock_news = [
            f"{symbol} shows strong quarterly earnings",
            f"Market volatility affects {symbol} performance",
            f"Analysts upgrade {symbol} target price",
            f"Industry trends favor {symbol} growth prospects"
        ]
        
        sentiments = []
        for news in mock_news:
            sentiment = self.analyze_text_sentiment(news)
            sentiments.append(sentiment)
        
        # Calculate average sentiment
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0
        return avg_sentiment, len(mock_news)
    
    def save_sentiment_data(self, symbol, sentiment_score, news_count):
        """Save sentiment data to database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO sentiment_data (symbol, date, sentiment_score, news_count)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (symbol, date) DO UPDATE SET
                sentiment_score = EXCLUDED.sentiment_score,
                news_count = EXCLUDED.news_count
            """, (
                symbol,
                datetime.now().date(),
                sentiment_score,
                news_count
            ))
            
            conn.commit()
            print(f"Saved sentiment data for {symbol}: {sentiment_score}")
        
        except Exception as e:
            print(f"Error saving sentiment data: {e}")
            conn.rollback()
        
        finally:
            cursor.close()
            conn.close()
    
    def get_sentiment_data(self, symbol, days=30):
        """Retrieve sentiment data from database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT date, sentiment_score, news_count
                FROM sentiment_data
                WHERE symbol = %s AND date >= CURRENT_DATE - INTERVAL '%s days'
                ORDER BY date
            """, (symbol, days))
            
            data = cursor.fetchall()
            return data
        
        except Exception as e:
            print(f"Error retrieving sentiment data: {e}")
            return []
        
        finally:
            cursor.close()
            conn.close()
    
    def update_sentiment_for_symbols(self, symbols):
        """Update sentiment data for multiple symbols"""
        for symbol in symbols:
            sentiment_score, news_count = self.get_news_sentiment(symbol)
            self.save_sentiment_data(symbol, sentiment_score, news_count)
