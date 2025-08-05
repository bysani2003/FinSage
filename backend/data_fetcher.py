import yfinance as yf
import pandas as pd
import psycopg2
from datetime import datetime, timedelta
import os

class DataFetcher:
    def __init__(self):
        self.db_url = os.getenv('DATABASE_URL', 'postgresql://postgres:password@localhost:5432/stockdb')
    
    def get_connection(self):
        return psycopg2.connect(self.db_url)
    
    def fetch_stock_data(self, symbol, period="1y"):
        """Fetch stock data from Yahoo Finance"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            return data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def save_stock_data(self, symbol, data):
        """Save stock data to database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            for date, row in data.iterrows():
                cursor.execute("""
                    INSERT INTO stocks (symbol, date, open_price, high_price, low_price, close_price, volume)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol, date) DO UPDATE SET
                    open_price = EXCLUDED.open_price,
                    high_price = EXCLUDED.high_price,
                    low_price = EXCLUDED.low_price,
                    close_price = EXCLUDED.close_price,
                    volume = EXCLUDED.volume
                """, (
                    symbol,
                    date.date(),
                    float(row['Open']),
                    float(row['High']),
                    float(row['Low']),
                    float(row['Close']),
                    int(row['Volume'])
                ))
            
            conn.commit()
            print(f"Saved {len(data)} records for {symbol}")
        
        except Exception as e:
            print(f"Error saving data: {e}")
            conn.rollback()
        
        finally:
            cursor.close()
            conn.close()
    
    def get_stock_data_from_db(self, symbol, days=365):
        """Retrieve stock data from database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT date, open_price, high_price, low_price, close_price, volume
                FROM stocks
                WHERE symbol = %s AND date >= %s
                ORDER BY date
            """, (symbol, datetime.now() - timedelta(days=days)))
            
            data = cursor.fetchall()
            columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            df = pd.DataFrame(data, columns=columns)
            df.set_index('Date', inplace=True)
            return df
        
        except Exception as e:
            print(f"Error retrieving data: {e}")
            return None
        
        finally:
            cursor.close()
            conn.close()
    
    def update_stock_data(self, symbols):
        """Update stock data for multiple symbols"""
        for symbol in symbols:
            print(f"Updating data for {symbol}")
            data = self.fetch_stock_data(symbol)
            if data is not None:
                self.save_stock_data(symbol, data)
