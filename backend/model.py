import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os
import psycopg2
from datetime import datetime, timedelta

class StockPredictor:
    def __init__(self):
        self.model = None
        self.model_type = "RandomForest"
        self.db_url = os.getenv('DATABASE_URL', 'postgresql://postgres:password@localhost:5432/stockdb')
        self.features = ['Open', 'High', 'Low', 'Volume', 'MA_5', 'MA_20', 'RSI', 'Price_Change', 'Volume_Change']
    
    def get_connection(self):
        return psycopg2.connect(self.db_url)
    
    def create_features(self, data):
        """Create technical indicators and features"""
        df = data.copy()
        
        # Moving averages
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        
        # RSI calculation
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Price changes
        df['Price_Change'] = df['Close'].pct_change()
        df['Volume_Change'] = df['Volume'].pct_change()
        
        # Target variable (next day's closing price)
        df['Target'] = df['Close'].shift(-1)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        return df
    
    def prepare_data(self, symbol):
        """Prepare training data from database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT date, open_price, high_price, low_price, close_price, volume
                FROM stocks
                WHERE symbol = %s
                ORDER BY date
            """, (symbol,))
            
            data = cursor.fetchall()
            columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            df = pd.DataFrame(data, columns=columns)
            df.set_index('Date', inplace=True)
            
            if len(df) < 30:  # Need minimum data for features
                return None, None
            
            # Create features
            df_features = self.create_features(df)
            
            # Prepare X and y
            X = df_features[self.features]
            y = df_features['Target']
            
            return X, y
        
        except Exception as e:
            print(f"Error preparing data: {e}")
            return None, None
        
        finally:
            cursor.close()
            conn.close()
    
    def train_model(self, symbol):
        """Train the prediction model"""
        X, y = self.prepare_data(symbol)
        
        if X is None or len(X) < 30:
            return False, "Insufficient data for training"
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize model
        if self.model_type == "RandomForest":
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            self.model = LinearRegression()
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Save model
        model_path = f"models/{symbol}_{self.model_type}_model.pkl"
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.model, model_path)
        
        return True, f"Model trained successfully. MSE: {mse:.2f}, MAE: {mae:.2f}"
    
    def load_model(self, symbol):
        """Load a trained model"""
        model_path = f"models/{symbol}_{self.model_type}_model.pkl"
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            return True
        return False
    
    def predict_price(self, symbol, days_ahead=1):
        """Predict stock price"""
        if self.model is None and not self.load_model(symbol):
            # Train model if not exists
            success, message = self.train_model(symbol)
            if not success:
                return None, message
        
        # Get recent data
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT date, open_price, high_price, low_price, close_price, volume
                FROM stocks
                WHERE symbol = %s
                ORDER BY date DESC
                LIMIT 30
            """, (symbol,))
            
            data = cursor.fetchall()
            columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            df = pd.DataFrame(data, columns=columns)
            df = df.sort_values('Date')
            df.set_index('Date', inplace=True)
            
            # Create features
            df_features = self.create_features(df)
            
            if len(df_features) == 0:
                return None, "Insufficient data for prediction"
            
            # Get latest features
            latest_features = df_features[self.features].iloc[-1:].values
            
            # Make prediction
            prediction = self.model.predict(latest_features)[0]
            
            # Save prediction to database
            self.save_prediction(symbol, prediction)
            
            return prediction, "Prediction successful"
        
        except Exception as e:
            return None, f"Error making prediction: {e}"
        
        finally:
            cursor.close()
            conn.close()
    
    def save_prediction(self, symbol, predicted_price):
        """Save prediction to database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO predictions (symbol, prediction_date, predicted_price, model_version)
                VALUES (%s, %s, %s, %s)
            """, (
                symbol,
                (datetime.now() + timedelta(days=1)).date(),
                predicted_price,
                self.model_type
            ))
            
            conn.commit()
        
        except Exception as e:
            print(f"Error saving prediction: {e}")
            conn.rollback()
        
        finally:
            cursor.close()
            conn.close()
    
    def get_prediction_accuracy(self, symbol, days=30):
        """Calculate prediction accuracy"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT p.predicted_price, s.close_price
                FROM predictions p
                JOIN stocks s ON p.symbol = s.symbol AND p.prediction_date = s.date
                WHERE p.symbol = %s AND p.prediction_date >= CURRENT_DATE - INTERVAL '%s days'
            """, (symbol, days))
            
            data = cursor.fetchall()
            if not data:
                return None
            
            predictions = [row[0] for row in data]
            actuals = [row[1] for row in data]
            
            mae = mean_absolute_error(actuals, predictions)
            mape = np.mean(np.abs((np.array(actuals) - np.array(predictions)) / np.array(actuals))) * 100
            
            return {"mae": mae, "mape": mape, "samples": len(data)}
        
        except Exception as e:
            print(f"Error calculating accuracy: {e}")
            return None
        
        finally:
            cursor.close()
            conn.close()
