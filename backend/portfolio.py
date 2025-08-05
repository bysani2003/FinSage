import psycopg2
import pandas as pd
from datetime import datetime, date
import os
from data_fetcher import DataFetcher

class PortfolioManager:
    def __init__(self):
        self.db_url = os.getenv('DATABASE_URL', 'postgresql://postgres:password@localhost:5432/stockdb')
        self.data_fetcher = DataFetcher()
    
    def get_connection(self):
        return psycopg2.connect(self.db_url)
    
    def create_portfolio(self, user_id, name, description=""):
        """Create a new portfolio"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO portfolios (user_id, name, description)
                VALUES (%s, %s, %s)
                RETURNING id
            """, (user_id, name, description))
            
            portfolio_id = cursor.fetchone()[0]
            conn.commit()
            return portfolio_id
        
        except Exception as e:
            print(f"Error creating portfolio: {e}")
            conn.rollback()
            return None
        
        finally:
            cursor.close()
            conn.close()
    
    def get_portfolios(self, user_id):
        """Get all portfolios for a user"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT id, name, description, created_at
                FROM portfolios
                WHERE user_id = %s
                ORDER BY created_at DESC
            """, (user_id,))
            
            portfolios = cursor.fetchall()
            return [
                {
                    "id": p[0],
                    "name": p[1],
                    "description": p[2],
                    "created_at": p[3].isoformat()
                }
                for p in portfolios
            ]
        
        except Exception as e:
            print(f"Error getting portfolios: {e}")
            return []
        
        finally:
            cursor.close()
            conn.close()
    
    def add_holding(self, portfolio_id, symbol, quantity, purchase_price, purchase_date=None):
        """Add a stock holding to portfolio"""
        if purchase_date is None:
            purchase_date = date.today()
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO portfolio_holdings (portfolio_id, symbol, quantity, purchase_price, purchase_date)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
            """, (portfolio_id, symbol.upper(), quantity, purchase_price, purchase_date))
            
            holding_id = cursor.fetchone()[0]
            conn.commit()
            return holding_id
        
        except Exception as e:
            print(f"Error adding holding: {e}")
            conn.rollback()
            return None
        
        finally:
            cursor.close()
            conn.close()
    
    def get_portfolio_holdings(self, portfolio_id):
        """Get all holdings in a portfolio with current prices"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT h.id, h.symbol, h.quantity, h.purchase_price, h.purchase_date, h.created_at
                FROM portfolio_holdings h
                WHERE h.portfolio_id = %s
                ORDER BY h.created_at DESC
            """, (portfolio_id,))
            
            holdings = cursor.fetchall()
            result = []
            
            for holding in holdings:
                # Get current price
                current_price = self.get_current_price(holding[1])
                
                holding_data = {
                    "id": holding[0],
                    "symbol": holding[1],
                    "quantity": float(holding[2]),
                    "purchase_price": float(holding[3]) if holding[3] else 0,
                    "purchase_date": holding[4].isoformat() if holding[4] else None,
                    "current_price": current_price,
                    "market_value": float(holding[2]) * current_price if current_price else 0,
                    "cost_basis": float(holding[2]) * float(holding[3]) if holding[3] else 0,
                    "gain_loss": 0,
                    "gain_loss_percent": 0
                }
                
                # Calculate gains/losses
                if holding_data["cost_basis"] > 0 and current_price:
                    holding_data["gain_loss"] = holding_data["market_value"] - holding_data["cost_basis"]
                    holding_data["gain_loss_percent"] = (holding_data["gain_loss"] / holding_data["cost_basis"]) * 100
                
                result.append(holding_data)
            
            return result
        
        except Exception as e:
            print(f"Error getting holdings: {e}")
            return []
        
        finally:
            cursor.close()
            conn.close()
    
    def get_current_price(self, symbol):
        """Get current stock price"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT close_price
                FROM stocks
                WHERE symbol = %s
                ORDER BY date DESC
                LIMIT 1
            """, (symbol,))
            
            result = cursor.fetchone()
            return float(result[0]) if result else None
        
        except Exception as e:
            print(f"Error getting current price: {e}")
            return None
        
        finally:
            cursor.close()
            conn.close()
    
    def get_portfolio_summary(self, portfolio_id):
        """Get portfolio performance summary"""
        holdings = self.get_portfolio_holdings(portfolio_id)
        
        if not holdings:
            return {
                "total_value": 0,
                "total_cost": 0,
                "total_gain_loss": 0,
                "total_gain_loss_percent": 0,
                "holdings_count": 0
            }
        
        total_value = sum(h["market_value"] for h in holdings)
        total_cost = sum(h["cost_basis"] for h in holdings)
        total_gain_loss = total_value - total_cost
        total_gain_loss_percent = (total_gain_loss / total_cost * 100) if total_cost > 0 else 0
        
        return {
            "total_value": total_value,
            "total_cost": total_cost,
            "total_gain_loss": total_gain_loss,
            "total_gain_loss_percent": total_gain_loss_percent,
            "holdings_count": len(holdings),
            "holdings": holdings
        }
    
    def remove_holding(self, holding_id):
        """Remove a holding from portfolio"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                DELETE FROM portfolio_holdings
                WHERE id = %s
            """, (holding_id,))
            
            conn.commit()
            return cursor.rowcount > 0
        
        except Exception as e:
            print(f"Error removing holding: {e}")
            conn.rollback()
            return False
        
        finally:
            cursor.close()
            conn.close()
    
    def update_holding(self, holding_id, quantity=None, purchase_price=None):
        """Update holding quantity or purchase price"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            update_fields = []
            params = []
            
            if quantity is not None:
                update_fields.append("quantity = %s")
                params.append(quantity)
            
            if purchase_price is not None:
                update_fields.append("purchase_price = %s")
                params.append(purchase_price)
            
            if not update_fields:
                return False
            
            params.append(holding_id)
            
            cursor.execute(f"""
                UPDATE portfolio_holdings
                SET {', '.join(update_fields)}
                WHERE id = %s
            """, params)
            
            conn.commit()
            return cursor.rowcount > 0
        
        except Exception as e:
            print(f"Error updating holding: {e}")
            conn.rollback()
            return False
        
        finally:
            cursor.close()
            conn.close()
    
    def get_portfolio_performance_history(self, portfolio_id, days=30):
        """Get historical performance of portfolio"""
        holdings = self.get_portfolio_holdings(portfolio_id)
        
        if not holdings:
            return []
        
        symbols = [h["symbol"] for h in holdings]
        performance_data = []
        
        # Get historical data for each symbol
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            for i in range(days):
                date_offset = days - i
                cursor.execute("""
                    SELECT symbol, close_price, date
                    FROM stocks
                    WHERE symbol = ANY(%s) AND date = (
                        SELECT date FROM stocks 
                        WHERE date <= CURRENT_DATE - INTERVAL '%s days'
                        ORDER BY date DESC LIMIT 1
                    )
                """, (symbols, date_offset))
                
                daily_prices = cursor.fetchall()
                daily_value = 0
                
                for symbol, price, price_date in daily_prices:
                    # Find holding for this symbol
                    holding = next((h for h in holdings if h["symbol"] == symbol), None)
                    if holding:
                        daily_value += holding["quantity"] * float(price)
                
                if daily_value > 0:
                    performance_data.append({
                        "date": price_date.isoformat() if daily_prices else None,
                        "value": daily_value
                    })
            
            return performance_data
        
        except Exception as e:
            print(f"Error getting performance history: {e}")
            return []
        
        finally:
            cursor.close()
            conn.close()
