import psycopg2
from datetime import datetime
import os
from data_fetcher import DataFetcher

class AlertManager:
    def __init__(self):
        self.db_url = os.getenv('DATABASE_URL', 'postgresql://postgres:password@localhost:5432/stockdb')
        self.data_fetcher = DataFetcher()
    
    def get_connection(self):
        return psycopg2.connect(self.db_url)
    
    def create_alert(self, user_id, symbol, target_price, alert_type):
        """Create a price alert"""
        if alert_type not in ['above', 'below']:
            return None, "Invalid alert type. Must be 'above' or 'below'"
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO price_alerts (user_id, symbol, target_price, alert_type)
                VALUES (%s, %s, %s, %s)
                RETURNING id
            """, (user_id, symbol.upper(), target_price, alert_type))
            
            alert_id = cursor.fetchone()[0]
            conn.commit()
            return alert_id, "Alert created successfully"
        
        except Exception as e:
            print(f"Error creating alert: {e}")
            conn.rollback()
            return None, f"Error creating alert: {e}"
        
        finally:
            cursor.close()
            conn.close()
    
    def get_user_alerts(self, user_id, active_only=True):
        """Get all alerts for a user"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            where_clause = "WHERE user_id = %s"
            params = [user_id]
            
            if active_only:
                where_clause += " AND is_active = TRUE"
            
            cursor.execute(f"""
                SELECT id, symbol, target_price, alert_type, is_active, triggered_at, created_at
                FROM price_alerts
                {where_clause}
                ORDER BY created_at DESC
            """, params)
            
            alerts = cursor.fetchall()
            result = []
            
            for alert in alerts:
                # Get current price for comparison
                current_price = self.get_current_price(alert[1])
                
                alert_data = {
                    "id": alert[0],
                    "symbol": alert[1],
                    "target_price": float(alert[2]),
                    "alert_type": alert[3],
                    "is_active": alert[4],
                    "triggered_at": alert[5].isoformat() if alert[5] else None,
                    "created_at": alert[6].isoformat(),
                    "current_price": current_price,
                    "distance_to_target": self.calculate_distance_to_target(
                        current_price, float(alert[2]), alert[3]
                    ) if current_price else None
                }
                
                result.append(alert_data)
            
            return result
        
        except Exception as e:
            print(f"Error getting alerts: {e}")
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
    
    def calculate_distance_to_target(self, current_price, target_price, alert_type):
        """Calculate distance to target price"""
        if current_price is None:
            return None
        
        if alert_type == "above":
            distance = target_price - current_price
            percentage = (distance / current_price) * 100
        else:  # below
            distance = current_price - target_price
            percentage = (distance / current_price) * 100
        
        return {
            "price_distance": distance,
            "percentage_distance": percentage
        }
    
    def check_triggered_alerts(self):
        """Check for triggered alerts and mark them"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Get all active alerts
            cursor.execute("""
                SELECT a.id, a.symbol, a.target_price, a.alert_type, a.user_id,
                       s.close_price
                FROM price_alerts a
                JOIN (
                    SELECT symbol, close_price,
                           ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date DESC) as rn
                    FROM stocks
                ) s ON a.symbol = s.symbol AND s.rn = 1
                WHERE a.is_active = TRUE
            """)
            
            alerts = cursor.fetchall()
            triggered_alerts = []
            
            for alert in alerts:
                alert_id, symbol, target_price, alert_type, user_id, current_price = alert
                
                is_triggered = False
                if alert_type == "above" and current_price >= target_price:
                    is_triggered = True
                elif alert_type == "below" and current_price <= target_price:
                    is_triggered = True
                
                if is_triggered:
                    # Mark alert as triggered
                    cursor.execute("""
                        UPDATE price_alerts
                        SET is_active = FALSE, triggered_at = %s
                        WHERE id = %s
                    """, (datetime.now(), alert_id))
                    
                    triggered_alerts.append({
                        "id": alert_id,
                        "user_id": user_id,
                        "symbol": symbol,
                        "target_price": float(target_price),
                        "current_price": float(current_price),
                        "alert_type": alert_type
                    })
            
            conn.commit()
            return triggered_alerts
        
        except Exception as e:
            print(f"Error checking triggered alerts: {e}")
            conn.rollback()
            return []
        
        finally:
            cursor.close()
            conn.close()
    
    def delete_alert(self, alert_id, user_id):
        """Delete an alert"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                DELETE FROM price_alerts
                WHERE id = %s AND user_id = %s
            """, (alert_id, user_id))
            
            conn.commit()
            return cursor.rowcount > 0
        
        except Exception as e:
            print(f"Error deleting alert: {e}")
            conn.rollback()
            return False
        
        finally:
            cursor.close()
            conn.close()
    
    def toggle_alert(self, alert_id, user_id, is_active=None):
        """Toggle alert active status"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            if is_active is None:
                # Toggle current status
                cursor.execute("""
                    UPDATE price_alerts
                    SET is_active = NOT is_active
                    WHERE id = %s AND user_id = %s
                """, (alert_id, user_id))
            else:
                # Set specific status
                cursor.execute("""
                    UPDATE price_alerts
                    SET is_active = %s
                    WHERE id = %s AND user_id = %s
                """, (is_active, alert_id, user_id))
            
            conn.commit()
            return cursor.rowcount > 0
        
        except Exception as e:
            print(f"Error toggling alert: {e}")
            conn.rollback()
            return False
        
        finally:
            cursor.close()
            conn.close()
    
    def get_alert_summary(self, user_id):
        """Get alert summary for user"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_alerts,
                    COUNT(CASE WHEN is_active = TRUE THEN 1 END) as active_alerts,
                    COUNT(CASE WHEN triggered_at IS NOT NULL THEN 1 END) as triggered_alerts
                FROM price_alerts
                WHERE user_id = %s
            """, (user_id,))
            
            result = cursor.fetchone()
            return {
                "total_alerts": result[0],
                "active_alerts": result[1],
                "triggered_alerts": result[2]
            }
        
        except Exception as e:
            print(f"Error getting alert summary: {e}")
            return {"total_alerts": 0, "active_alerts": 0, "triggered_alerts": 0}
        
        finally:
            cursor.close()
            conn.close()
