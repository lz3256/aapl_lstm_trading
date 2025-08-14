import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import alpaca_trade_api as tradeapi
from tensorflow.keras.models import load_model
import json
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

class AAPLLSTMLiveTrading:
    def __init__(self, api_key, api_secret, base_url='https://paper-api.alpaca.markets'):
        """Initialize with Alpaca API credentials"""
        self.api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
        self.symbol = 'AAPL'
        self.sequence_length = 10
        self.model = None
        self.position_size = int(os.getenv('POSITION_SIZE', 100))  # From .env or default 100
        self.min_confidence = float(os.getenv('MIN_CONFIDENCE', 0.6))  # From .env or default 0.6
        self.last_trade_time = None
        self.min_trade_interval = timedelta(minutes=30)  # Trade every 30 minutes
        self.testing_mode = False  # Set to True for after-hours testing
        
    def get_historical_data(self, days=5):
        """Get historical 30-minute bars from Alpaca"""
        print(f"Fetching historical data for {self.symbol}...")
        
        end = datetime.now()
        start = end - timedelta(days=days)
        
        try:
            # Get 30-minute bars using IEX feed (free for paper trading)
            bars = self.api.get_bars(
                self.symbol,
                '30Min',
                start=start.strftime('%Y-%m-%d'),
                end=end.strftime('%Y-%m-%d'),
                limit=100,  # Reduced from 1000
                feed='iex'  # Use IEX feed
            ).df
            
        except Exception as e:
            print(f"Error fetching bars: {e}")
            return None
            
        if bars.empty:
            print("No historical data available")
            return None
            
        print(f"Fetched {len(bars)} bars")
        print(f"Date range: {bars.index[0]} to {bars.index[-1]}")
        
        # Rename columns to match your training data format
        bars = bars.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        
        return bars
    
    def get_latest_price(self):
        """Get the latest price from Alpaca"""
        try:
            # Use IEX feed for latest quote
            quote = self.api.get_latest_quote(self.symbol, feed='iex')
            return {
                'price': quote.ask_price if quote.ask_price > 0 else quote.bid_price,
                'bid': quote.bid_price,
                'ask': quote.ask_price,
                'timestamp': quote.timestamp
            }
        except Exception as e:
            print(f"Error getting latest price: {e}")
            # Fallback to latest trade if quote fails
            try:
                trade = self.api.get_latest_trade(self.symbol, feed='iex')
                return {
                    'price': trade.price,
                    'bid': trade.price,
                    'ask': trade.price,
                    'timestamp': trade.timestamp
                }
            except Exception as e2:
                print(f"Error getting latest trade: {e2}")
                return None
    
    def get_current_position(self):
        """Get current position in AAPL"""
        try:
            position = self.api.get_position(self.symbol)
            return {
                'qty': int(position.qty),
                'side': position.side,
                'avg_entry_price': float(position.avg_entry_price),
                'market_value': float(position.market_value),
                'unrealized_pl': float(position.unrealized_pl)
            }
        except:
            return None
    
    def calculate_returns(self, data):
        """Calculate returns from price data"""
        returns = data['Open'].pct_change().dropna()
        return returns
    
    def predict_next_movement(self, returns):
        """Predict next movement using the trained model"""
        if self.model is None:
            raise ValueError("Model not loaded!")
        
        if len(returns) < self.sequence_length:
            print(f"Need at least {self.sequence_length} returns for prediction")
            return None, None
        
        # Get last 10 returns
        last_sequence = returns.iloc[-self.sequence_length:].values.reshape(1, self.sequence_length, 1)
        
        # Predict
        prediction = self.model.predict(last_sequence, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        
        # Convert to movement direction
        predicted_movement = 1 if predicted_class == 1 else -1
        
        return predicted_movement, confidence
    
    def place_order(self, side, qty=None):
        """Place a market order through Alpaca"""
        if qty is None:
            qty = self.position_size
            
        try:
            order = self.api.submit_order(
                symbol=self.symbol,
                qty=qty,
                side=side,
                type='market',
                time_in_force='day'
            )
            
            print(f"   {side.upper()} order placed:")
            print(f"   Order ID: {order.id}")
            print(f"   Quantity: {qty} shares")
            print(f"   Status: {order.status}")
            
            return order
            
        except Exception as e:
            print(f"Error placing order: {e}")
            return None
    
    def manage_position(self, predicted_movement, confidence):
        """Manage position based on prediction"""
        current_position = self.get_current_position()
        latest_price = self.get_latest_price()
        
        if latest_price is None:
            print("Cannot get latest price")
            return
        
        print(f"\nPosition Management:")
        print(f"Current price: ${latest_price['price']:.2f}")
        print(f"Prediction: {'RISE' if predicted_movement == 1 else 'FALL'} (confidence: {confidence:.1%})")
        
        # Check if we have a position
        if current_position:
            print(f"Current position: {current_position['qty']} shares @ ${current_position['avg_entry_price']:.2f}")
            print(f"Unrealized P&L: ${current_position['unrealized_pl']:.2f}")
            
            # If we're long and predict fall, or short and predict rise, close position
            if (current_position['side'] == 'long' and predicted_movement == -1) or \
               (current_position['side'] == 'short' and predicted_movement == 1):
                print(f"Closing position due to direction change")
                
                # Close current position
                close_side = 'sell' if current_position['side'] == 'long' else 'buy'
                self.place_order(close_side, abs(current_position['qty']))
                
                # Wait a moment for order to process
                time.sleep(2)
                
                # Open new position in predicted direction
                new_side = 'buy' if predicted_movement == 1 else 'sell'
                self.place_order(new_side)
            else:
                print(f"Holding current {current_position['side']} position")
        else:
            # No position, open one based on prediction
            print("No current position")
            
            if confidence > self.min_confidence:  # Only trade if confidence is high enough
                side = 'buy' if predicted_movement == 1 else 'sell'
                print(f"Opening {side} position")
                self.place_order(side)
            else:
                print(f"Confidence too low ({confidence:.1%} < {self.min_confidence:.1%}), skipping trade")
    
    def load_trained_model(self, model_path='models/aapl_lstm_model.h5'):
        """Load the trained LSTM model"""
        try:
            # Check if model file exists
            if not os.path.exists(model_path):
                print(f"Model file not found at: {model_path}")
                print(f"Current directory: {os.getcwd()}")
                print(f"Looking for: {os.path.abspath(model_path)}")
                return False
                
            self.model = load_model(model_path)
            print(f"Model loaded from {model_path}")
            
            # Try to load model configuration
            config_path = model_path.replace('.h5', '_config.json').replace('aapl_lstm_model', 'model')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                print(f"📄 Model config loaded")
                print(f"   - Trained on: {config.get('training_date', 'Unknown')}")
                print(f"   - Sequence length: {config.get('sequence_length', 'Unknown')}")
                if 'backtest_results' in config:
                    print(f"   - Backtest accuracy: {config['backtest_results'].get('accuracy', 0):.1%}")
            
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Make sure you've trained the model first with: python src/train_model.py")
            return False
    
    def run_live_trading(self, check_interval=60):
        """Run the live trading loop"""
        print("=" * 60)
        print("AAPL LSTM Live Trading with Alpaca")
        print("=" * 60)
        print(f"Symbol: {self.symbol}")
        print(f"Strategy: LSTM prediction on 30-min bars")
        print(f"Check interval: {check_interval} seconds")
        print(f"Position size: {self.position_size} shares")
        print(f"Data feed: IEX (free for paper trading)")
        print()
        
        # Note about data feeds
        print("   Note: Using IEX data feed (free)")
        print("   - Suitable for paper trading")
        print("   - May have slight delays vs SIP feed")
        print()
        
        # Main trading loop
        while True:
            try:
                current_time = datetime.now()
                print(f"\n{'='*50}")
                print(f"{current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Check if market is open
                clock = self.api.get_clock()
                if not clock.is_open:
                    print("Market is closed")
                    print(f"Next open: {clock.next_open}")
                    time.sleep(check_interval)
                    continue
                
                # Check if we should trade (every 30 minutes)
                if self.last_trade_time and (current_time - self.last_trade_time) < self.min_trade_interval:
                    remaining = self.min_trade_interval - (current_time - self.last_trade_time)
                    print(f"Next trade check in: {remaining.seconds // 60} minutes")
                    time.sleep(check_interval)
                    continue
                
                # Get historical data - we have 40 bars available
                historical_data = self.get_historical_data(days=5)  # 5 days is enough
                if historical_data is None or len(historical_data) < 15:  # Need at least 15 bars (10 for sequence + buffer)
                    print(f"Insufficient historical data (need at least 15 bars, got {len(historical_data) if historical_data is not None else 0})")
                    time.sleep(check_interval)
                    continue
                
                # Calculate returns
                returns = self.calculate_returns(historical_data)
                
                # Make prediction
                predicted_movement, confidence = self.predict_next_movement(returns)
                
                if predicted_movement is not None:
                    # Manage position based on prediction
                    self.manage_position(predicted_movement, confidence)
                    self.last_trade_time = current_time
                
                # Display account info
                account = self.api.get_account()
                print(f"\n Account Status:")
                print(f"   Buying Power: ${float(account.buying_power):,.2f}")
                print(f"   Portfolio Value: ${float(account.portfolio_value):,.2f}")
                
                print(f"\n Next check in {check_interval} seconds...")
                
            except KeyboardInterrupt:
                print("\n\n Trading stopped by user")
                break
            except Exception as e:
                print(f"\n Error in trading loop: {e}")
                print(" Retrying in 60 seconds...")
                time.sleep(60)
            
            time.sleep(check_interval)

def save_trained_model(strategy_instance, filepath='aapl_lstm_model.h5'):
    """Helper function to save your trained model"""
    if strategy_instance.model:
        strategy_instance.model.save(filepath)
        print(f" Model saved to {filepath}")
    else:
        print(" No model to save")

def main():
    """Main function to run live trading"""
    # Load Alpaca API credentials from environment variables
    API_KEY = os.getenv('APCA_API_KEY_ID')
    API_SECRET = os.getenv('APCA_API_SECRET_KEY')
    BASE_URL = os.getenv('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets')
    
    # Validate credentials
    if not API_KEY or not API_SECRET:
        print(" Error: API credentials not found!")
        print(" Please create a .env file with:")
        print("   APCA_API_KEY_ID=your_api_key")
        print("   APCA_API_SECRET_KEY=your_secret_key")
        print("   APCA_API_BASE_URL=https://paper-api.alpaca.markets")
        return
    
    print(" API credentials loaded from .env file")
    print(f" Using base URL: {BASE_URL}")
    
    # Initialize live trading
    trader = AAPLLSTMLiveTrading(API_KEY, API_SECRET, BASE_URL)
    
    # Load the trained model
    # Note: You need to save your trained model first!
    if not trader.load_trained_model('models/aapl_lstm_model.h5'):
        print("\n  To save your trained model, run:")
        print("python src/train_model.py")
        return
    
    # Run live trading
    # Check interval from .env or default to 60 seconds
    check_interval = int(os.getenv('CHECK_INTERVAL', 60))
    trader.run_live_trading(check_interval=check_interval)

if __name__ == "__main__":
    main()
