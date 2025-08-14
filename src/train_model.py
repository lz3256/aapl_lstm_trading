#!/usr/bin/env python3
"""
Train and save LSTM model for live trading
Run this once to create the model file that will be used for live trading
"""

import sys
import os
from datetime import datetime
import json

# Add parent directory to path to import the original strategy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your original strategy class
# Note: You'll need to copy your aapl_lstm_open_price.py to the src directory first
from aapl_lstm_open_price import AAPLLSTMSimpleStrategy

def train_and_save_model():
    """Train the model and save it for live trading"""
    print("🚀 Training LSTM model and saving for live trading...")
    print("=" * 60)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Initialize strategy with correct path to CSV
    strategy = AAPLLSTMSimpleStrategy(
        sequence_length=10,
        csv_file="data/aapl_alpaca_30min_20250722.csv"  # Updated path
    )
    
    # Run the strategy (this trains the model)
    predicted_movement, confidence, metrics = strategy.run_strategy()
    
    if strategy.model is not None:
        # Save the trained model
        model_path = 'models/aapl_lstm_model.h5'
        strategy.model.save(model_path)
        print(f"\n✅ Model saved to {model_path}")
        print("📊 You can now use this model for live trading!")
        
        # Also save the model configuration
        config = {
            'sequence_length': strategy.sequence_length,
            'trained_on': strategy.csv_file,
            'model_path': model_path,
            'training_date': str(datetime.now()),
            'last_prediction': {
                'movement': int(predicted_movement) if predicted_movement else None,
                'confidence': float(confidence) if confidence else None
            },
            'backtest_results': {
                'accuracy': metrics['backtest_results'].get('accuracy', 0),
                'total_return': metrics['backtest_results'].get('total_return', 0),
                'win_rate': metrics['backtest_results'].get('win_rate', 0)
            } if 'backtest_results' in metrics else {}
        }
        
        config_path = 'models/model_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"📄 Configuration saved to {config_path}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 MODEL TRAINING COMPLETE")
        print("=" * 60)
        print(f"✅ Model: {model_path}")
        print(f"✅ Config: {config_path}")
        print(f"📈 Last prediction: {config['last_prediction']}")
        if config['backtest_results']:
            print(f"📊 Backtest accuracy: {config['backtest_results']['accuracy']:.1%}")
            print(f"💰 Backtest return: {config['backtest_results']['total_return']:.2%}")
        print("\n🚀 Ready for live trading! Run: python src/live_trading.py")
        
        return True
    else:
        print("❌ Failed to train model")
        return False

def main():
    """Main function"""
    train_and_save_model()

if __name__ == "__main__":
    main()
