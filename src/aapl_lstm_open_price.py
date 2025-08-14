import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, LayerNormalization
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AAPLLSTMSimpleStrategy:
    def __init__(self, sequence_length=10, csv_file="aapl_alpaca_30min_20250722.csv"):
        self.sequence_length = sequence_length
        self.csv_file = csv_file
        self.model = None
        self.data = None
        
    def load_data_from_csv(self):
        """Load AAPL data from the specific CSV file"""
        try:
            print(f"📂 Loading data from: {self.csv_file}")
            
            # Load CSV with proper column names
            data = pd.read_csv(self.csv_file)
            
            # Convert timestamp to datetime and set as index
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data = data.set_index('timestamp')
            
            # Rename columns to standard OHLCV format
            data = data.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            # Sort by timestamp to ensure chronological order
            data = data.sort_index()
            
            print(f"✅ Loaded {len(data)} rows")
            print(f"📊 Data range: {data.index[0]} to {data.index[-1]}")
            print(f"📈 Open price range: ${data['Open'].min():.2f} - ${data['Open'].max():.2f}")
            
            # Clean data - remove any NaN values
            data = data.dropna()
            print(f"📊 After cleaning: {len(data)} rows")
            
            return data
            
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            return None
    
    def split_train_test_data(self, data, recent_months=3):
        """Split data using only recent months"""
        # Sort by date to ensure chronological order
        data = data.sort_index()
        
        # Use only the most recent months of data
        cutoff_date = data.index[-1] - timedelta(days=recent_months * 30)
        recent_data = data[data.index >= cutoff_date]
        
        print(f"📅 Using recent {recent_months} months:")
        print(f"📊 Recent subset: {len(recent_data)} bars ({recent_data.index[0]} to {recent_data.index[-1]})")
        
        # Calculate split point (80/20)
        split_point = int(len(recent_data) * 0.8)
        
        train_data = recent_data.iloc[:split_point]
        test_data = recent_data.iloc[split_point:]
        
        print(f"\n📅 Data Split:")
        print(f"📚 Training: {train_data.index[0]} to {train_data.index[-1]} ({len(train_data)} bars)")
        print(f"🧪 Testing:  {test_data.index[0]} to {test_data.index[-1]} ({len(test_data)} bars)")
        
        return train_data, test_data
    
    def create_returns_only(self, data):
        """Create ONLY open price returns - no other features"""
        returns = data['Open'].pct_change().dropna()
        
        print(f"📊 Created ONLY open price returns (no other features)")
        print(f"📈 Return statistics:")
        print(f"   - Count: {len(returns)}")
        print(f"   - Mean: {returns.mean():.6f}")
        print(f"   - Std:  {returns.std():.6f}")
        print(f"   - Min:  {returns.min():.4f}")
        print(f"   - Max:  {returns.max():.4f}")
        
        return returns
    
    def create_future_returns(self, data):
        """Create future returns for prediction"""
        future_returns = data['Open'].pct_change().shift(-1).dropna()
        print(f"🎯 Created {len(future_returns)} future return targets")
        return future_returns
    
    def create_labels(self, returns, threshold=0.0002):
        """Create binary labels: Rise vs Fall"""
        labels = []
        for ret in returns:
            if ret > threshold:
                labels.append(1)  # Rise
            else:
                labels.append(-1)  # Fall
        
        labels_series = pd.Series(labels, index=returns.index)
        rise_pct = (labels_series == 1).mean() * 100
        print(f"📊 Labels: {rise_pct:.1f}% Rise, {100-rise_pct:.1f}% Fall")
        print(f"🎚️ Threshold: {threshold*100:.3f}% (moves > {threshold*100:.3f}% = Rise)")
        
        # Adjust if too imbalanced
        if rise_pct < 25:
            print(f"⚠️  Imbalanced. Using 30th percentile for balance...")
            threshold_30pct = returns.quantile(0.7)
            balanced_labels = []
            for ret in returns:
                if ret > threshold_30pct:
                    balanced_labels.append(1)
                else:
                    balanced_labels.append(-1)
            
            labels_series = pd.Series(balanced_labels, index=returns.index)
            balanced_rise_pct = (labels_series == 1).mean() * 100
            print(f"🔄 Adjusted: {balanced_rise_pct:.1f}% Rise, {100-balanced_rise_pct:.1f}% Fall")
        
        return labels_series
    
    def prepare_return_sequences(self, returns, labels):
        """Prepare sequences using ONLY past 20 returns"""
        sequences = []
        targets = []
        
        # Align returns and labels
        common_index = returns.index.intersection(labels.index)
        returns_aligned = returns.loc[common_index]
        labels_aligned = labels.loc[common_index]
        
        print(f"📊 Aligned data: {len(common_index)} time points")
        print(f"📊 Using ONLY past {self.sequence_length} returns (no other features)")
        
        # Create sequences: past 10 returns → predict next movement
        for i in range(self.sequence_length, len(returns_aligned)):
            # Get sequence of past 10 returns only
            seq = returns_aligned.iloc[i-self.sequence_length:i].values.reshape(-1, 1)  # Shape: (10, 1)
            target = labels_aligned.iloc[i]
            
            sequences.append(seq)
            targets.append(target)
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        print(f"🔢 Created {len(sequences)} sequences")
        print(f"📊 Input shape: {sequences.shape} (samples, {self.sequence_length} timesteps, 1 feature)")
        print(f"📊 Each sequence: {self.sequence_length} past returns → predict next movement")
        print(f"📊 Target distribution: Rise={np.sum(targets==1)}, Fall={np.sum(targets==-1)}")
        
        return sequences, targets
    
    def build_model(self, input_shape):
        """Build LSTM model for return sequence prediction"""
        model = Sequential([
            # First LSTM layer
            LSTM(64, return_sequences=True, input_shape=input_shape,
                 kernel_regularizer=l1_l2(0.01, 0.01)),
            LayerNormalization(),
            Dropout(0.3),
            
            # Second LSTM layer
            LSTM(32, return_sequences=True,
                 kernel_regularizer=l1_l2(0.01, 0.01)),
            LayerNormalization(),
            Dropout(0.3),
            
            # Final LSTM layer
            LSTM(16, return_sequences=False,
                 kernel_regularizer=l1_l2(0.01, 0.01)),
            LayerNormalization(),
            Dropout(0.4),
            
            # Dense layers
            Dense(32, activation='relu', kernel_regularizer=l1_l2(0.01, 0.01)),
            LayerNormalization(),
            Dropout(0.4),
            
            Dense(16, activation='relu'),
            Dropout(0.3),
            
            Dense(8, activation='relu'),
            
            # Output layer
            Dense(2, activation='softmax')
        ])
        
        # Learning rate scheduling
        lr_schedule = ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=50,
            decay_rate=0.9,
            staircase=True
        )
        
        optimizer = AdamW(learning_rate=lr_schedule, weight_decay=0.01)
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, X, y, validation_split=0.2, epochs=40):
        """Train the LSTM model"""
        print(f"\n🚀 Training LSTM on return sequences...")
        print(f"📊 Training shape: {X.shape}")
        
        # Convert labels: -1 -> 0 (Fall), 1 -> 1 (Rise)
        y_categorical = (y + 1) // 2
        unique_labels, counts = np.unique(y_categorical, return_counts=True)
        print(f"📊 Label distribution: {dict(zip(unique_labels, counts))}")
        
        # Class weights for balance
        classes = np.unique(y_categorical)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_categorical)
        class_weight_dict = dict(zip(classes, class_weights))
        print(f"⚖️  Class weights: {class_weight_dict}")
        
        # Build model
        print(f"🏗️  Building LSTM for return sequences...")
        self.model = self.build_model((X.shape[1], X.shape[2]))
        
        print("🏗️  Model architecture:")
        self.model.summary()
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=6, min_lr=1e-7, verbose=1)
        ]
        
        print(f"🎯 Training for {epochs} epochs...")
        
        # Train
        history = self.model.fit(
            X, y_categorical,
            epochs=epochs,
            batch_size=32,
            validation_split=validation_split,
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=1
        )
        
        print(f"✅ Training completed!")
        return history
    
    def predict_movement(self, returns):
        """Predict next movement using past 10 returns"""
        if self.model is None:
            raise ValueError("Model not trained!")
        
        # Get last 10 returns
        last_sequence = returns.iloc[-10:].values.reshape(1, 10, 1)
        
        # Predict
        prediction = self.model.predict(last_sequence, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        
        # Convert back to -1 (Fall), 1 (Rise)
        predicted_movement = 1 if predicted_class == 1 else -1
        
        return predicted_movement, confidence, prediction[0]
    
    def backtest(self, test_returns, test_labels, test_data):
        """Backtest using only return sequences"""
        print(f"\n🧪 Backtesting with return sequences only...")
        print(f"📊 Using ONLY past 20 returns for each prediction")
        
        predictions = []
        confidences = []
        actual_labels = []
        actual_returns_for_pnl = []
        
        # Align returns and labels
        common_index = test_returns.index.intersection(test_labels.index)
        test_returns_aligned = test_returns.loc[common_index]
        test_labels_aligned = test_labels.loc[common_index]
        
        print(f"📊 Test data: {len(test_returns_aligned)} time points")
        
        # Make predictions (need 20 past returns)
        total_predictions = len(test_returns_aligned) - 20
        print(f"📈 Making {total_predictions} predictions...")
        
        for i in range(20, len(test_returns_aligned)):
            if i % 20 == 0:
                print(f"📊 Progress: {i-20}/{total_predictions}")
                
            try:
                # Get past 20 returns
                past_returns = test_returns_aligned.iloc[i-20:i]
                timestamp = test_returns_aligned.index[i]
                actual_label = test_labels_aligned.iloc[i]
                actual_return = test_returns_aligned.iloc[i]  # Use actual return for P&L
                
                # Predict
                pred, conf, _ = self.predict_movement(past_returns)
                
                predictions.append(pred)
                confidences.append(conf)
                actual_labels.append(actual_label)
                actual_returns_for_pnl.append(actual_return)
                
            except Exception as e:
                print(f"❌ Error at {i}: {e}")
                continue
        
        print(f"✅ Completed {len(predictions)} predictions")
        
        if len(predictions) == 0:
            return {}
        
        predictions = np.array(predictions)
        actual_labels = np.array(actual_labels)
        actual_returns_for_pnl = np.array(actual_returns_for_pnl)
        confidences = np.array(confidences)
        
        # Performance metrics
        accuracy = accuracy_score(actual_labels, predictions)
        
        print(f"\n📊 Backtest Results (Returns Only):")
        print(f"🎯 Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"📈 Total predictions: {len(predictions)}")
        print(f"📊 Breakdown: Rise={np.sum(predictions==1)}, Fall={np.sum(predictions==-1)}")
        print(f"🎚️ Avg confidence: {np.mean(confidences):.3f}")
        
        # Confusion matrix
        cm = confusion_matrix(actual_labels, predictions, labels=[-1, 1])
        print(f"\n📋 Confusion Matrix:")
        print(f"           Predicted")
        print(f"Actual   Fall  Rise")
        print(f"Fall      {cm[0,0]:3d}   {cm[0,1]:3d}")
        print(f"Rise      {cm[1,0]:3d}   {cm[1,1]:3d}")
        
        # Trading performance using actual returns
        pnl_trades = []
        for i, pred in enumerate(predictions):
            actual_return = actual_returns_for_pnl[i]
            if pred == 1:  # Predicted rise -> Long
                pnl = actual_return
            else:  # Predicted fall -> Short
                pnl = -actual_return
            pnl_trades.append(pnl)
        
        if pnl_trades:
            total_return = sum(pnl_trades)
            win_rate = sum(1 for pnl in pnl_trades if pnl > 0) / len(pnl_trades)
            avg_return = np.mean(pnl_trades)
            
            print(f"\n💰 Trading Performance (Returns Only):")
            print(f"📈 Total Return: {total_return:.4f} ({total_return*100:.2f}%)")
            print(f"🎯 Win Rate: {win_rate:.3f} ({win_rate*100:.1f}%)")
            print(f"📊 Avg Trade: {avg_return:.6f} ({avg_return*100:.4f}%)")
            print(f"🏆 Best Trade: {max(pnl_trades):.4f} ({max(pnl_trades)*100:.2f}%)")
            print(f"💥 Worst Trade: {min(pnl_trades):.4f} ({min(pnl_trades)*100:.2f}%)")
            
            # Check prediction diversity
            if np.sum(predictions==-1) == len(predictions):
                print(f"⚠️  Model predicted ALL SHORT positions")
            elif np.sum(predictions==1) == len(predictions):
                print(f"⚠️  Model predicted ALL LONG positions")
            else:
                print(f"✅ Diverse predictions: {np.sum(predictions==1)} Long, {np.sum(predictions==-1)} Short")
        
        return {
            'accuracy': accuracy,
            'total_return': total_return if pnl_trades else 0,
            'win_rate': win_rate if pnl_trades else 0,
            'predictions': predictions,
            'actual': actual_labels,
            'confidences': confidences,
            'returns': pnl_trades if pnl_trades else []
        }
    
    def run_strategy(self):
        """Run the complete strategy using only return sequences"""
        print("=" * 60)
        print("🍎 AAPL LSTM - Returns Only Strategy")
        print("=" * 60)
        print(f"📊 Input: Past {self.sequence_length} open price returns ONLY")
        print(f"🎯 Output: Next 30-min open price movement")
        print(f"🚫 NO complex features - just pure return sequences")
        print()
        
        # 1. Load data
        self.data = self.load_data_from_csv()
        if self.data is None:
            return None, 0, {}
        
        # 2. Split data
        train_data, test_data = self.split_train_test_data(self.data, recent_months=3)
        
        # 3. Create ONLY returns (no other features)
        train_returns = self.create_returns_only(train_data)
        test_returns = self.create_returns_only(test_data)
        
        # 4. Create labels
        train_future_returns = self.create_future_returns(train_data)
        test_future_returns = self.create_future_returns(test_data)
        train_labels = self.create_labels(train_future_returns, threshold=0.0003)
        test_labels = self.create_labels(test_future_returns, threshold=0.0003)
        
        # 5. Prepare sequences (10 returns → 1 prediction)
        print(f"\n📊 Preparing return sequences...")
        X_train, y_train = self.prepare_return_sequences(train_returns, train_labels)
        
        if len(X_train) < 50:
            print("❌ Insufficient training data")
            return None, 0, {}
        
        # 6. Train model
        print(f"\n🎯 Training phase...")
        history = self.train_model(X_train, y_train, epochs=30)
        
        # 7. Backtest
        print(f"\n🧪 Backtest phase...")
        backtest_results = self.backtest(test_returns, test_labels, test_data)
        
        # 8. Current prediction
        print(f"\n🔮 Current Prediction")
        print("-" * 40)
        
        current_returns = self.create_returns_only(self.data)
        if len(current_returns) >= 10:
            predicted_movement, confidence, prob_dist = self.predict_movement(current_returns)
            
            movement_names = {-1: "📉 FALL", 1: "📈 RISE"}
            action_names = {-1: "SHORT", 1: "LONG"}
            
            print(f"Prediction: {movement_names[predicted_movement]}")
            print(f"Confidence: {confidence:.1%}")
            print(f"Probabilities: Fall={prob_dist[0]:.3f}, Rise={prob_dist[1]:.3f}")
            print(f"Action: {action_names[predicted_movement]} AAPL")
            print(f"Current open: ${self.data['Open'].iloc[-1]:.2f}")
            print(f"Last timestamp: {self.data.index[-1]}")
            print(f"📊 Simple Strategy: Past 10 returns → next movement")
            
            return predicted_movement, confidence, {
                'training_history': history,
                'backtest_results': backtest_results
            }
        else:
            print("❌ Insufficient data for prediction")
            return None, 0, {'backtest_results': backtest_results}

def main():
    """Main function"""
    print("🚀 AAPL LSTM - Pure Returns Strategy")
    print("=" * 50)
    
    # Initialize strategy
    strategy = AAPLLSTMSimpleStrategy(
        sequence_length=10,
        csv_file="aapl_alpaca_30min_20250722.csv"
    )
    
    predicted_movement, confidence, metrics = strategy.run_strategy()
    
    if predicted_movement is not None:
        print("\n" + "=" * 60)
        print("🎯 FINAL RECOMMENDATION")
        print("=" * 60)
        
        if predicted_movement == 1:
            print("📈 BUY: Long AAPL")
            print("💰 Strategy: Next 30-min open expected to rise")
        else:
            print("📉 SELL: Short AAPL")
            print("💰 Strategy: Next 30-min open expected to fall")
        
        print(f"🎯 Confidence: {confidence:.1%}")
        print("📊 Pure approach: Only past 10 returns used")
        print("🚫 No complex features - just return patterns")

if __name__ == "__main__":
    main()


