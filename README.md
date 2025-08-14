# aapl_lstm_trading
An automated trading system that uses LSTM neural networks to predict Apple (AAPL) stock price movements and execute trades via Alpaca API.

## Strategy Overview
This trading bot uses machine learning to predict short-term price movements:

Timeframe: 30-minute bars
Input: Last 10 open price returns
Output: Binary prediction (RISE or FALL)
Action: Automatically places buy/sell orders based on predictions
Position Size: 100 shares (configurable)

## How It Works

Data Collection: Fetches real-time 30-minute AAPL bars from Alpaca
Feature Engineering: Calculates price returns from open prices
Prediction: LSTM model analyzes the last 10 returns to predict next movement
Execution: Places market orders based on prediction

Predicts RISE → BUY (go long)
Predicts FALL → SELL (go short)


Position Management: Automatically reverses positions when prediction changes

Quick Start
Prerequisites

Python 3.8+
Alpaca Paper Trading Account (Sign up free)
~50MB for historical data and model files

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/lz3256/aapl-lstm-trading.git
cd aapl-lstm-trading
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables:**
```bash
cp .env.example .env
#Edit .env with your Alpaca API credentials
```

4. **Train the model:**
```bash
python train.py
```
It will use the data until 2025/7/22 to train the needed LSTM model, if you wish to use new data just replace it with new data in the same route

5. **Run Connection Test:**
```bash
python test_alpaca_connection.py
```

6. **Run live trading:**
```bash
python train.py
```

7. **Check Current Positions:**
```bash
python check_alpaca_positions.py
```

## Configuration
Edit `.env` file to configure:

```
# Alpaca API (required)
APCA_API_KEY_ID=your_api_key
APCA_API_SECRET_KEY=your_secret_key
APCA_API_BASE_URL=https://paper-api.alpaca.markets
```

## Performance
<img width="590" height="403" alt="image" src="https://github.com/user-attachments/assets/ca68a8fa-fb3b-4074-972f-3a744d8c0c4c" />





