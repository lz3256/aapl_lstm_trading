## aapl_lstm_trading
An automated trading system that uses LSTM neural networks to predict Apple (AAPL) stock price movements and execute trades via Alpaca API.

# Strategy Overview
This trading bot uses machine learning to predict short-term price movements:

Timeframe: 30-minute bars
Input: Last 10 open price returns
Output: Binary prediction (RISE or FALL)
Action: Automatically places buy/sell orders based on predictions
Position Size: 100 shares (configurable)

# How It Works

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

# Installation

1.Clone the repository:
```bash git clone https://github.com/lz3256/aapl-lstm-trading.git
cd aapl-lstm-trading ```


