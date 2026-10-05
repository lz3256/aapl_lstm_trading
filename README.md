# AAPL LSTM Trading Prototype

A research prototype connecting return-sequence classification with an Alpaca paper-trading adapter. The project covers CSV preparation, chronological splitting, LSTM training, model persistence, account diagnostics, and position management.

## What is implemented

- A stacked LSTM classifier with 64/32/16 recurrent units, dropout, layer normalization, regularization, class weighting, and early stopping.
- Ten open-price returns per training input, with a two-class softmax output.
- An 80/20 chronological split within the most recent approximately 90 days of the supplied CSV.
- Saved model and configuration files under `models/`.
- Alpaca account/connection checks and an order adapter with confidence and position-size settings.

## Data and prediction target

The included `data/aapl_alpaca_30min_20250722.csv` contains **1,589 AAPL bars**, from **2025-01-23 through 2025-07-21**, with timestamp, OHLC, volume, trade count, and VWAP. The filename is an export label, not the final observation date. The model uses open-price returns rather than all available columns.

The training target is based on the following open-to-open return. `run_strategy()` uses a 0.0003 cutoff; observations at or below the cutoff form the other class. If positive labels are rare, `create_labels()` changes the threshold using a quantile of the supplied target series. This means the labels are not simply positive versus negative returns. See [method and evaluation notes](docs/METHOD.md).

## Offline training

Run these commands from the repository root in a separate Python environment:

```bash
git clone https://github.com/lz3256/aapl_lstm_trading.git
cd aapl_lstm_trading
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/train_model.py
```

Training uses the bundled CSV and does not place orders. TensorFlow support depends on your Python version and platform. Model weights are generated locally and are not included in the repository.

## Broker integration

Copy `.env.example` to `.env` and supply credentials from your own Alpaca paper account. The example contains no credentials. Connection and position checks are separate from the trading loop:

```bash
cp .env.example .env
python src/test_alpaca_connection.py
python src/check_alpaca_positions.py
```

`src/live_trading.py` contains the order-submission loop. It defaults to the paper endpoint but accepts an endpoint from the environment. Starting that script can submit orders; review its configuration and [integration notes](docs/METHOD.md) before using it.

| Setting | Default | Purpose |
| --- | --- | --- |
| `APCA_API_BASE_URL` | `https://paper-api.alpaca.markets` | Broker endpoint |
| `POSITION_SIZE` | 100 | Shares per position |
| `MIN_CONFIDENCE` | 0.6 | Classification-confidence threshold |
| `CHECK_INTERVAL` | 60 | Loop interval in seconds |

## Evaluation status

The existing backtest is a prototype diagnostic. Its P&L uses a return at the target timestamp while the classifier label uses the following return; the backtest also starts from a hard-coded 20-return window although training uses ten. Threshold fallback is applied separately to train and test labels. These details require correction and a fresh evaluation before results can support performance claims.

Transaction costs, slippage, borrow costs, and a complete portfolio accounting model are not included. The original screenshot is retained below as a historical artifact, not a verified performance benchmark.

<img width="590" alt="Historical AAPL prototype output" src="https://github.com/user-attachments/assets/ca68a8fa-fb3b-4074-972f-3a744d8c0c4c" />

## Repository map

- `src/aapl_lstm_open_price.py`: features, labels, classifier, and diagnostic backtest.
- `src/train_model.py`: offline training and model persistence.
- `src/live_trading.py`: market-data and order integration.
- `src/test_alpaca_connection.py`, `src/check_alpaca_positions.py`: account diagnostics.
- `data/`: the supplied historical CSV.
- `docs/METHOD.md`: target definitions, known evaluation issues, and next experiments.

## Credits

Project repository maintained by [@lz3256](https://github.com/lz3256). Documentation was reviewed and expanded with OpenAI Codex by inspecting the existing source. Market data and broker services are attributed to Alpaca; dependencies retain their respective licenses.
