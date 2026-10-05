# AAPL Method and Evaluation Notes

## Research question

Can a short sequence of open-price returns provide useful information about a later open-price movement, and how can a saved classifier be connected to a paper-trading adapter?

## Implemented pipeline

`load_data_from_csv()` sorts bars by timestamp, standardizes OHLC names, and drops missing rows. `split_train_test_data()` retains approximately the last 90 days and splits observations chronologically 80/20. Returns and shifted future returns are computed separately in each split. Training inputs contain ten return observations preceding the target timestamp.

The LSTM has 64/32/16 recurrent units and a dense classifier. Training uses class weights, validation-loss early stopping, dropout, normalization, and regularization. The script writes an HDF5 model and JSON configuration.

## Issues to resolve before interpreting performance

1. **Prediction and execution timing:** map each input, target open, decision time, entry, and exit to explicit bar timestamps. Training labels use a shifted following return, whereas current backtest P&L uses the return at the label timestamp.
2. **Label consistency:** the low-positive-share fallback computes a threshold separately in each split. Estimate any adaptive threshold on training data and reuse it unchanged for validation/test.
3. **Sequence consistency:** the diagnostic backtest uses hard-coded 20-observation slices; training and the broker adapter use ten. Check the effective prediction window and warmup rules together.
4. **Chronological validation:** reserve explicit validation and future test periods, preserve boundaries, and report baselines and class proportions. Repeated inspection of one test period does not create new evidence.
5. **P&L accounting:** the existing sum of signed bar returns is not a compounded, cost-adjusted portfolio return. Add turnover, transaction and borrow costs, execution delays, and position accounting before strategy comparisons.
6. **Order lifecycle:** inspect open orders, fills, position reversals, market sessions, and restart behavior before running the adapter. Classifier confidence alone is not a position-risk measure.

These are implementation-derived limitations. This documentation update does not modify training behavior, regenerate metrics, authenticate to a broker, or run an order loop.

## Suggested experiments

After correcting timing and labels, compare the classifier with a majority-class baseline, a lagged-return baseline, and logistic regression using the same split and features. Report balanced accuracy, class confusion, calibration, turnover, and cost-adjusted returns separately. Use new future data for confirmation.
