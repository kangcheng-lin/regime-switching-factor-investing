# Regime Switching Factor Investing

This project focuses on building a stock-level factor investing framework with:

- Market regime detection (HMM)
- Dynamic factor/model switching
- Portfolio construction and backtesting

## Status

### Regime Detection Pipeline (Completed)

A full end-to-end regime detection pipeline has been implemented and validated using SPY data:

- Data download and preprocessing (`download_spy.py`)
- Feature engineering (`build_features.py`)
- Hidden Markov Model training (`fit_hmm.py`)
- Regime visualization (`plot_regimes.py`)

---
