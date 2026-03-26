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

### Stock Universe Construction (Completed)

A clean, tradable stock universe has been constructed from exchange-level listings:

- Source: NASDAQ, NYSE, and AMEX listings
- Initial dataset: ~7,000+ securities
- Final universe: ~4,500+ securities

Filtering methodology:
- Keep only securities labeled as:
  - **Common Stock**
  - **Common Shares**
- Exclude:
  - ADR / ADS (e.g., American Depositary Shares)
  - ETFs, ETNs, funds, trusts
  - Preferred stocks, warrants, notes, and other non-equity instruments

Key design decision:
- Universe is defined as **US-listed common stocks**
- Includes foreign companies listed in the US (e.g., ALV)
- Focus is on tradability rather than company domicile
