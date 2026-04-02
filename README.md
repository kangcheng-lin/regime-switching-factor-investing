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

A clean, tradable U.S. equity universe has been constructed using exchange-level listings and validated filtering rules.

#### Data Sources

- Active listings sourced from official **NASDAQ / NYSE / AMEX exchange data**
- Delisted securities sourced from **Financial Modeling Prep (FMP)**
- Security lifecycle data providing **entry and exit dates (if any)** for each ticker

#### Methodology

The universe is constructed by combining:
- Currently listed securities (active universe)
- Historically listed securities (delisted universe)

and applying strict filtering criteria.

#### Inclusion Criteria

- Common equity instruments:
  - **Common Stock(s)**
  - **Common Share(s)**
  - **Ordinary Share(s)**
- Listed on:
  - NASDAQ
  - NYSE
  - AMEX

#### Exclusion Criteria

- Depositary instruments:
  - ADR / ADS
- Funds and structured products:
  - ETFs, ETNs, mutual funds, trusts
- Real estate structures:
  - REITs
- Corporate structures:
  - SPACs / blank-check companies
- Non-common equity:
  - Preferred shares (all types)
- Derivative-like instruments:
  - Warrants, rights, units
- Debt-like instruments:
  - Notes, bonds
- Hybrid / credit vehicles:
  - BDCs and similar structures

#### Lifecycle Integration

Each security is mapped to a lifecycle record containing:

- **Entry date**: first appearance in the dataset / tradable start
- **Exit date**: delisting or last tradable date (if applicable)

This enables:
- Time-consistent universe construction
- Proper handling of **entry and exit of securities over time**
- Compatibility with **backtesting without survivorship bias**

#### Final Universe

- Initial candidate universe: ~7,000+ securities
- Final filtered universe: ~5,400+ securities
- Final filtered universe with usable tickers: ~4,900 securities

#### Key Design Decisions

- Universe is defined by **listing venue (U.S. exchanges)**, not company domicile
- Foreign companies listed in the U.S. are included (e.g., non-U.S. ordinary shares)
- Filtering emphasizes **tradability and instrument type consistency**
- Lifecycle data is used to ensure **time-aware universe membership**
- Usability is defined based on the availability of data from both **FMP data (primary)** and **Yahoo Finance data (fallback)**

---