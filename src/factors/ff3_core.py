from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import math
import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class FF3Config:
    lifecycle_path: str
    balance_sheet_dir: str
    income_statement_dir: str
    market_cap_dir: str
    price_cache_path: str = "data/raw/prices/yahoo_adjusted_close.parquet"
    calendar_path: str = "data/processed/calendar/trading_calendar.csv"
    output_dir: str = "results/tables/ff3"
    rebalance_frequency: str = "monthly"  # supported: monthly, weekly
    long_quantile: float = 0.20
    short_quantile: float = 0.20
    long_gross: float = 0.50
    short_gross: float = 0.50
    min_ttm_quarters: int = 4
    price_batch_size: int = 100  # number of tickers to download from yfinance in a single batch
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    # --- New: local FMP price support ---
    fmp_price_dir: Optional[str] = None
    prefer_fmp_prices: bool = True
    use_yahoo_fallback: bool = True


class FF3Core:
    """Heavy-lifting component for the FF3 backtest pipeline.

    Responsibilities:
    - point-in-time lifecycle universe
    - price download/cache
    - point-in-time fundamentals
    - FF3 signal construction
    - portfolio membership / weights
    - holding-period return calculation
    """

    def __init__(self, config: FF3Config):
        self.config = config
        self.lifecycle = self._load_lifecycle(config.lifecycle_path)
        self.balance_index = self._build_file_index(config.balance_sheet_dir)
        self.income_index = self._build_file_index(config.income_statement_dir)
        self.market_cap_index = self._build_file_index(config.market_cap_dir)
        self._balance_cache: Dict[str, pd.DataFrame] = {}
        self._income_cache: Dict[str, pd.DataFrame] = {}
        self._market_cap_cache: Dict[str, pd.DataFrame] = {}
        self._price_panel: Optional[pd.DataFrame] = None
        # --- New: FMP price index ---
        self.fmp_price_index = self._build_file_index(config.fmp_price_dir) if config.fmp_price_dir else {}
        self._fmp_price_cache: Dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Calendar / prices
    # ------------------------------------------------------------------
    def build_rebalance_calendar(self) -> pd.DataFrame:
        calendar_path = Path(self.config.calendar_path)
        if not calendar_path.exists():
            raise FileNotFoundError(f"Trading calendar not found: {calendar_path}")

        cal = pd.read_csv(calendar_path)

        cal["trading_date"] = pd.to_datetime(cal["trading_date"])
        cal["next_trading_date"] = pd.to_datetime(cal["next_trading_date"])
        if "week_end" in cal.columns:
            cal["week_end"] = pd.to_datetime(cal["week_end"])

        if self.config.start_date is not None:
            cal = cal.loc[cal["trading_date"] >= pd.Timestamp(self.config.start_date)].copy()

        if self.config.end_date is not None:
            cal = cal.loc[cal["trading_date"] <= pd.Timestamp(self.config.end_date)].copy()

        if self.config.rebalance_frequency == "monthly":
            cal = cal.loc[cal["is_month_end"] == True].copy()
        elif self.config.rebalance_frequency == "weekly":
            cal = cal.loc[cal["is_week_end"] == True].copy()
        else:
            raise ValueError(f"Unsupported rebalance_frequency={self.config.rebalance_frequency!r}")

        cal = cal.sort_values("trading_date").reset_index(drop=True)

        out = cal[["trading_date", "next_trading_date"]].copy()
        out = out.rename(
            columns={
                "trading_date": "signal_date",
                "next_trading_date": "execution_date",
            }
        )

        out["next_execution_date"] = out["execution_date"].shift(-1)
        out = out.dropna(subset=["signal_date", "execution_date", "next_execution_date"]).reset_index(drop=True)

        return out

    def load_or_download_price_panel(self, tickers: Iterable[str]) -> pd.DataFrame:
        cache_path = Path(self.config.price_cache_path)

        if cache_path.exists():
            print(f"Found price cache: {cache_path}")
            price_panel = pd.read_parquet(cache_path)
            price_panel.index = pd.to_datetime(price_panel.index)
            price_panel = price_panel.sort_index()

            tickers = sorted(set(tickers))

            if self._is_price_cache_valid(price_panel, tickers):
                print("Using validated price cache.")
                self._price_panel = price_panel
                return self._price_panel

            print("Price cache failed validation. Redownloading...")
            print(f"Overwriting cache at {cache_path}")

        tickers = sorted(set(tickers))

        fmp_panel = pd.DataFrame()
        remaining_tickers = tickers.copy()

        # --- Step 1: Load FMP prices ---
        if self.config.prefer_fmp_prices and self.fmp_price_index:
            print("Loading prices from local FMP folder...")

            fmp_panel = self._build_fmp_price_panel(tickers)

            loaded_tickers = list(fmp_panel.columns)
            remaining_tickers = sorted(set(tickers) - set(loaded_tickers))

            print("FMP tickers loaded:", sorted(loaded_tickers))
            print("Yahoo fallback tickers:", remaining_tickers)

            print(f"FMP loaded: {len(loaded_tickers)} tickers")
            print(f"Remaining for Yahoo: {len(remaining_tickers)} tickers")

        # --- Step 2: Yahoo fallback ---
        yahoo_panel = pd.DataFrame()

        if self.config.use_yahoo_fallback and remaining_tickers:
            print(f"Downloading remaining {len(remaining_tickers)} tickers from Yahoo...")

            batches = self._chunk_list(remaining_tickers, self.config.price_batch_size)
            batch_panels = []

            for i, batch in enumerate(batches, start=1):
                print(f"Yahoo batch {i}/{len(batches)}: {len(batch)} tickers")

                data = yf.download(
                    tickers=batch,
                    start=self.config.start_date,
                    end=self.config.end_date,
                    auto_adjust=False,
                    progress=False,
                    group_by="ticker",
                    threads=True,
                )

                if data.empty:
                    continue

                panel = self._extract_adjusted_close_panel(data, batch)

                if not panel.empty:
                    batch_panels.append(panel)

            if batch_panels:
                yahoo_panel = pd.concat(batch_panels, axis=1)

        # --- Step 3: Merge ---
        if not fmp_panel.empty and not yahoo_panel.empty:
            price_panel = pd.concat([fmp_panel, yahoo_panel], axis=1)
        elif not fmp_panel.empty:
            price_panel = fmp_panel
        elif not yahoo_panel.empty:
            price_panel = yahoo_panel
        else:
            raise ValueError("No price data available from FMP or Yahoo.")

        price_panel = price_panel.sort_index()
        price_panel = price_panel.loc[:, ~price_panel.columns.duplicated()]
        price_panel = price_panel.reindex(sorted(price_panel.columns), axis=1)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        price_panel.to_parquet(cache_path)

        self._price_panel = price_panel
        return self._price_panel

    # ------------------------------------------------------------------
    # Signal / portfolio construction
    # ------------------------------------------------------------------
    def build_signals_for_date(self, signal_date: pd.Timestamp) -> pd.DataFrame:
        active_universe = self.get_universe_at_date(signal_date)

        # Debugging output
        # print(f"\\nSignal date: {signal_date}")
        # print(f"Active universe count: {len(active_universe)}")
        # print(f"Active universe tickers: {active_universe}")

        records: List[Dict[str, object]] = []

        for ticker in active_universe:
            market_cap = self.get_market_cap_at_date(ticker, signal_date)
            adj_close = self.get_price_at_date(ticker, signal_date)
            book_equity = self.get_book_equity_at_date(ticker, signal_date)
            ttm_net_income = self.get_ttm_net_income_at_date(ticker, signal_date)
            records.append(
                {
                    "signal_date": signal_date,
                    "ticker": ticker,
                    "adj_close_t": adj_close,
                    "market_cap": market_cap,
                    "book_equity": book_equity,
                    "ttm_net_income": ttm_net_income,
                }
            )

        df = pd.DataFrame(records)

        # Debugging output
        # print("Raw records shape:", df.shape)
        # if not df.empty:
        #     print(df[["ticker", "adj_close_t", "market_cap", "book_equity", "ttm_net_income"]].head())
        #     print("Non-null counts:")
        #     print(df[["adj_close_t", "market_cap", "book_equity", "ttm_net_income"]].notna().sum())

        if df.empty:
            return df

        df = self.apply_signal_filters(df)

        # Debugging output
        # print("Post-filter shape:", df.shape)

        if df.empty:
            return df

        df["pb"] = df["market_cap"] / df["book_equity"]
        df["ep"] = df["ttm_net_income"] / df["market_cap"]
        df["size_signal"] = -df["market_cap"]
        df["pb_signal"] = -df["pb"]
        df["ep_signal"] = df["ep"]

        df["rank_size"] = df["size_signal"].rank(method="first", ascending=False)
        df["rank_pb"] = df["pb_signal"].rank(method="first", ascending=False)
        df["rank_ep"] = df["ep_signal"].rank(method="first", ascending=False)
        df["ff3_score"] = df["rank_size"] + df["rank_pb"] + df["rank_ep"]

        return df.sort_values(["ff3_score", "ticker"], ascending=[True, True]).reset_index(drop=True)

    def form_portfolio_membership(
        self,
        signals_at_t: pd.DataFrame,
        signal_date: pd.Timestamp,
        execution_date: pd.Timestamp,
    ) -> pd.DataFrame:
        if signals_at_t.empty:
            return pd.DataFrame()

        n = len(signals_at_t)
        n_long = max(1, math.floor(n * self.config.long_quantile))
        n_short = max(1, math.floor(n * self.config.short_quantile))

        ranked = signals_at_t.sort_values(["ff3_score", "ticker"], ascending=[True, True]).copy()
        longs = ranked.head(n_long).copy()
        shorts = ranked.tail(n_short).copy()
        longs["side"] = "long"
        shorts["side"] = "short"

        membership = pd.concat([longs, shorts], ignore_index=True)
        membership["signal_date"] = signal_date
        membership["execution_date"] = execution_date
        membership["adj_close_e"] = membership["ticker"].map(lambda t: self.get_price_at_date(t, execution_date))
        membership = membership.dropna(subset=["adj_close_e"]).reset_index(drop=True)
        return membership

    def assign_weights(self, membership_at_t: pd.DataFrame) -> pd.DataFrame:
        if membership_at_t.empty:
            return pd.DataFrame()

        df = membership_at_t.copy()
        long_mask = df["side"] == "long"
        short_mask = df["side"] == "short"
        n_long = int(long_mask.sum())
        n_short = int(short_mask.sum())
        if n_long == 0 or n_short == 0:
            return pd.DataFrame()

        df.loc[long_mask, "weight"] = self.config.long_gross / n_long
        df.loc[short_mask, "weight"] = -self.config.short_gross / n_short
        return df

    def compute_holding_period_asset_returns(
        self,
        weights_at_e: pd.DataFrame,
        execution_date: pd.Timestamp,
        next_execution_date: pd.Timestamp,
    ) -> pd.DataFrame:
        if weights_at_e.empty:
            return pd.DataFrame()

        records: List[Dict[str, object]] = []
        for row in weights_at_e.itertuples(index=False):
            entry_price = row.adj_close_e
            exit_price = self.get_price_at_date(row.ticker, next_execution_date)
            actual_exit_date = next_execution_date

            if pd.isna(exit_price):
                fallback = self.get_last_available_price_before(row.ticker, next_execution_date)
                if fallback is None:
                    continue
                actual_exit_date, exit_price = fallback

            asset_return = (exit_price / entry_price) - 1.0
            records.append(
                {
                    "signal_date": row.signal_date,
                    "execution_date": execution_date,
                    "next_execution_date": next_execution_date,
                    "actual_exit_date": actual_exit_date,
                    "ticker": row.ticker,
                    "side": row.side,
                    "weight": row.weight,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "asset_return": asset_return,
                    "weighted_return": row.weight * asset_return,
                }
            )
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Point-in-time accessors
    # ------------------------------------------------------------------
    def get_universe_at_date(self, date: pd.Timestamp) -> List[str]:
        mask = (self.lifecycle["entry_date"] <= date) & (self.lifecycle["exit_date"] >= date)
        return sorted(self.lifecycle.loc[mask, "ticker"].dropna().astype(str).unique().tolist())

    def get_price_at_date(self, ticker: str, date: pd.Timestamp) -> Optional[float]:
        if self._price_panel is None or ticker not in self._price_panel.columns or date not in self._price_panel.index:
            return np.nan
        value = self._price_panel.at[date, ticker]
        return float(value) if pd.notna(value) else np.nan

    def get_last_available_price_before(self, ticker: str, date: pd.Timestamp) -> Optional[Tuple[pd.Timestamp, float]]:
        if self._price_panel is None or ticker not in self._price_panel.columns:
            return None
        series = self._price_panel.loc[self._price_panel.index <= date, ticker].dropna()
        if series.empty:
            return None
        return pd.Timestamp(series.index[-1]), float(series.iloc[-1])

    def get_market_cap_at_date(self, ticker: str, date: pd.Timestamp) -> Optional[float]:
        df = self._load_market_cap(ticker)
        if df.empty:
            return np.nan
        eligible = df.loc[df["availability_date"] <= date]
        if eligible.empty:
            return np.nan
        return float(eligible.iloc[-1]["marketCap"])

    def get_book_equity_at_date(self, ticker: str, date: pd.Timestamp) -> Optional[float]:
        df = self._load_balance_sheet(ticker)
        if df.empty:
            return np.nan
        eligible = df.loc[df["availability_date"] <= date]
        if eligible.empty:
            return np.nan
        return float(eligible.iloc[-1]["totalStockholdersEquity"])

    def get_ttm_net_income_at_date(self, ticker: str, date: pd.Timestamp) -> Optional[float]:
        df = self._load_income_statement(ticker)
        if df.empty:
            return np.nan
        eligible = df.loc[df["availability_date"] <= date, ["availability_date", "netIncome"]].dropna()
        eligible = eligible.sort_values("availability_date")
        if len(eligible) < self.config.min_ttm_quarters:
            return np.nan
        return float(eligible.tail(self.config.min_ttm_quarters)["netIncome"].sum())

    # ------------------------------------------------------------------
    # CSV loaders
    # ------------------------------------------------------------------
    def _load_lifecycle(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}
        ticker_col = self._first_existing(cols, ["ticker", "symbol"])
        entry_col = self._first_existing(cols, ["entry_date", "start_date", "ipo_date"])
        exit_col = self._first_existing(cols, ["exit_date", "end_date", "delisted_date"])
        if ticker_col is None or entry_col is None or exit_col is None:
            raise ValueError("Lifecycle file must contain ticker, entry_date, and exit_date columns (or aliases).")

        out = df[[ticker_col, entry_col, exit_col]].copy()
        out.columns = ["ticker", "entry_date", "exit_date"]
        out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
        out["entry_date"] = pd.to_datetime(out["entry_date"], errors="coerce")
        out["exit_date"] = pd.to_datetime(out["exit_date"], errors="coerce")
        return out.dropna(subset=["ticker", "entry_date", "exit_date"]).reset_index(drop=True)

    def _build_file_index(self, directory: str) -> Dict[str, Path]:
        base = Path(directory)
        if not base.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        return {path.stem.upper(): path for path in base.glob("*.csv")}

    def _load_balance_sheet(self, ticker: str) -> pd.DataFrame:
        ticker = ticker.upper()
        if ticker in self._balance_cache:
            return self._balance_cache[ticker]
        path = self.balance_index.get(ticker)
        if path is None:
            self._balance_cache[ticker] = pd.DataFrame()
            return self._balance_cache[ticker]

        df = pd.read_csv(path)
        df = self._normalize_availability_date(df)
        if "totalStockholdersEquity" not in df.columns:
            raise ValueError(f"Missing totalStockholdersEquity in balance sheet file: {path}")
        df = df[["availability_date", "totalStockholdersEquity"]].copy()
        df["totalStockholdersEquity"] = pd.to_numeric(df["totalStockholdersEquity"], errors="coerce")
        df = df.dropna(subset=["availability_date"]).sort_values("availability_date").reset_index(drop=True)
        self._balance_cache[ticker] = df
        return df

    def _load_income_statement(self, ticker: str) -> pd.DataFrame:
        ticker = ticker.upper()
        if ticker in self._income_cache:
            return self._income_cache[ticker]
        path = self.income_index.get(ticker)
        if path is None:
            self._income_cache[ticker] = pd.DataFrame()
            return self._income_cache[ticker]

        df = pd.read_csv(path)
        df = self._normalize_availability_date(df)
        if "netIncome" not in df.columns:
            raise ValueError(f"Missing netIncome in income statement file: {path}")
        df = df[["availability_date", "netIncome"]].copy()
        df["netIncome"] = pd.to_numeric(df["netIncome"], errors="coerce")
        df = df.dropna(subset=["availability_date"]).sort_values("availability_date").reset_index(drop=True)
        self._income_cache[ticker] = df
        return df

    def _load_market_cap(self, ticker: str) -> pd.DataFrame:
        ticker = ticker.upper()
        if ticker in self._market_cap_cache:
            return self._market_cap_cache[ticker]
        path = self.market_cap_index.get(ticker)
        if path is None:
            self._market_cap_cache[ticker] = pd.DataFrame()
            return self._market_cap_cache[ticker]

        df = pd.read_csv(path)
        df = self._normalize_availability_date(df)
        cols = {c.lower(): c for c in df.columns}
        mcap_col = self._first_existing(cols, ["marketcap", "market_cap"])
        if mcap_col is None:
            raise ValueError(f"Missing marketCap in market cap file: {path}")
        df = df[["availability_date", mcap_col]].copy()
        df.columns = ["availability_date", "marketCap"]
        df["marketCap"] = pd.to_numeric(df["marketCap"], errors="coerce")
        df = df.dropna(subset=["availability_date"]).sort_values("availability_date").reset_index(drop=True)
        self._market_cap_cache[ticker] = df
        return df
    
    def _load_fmp_price(self, ticker: str) -> pd.DataFrame:
        ticker = ticker.upper()

        if ticker in self._fmp_price_cache:
            return self._fmp_price_cache[ticker]

        path = self.fmp_price_index.get(ticker)
        if path is None:
            self._fmp_price_cache[ticker] = pd.DataFrame()
            return self._fmp_price_cache[ticker]

        df = pd.read_csv(path)

        if "date" not in df.columns:
            raise ValueError(f"Missing 'date' column in FMP price file: {path}")

        # Prefer adjClose
        if "adjClose" in df.columns:
            price_col = "adjClose"
        elif "close" in df.columns:
            price_col = "close"
        else:
            raise ValueError(f"No price column found in FMP file: {path}")

        df = df[["date", price_col]].copy()
        df.columns = ["date", "price"]

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["price"] = pd.to_numeric(df["price"], errors="coerce")

        df = df.dropna(subset=["date"]).drop_duplicates(subset=["date"])
        df = df.sort_values("date").set_index("date")

        # Apply date filter
        if self.config.start_date:
            df = df.loc[df.index >= pd.Timestamp(self.config.start_date)]
        if self.config.end_date:
            df = df.loc[df.index <= pd.Timestamp(self.config.end_date)]

        self._fmp_price_cache[ticker] = df
        return df
    
    def _build_fmp_price_panel(self, tickers: List[str]) -> pd.DataFrame:
        panels = {}

        for ticker in tickers:
            df = self._load_fmp_price(ticker)
            if df.empty:
                continue
            panels[ticker] = df["price"]

        if not panels:
            return pd.DataFrame()

        panel = pd.DataFrame(panels)
        panel.index = pd.to_datetime(panel.index)
        return panel.sort_index()

    def _normalize_availability_date(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = {c.lower(): c for c in df.columns}
        preferred = self._first_existing(cols, ["accepteddate", "fillingdate", "filingdate", "date"])
        if preferred is None:
            raise ValueError("CSV must contain one of: acceptedDate, fillingDate, filingDate, or date.")
        out = df.copy()
        out["availability_date"] = pd.to_datetime(out[preferred], errors="coerce")
        return out

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------
    def apply_signal_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        required = ["adj_close_t", "market_cap", "book_equity", "ttm_net_income"]
        out = df.dropna(subset=required).copy()
        out = out[
            (out["adj_close_t"] > 0)
            & (out["market_cap"] > 0)
            & (out["book_equity"] > 0)
            & (out["ttm_net_income"] > 0)
        ].copy()
        return out

    def save_outputs(self, outputs: Dict[str, pd.DataFrame]) -> None:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, df in outputs.items():
            df.to_csv(output_dir / f"{name}.csv", index=False)

    def _extract_adjusted_close_panel(self, data: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
        if isinstance(data.columns, pd.MultiIndex):
            cols: Dict[str, pd.Series] = {}
            for ticker in tickers:
                if (ticker, "Adj Close") in data.columns:
                    cols[ticker] = data[(ticker, "Adj Close")]
                elif (ticker, "Close") in data.columns:
                    cols[ticker] = data[(ticker, "Close")]
            panel = pd.DataFrame(cols)
        else:
            if "Adj Close" in data.columns and len(tickers) == 1:
                panel = pd.DataFrame({tickers[0]: data["Adj Close"]})
            elif "Close" in data.columns and len(tickers) == 1:
                panel = pd.DataFrame({tickers[0]: data["Close"]})
            else:
                raise ValueError("Could not locate adjusted close prices in Yahoo output.")
        panel.index = pd.to_datetime(panel.index)
        return panel.sort_index()
    
    def _is_price_cache_valid(self, price_panel: pd.DataFrame, tickers: list[str]) -> bool:
        if price_panel.empty:
            print("Price cache invalid: empty panel.")
            return False

        # Ticker coverage
        missing_tickers = sorted(set(tickers) - set(price_panel.columns))
        if missing_tickers:
            print(f"Price cache invalid: missing {len(missing_tickers)} tickers.")
            print("Sample missing tickers:", missing_tickers[:10])
            return False

        # Date coverage
        cache_start = pd.Timestamp(price_panel.index.min())
        cache_end = pd.Timestamp(price_panel.index.max())

        if self.config.start_date is not None:
            requested_start = pd.Timestamp(self.config.start_date)
            if cache_start > requested_start:
                print(
                    f"Price cache invalid: cache starts at {cache_start.date()}, "
                    f"later than requested start {requested_start.date()}."
                )
                return False

        if self.config.end_date is not None:
            requested_end = pd.Timestamp(self.config.end_date)
            if cache_end < requested_end:
                print(
                    f"Price cache invalid: cache ends at {cache_end.date()}, "
                    f"earlier than requested end {requested_end.date()}."
                )
                return False

        # Requested slice should not be empty
        sliced = price_panel.copy()
        if self.config.start_date is not None:
            sliced = sliced.loc[sliced.index >= pd.Timestamp(self.config.start_date)]
        if self.config.end_date is not None:
            sliced = sliced.loc[sliced.index <= pd.Timestamp(self.config.end_date)]

        if sliced.empty:
            print("Price cache invalid: requested date slice is empty.")
            return False

        # At least one requested ticker should have some data in the requested slice,
        # and ideally most should.
        non_null_counts = sliced[tickers].notna().sum()
        if (non_null_counts == 0).all():
            print("Price cache invalid: all requested tickers are fully null in requested slice.")
            return False

        return True

    def _universe_tickers_from_lifecycle(self) -> List[str]:
        return sorted(self.lifecycle["ticker"].dropna().astype(str).str.upper().unique().tolist())

    @staticmethod
    def _first_existing(mapping: Dict[str, str], candidates: List[str]) -> Optional[str]:
        for candidate in candidates:
            if candidate in mapping:
                return mapping[candidate]
        return None
    
    @staticmethod
    def _chunk_list(items: List[str], chunk_size: int) -> List[List[str]]:
        return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
    