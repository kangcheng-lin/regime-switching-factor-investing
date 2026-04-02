from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import yfinance as yf


# ============================================================
# CONFIG
# ============================================================
LIFECYCLE_PATH = "data/processed/universe/security_lifecycle_filtered.csv"
FMP_PRICE_DIR = "data/raw/prices/fmp"

# Output files
FULL_OUTPUT_PATH = "data/processed/universe/price_availability_full.csv"
USABLE_OUTPUT_PATH = "data/processed/universe/price_available_universe.csv"
REMOVED_OUTPUT_PATH = "data/processed/universe/price_unavailable_universe.csv"

# Yahoo probing controls
YAHOO_SLEEP_SECONDS = 0.5
PROBE_CALENDAR_DAYS = 14
TODAY = pd.Timestamp.today().normalize()

# Optional debug tickers
DEBUG_TICKERS = {
    "UBER", "MSFT", "NVDA", "TSLA", "META"
}


# ============================================================
# HELPERS
# ============================================================
def first_existing(mapping: dict[str, str], candidates: list[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in mapping:
            return mapping[candidate]
    return None


def load_lifecycle(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    cols = {c.lower(): c for c in df.columns}

    ticker_col = first_existing(cols, ["ticker", "symbol"])
    entry_col = first_existing(cols, ["entry_date", "start_date", "ipo_date"])
    exit_col = first_existing(cols, ["exit_date", "end_date", "delisted_date"])

    if ticker_col is None or entry_col is None or exit_col is None:
        raise ValueError(
            "Lifecycle file must contain ticker/symbol, entry_date/start_date/ipo_date, "
            "and exit_date/end_date/delisted_date columns."
        )

    out = df[[ticker_col, entry_col, exit_col]].copy()
    out.columns = ["ticker", "entry_date", "exit_date"]

    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    out["entry_date"] = pd.to_datetime(out["entry_date"], errors="coerce")
    out["exit_date"] = pd.to_datetime(out["exit_date"], errors="coerce")

    out = out.dropna(subset=["ticker", "entry_date"]).reset_index(drop=True)

    # Missing exit date means still active through TODAY
    out["exit_date_filled"] = out["exit_date"].fillna(TODAY)

    # Remove invalid rows
    out = out.loc[out["exit_date_filled"] >= out["entry_date"]].copy()

    # Keep the widest lifecycle range per ticker for screening purposes
    out = (
        out.groupby("ticker", as_index=False)
        .agg(
            entry_date=("entry_date", "min"),
            exit_date=("exit_date", "max"),
            exit_date_filled=("exit_date_filled", "max"),
        )
        .sort_values("ticker")
        .reset_index(drop=True)
    )

    return out


def compute_probe_window(
    entry_date: pd.Timestamp,
    exit_date_filled: pd.Timestamp,
    probe_calendar_days: int = PROBE_CALENDAR_DAYS,
) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Choose a short probe window INSIDE the valid lifecycle range.

    Current rule:
    - probe_end = exit_date_filled
    - probe_start = max(entry_date, probe_end - probe_calendar_days)

    This keeps the probe inside the valid lifecycle window.
    """
    valid_start = pd.Timestamp(entry_date).normalize()
    valid_end = pd.Timestamp(exit_date_filled).normalize()

    if valid_end < valid_start:
        return None

    probe_end = valid_end
    probe_start = max(valid_start, probe_end - pd.Timedelta(days=probe_calendar_days))

    if probe_end < probe_start:
        return None

    return probe_start, probe_end


def check_fmp_availability(
    ticker: str,
    entry_date: pd.Timestamp,
    exit_date_filled: pd.Timestamp,
    fmp_dir: str,
) -> tuple[bool, Optional[pd.Timestamp], Optional[pd.Timestamp], int, str]:
    """
    Return whether local FMP file contains at least one usable price
    inside the ticker's valid lifecycle range.
    """
    path = Path(fmp_dir) / f"{ticker}.csv"
    if not path.exists():
        return False, None, None, 0, "missing_fmp_file"

    try:
        df = pd.read_csv(path)
    except Exception as e:
        return False, None, None, 0, f"fmp_read_error: {type(e).__name__}"

    if "date" not in df.columns:
        return False, None, None, 0, "fmp_missing_date_column"

    if "adjClose" in df.columns:
        price_col = "adjClose"
    elif "close" in df.columns:
        price_col = "close"
    else:
        return False, None, None, 0, "fmp_missing_price_column"

    df = df[["date", price_col]].copy()
    df.columns = ["date", "price"]

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    df = df.dropna(subset=["date"]).sort_values("date")
    df = df.drop_duplicates(subset=["date"], keep="last")

    mask = (
        (df["date"] >= pd.Timestamp(entry_date).normalize()) &
        (df["date"] <= pd.Timestamp(exit_date_filled).normalize())
    )
    in_range = df.loc[mask].copy()
    in_range = in_range.loc[in_range["price"].notna()]

    if in_range.empty:
        return False, None, None, 0, "fmp_no_prices_in_lifecycle_range"

    first_date = pd.Timestamp(in_range["date"].min())
    last_date = pd.Timestamp(in_range["date"].max())
    n_obs = int(in_range["price"].notna().sum())

    return True, first_date, last_date, n_obs, "fmp_ok"


def extract_yahoo_price_series(data: pd.DataFrame, ticker: str) -> Optional[pd.Series]:
    """
    Robustly extract a usable Yahoo price series from yf.download output.

    Based on your notebook feedback, Yahoo on your machine returns
    MultiIndex columns with layout: (Ticker, Price)

    With auto_adjust=False, we prefer:
    1) Adj Close
    2) Close

    We also keep fallback support for reverse MultiIndex layout and flat columns.
    """
    series = None

    if isinstance(data.columns, pd.MultiIndex):
        # Primary expected layout from your notebook: (Ticker, Price)
        if (ticker, "Adj Close") in data.columns:
            series = data[(ticker, "Adj Close")]
        elif (ticker, "Close") in data.columns:
            series = data[(ticker, "Close")]

        # Fallback in case layout is reversed: (Price, Ticker)
        elif ("Adj Close", ticker) in data.columns:
            series = data[("Adj Close", ticker)]
        elif ("Close", ticker) in data.columns:
            series = data[("Close", ticker)]

    else:
        if "Adj Close" in data.columns:
            series = data["Adj Close"]
        elif "Close" in data.columns:
            series = data["Close"]

    if series is None:
        return None

    series = pd.to_numeric(series, errors="coerce").dropna()

    if series.empty:
        return None

    return series


def check_yahoo_availability(
    ticker: str,
    entry_date: pd.Timestamp,
    exit_date_filled: pd.Timestamp,
) -> tuple[
    bool,
    Optional[pd.Timestamp],
    Optional[pd.Timestamp],
    int,
    Optional[pd.Timestamp],
    Optional[pd.Timestamp],
    str,
]:
    """
    Check whether Yahoo has at least one usable price inside a short
    lifecycle-respecting probe window.
    """
    window = compute_probe_window(entry_date, exit_date_filled)
    if window is None:
        return False, None, None, 0, None, None, "invalid_probe_window"

    probe_start, probe_end = window

    try:
        data = yf.download(
            tickers=ticker,
            start=probe_start.strftime("%Y-%m-%d"),
            end=(probe_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=False,
        )
    except Exception as e:
        return False, None, None, 0, probe_start, probe_end, f"yahoo_exception: {type(e).__name__}"

    if data.empty:
        return False, None, None, 0, probe_start, probe_end, "yahoo_empty_download"

    series = extract_yahoo_price_series(data, ticker)

    if series is None:
        return False, None, None, 0, probe_start, probe_end, "yahoo_missing_price_column"

    first_date = pd.Timestamp(series.index.min())
    last_date = pd.Timestamp(series.index.max())
    n_obs = int(series.notna().sum())

    return True, first_date, last_date, n_obs, probe_start, probe_end, "yahoo_ok"


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    lifecycle = load_lifecycle(LIFECYCLE_PATH)

    print(f"Total lifecycle tickers: {len(lifecycle)}")

    records = []

    fmp_usable_count = 0
    yahoo_usable_count = 0

    for i, row in enumerate(lifecycle.itertuples(index=False), start=1):
        ticker = row.ticker
        entry_date = pd.Timestamp(row.entry_date)
        exit_date = row.exit_date if pd.notna(row.exit_date) else pd.NaT
        exit_date_filled = pd.Timestamp(row.exit_date_filled)

        if i % 100 == 0 or i == 1 or i == len(lifecycle):
            print(f"Processing {i}/{len(lifecycle)}: {ticker}")

        # ----------------------------------------------------
        # Step 1: FMP first
        # ----------------------------------------------------
        fmp_usable, fmp_first_date, fmp_last_date, fmp_n_obs, fmp_reason = check_fmp_availability(
            ticker=ticker,
            entry_date=entry_date,
            exit_date_filled=exit_date_filled,
            fmp_dir=FMP_PRICE_DIR,
        )

        # ----------------------------------------------------
        # Step 2: Yahoo fallback only if FMP not usable
        # ----------------------------------------------------
        yahoo_usable = False
        yahoo_first_date = None
        yahoo_last_date = None
        yahoo_n_obs = 0
        yahoo_probe_start = None
        yahoo_probe_end = None
        yahoo_reason = "not_checked_because_fmp_ok"

        if not fmp_usable:
            yahoo_usable, yahoo_first_date, yahoo_last_date, yahoo_n_obs, yahoo_probe_start, yahoo_probe_end, yahoo_reason = (
                check_yahoo_availability(
                    ticker=ticker,
                    entry_date=entry_date,
                    exit_date_filled=exit_date_filled,
                )
            )
            time.sleep(YAHOO_SLEEP_SECONDS)

        usable = fmp_usable or yahoo_usable

        if fmp_usable:
            source = "FMP"
            final_reason = fmp_reason
            fmp_usable_count += 1
        elif yahoo_usable:
            source = "Yahoo"
            final_reason = yahoo_reason
            yahoo_usable_count += 1
        else:
            source = "none"
            final_reason = yahoo_reason if not fmp_usable else fmp_reason

        if ticker in DEBUG_TICKERS:
            print(
                f"[DEBUG] {ticker} | "
                f"fmp_usable={fmp_usable} ({fmp_reason}) | "
                f"yahoo_usable={yahoo_usable} ({yahoo_reason}) | "
                f"source={source} | "
                f"probe=({yahoo_probe_start}, {yahoo_probe_end})"
            )

        records.append(
            {
                "ticker": ticker,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "exit_date_filled": exit_date_filled,

                "fmp_usable": fmp_usable,
                "fmp_first_date": fmp_first_date,
                "fmp_last_date": fmp_last_date,
                "fmp_n_obs": fmp_n_obs,
                "fmp_reason": fmp_reason,

                "yahoo_usable": yahoo_usable,
                "yahoo_first_date": yahoo_first_date,
                "yahoo_last_date": yahoo_last_date,
                "yahoo_n_obs": yahoo_n_obs,
                "yahoo_probe_start": yahoo_probe_start,
                "yahoo_probe_end": yahoo_probe_end,
                "yahoo_reason": yahoo_reason,

                "usable": usable,
                "source": source,
                "final_reason": final_reason,
            }
        )

    result = pd.DataFrame(records).sort_values("ticker").reset_index(drop=True)
    usable_result = result.loc[result["usable"]].copy()
    removed_result = result.loc[~result["usable"]].copy()

    print()
    print("========== SUMMARY ==========")
    print(f"Total lifecycle tickers: {len(result)}")
    print(f"Usable from FMP: {fmp_usable_count}")
    print(f"Usable from Yahoo fallback: {yahoo_usable_count}")
    print(f"Total usable tickers: {len(usable_result)}")
    print(f"Removed tickers: {len(removed_result)}")

    full_output_path = Path(FULL_OUTPUT_PATH)
    usable_output_path = Path(USABLE_OUTPUT_PATH)
    removed_output_path = Path(REMOVED_OUTPUT_PATH)

    full_output_path.parent.mkdir(parents=True, exist_ok=True)

    result.to_csv(full_output_path, index=False)
    usable_result.to_csv(usable_output_path, index=False)
    removed_result.to_csv(removed_output_path, index=False)

    print(f"Saved full result table to: {full_output_path}")
    print(f"Saved usable universe to: {usable_output_path}")
    print(f"Saved removed tickers to: {removed_output_path}")


if __name__ == "__main__":
    main()