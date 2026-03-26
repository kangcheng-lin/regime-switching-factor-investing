from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def get_project_root() -> Path:
    """
    Assumes this script lives in:
    src/data/build_security_lifecycle.py
    """
    return Path(__file__).resolve().parents[2]


def safe_read_csv(csv_path: Path, usecols: Optional[list[str]] = None) -> pd.DataFrame:
    """
    Read a CSV safely with a small wrapper so errors are easier to debug.
    """
    try:
        return pd.read_csv(csv_path, usecols=usecols)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV: {csv_path}") from e


def get_first_market_cap_dates(market_cap_dir: Path) -> pd.DataFrame:
    """
    For each ticker CSV in market_cap_history_csv, read the earliest available
    market cap date.

    Expected file pattern:
        AAPL.csv
        UBER.csv
        ...

    Expected columns inside each CSV:
        - date
        - marketCap
        - possibly symbol
    """
    records: list[dict] = []

    csv_files = sorted(market_cap_dir.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in market cap directory: {market_cap_dir}")

    for csv_file in csv_files:
        symbol = csv_file.stem.upper().strip()

        df = safe_read_csv(csv_file, usecols=["date"])
        if df.empty:
            continue

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")

        if df.empty:
            continue

        first_date = df["date"].iloc[0]

        records.append(
            {
                "symbol": symbol,
                "first_market_cap_date": first_date,
            }
        )

    return pd.DataFrame(records)


def get_first_filing_dates(statement_dir: Path, output_col_name: str) -> pd.DataFrame:
    """
    Generic helper for quarterly statement folders:
        - income_statement_quarter_csv
        - balance_sheet_quarter_csv
        - cash_flow_quarter_csv

    We read only 'filingDate' because that is the economically correct
    availability date for the market.
    """
    records: list[dict] = []

    csv_files = sorted(statement_dir.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in statement directory: {statement_dir}")

    for csv_file in csv_files:
        symbol = csv_file.stem.upper().strip()

        try:
            df = pd.read_csv(csv_file, usecols=["filingDate"])
        except Exception as e:
            print(f"Skipping {csv_file.name}: {e}")
            continue

        if df.empty:
            continue

        df["filingDate"] = pd.to_datetime(df["filingDate"], errors="coerce")
        df = df.dropna(subset=["filingDate"]).sort_values("filingDate")

        if df.empty:
            continue

        first_date = df["filingDate"].iloc[0]

        records.append(
            {
                "symbol": symbol,
                output_col_name: first_date,
            }
        )

    return pd.DataFrame(records)


def load_delisted_table(delisted_path: Path) -> pd.DataFrame:
    """
    Read delisted ticker reference file.

    Expected columns:
        - symbol
        - delistedDate
        - ipoDate (optional for future use)
    """
    if not delisted_path.exists():
        raise FileNotFoundError(f"Delisted file not found: {delisted_path}")

    df = safe_read_csv(delisted_path)

    required_cols = ["symbol", "delistedDate"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in delisted file: {missing}")

    df = df.copy()
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df["delistedDate"] = pd.to_datetime(df["delistedDate"], errors="coerce")

    df = df.dropna(subset=["symbol"]).drop_duplicates(subset=["symbol"], keep="first")

    return df[["symbol", "delistedDate"]]


def build_security_lifecycle(
    market_cap_dir: Path,
    income_dir: Path,
    balance_dir: Path,
    cashflow_dir: Path,
    delisted_path: Path,
) -> pd.DataFrame:
    """
    Build a lifecycle table with:
        - first_market_cap_date
        - first_income_filing_date
        - first_balance_filing_date
        - first_cashflow_filing_date
        - entry_date = max(all first-available dates)
        - delisted_date
        - exit_date
    """
    market_cap_df = get_first_market_cap_dates(market_cap_dir)
    income_df = get_first_filing_dates(income_dir, "first_income_filing_date")
    balance_df = get_first_filing_dates(balance_dir, "first_balance_filing_date")
    cashflow_df = get_first_filing_dates(cashflow_dir, "first_cashflow_filing_date")
    delisted_df = load_delisted_table(delisted_path)

    lifecycle = market_cap_df.merge(income_df, on="symbol", how="outer")
    lifecycle = lifecycle.merge(balance_df, on="symbol", how="outer")
    lifecycle = lifecycle.merge(cashflow_df, on="symbol", how="outer")
    lifecycle = lifecycle.merge(delisted_df, on="symbol", how="left")

    required_date_cols = [
        "first_market_cap_date",
        "first_income_filing_date",
        "first_balance_filing_date",
        "first_cashflow_filing_date",
    ]

    # Keep only stocks that have all core inputs required for factor usability.
    lifecycle = lifecycle.dropna(subset=required_date_cols).copy()

    lifecycle["entry_date"] = lifecycle[required_date_cols].max(axis=1)
    lifecycle["exit_date"] = lifecycle["delistedDate"]

    lifecycle = lifecycle.sort_values("symbol").reset_index(drop=True)

    return lifecycle[
        [
            "symbol",
            "first_market_cap_date",
            "first_income_filing_date",
            "first_balance_filing_date",
            "first_cashflow_filing_date",
            "entry_date",
            "delistedDate",
            "exit_date",
        ]
    ]


def main() -> None:
    project_root = get_project_root()

    fundamentals_dir = project_root / "data" / "raw" / "fundamentals"

    market_cap_dir = fundamentals_dir / "market_cap_history_csv"
    income_dir = fundamentals_dir / "income_statement_quarter_csv"
    balance_dir = fundamentals_dir / "balance_sheet_quarter_csv"
    cashflow_dir = fundamentals_dir / "cash_flow_quarter_csv"
    delisted_path = fundamentals_dir / "fmp_tickers" / "delisted_tickers_jan_28_2026.csv"

    output_path = project_root / "data" / "processed" / "universe" / "security_lifecycle.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lifecycle = build_security_lifecycle(
        market_cap_dir=market_cap_dir,
        income_dir=income_dir,
        balance_dir=balance_dir,
        cashflow_dir=cashflow_dir,
        delisted_path=delisted_path,
    )

    lifecycle.to_csv(output_path, index=False)

    print("Security lifecycle build complete.")
    print(f"Number of securities: {len(lifecycle)}")
    print(f"Saved to: {output_path}")
    print("\nSample:")
    print(lifecycle.head())


if __name__ == "__main__":
    main()