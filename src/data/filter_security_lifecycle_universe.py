from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data/processed/universe/security_lifecycle.csv"
ACTIVE_FILE = PROJECT_ROOT / "data/processed/universe/us_common_stocks.csv"
DELISTED_FILE = PROJECT_ROOT / "data/processed/universe/delisted_us_common_stocks_reference.csv"

OUTPUT_FILE = PROJECT_ROOT / "data/processed/universe/security_lifecycle_filtered.csv"


def get_symbol_column(df: pd.DataFrame, file_label: str) -> str:
    for col in df.columns:
        if str(col).strip().lower() == "symbol":
            return col

    raise KeyError(
        f"No symbol column found in {file_label}. "
        f"Available columns: {list(df.columns)}"
    )


def clean_symbol_series(series: pd.Series) -> pd.Series:
    cleaned = series.copy()

    cleaned = cleaned.where(cleaned.notna(), pd.NA)

    cleaned = cleaned.map(
        lambda x: str(x).strip().upper() if pd.notna(x) else pd.NA
    )

    cleaned = cleaned.replace({"": pd.NA})

    return cleaned


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

    if not ACTIVE_FILE.exists():
        raise FileNotFoundError(f"Missing active reference file: {ACTIVE_FILE}")

    if not DELISTED_FILE.exists():
        raise FileNotFoundError(f"Missing delisted reference file: {DELISTED_FILE}")

    df = pd.read_csv(INPUT_FILE)
    active_df = pd.read_csv(ACTIVE_FILE)
    delisted_df = pd.read_csv(DELISTED_FILE)

    print(f"Security lifecycle rows: {len(df)}")
    print(f"Active reference rows: {len(active_df)}")
    print(f"Delisted reference rows: {len(delisted_df)}")

    main_symbol_col = get_symbol_column(df, "security_lifecycle.csv")
    active_symbol_col = get_symbol_column(active_df, "us_common_stocks.csv")
    delisted_symbol_col = get_symbol_column(
        delisted_df,
        "delisted_us_common_stocks_reference.csv",
    )

    df[main_symbol_col] = clean_symbol_series(df[main_symbol_col])
    active_df[active_symbol_col] = clean_symbol_series(active_df[active_symbol_col])
    delisted_df[delisted_symbol_col] = clean_symbol_series(delisted_df[delisted_symbol_col])

    df = df[df[main_symbol_col].notna()].copy()
    active_df = active_df[active_df[active_symbol_col].notna()].copy()
    delisted_df = delisted_df[delisted_df[delisted_symbol_col].notna()].copy()

    active_tickers = set(active_df[active_symbol_col])
    delisted_tickers = set(delisted_df[delisted_symbol_col])
    valid_tickers = active_tickers.union(delisted_tickers)

    print(f"Unique active tickers: {len(active_tickers)}")
    print(f"Unique delisted tickers: {len(delisted_tickers)}")
    print(f"Total valid tickers: {len(valid_tickers)}")

    filtered_df = df[df[main_symbol_col].isin(valid_tickers)].copy()

    print(f"Filtered rows kept: {len(filtered_df)}")
    print(f"Rows removed: {len(df) - len(filtered_df)}")
    print(f"Unique symbols in filtered: {filtered_df[main_symbol_col].nunique()}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved filtered security lifecycle file to: {OUTPUT_FILE}")
    print("\nSample output:")
    print(filtered_df.head())


if __name__ == "__main__":
    main()