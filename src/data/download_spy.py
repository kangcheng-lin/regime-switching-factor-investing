from pathlib import Path
import pandas as pd
import yfinance as yf


def main() -> None:
    # Create output folder if it does not exist
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download daily OHLCV data for SPY
    # Keep auto_adjust=False so we retain raw OHLC columns clearly at this stage.
    df = yf.download(
        "SPY",
        start="1993-01-01",
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        raise ValueError("Downloaded dataframe is empty.")

    # Flatten columns if yfinance returns a MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Standardize column names
    df = df.reset_index()
    df.columns = [str(col).strip().lower().replace(" ", "_") for col in df.columns]

    # Save to csv
    output_path = output_dir / "spy_daily.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df)} rows to {output_path}")
    print(df.head())


if __name__ == "__main__":
    main()