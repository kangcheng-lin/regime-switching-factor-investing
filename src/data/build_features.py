from pathlib import Path
import pandas as pd


def main() -> None:
    
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)

    data_path = data_dir / "spy_daily.csv"


    if not data_path.exists():
        raise FileNotFoundError("SPY data not found. Run download_spy.py first.")

    df = pd.read_csv(data_path)

    # Ensure sorted by date
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # --- Feature 1: Daily Return ---
    df["return"] = df["close"].pct_change()

    # --- Feature 2: Rolling Volatility (10-day) ---
    df["volatility"] = df["return"].rolling(window=10).std()

    # Drop initial NaNs
    df = df.dropna().reset_index(drop=True)

    # Save
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "spy_features.csv"

    df.to_csv(output_path, index=False)

    print(f"Saved features to {output_path}")
    print(df[["date", "return", "volatility"]].head())


if __name__ == "__main__":
    main()