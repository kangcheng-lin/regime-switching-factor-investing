from pathlib import Path
import pandas as pd


RAW_FILE = Path("data/raw/nasdaq_nyse_amex.csv")
OUTPUT_FILE = Path("data/processed/us_common_stocks.csv")


def main() -> None:
    df = pd.read_csv(RAW_FILE)

    print(f"Initial number of securities: {len(df)}")

    keep_pattern = r"common stock|common shares"

    df = df[df["Name"].str.contains(keep_pattern, case=False, na=False)].copy()

    print(f"After keeping common stock / common shares only: {len(df)}")

    df = df.sort_values("Symbol").reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved cleaned universe to: {OUTPUT_FILE}")
    print("\nSample:")
    print(df.head())


if __name__ == "__main__":
    main()