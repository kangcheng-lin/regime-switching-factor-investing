from pathlib import Path
import pandas as pd


RAW_FILE = Path("data/raw/nasdaq_nyse_amex.csv")
OUTPUT_FILE = Path("data/processed/universe/us_common_stocks.csv")


def main() -> None:
    df = pd.read_csv(RAW_FILE)

    print(f"Initial number of securities: {len(df)}")

    # Step 1: Keep common / ordinary shares (singular + plural)
    include_pattern = r"common stocks?|common shares?|ordinary shares?"
    df = df[df["Name"].str.contains(include_pattern, case=False, na=False)].copy()

    print(f"After including common/ordinary shares: {len(df)}")

    # Step 2: Exclude ADR / ADS
    # Covers:
    # - American Depositary Shares
    # - ADR / ADS abbreviations
    exclude_pattern = r"depositary|adr|ads"
    df = df[~df["Name"].str.contains(exclude_pattern, case=False, na=False)].copy()

    print(f"After excluding ADR/ADS: {len(df)}")

    df = df.sort_values("Symbol").reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved cleaned universe to: {OUTPUT_FILE}")
    print("\nSample:")
    print(df.head())


if __name__ == "__main__":
    main()