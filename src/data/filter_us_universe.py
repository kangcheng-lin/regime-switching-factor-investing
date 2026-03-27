from pathlib import Path
import pandas as pd


RAW_FILE = Path("data/raw/nasdaq_nyse_amex.csv")
OUTPUT_FILE = Path("data/processed/universe/us_common_stocks.csv")


def main() -> None:
    df = pd.read_csv(RAW_FILE)

    print(f"Initial number of securities: {len(df)}")

    
    # Step 1: Start with the candidate pool we want:
    # common stock / common shares / ordinary shares

    include_pattern = (
        r"common stock|common shares|ordinary share|ordinary shares"
    )

    df = df[df["Name"].str.contains(include_pattern, case=False, na=False)].copy()

    print(f"After keeping common stock / common shares / ordinary shares: {len(df)}")

    
    # Step 2: Remove things we DO NOT want, even if they passed Step 1

    exclude_name_pattern = (
        r"adr|ads|depositary|depository|"
        r"etf|etn|mutual fund|closed-end fund|fund|trust|"
        r"reit|real estate investment trust|real estate trust|"
        r"spac|blank check|acquisition corp|acquisition corporation|"
        r"preferred|convertible preferred|"
        r"warrant|right|unit|"
        r"note|bond|senior note|convertible note|"
        r"beneficial interest"
    )

    df = df[~df["Name"].str.contains(exclude_name_pattern, case=False, na=False)].copy()

    print(f"After excluding ADR / ETF / SPAC / REIT / etc. by Name: {len(df)}")

    
    # Step 3: Extra protection using Industry column (if available)
    
    if "Industry" in df.columns:
        exclude_industry_pattern = (
            r"business development|bdc|"
            r"reit|real estate investment trust|"
            r"closed-end fund|etf|etn|trust"
        )

        df = df[
            ~df["Industry"].str.contains(exclude_industry_pattern, case=False, na=False)
        ].copy()

        print(f"After excluding BDC / REIT / fund-like Industry entries: {len(df)}")

    
    # Final cleanup
    
    df = df.sort_values("Symbol").reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved cleaned universe to: {OUTPUT_FILE}")
    print("\nSample:")
    print(df.head())


if __name__ == "__main__":
    main()