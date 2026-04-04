import pandas as pd
import yfinance as yf
from pathlib import Path


def build_trading_calendar(
    start_date: str = "2000-01-01",
    end_date: str = None,
    output_path: str = "data/processed/calendar/trading_calendar.csv",
):
    """
    Build a universal US trading calendar using SPY prices.

    Output columns:
    - trading_date
    - year
    - month
    - week
    - is_month_end
    - is_week_end
    - next_trading_date
    """

    print("Downloading SPY data from Yahoo Finance...")

    spy = yf.download(
        "SPY",
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=False,
    )

    if spy.empty:
        raise ValueError("Failed to download SPY data.")

    df = pd.DataFrame(index=pd.to_datetime(spy.index))
    df = df.sort_index()

    # Core columns
    df["trading_date"] = df.index
    df["year"] = df.index.year
    df["month"] = df.index.month
    df["week_end"] = df.index.to_period("W-FRI").end_time.normalize()

    # Identify month-end trading days
    df["is_month_end"] = False
    month_end_idx = df.groupby(["year", "month"]).tail(1).index
    df.loc[month_end_idx, "is_month_end"] = True

    # Identify week-end trading days
    df["is_week_end"] = False
    week_end_idx = df.groupby("week_end").tail(1).index
    df.loc[week_end_idx, "is_week_end"] = True

    # Next trading day (for execution)
    df["next_trading_date"] = df["trading_date"].shift(-1)

    # Drop last row (no next trading day)
    df = df.dropna(subset=["next_trading_date"])

    # Reset index
    df = df.reset_index(drop=True)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"Trading calendar saved to: {output_path}")
    print(f"Total trading days: {len(df)}")

    return df


if __name__ == "__main__":
    calendar = build_trading_calendar(
        start_date="2000-01-01",
        end_date=None,
    )

    print(calendar.head())
    print(calendar.tail())