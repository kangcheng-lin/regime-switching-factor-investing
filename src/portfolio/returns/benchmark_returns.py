from __future__ import annotations

import pandas as pd


def build_benchmark_price_map(
    regime_df: pd.DataFrame,
    date_col: str = "date",
    price_col: str = "adj_close",
) -> pd.Series:
    """
    Build a date -> adjusted close lookup series for the benchmark.

    Parameters
    ----------
    regime_df : pd.DataFrame
        Typically the standardized SPY regime table.
    date_col : str
        Date column name in regime_df.
    price_col : str
        Adjusted close column name in regime_df.

    Returns
    -------
    pd.Series
        Index = pd.Timestamp dates
        Values = adjusted close prices
    """
    required = [date_col, price_col]
    missing = [col for col in required if col not in regime_df.columns]
    if missing:
        raise ValueError(
            f"regime_df is missing required columns: {missing}. "
            f"Available columns: {list(regime_df.columns)}"
        )

    prices = regime_df[[date_col, price_col]].copy()
    prices[date_col] = pd.to_datetime(prices[date_col], errors="coerce")
    prices[price_col] = pd.to_numeric(prices[price_col], errors="coerce")

    prices = prices.dropna(subset=[date_col, price_col]).copy()
    prices = prices.drop_duplicates(subset=[date_col], keep="last")
    prices = prices.sort_values(date_col)

    price_series = prices.set_index(date_col)[price_col]
    price_series.name = "benchmark_adj_close"

    return price_series


def attach_benchmark_returns(
    returns_df: pd.DataFrame,
    benchmark_prices: pd.Series,
    execution_col: str = "execution_date",
    exit_col: str = "next_execution_date",
    benchmark_name: str = "spy",
    drop_missing_prices: bool = False,
) -> pd.DataFrame:
    """
    Attach benchmark holding-period returns to each strategy return row.

    Benchmark return is computed exactly like the strategy holding-period return:
        benchmark_return = exit_price / entry_price - 1

    Parameters
    ----------
    returns_df : pd.DataFrame
        Portfolio return table with execution and exit dates.
    benchmark_prices : pd.Series
        Date-indexed adjusted close prices.
    execution_col : str
        Entry date column in returns_df.
    exit_col : str
        Exit date column in returns_df.
    benchmark_name : str
        Prefix for added columns, e.g. 'spy'.
    drop_missing_prices : bool
        If True, drop rows with missing benchmark entry/exit price.

    Returns
    -------
    pd.DataFrame
        Original returns_df plus:
        - {benchmark_name}_entry_price
        - {benchmark_name}_exit_price
        - {benchmark_name}_return
    """
    for col in [execution_col, exit_col]:
        if col not in returns_df.columns:
            raise ValueError(
                f"returns_df must contain {col!r}. "
                f"Available columns: {list(returns_df.columns)}"
            )

    if not isinstance(benchmark_prices, pd.Series):
        raise TypeError("benchmark_prices must be a pandas Series.")

    if benchmark_prices.empty:
        raise ValueError("benchmark_prices is empty.")

    out = returns_df.copy()
    out[execution_col] = pd.to_datetime(out[execution_col], errors="coerce")
    out[exit_col] = pd.to_datetime(out[exit_col], errors="coerce")

    entry_col_name = f"{benchmark_name}_entry_price"
    exit_col_name = f"{benchmark_name}_exit_price"
    return_col_name = f"{benchmark_name}_return"

    out[entry_col_name] = out[execution_col].map(benchmark_prices)
    out[exit_col_name] = out[exit_col].map(benchmark_prices)

    valid_mask = (
        out[entry_col_name].notna()
        & out[exit_col_name].notna()
        & (out[entry_col_name] != 0)
    )

    out[return_col_name] = pd.NA
    out.loc[valid_mask, return_col_name] = (
        out.loc[valid_mask, exit_col_name] / out.loc[valid_mask, entry_col_name] - 1.0
    )

    out[return_col_name] = pd.to_numeric(out[return_col_name], errors="coerce")

    if drop_missing_prices:
        out = out.dropna(subset=[entry_col_name, exit_col_name, return_col_name]).reset_index(drop=True)

    return out