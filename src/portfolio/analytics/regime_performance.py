from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def load_strategy_portfolio_returns(
    path: str,
    strategy_name: str,
) -> pd.DataFrame:
    """
    Load one strategy's portfolio return table and add a strategy column.

    Expected columns:
    - signal_date
    - execution_date
    - next_execution_date
    - portfolio_return
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Portfolio return file not found: {file_path}")

    df = pd.read_csv(file_path)

    required = [
        "signal_date",
        "execution_date",
        "next_execution_date",
        "portfolio_return",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Strategy file {file_path} is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    out = df.copy()
    for col in ["signal_date", "execution_date", "next_execution_date"]:
        out[col] = pd.to_datetime(out[col], errors="coerce")

    out["portfolio_return"] = pd.to_numeric(out["portfolio_return"], errors="coerce")
    out["strategy"] = strategy_name

    out = out.dropna(
        subset=["signal_date", "execution_date", "next_execution_date", "portfolio_return"]
    ).reset_index(drop=True)

    ordered_cols = [
        "strategy",
        "signal_date",
        "execution_date",
        "next_execution_date",
        "portfolio_return",
    ]
    remaining_cols = [c for c in out.columns if c not in ordered_cols]
    out = out[ordered_cols + remaining_cols]

    return out


def stack_strategy_returns(
    strategy_tables: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Combine multiple strategy return tables into one long-format table.

    Parameters
    ----------
    strategy_tables : dict[str, pd.DataFrame]
        Mapping from strategy name to already-loaded DataFrame.

    Returns
    -------
    pd.DataFrame
        Long-format stacked table.
    """
    if not strategy_tables:
        return pd.DataFrame()

    frames = []
    for strategy_name, df in strategy_tables.items():
        if df.empty:
            continue

        temp = df.copy()
        if "strategy" not in temp.columns:
            temp["strategy"] = strategy_name

        frames.append(temp)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["strategy", "signal_date"]).reset_index(drop=True)
    return out


def add_excess_return(
    df: pd.DataFrame,
    strategy_return_col: str = "portfolio_return",
    benchmark_return_col: str = "spy_return",
    output_col: str = "excess_return",
) -> pd.DataFrame:
    """
    Add excess return relative to the benchmark.

    excess_return = portfolio_return - benchmark_return
    """
    required = [strategy_return_col, benchmark_return_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Input DataFrame is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    out = df.copy()
    out[strategy_return_col] = pd.to_numeric(out[strategy_return_col], errors="coerce")
    out[benchmark_return_col] = pd.to_numeric(out[benchmark_return_col], errors="coerce")
    out[output_col] = out[strategy_return_col] - out[benchmark_return_col]
    return out


def _annualized_return_from_simple_returns(
    returns: pd.Series,
    periods_per_year: int,
) -> float:
    returns = pd.to_numeric(returns, errors="coerce").dropna()
    n = len(returns)
    if n == 0:
        return np.nan

    total_growth = (1.0 + returns).prod()
    if total_growth <= 0:
        return np.nan

    return total_growth ** (periods_per_year / n) - 1.0


def _annualized_volatility(
    returns: pd.Series,
    periods_per_year: int,
) -> float:
    returns = pd.to_numeric(returns, errors="coerce").dropna()
    if len(returns) < 2:
        return np.nan
    return returns.std(ddof=1) * np.sqrt(periods_per_year)


def _subset_cumulative_return(
    returns: pd.Series,
) -> float:
    returns = pd.to_numeric(returns, errors="coerce").dropna()
    if len(returns) == 0:
        return np.nan
    return (1.0 + returns).prod() - 1.0


def _mean_return(
    returns: pd.Series,
) -> float:
    returns = pd.to_numeric(returns, errors="coerce").dropna()
    if len(returns) == 0:
        return np.nan
    return returns.mean()


def compute_regime_summary(
    df: pd.DataFrame,
    strategy_col: str = "strategy",
    regime_col: str = "regime",
    return_col: str = "portfolio_return",
    benchmark_return_col: str = "spy_return",
    excess_return_col: str = "excess_return",
    periods_per_year: int = 12,
) -> pd.DataFrame:
    """
    Compute summary metrics by strategy x regime.

    Notes
    -----
    The subset cumulative return is computed within each regime subset:
        (1 + r).prod() - 1
    This is a conditional summary statistic, not a continuous tradable path.
    """
    required = [
        strategy_col,
        regime_col,
        return_col,
        benchmark_return_col,
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Input DataFrame is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    work = df.copy()

    if excess_return_col not in work.columns:
        work = add_excess_return(
            work,
            strategy_return_col=return_col,
            benchmark_return_col=benchmark_return_col,
            output_col=excess_return_col,
        )

    group_cols = [strategy_col, regime_col]
    rows = []

    for keys, group in work.groupby(group_cols, dropna=False):
        strategy_value, regime_value = keys

        strat_returns = pd.to_numeric(group[return_col], errors="coerce").dropna()
        spy_returns = pd.to_numeric(group[benchmark_return_col], errors="coerce").dropna()
        excess_returns = pd.to_numeric(group[excess_return_col], errors="coerce").dropna()

        n_periods = len(strat_returns)
        mean_return = _mean_return(strat_returns)
        ann_return = _annualized_return_from_simple_returns(strat_returns, periods_per_year)
        ann_vol = _annualized_volatility(strat_returns, periods_per_year)

        if pd.notna(ann_vol) and ann_vol != 0 and pd.notna(ann_return):
            sharpe = ann_return / ann_vol
        else:
            sharpe = np.nan

        subset_cum_return = _subset_cumulative_return(strat_returns)
        mean_spy_return = _mean_return(spy_returns)
        mean_excess_return = _mean_return(excess_returns)

        rows.append(
            {
                strategy_col: strategy_value,
                regime_col: regime_value,
                "n_periods": n_periods,
                "mean_return": mean_return,
                "ann_return": ann_return,
                "ann_vol": ann_vol,
                "sharpe": sharpe,
                "subset_cum_return": subset_cum_return,
                "mean_spy_return": mean_spy_return,
                "mean_excess_return": mean_excess_return,
            }
        )

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    out = out.sort_values([strategy_col, regime_col]).reset_index(drop=True)
    return out