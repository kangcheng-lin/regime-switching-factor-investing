from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_regime_table(
    path: str,
    date_col: str = "date",
    regime_col: str = "regime",
    price_col: str = "adj_close",
) -> pd.DataFrame:
    """
    Load the SPY regime table and standardize required columns.

    Expected raw columns:
    - date
    - regime
    - adj_close

    Returns
    -------
    pd.DataFrame
        Columns: date, regime, adj_close
        Sorted by date, deduplicated on date.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Regime file not found: {file_path}")

    df = pd.read_csv(file_path)

    required = [date_col, regime_col, price_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Regime file is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    out = df[[date_col, regime_col, price_col]].copy()
    out.columns = ["date", "regime", "adj_close"]

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["adj_close"] = pd.to_numeric(out["adj_close"], errors="coerce")

    out = out.dropna(subset=["date", "regime", "adj_close"]).copy()
    out = out.drop_duplicates(subset=["date"], keep="last")
    out = out.sort_values("date").reset_index(drop=True)

    return out


def assign_regime_to_returns(
    returns_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    assignment_date_col: str = "signal_date",
    drop_missing_regime: bool = False,
) -> pd.DataFrame:
    """
    Attach regime labels to portfolio return rows using the chosen date column.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Strategy portfolio return table. Must contain assignment_date_col.
    regime_df : pd.DataFrame
        Output of load_regime_table(), must contain columns: date, regime.
    assignment_date_col : str
        Column in returns_df used for regime assignment.
        Default is signal_date to avoid look-ahead bias.
    drop_missing_regime : bool
        If True, drop rows that fail to match a regime.
        If False, keep them and leave regime as NaN for inspection.

    Returns
    -------
    pd.DataFrame
        Original returns_df plus:
        - assigned_regime_date
        - regime
    """
    if assignment_date_col not in returns_df.columns:
        raise ValueError(
            f"returns_df must contain {assignment_date_col!r}. "
            f"Available columns: {list(returns_df.columns)}"
        )

    required_regime_cols = {"date", "regime"}
    missing = required_regime_cols - set(regime_df.columns)
    if missing:
        raise ValueError(
            f"regime_df must contain columns {sorted(required_regime_cols)}. "
            f"Missing: {sorted(missing)}"
        )

    out = returns_df.copy()
    out[assignment_date_col] = pd.to_datetime(out[assignment_date_col], errors="coerce")

    regime_map = regime_df[["date", "regime"]].copy()
    regime_map = regime_map.rename(columns={"date": assignment_date_col})
    regime_map = regime_map.drop_duplicates(subset=[assignment_date_col], keep="last")

    out = out.merge(
        regime_map,
        on=assignment_date_col,
        how="left",
        validate="m:1",
    )

    out["assigned_regime_date"] = out[assignment_date_col]

    ordered_cols = list(returns_df.columns) + ["assigned_regime_date", "regime"]
    out = out[ordered_cols]

    if drop_missing_regime:
        out = out.dropna(subset=["regime"]).reset_index(drop=True)

    return out