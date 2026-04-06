from __future__ import annotations

import pandas as pd

from src.strategies.base.cross_sectional_base import CrossSectionalBacktestBase
from src.strategies.base.strategy_interface import BaseCrossSectionalStrategy


class AQRStrategy(BaseCrossSectionalStrategy):
    """
    Simplified AQR-style factor model for this project.

    Signals:
    - Value: higher book_equity / market_cap
    - Quality: higher operatingIncome / revenue
    - Momentum: higher rebalance-to-rebalance return

    Portfolio:
    - long-short
    - equal-weight composite rank
    """

    @property
    def name(self) -> str:
        return "aqr"

    @property
    def score_column(self) -> str:
        return "aqr_score"

    @property
    def sort_ascending(self) -> bool:
        return True

    def enrich_raw_snapshot(
        self,
        base: CrossSectionalBacktestBase,
        raw_df: pd.DataFrame,
        signal_date: pd.Timestamp,
    ) -> pd.DataFrame:
        df = raw_df.copy()

        df["mom_1m"] = df["ticker"].map(
            lambda t: base.get_rebalance_return(ticker=t, signal_date=signal_date)
        )

        return df

    def build_signals(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        df = raw_df.copy()

        # Strategy-specific sanity filters
        df = df[
            (df["book_equity"] > 1e6) &
            (df["market_cap"] > 1e6) &
            (df["revenue"] > 1e6) &
            (df["operatingIncome"].notna()) &
            (df["operatingIncome"] != 0)
        ].copy()

        df = df.dropna(subset=["mom_1m"]).copy()

        if df.empty:
            df[self.score_column] = pd.Series(dtype=float)
            return df

        # Signals
        df["value_signal"] = df["book_equity"] / df["market_cap"]
        df["quality_signal"] = df["operatingIncome"] / df["revenue"]
        df["momentum_signal"] = df["mom_1m"]

        # Ranks: higher is better for all three, so ascending=False
        df["rank_value"] = df["value_signal"].rank(method="first", ascending=False)
        df["rank_quality"] = df["quality_signal"].rank(method="first", ascending=False)
        df["rank_momentum"] = df["momentum_signal"].rank(method="first", ascending=False)

        df[self.score_column] = (
            df["rank_value"]
            + df["rank_quality"]
            + df["rank_momentum"]
        )

        return df.sort_values([self.score_column, "ticker"], ascending=[True, True]).reset_index(drop=True)