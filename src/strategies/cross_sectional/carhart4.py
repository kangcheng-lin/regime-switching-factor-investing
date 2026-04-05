from __future__ import annotations

import pandas as pd

from src.strategies.base.cross_sectional_base import CrossSectionalBacktestBase
from src.strategies.base.strategy_interface import BaseCrossSectionalStrategy


class Carhart4Strategy(BaseCrossSectionalStrategy):
    """
    Carhart Four-Factor Model:
    - Size (small cap preferred)
    - Value (low P/B preferred)
    - Profitability (high E/P preferred)
    - Momentum (high 1-month return preferred)
    """

    @property
    def name(self) -> str:
        return "carhart4"

    @property
    def score_column(self) -> str:
        return "carhart4_score"

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

        df = df.dropna(subset=["mom_1m"]).copy()
        if df.empty:
            df[self.score_column] = pd.Series(dtype=float)
            return df

        # Core ratios
        df["pb"] = df["market_cap"] / df["book_equity"]
        df["ep"] = df["ttm_net_income"] / df["market_cap"]

        # Signals
        df["size_signal"] = -df["market_cap"]
        df["pb_signal"] = -df["pb"]
        df["ep_signal"] = df["ep"]
        df["mom_signal"] = df["mom_1m"]

        # Ranks
        df["rank_size"] = df["size_signal"].rank(method="first", ascending=False)
        df["rank_pb"] = df["pb_signal"].rank(method="first", ascending=False)
        df["rank_ep"] = df["ep_signal"].rank(method="first", ascending=False)
        df["rank_mom"] = df["mom_signal"].rank(method="first", ascending=False)

        # Equal-weight composite (4 factors now)
        df[self.score_column] = (
            df["rank_size"]
            + df["rank_pb"]
            + df["rank_ep"]
            + df["rank_mom"]
        )

        return df.sort_values([self.score_column, "ticker"], ascending=[True, True]).reset_index(drop=True)