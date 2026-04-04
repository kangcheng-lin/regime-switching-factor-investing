from __future__ import annotations

import pandas as pd

from src.strategies.base.strategy_interface import BaseCrossSectionalStrategy


class FF3Strategy(BaseCrossSectionalStrategy):
    """
    Fama-French-style cross-sectional ranking strategy.

    Current implementation follows the existing project design:
    - size proxy: smaller market cap preferred
    - value proxy: lower price-to-book preferred
    - profitability proxy: higher earnings-to-price preferred

    Scores are rank-based and summed equally.
    Lower composite score is better.
    """

    @property
    def name(self) -> str:
        return "ff3"

    @property
    def score_column(self) -> str:
        return "ff3_score"

    @property
    def sort_ascending(self) -> bool:
        return True

    def build_signals(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        df = raw_df.copy()

        # Core ratios
        df["pb"] = df["market_cap"] / df["book_equity"]
        df["ep"] = df["ttm_net_income"] / df["market_cap"]

        # Signal directions
        # Smaller market cap is preferred
        df["size_signal"] = -df["market_cap"]

        # Lower price-to-book is preferred
        df["pb_signal"] = -df["pb"]

        # Higher earnings-to-price is preferred
        df["ep_signal"] = df["ep"]

        # Cross-sectional ranks
        df["rank_size"] = df["size_signal"].rank(method="first", ascending=False)
        df["rank_pb"] = df["pb_signal"].rank(method="first", ascending=False)
        df["rank_ep"] = df["ep_signal"].rank(method="first", ascending=False)

        # Equal-weight composite
        df[self.score_column] = df["rank_size"] + df["rank_pb"] + df["rank_ep"]

        return df.sort_values([self.score_column, "ticker"], ascending=[True, True]).reset_index(drop=True)