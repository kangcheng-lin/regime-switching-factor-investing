from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.base.cross_sectional_base import CrossSectionalBacktestBase
from src.strategies.base.strategy_interface import BaseCrossSectionalStrategy


class ValueStrategy(BaseCrossSectionalStrategy):
    """
    Paper-faithful Value strategy (long-only in portfolio construction).

    Signals:
    - higher dividend-per-share proxy
    - lower price-to-book
    - higher free cash flow yield
    - more negative rebalance-to-rebalance return (mean reversion)
    """

    @property
    def name(self) -> str:
        return "value"

    @property
    def score_column(self) -> str:
        return "value_score"

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

        # shares outstanding proxy = market cap / price
        df["shares_outstanding_proxy"] = np.where(
            (df["adj_close_t"].notna()) & (df["adj_close_t"] != 0),
            df["market_cap"] / df["adj_close_t"],
            np.nan,
        )

        # Dividend-per-share proxy:
        # commonDividendsPaid is usually negative cash outflow, so take absolute value
        df["dps_proxy"] = np.where(
            (df["shares_outstanding_proxy"].notna()) & (df["shares_outstanding_proxy"] != 0),
            np.abs(df["ttm_common_dividends_paid"]) / df["shares_outstanding_proxy"],
            np.nan,
        )

        return df

    def build_signals(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        df = raw_df.copy()

        # Strategy-specific required fields
        required = [
            "market_cap",
            "book_equity",
            "ttm_free_cash_flow",
            "ttm_common_dividends_paid",
            "mom_1m",
            "dps_proxy",
        ]
        df = df.dropna(subset=required).copy()

        if df.empty:
            df[self.score_column] = pd.Series(dtype=float)
            return df

        # Core ratios / signals
        df["pb"] = df["market_cap"] / df["book_equity"]
        df["pb_signal"] = -df["pb"]  # lower P/B is better

        df["fcf_yield"] = df["ttm_free_cash_flow"] / df["market_cap"]
        df["fcf_yield_signal"] = df["fcf_yield"]  # higher is better

        df["dividend_signal"] = df["dps_proxy"]  # higher dividend/share is better

        # More negative past return is preferred for mean reversion
        df["reversal_signal"] = -df["mom_1m"]

        # Ranks
        df["rank_dividend"] = df["dividend_signal"].rank(method="first", ascending=False)
        df["rank_pb"] = df["pb_signal"].rank(method="first", ascending=False)
        df["rank_fcf"] = df["fcf_yield_signal"].rank(method="first", ascending=False)
        df["rank_reversal"] = df["reversal_signal"].rank(method="first", ascending=False)

        df[self.score_column] = (
            df["rank_dividend"]
            + df["rank_pb"]
            + df["rank_fcf"]
            + df["rank_reversal"]
        )

        return df.sort_values([self.score_column, "ticker"], ascending=[True, True]).reset_index(drop=True)