from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from src.strategies.base.cross_sectional_base import CrossSectionalBacktestBase


@dataclass
class BacktestResult:
    """
    Container for all standard backtest outputs.
    """
    rebalance_calendar: pd.DataFrame
    signals_at_T: pd.DataFrame
    portfolio_membership_at_T: pd.DataFrame
    weights_at_E: pd.DataFrame
    asset_returns: pd.DataFrame
    portfolio_returns: pd.DataFrame

    def to_dict(self) -> Dict[str, pd.DataFrame]:
        return {
            "rebalance_calendar": self.rebalance_calendar,
            "signals_at_T": self.signals_at_T,
            "portfolio_membership_at_T": self.portfolio_membership_at_T,
            "weights_at_E": self.weights_at_E,
            "asset_returns": self.asset_returns,
            "portfolio_returns": self.portfolio_returns,
        }


class BacktestEngine:
    """
    Generic backtest engine for cross-sectional strategies.

    The engine assumes the supplied base object already knows how to:
    - build the rebalance calendar
    - load price data
    - build signals for a date
    - form portfolio membership
    - assign weights
    - compute holding-period returns
    - save outputs
    """

    def __init__(self, base: CrossSectionalBacktestBase):
        self.base = base

    def run(self, save_outputs: bool = True) -> BacktestResult:
        calendar = self.base.build_rebalance_calendar()

        # Load price panel once before signal construction
        tickers = self.base.universe_tickers_from_lifecycle()
        self.base.load_or_download_price_panel(tickers)

        all_signals: List[pd.DataFrame] = []
        all_membership: List[pd.DataFrame] = []
        all_weights: List[pd.DataFrame] = []
        all_asset_returns: List[pd.DataFrame] = []

        for row in calendar.itertuples(index=False):
            signal_date = pd.Timestamp(row.signal_date)
            execution_date = pd.Timestamp(row.execution_date)
            next_execution_date = pd.Timestamp(row.next_execution_date)

            signals_at_t = self.base.build_signals_for_date(signal_date)
            if signals_at_t.empty:
                continue

            membership_at_t = self.base.form_portfolio_membership(
                signals_at_t=signals_at_t,
                signal_date=signal_date,
                execution_date=execution_date,
            )
            if membership_at_t.empty:
                continue

            weights_at_e = self.base.assign_weights(membership_at_t)
            if weights_at_e.empty:
                continue

            asset_returns = self.base.compute_holding_period_asset_returns(
                weights_at_e=weights_at_e,
                execution_date=execution_date,
                next_execution_date=next_execution_date,
            )
            if asset_returns.empty:
                continue

            all_signals.append(signals_at_t)
            all_membership.append(membership_at_t)
            all_weights.append(weights_at_e)
            all_asset_returns.append(asset_returns)

        asset_returns_df = self._concat_or_empty(all_asset_returns)

        result = BacktestResult(
            rebalance_calendar=calendar,
            signals_at_T=self._concat_or_empty(all_signals),
            portfolio_membership_at_T=self._concat_or_empty(all_membership),
            weights_at_E=self._concat_or_empty(all_weights),
            asset_returns=asset_returns_df,
            portfolio_returns=self._build_portfolio_return_table(asset_returns_df),
        )

        if save_outputs:
            self.base.save_outputs(result.to_dict())

        return result

    @staticmethod
    def _concat_or_empty(frames: List[pd.DataFrame]) -> pd.DataFrame:
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _build_portfolio_return_table(asset_returns: pd.DataFrame) -> pd.DataFrame:
        if asset_returns.empty:
            return pd.DataFrame()

        return (
            asset_returns.groupby(
                ["signal_date", "execution_date", "next_execution_date"],
                as_index=False,
            )["weighted_return"]
            .sum()
            .rename(columns={"weighted_return": "portfolio_return"})
        )