from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from src.strategies.base.cross_sectional_base import CrossSectionalBacktestBase


class BaseCrossSectionalStrategy(ABC):
    """
    Interface for cross-sectional stock-selection strategies.

    Each strategy should:
    1. Optionally enrich the raw point-in-time cross section with additional features
       (e.g. 1-month return, operating margin, free cash flow yield, etc.)
    2. Build ranked/scored signals from that cross section
    3. Return a DataFrame that includes a strategy-specific score column
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""
        raise NotImplementedError

    @property
    @abstractmethod
    def score_column(self) -> str:
        """
        Column used for portfolio ranking.
        Lower values are assumed to be better unless sort_ascending=False.
        """
        raise NotImplementedError

    @property
    def sort_ascending(self) -> bool:
        """
        Whether better scores are lower values.

        Example:
        - FF3 composite rank sum: lower is better -> True
        - A hypothetical model where higher score is better -> False
        """
        return True

    def enrich_raw_snapshot(
        self,
        base: "CrossSectionalBacktestBase",
        raw_df: pd.DataFrame,
        signal_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Optional hook for adding strategy-specific raw features before scoring.

        Default behavior: no extra features added.

        Examples of future use:
        - Modified FF: add 1-month price change
        - Carhart: add momentum factor
        - AQR: add operating-income-to-revenue
        """
        return raw_df

    @abstractmethod
    def build_signals(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert a raw filtered cross-section into scored/ranked strategy signals.

        Must return a DataFrame containing self.score_column.
        """
        raise NotImplementedError