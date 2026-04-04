from __future__ import annotations

import pandas as pd

from ff3_core import FF3Config, FF3Core


class FF3Pipeline:
    """Thin orchestrator for the FF3 workflow.

    This class coordinates the full monthly/weekly process, while the
    heavy lifting lives in FF3Core.
    """

    def __init__(self, config: FF3Config):
        self.config = config
        self.core = FF3Core(config)

    def run(self) -> dict[str, pd.DataFrame]:
        calendar = self.core.build_rebalance_calendar()

        # Load price panel once before signal construction
        tickers = self.core._universe_tickers_from_lifecycle()
        self.core.load_or_download_price_panel(tickers)

        all_signals: list[pd.DataFrame] = []
        all_membership: list[pd.DataFrame] = []
        all_weights: list[pd.DataFrame] = []
        all_asset_returns: list[pd.DataFrame] = []

        for row in calendar.itertuples(index=False):
            signal_date = pd.Timestamp(row.signal_date)
            execution_date = pd.Timestamp(row.execution_date)
            next_execution_date = pd.Timestamp(row.next_execution_date)

            signals_at_t = self.core.build_signals_for_date(signal_date)
            if signals_at_t.empty:
                continue

            membership_at_t = self.core.form_portfolio_membership(
                signals_at_t=signals_at_t,
                signal_date=signal_date,
                execution_date=execution_date,
            )
            if membership_at_t.empty:
                continue

            weights_at_e = self.core.assign_weights(membership_at_t)
            if weights_at_e.empty:
                continue

            asset_returns = self.core.compute_holding_period_asset_returns(
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

        outputs = {
            "rebalance_calendar": calendar,
            "signals_at_T": self._concat_or_empty(all_signals),
            "portfolio_membership_at_T": self._concat_or_empty(all_membership),
            "weights_at_E": self._concat_or_empty(all_weights),
            "asset_returns": self._concat_or_empty(all_asset_returns),
            "portfolio_returns": self._build_portfolio_return_table(self._concat_or_empty(all_asset_returns)),
        }
        self.core.save_outputs(outputs)
        return outputs

    @staticmethod
    def _concat_or_empty(frames: list[pd.DataFrame]) -> pd.DataFrame:
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _build_portfolio_return_table(asset_returns: pd.DataFrame) -> pd.DataFrame:
        if asset_returns.empty:
            return pd.DataFrame()
        return (
            asset_returns.groupby(["signal_date", "execution_date", "next_execution_date"], as_index=False)["weighted_return"]
            .sum()
            .rename(columns={"weighted_return": "portfolio_return"})
        )


if __name__ == "__main__":
    config = FF3Config(
        lifecycle_path="data/processed/universe/price_available_universe.csv",
        balance_sheet_dir="data/processed/fundamentals_clean/balance_sheet_quarter_csv",
        income_statement_dir="data/processed/fundamentals_clean/income_statement_quarter_csv",
        market_cap_dir="data/raw/fundamentals/market_cap_history_csv",
        price_cache_path="data/raw/prices/yahoo_adjusted_close.parquet",
        calendar_path="data/processed/calendar/trading_calendar.csv",
        output_dir="results/tables/ff3_full_price_available_universe_no_fmp_v2",
        rebalance_frequency="monthly",
        long_quantile=0.20,
        short_quantile=0.20,
        long_gross=0.50,
        short_gross=0.50,
        price_batch_size=50,
        start_date="1996-01-01",
        end_date="2026-01-01",
        fmp_price_dir="data/raw/prices/fmp",
        prefer_fmp_prices=False,
        use_yahoo_fallback=True,
    )

    pipeline = FF3Pipeline(config)
    outputs = pipeline.run()
    for name, df in outputs.items():
        print(f"{name}: {df.shape}")
