from __future__ import annotations

from src.backtest.engine import BacktestEngine
from src.strategies.base.cross_sectional_base import (
    CrossSectionalBacktestBase,
    CrossSectionalConfig,
)
from src.strategies.cross_sectional.carhart4 import Carhart4Strategy


def main() -> None:
    config = CrossSectionalConfig(
        lifecycle_path="data/processed/universe/price_available_universe.csv",
        balance_sheet_dir="data/processed/fundamentals_clean/balance_sheet_quarter_csv",
        income_statement_dir="data/processed/fundamentals_clean/income_statement_quarter_csv",
        market_cap_dir="data/raw/fundamentals/market_cap_history_csv",
        price_cache_path="data/raw/prices/yahoo_adjusted_close.parquet",
        calendar_path="data/processed/calendar/trading_calendar.csv",
        output_dir="results/tables/carhart4_full_price_available_universe_modular",
        rebalance_frequency="monthly",
        long_quantile=0.20,
        short_quantile=0.20,
        long_gross=0.50,
        short_gross=0.50,
        min_ttm_quarters=4,
        price_batch_size=50,
        start_date="1996-01-01",
        end_date="2026-01-01",
        fmp_price_dir="data/raw/prices/fmp",
        prefer_fmp_prices=False,
        use_yahoo_fallback=True,
        min_price=5.0,
        min_market_cap=1e6,
        min_book_equity=1e6,
        allow_negative_earnings=True,
    )

    strategy = Carhart4Strategy()
    base = CrossSectionalBacktestBase(config=config, strategy=strategy)
    engine = BacktestEngine(base=base)

    result = engine.run(save_outputs=True)

    print(f"Strategy: {strategy.name}")
    print(f"rebalance_calendar: {result.rebalance_calendar.shape}")
    print(f"signals_at_T: {result.signals_at_T.shape}")
    print(f"portfolio_membership_at_T: {result.portfolio_membership_at_T.shape}")
    print(f"weights_at_E: {result.weights_at_E.shape}")
    print(f"asset_returns: {result.asset_returns.shape}")
    print(f"portfolio_returns: {result.portfolio_returns.shape}")


if __name__ == "__main__":
    main()