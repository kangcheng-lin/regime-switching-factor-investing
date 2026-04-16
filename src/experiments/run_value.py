from __future__ import annotations

from src.backtest.engine import BacktestEngine
from src.strategies.base.cross_sectional_base import (
    CrossSectionalBacktestBase,
    CrossSectionalConfig,
)
from src.strategies.cross_sectional.value import ValueStrategy


def main() -> None:
    # Universe control:
    use_top_n = True
    universe_top_n = 1000
    sort_field = "market_cap"
    if use_top_n:
        universe_tag = f"top{universe_top_n}_{sort_field}"
    else:
        universe_tag = "full_universe"
    
    # Portfolio mode:
        # - long_short: build long and short legs
        # - long_only: only take long positions
    portfolio_mode="long_only"
    long_n = 50
    short_n = 50
    selection_n = 50
    if portfolio_mode == "long_short":
        selection_tag = f"top{long_n}_long_top{short_n}_short"
    else:
        selection_tag = f"top{selection_n}_long_only"

    config = CrossSectionalConfig(
        lifecycle_path="data/processed/universe/price_available_universe.csv",
        balance_sheet_dir="data/processed/fundamentals_clean/balance_sheet_quarter_csv",
        income_statement_dir="data/processed/fundamentals_clean/income_statement_quarter_csv",
        cash_flow_dir="data/processed/fundamentals_clean/cash_flow_quarter_csv",
        market_cap_dir="data/raw/fundamentals/market_cap_history_csv",
        price_cache_path="data/raw/prices/yahoo_adjusted_close.parquet",
        calendar_path="data/processed/calendar/trading_calendar.csv",
        output_dir=f"results/tables/value_{universe_tag}_{selection_tag}",
        
        rebalance_frequency="monthly", # monthly or weekly
        
        # Portfolio mode:
        # - long_short: build long and short legs
        # - long_only: only take long positions
        portfolio_mode=portfolio_mode,
        
        selection_mode="top_n", # "quantile" or "top_n"

        # Mode 1: quantile-based (e.g., top 20%), unused when selection_mode="top_n"
        long_quantile=0.20,
        short_quantile=0.20,
        selection_quantile=0.20,
        
        # Mode 2: fixed-count selection (e.g., top 50)
        long_n=long_n,
        short_n=short_n,
        selection_n=selection_n, # used for long_only mode

        # Gross exposure per side
        long_gross=0.50,
        short_gross=0.50,

        min_ttm_quarters=4, # minimum quarters required for TTM metrics
        price_batch_size=50,
        start_date="2006-01-01",
        end_date="2026-01-01",
        fmp_price_dir="data/raw/prices/fmp",
        prefer_fmp_prices=False,
        use_yahoo_fallback=True,
        
        # Generic tradability filters
        min_price=5.0,
        min_market_cap=1e6,
        min_book_equity=1e6,
        allow_negative_earnings=True,

        use_top_n_universe=use_top_n, # If True, restrict universe to top N by a given field (point-in-time)
        top_n=universe_top_n, # Number of stocks to keep (e.g., 2000 largest)
        universe_sort_field=sort_field, # Field used to rank universe (currently only "market_cap")
    )

    strategy = ValueStrategy()
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