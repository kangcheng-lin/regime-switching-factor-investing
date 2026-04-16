from __future__ import annotations

from pandas import Timestamp

from src.strategies.base.cross_sectional_base import (
    CrossSectionalBacktestBase,
    CrossSectionalConfig,
)
from src.strategies.cross_sectional.aqr import AQRStrategy


def inspect_pipeline(
    base: CrossSectionalBacktestBase,
    strategy: AQRStrategy,
    test_date: Timestamp,
    use_top_n: bool,
    top_n: int,
) -> dict[str, object]:
    base.config.use_top_n_universe = use_top_n
    base.config.top_n = top_n

    raw_df = base.build_raw_snapshot_for_date(test_date)
    enriched_df = strategy.enrich_raw_snapshot(base, raw_df.copy(), test_date)
    generic_filtered_df = base.apply_signal_filters(enriched_df)
    universe_filtered_df = base.apply_top_n_universe_filter(generic_filtered_df)

    print("duplicate tickers after generic filters:", generic_filtered_df["ticker"].duplicated().sum())
    print("duplicate tickers after top-N filter:", universe_filtered_df["ticker"].duplicated().sum())

    print("unique tickers after generic filters:", generic_filtered_df["ticker"].nunique())
    print("rows after generic filters:", len(generic_filtered_df))

    print("unique tickers after top-N filter:", universe_filtered_df["ticker"].nunique())
    print("rows after top-N filter:", len(universe_filtered_df))

    if universe_filtered_df.empty:
        signals_df = universe_filtered_df.copy()
    else:
        signals_df = strategy.build_signals(universe_filtered_df.copy())

    print(f"use_top_n_universe={use_top_n}, top_n={top_n}")
    print("raw snapshot rows:", len(raw_df))
    print("after generic filters:", len(generic_filtered_df))
    print("after apply_top_n_universe_filter:", len(universe_filtered_df))
    print("after strategy.build_signals:", len(signals_df))

    if not universe_filtered_df.empty:
        print()
        print("Top 5 after apply_top_n_universe_filter:")
        print(
            universe_filtered_df.sort_values("market_cap", ascending=False)[
                ["ticker", "market_cap"]
            ].head()
        )

        print()
        print("Bottom 5 after apply_top_n_universe_filter:")
        print(
            universe_filtered_df.sort_values("market_cap", ascending=True)[
                ["ticker", "market_cap"]
            ].head()
        )

    return {
        "raw_df": raw_df,
        "enriched_df": enriched_df,
        "generic_filtered_df": generic_filtered_df,
        "universe_filtered_df": universe_filtered_df,
        "signals_df": signals_df,
    }


def main() -> None:
    strategy = AQRStrategy()

    config = CrossSectionalConfig(
        lifecycle_path="data/processed/universe/price_available_universe.csv",
        balance_sheet_dir="data/processed/fundamentals_clean/balance_sheet_quarter_csv",
        income_statement_dir="data/processed/fundamentals_clean/income_statement_quarter_csv",
        cash_flow_dir="data/processed/fundamentals_clean/cash_flow_quarter_csv",
        market_cap_dir="data/raw/fundamentals/market_cap_history_csv",
        price_cache_path="data/raw/prices/yahoo_adjusted_close.parquet",
        calendar_path="data/processed/calendar/trading_calendar.csv",
        output_dir="results/tables/debug_topn_aqr",
        rebalance_frequency="monthly",
        portfolio_mode="long_short",
        long_quantile=0.20,
        short_quantile=0.20,
        selection_quantile=0.20,
        selection_mode="top_n",
        long_n=50,
        short_n=50,
        selection_n=50,
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
        use_top_n_universe=False,
        top_n=2000,
        universe_sort_field="market_cap",
    )

    base = CrossSectionalBacktestBase(config=config, strategy=strategy)

    tickers = base.universe_tickers_from_lifecycle()
    base.load_or_download_price_panel(tickers)

    calendar = base.build_rebalance_calendar()
    test_date = calendar.loc[calendar["signal_date"] >= "2010-01-01", "signal_date"].iloc[0]
    print("Test date:", test_date)
    print()

    print("=" * 60)
    print("CASE 1: full universe")
    print("=" * 60)
    full_out = inspect_pipeline(
        base=base,
        strategy=strategy,
        test_date=test_date,
        use_top_n=False,
        top_n=500,
    )

    print()
    print("=" * 60)
    print("CASE 2: top-N universe")
    print("=" * 60)
    topn_out = inspect_pipeline(
        base=base,
        strategy=strategy,
        test_date=test_date,
        use_top_n=True,
        top_n=500,
    )


if __name__ == "__main__":
    main()