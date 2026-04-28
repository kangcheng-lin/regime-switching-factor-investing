from AlgorithmImports import *


class ValueFactorMonthly(QCAlgorithm):
    """
    First QC port of the value strategy.

    Mapping to the current engine:
    - run_value.py config              -> Initialize()
    - engine loop                      -> QC scheduler
    - portfolio return calculation     -> handled by QC automatically
    """

    def Initialize(self):
        # ============================================================
        # 1) Backtest configuration
        # Corresponds to: run_value.py / CrossSectionalConfig
        # ============================================================
        self.SetStartDate(2006, 1, 1)
        self.SetEndDate(2026, 1, 1)
        self.SetCash(100000)

        # Optional: keep brokerage model simple for now
        self.SetBrokerageModel(BrokerageName.Default, AccountType.Margin)

        # ============================================================
        # 2) Strategy parameters
        # Corresponds to: run_value.py settings
        # ============================================================
        self.top_universe_n = 1000     # top 1000 by market cap approx
        self.selection_n = 50          # long-only top 50
        self.min_price = 5.0
        self.min_market_cap = 1e6
        self.min_book_equity = 1e6

        # ============================================================
        # 3) Internal state
        # These replace parts of the engine tables
        # ============================================================
        self.rebalance_flag = False
        self.pending_symbols = []
        self.current_universe = []
        self.latest_fine = {}

        # ============================================================
        # 4) Universe settings
        # QC-native approximation of the universe pipeline
        # ============================================================
        self.UniverseSettings.Resolution = Resolution.Daily
        self.UniverseSettings.DataNormalizationMode = DataNormalizationMode.Adjusted

        self.AddUniverse(self.CoarseSelectionFunction, self.FineSelectionFunction)

        # ============================================================
        # 5) Scheduling
        # We will later preserve the timing:
        # signal on last trading day of month
        # execute on first trading day of next month
        # ============================================================
        self.AddEquity("SPY", Resolution.Daily)
        self.spy = self.Symbol("SPY")

        self.Schedule.On(
            self.DateRules.MonthEnd(self.spy),
            self.TimeRules.AfterMarketOpen(self.spy, 30),
            self.ComputeMonthEndSignals
        )

        self.Schedule.On(
            self.DateRules.MonthStart(self.spy),
            self.TimeRules.AfterMarketOpen(self.spy, 30),
            self.ExecuteMonthStartRebalance
        )

        self.Debug("Initialize complete.")

    def CoarseSelectionFunction(self, coarse):
        """
        First-pass QC-native universe filter.

        Rough mapping to:
        - lifecycle / price-available universe
        - generic price filter
        """
        filtered = [
            c for c in coarse
            if c.HasFundamentalData
            and c.Price is not None
            and c.Price > self.min_price
        ]

        # Use dollar volume here only as a coarse prefilter for liquidity.
        # Final top-1000 approximation will be done in FineSelectionFunction.
        top_liquid = sorted(filtered, key=lambda c: c.DollarVolume, reverse=True)[:3000]
        return [c.Symbol for c in top_liquid]

    def FineSelectionFunction(self, fine):
        """
        Second-pass filter.

        Rough mapping to:
        - apply_signal_filters()
        - apply_top_n_universe_filter()
        """
        candidates = []

        for f in fine:
            try:
                market_cap = f.MarketCap
                book_equity = f.FinancialStatements.BalanceSheet.TotalEquity.Value
            except Exception:
                continue

            if market_cap is None or market_cap <= self.min_market_cap:
                continue

            if book_equity is None or book_equity <= self.min_book_equity:
                continue

            candidates.append(f)

        top_by_market_cap = sorted(
            candidates,
            key=lambda f: f.MarketCap if f.MarketCap is not None else 0,
            reverse=True
        )[:self.top_universe_n]

        self.current_universe = [f.Symbol for f in top_by_market_cap]
        self.latest_fine = {f.Symbol: f for f in top_by_market_cap}

        return self.current_universe

    def ComputeMonthEndSignals(self):
        """
        Placeholder for:
        - build_raw_snapshot_for_date()
        - enrich_raw_snapshot()
        - build_signals()
        """
        self.rebalance_flag = True
        self.Debug(f"Month-end signal event: {self.Time.date()} | universe size = {len(self.current_universe)}")

    def ExecuteMonthStartRebalance(self):
        """
        Placeholder for:
        - form_portfolio_membership()
        - assign_weights()
        """
        self.Debug(f"Month-start execution event: {self.Time.date()} | rebalance_flag = {self.rebalance_flag}")

    def OnSecuritiesChanged(self, changes):
        """
        Useful for debugging universe behavior in QC.
        """
        added = [x.Symbol.Value for x in changes.AddedSecurities]
        removed = [x.Symbol.Value for x in changes.RemovedSecurities]

        if added:
            self.Debug(f"Added: {len(added)}")
        if removed:
            self.Debug(f"Removed: {len(removed)}")

    def OnData(self, data):
        """
        Not used yet.
        Your strategy will be scheduler-driven, not OnData-driven.
        """
        pass