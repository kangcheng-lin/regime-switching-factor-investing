# File: src/data/deduplicate_fundamentals.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


@dataclass
class DedupConfig:
    lifecycle_path: str = "data/processed/universe/security_lifecycle_filtered.csv"
    balance_sheet_src: str = "data/raw/fundamentals/balance_sheet_quarter_csv"
    income_statement_src: str = "data/raw/fundamentals/income_statement_quarter_csv"
    cash_flow_src: str = "data/raw/fundamentals/cash_flow_quarter_csv"
    balance_sheet_dst: str = "data/processed/fundamentals_clean/balance_sheet_quarter_csv"
    income_statement_dst: str = "data/processed/fundamentals_clean/income_statement_quarter_csv"
    cash_flow_dst: str = "data/processed/fundamentals_clean/cash_flow_quarter_csv"

class FundamentalDeduplicator:
    """Clean quarterly fundamentals only for tickers in the lifecycle universe.

    Scope:
    - balance sheet
    - income statement
    - cash flow

    Deduplication rule:
    - use `date` as the statement-period identifier
    - use availability date priority:
      acceptedDate > fillingDate > filingDate > date
    - for duplicate rows with the same statement period, keep the row with the
      latest availability date

    Cleaned files are written to processed folders. Raw files are never overwritten.
    """

    def __init__(self, config: DedupConfig):
        self.config = config
        self.lifecycle = self._load_lifecycle(config.lifecycle_path)
        self.universe_tickers = self._load_universe_tickers()

    def run(self) -> dict[str, int]:
        stats: dict[str, int] = {"universe_tickers": len(self.universe_tickers)}
        stats.update(
            self._process_directory(
                src_dir=Path(self.config.balance_sheet_src),
                dst_dir=Path(self.config.balance_sheet_dst),
                label="balance_sheet",
            )
        )
        stats.update(
            self._process_directory(
                src_dir=Path(self.config.income_statement_src),
                dst_dir=Path(self.config.income_statement_dst),
                label="income_statement",
            )
        )
        stats.update(
            self._process_directory(
                src_dir=Path(self.config.cash_flow_src),
                dst_dir=Path(self.config.cash_flow_dst),
                label="cash_flow",
            )
        )
        return stats

    def _process_directory(self, src_dir: Path, dst_dir: Path, label: str) -> dict[str, int]:
        if not src_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {src_dir}")

        dst_dir.mkdir(parents=True, exist_ok=True)
        files_written = 0
        rows_before_total = 0
        rows_after_total = 0
        tickers_missing = 0

        for ticker in sorted(self.universe_tickers):
            src_path = src_dir / f"{ticker}.csv"
            if not src_path.exists():
                tickers_missing += 1
                continue

            raw = pd.read_csv(src_path)
            rows_before_total += len(raw)
            clean = self._deduplicate_quarterly_file(raw)
            rows_after_total += len(clean)

            dst_path = dst_dir / f"{ticker}.csv"
            clean.to_csv(dst_path, index=False)
            files_written += 1

        return {
            f"{label}_files_written": files_written,
            f"{label}_rows_before": rows_before_total,
            f"{label}_rows_after": rows_after_total,
            f"{label}_tickers_missing": tickers_missing,
        }

    def _deduplicate_quarterly_file(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        if "date" not in out.columns:
            raise ValueError("Quarterly statement file is missing required `date` column.")

        out["statement_date"] = pd.to_datetime(out["date"], errors="coerce", format="mixed")

        availability_col = self._first_existing(
            out.columns,
            ["acceptedDate", "fillingDate", "filingDate", "date"],
        )
        if availability_col is None:
            raise ValueError(
                "Quarterly statement file must contain one of: acceptedDate, fillingDate, filingDate, date"
            )
        out["availability_date"] = pd.to_datetime(out[availability_col], errors="coerce", format="mixed")

        out = out.dropna(subset=["statement_date", "availability_date"]).copy()
        out = out.sort_values(["statement_date", "availability_date"]).copy()
        out = out.drop_duplicates(subset=["statement_date"], keep="last").copy()
        out = out.sort_values(["statement_date", "availability_date"]).reset_index(drop=True)
        return out

    def _load_lifecycle(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}
        ticker_col = None
        for cand in ["ticker", "symbol"]:
            if cand in cols:
                ticker_col = cols[cand]
                break
        if ticker_col is None:
            raise ValueError("Lifecycle file must contain ticker or symbol column.")
        out = df[[ticker_col]].copy()
        out.columns = ["ticker"]
        out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
        return out.dropna(subset=["ticker"]).drop_duplicates().reset_index(drop=True)

    def _load_universe_tickers(self) -> set[str]:
        return set(self.lifecycle["ticker"].tolist())

    @staticmethod
    def _first_existing(columns: Iterable[str], candidates: list[str]) -> Optional[str]:
        colset = set(columns)
        for candidate in candidates:
            if candidate in colset:
                return candidate
        return None


if __name__ == "__main__":
    config = DedupConfig()
    cleaner = FundamentalDeduplicator(config)
    stats = cleaner.run()
    for key, value in stats.items():
        print(f"{key}: {value}")