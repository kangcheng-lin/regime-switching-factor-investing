from __future__ import annotations

import json
import re
from collections import defaultdict, OrderedDict
from pathlib import Path

import pandas as pd


PORT_PATTERN = re.compile(
    r"PORT\|(?P<date>\d{4}-\d{2}-\d{2})\|chunk=(?P<chunk>\d+)\|(?P<tickers>[A-Z0-9.\-_|]+)"
)


def parse_port_chunks_from_text(text: str) -> dict[str, dict[int, list[str]]]:
    """
    Parse all PORT chunk lines from one log text.

    Returns:
        {
            "2024-01-31": {
                1: ["WBD", "CVS", ...],
                2: [...],
                ...
            },
            ...
        }
    """
    chunk_map: dict[str, dict[int, list[str]]] = defaultdict(dict)

    for m in PORT_PATTERN.finditer(text):
        date = m.group("date")
        chunk_id = int(m.group("chunk"))
        tickers = [t for t in m.group("tickers").split("|") if t]
        chunk_map[date][chunk_id] = tickers

    return chunk_map


def merge_chunk_maps(chunk_maps: list[dict[str, dict[int, list[str]]]]) -> dict[str, dict[int, list[str]]]:
    """
    Merge chunk maps from multiple files.

    Later files overwrite earlier files only for exact (date, chunk_id) collisions.
    """
    merged: dict[str, dict[int, list[str]]] = defaultdict(dict)

    for cmap in chunk_maps:
        for date, chunk_dict in cmap.items():
            for chunk_id, tickers in chunk_dict.items():
                merged[date][chunk_id] = tickers

    return merged


def reconstruct_portfolios(chunk_map: dict[str, dict[int, list[str]]]) -> OrderedDict[str, list[str]]:
    """
    Reconstruct full portfolio for each date by concatenating chunks in chunk order.
    Removes duplicates while preserving order.
    """
    portfolios: OrderedDict[str, list[str]] = OrderedDict()

    for date in sorted(chunk_map.keys()):
        full_list: list[str] = []
        for chunk_id in sorted(chunk_map[date].keys()):
            full_list.extend(chunk_map[date][chunk_id])

        seen = set()
        deduped: list[str] = []
        for ticker in full_list:
            if ticker not in seen:
                deduped.append(ticker)
                seen.add(ticker)

        portfolios[date] = deduped

    return portfolios


def portfolios_to_wide_df(portfolios: dict[str, list[str]]) -> pd.DataFrame:
    """
    Wide format:
        date | n | ticker_1 | ticker_2 | ... | ticker_50
    """
    rows = []
    for date, symbols in portfolios.items():
        row = {"date": date, "n": len(symbols)}
        for i, sym in enumerate(symbols, start=1):
            row[f"ticker_{i}"] = sym
        rows.append(row)
    return pd.DataFrame(rows)


def portfolios_to_long_df(portfolios: dict[str, list[str]]) -> pd.DataFrame:
    """
    Long format:
        date | rank | ticker
    """
    rows = []
    for date, symbols in portfolios.items():
        for rank, sym in enumerate(symbols, start=1):
            rows.append({"date": date, "rank": rank, "ticker": sym})
    return pd.DataFrame(rows)


def extract_portfolios_from_logs_dir(log_dir: str | Path, pattern: str = "*.txt") -> OrderedDict[str, list[str]]:
    """
    Read all matching log files from a directory and reconstruct rebalance portfolios.
    """
    log_dir = Path(log_dir)
    files = sorted(log_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No log files matching {pattern!r} found in {log_dir}")

    chunk_maps = []
    for file in files:
        text = file.read_text(encoding="utf-8", errors="ignore")
        chunk_maps.append(parse_port_chunks_from_text(text))

    merged_chunks = merge_chunk_maps(chunk_maps)
    portfolios = reconstruct_portfolios(merged_chunks)
    return portfolios


def save_outputs(portfolios: dict[str, list[str]], output_dir: str | Path, prefix: str = "qc_portfolios") -> None:
    """
    Save JSON, wide CSV, and long CSV.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{prefix}.json"
    wide_path = output_dir / f"{prefix}_wide.csv"
    long_path = output_dir / f"{prefix}_long.csv"

    json_path.write_text(json.dumps(portfolios, indent=2), encoding="utf-8")
    portfolios_to_wide_df(portfolios).to_csv(wide_path, index=False)
    portfolios_to_long_df(portfolios).to_csv(long_path, index=False)

    print("Saved:")
    print(f"  {json_path}")
    print(f"  {wide_path}")
    print(f"  {long_path}")


def main():
    # Put all your yearly QC log txt files in this folder
    log_dir = Path(__file__).parent / "qc_logs"

    portfolios = extract_portfolios_from_logs_dir(log_dir, pattern="*.txt")

    print(f"Parsed {len(portfolios)} rebalance dates.")
    preview_dates = list(portfolios.keys())[:5]
    for date in preview_dates:
        print(date, len(portfolios[date]), portfolios[date][:10])

    save_outputs(portfolios, output_dir="qc_parsed", prefix="qc_value_portfolios")


if __name__ == "__main__":
    main()