from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Example command: python src/experiments/verification/build_position_lifecycle.py --input_dir results/tables/aqr_full_price_available_universe_modular --strategy aqr --output_dir results/verification

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build position lifecycle audit table from strategy output files."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing strategy output CSV files.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        help="Strategy name, e.g. aqr, value, ff3.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save verification output CSV files.",
    )
    return parser.parse_args()


def load_csv(path: Path, required_columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"File {path} is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    return df


def load_inputs(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    weights_path = input_dir / "weights_at_E.csv"
    asset_returns_path = input_dir / "asset_returns.csv"

    weights = load_csv(
        weights_path,
        required_columns=[
            "signal_date",
            "execution_date",
            "ticker",
            "side",
            "adj_close_e",
            "weight",
        ],
    )

    asset_returns = load_csv(
        asset_returns_path,
        required_columns=[
            "signal_date",
            "execution_date",
            "next_execution_date",
            "actual_exit_date",
            "ticker",
            "side",
            "entry_price",
            "exit_price",
            "asset_return",
            "weighted_return",
        ],
    )

    return weights, asset_returns


def standardize_dates(df: pd.DataFrame, date_columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in date_columns:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def prepare_weights_table(weights: pd.DataFrame, strategy: str) -> pd.DataFrame:
    out = weights.copy()
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    out["side"] = out["side"].astype(str).str.lower().str.strip()
    out["strategy"] = strategy

    out = out.rename(columns={"adj_close_e": "current_execution_price"})

    keep_cols = [
        "strategy",
        "signal_date",
        "execution_date",
        "ticker",
        "side",
        "current_execution_price",
        "weight",
    ]
    return out[keep_cols].copy()


def prepare_asset_returns_table(asset_returns: pd.DataFrame) -> pd.DataFrame:
    out = asset_returns.copy()
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    out["side"] = out["side"].astype(str).str.lower().str.strip()

    keep_cols = [
        "signal_date",
        "execution_date",
        "next_execution_date",
        "actual_exit_date",
        "ticker",
        "side",
        "entry_price",
        "exit_price",
        "asset_return",
        "weighted_return",
    ]
    return out[keep_cols].copy()


def merge_core_tables(
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
) -> pd.DataFrame:
    merge_keys = ["signal_date", "execution_date", "ticker", "side"]

    merged = weights.merge(
        asset_returns,
        on=merge_keys,
        how="inner",
        validate="1:1",
    )

    merged = merged.sort_values(
        ["ticker", "side", "execution_date"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    return merged


def add_holding_spell_columns(merged: pd.DataFrame) -> pd.DataFrame:
    out = merged.copy()

    group_keys = ["ticker", "side"]

    out = out.sort_values(
        ["ticker", "side", "execution_date"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    out["prev_signal_date"] = out.groupby(group_keys)["signal_date"].shift(1)
    out["prev_execution_date"] = out.groupby(group_keys)["execution_date"].shift(1)
    out["prev_next_execution_date"] = out.groupby(group_keys)["next_execution_date"].shift(1)

    out["was_held_previous_rebalance"] = (
        out["prev_next_execution_date"].notna()
        & (out["prev_next_execution_date"] == out["execution_date"])
    )

    out["is_new_entry"] = ~out["was_held_previous_rebalance"]

    out["spell_id"] = out.groupby(group_keys)["is_new_entry"].cumsum()

    out["spell_entry_signal_date"] = out.groupby(group_keys + ["spell_id"])["signal_date"].transform("first")
    out["spell_entry_execution_date"] = out.groupby(group_keys + ["spell_id"])["execution_date"].transform("first")

    out["position_age_rebalances"] = out.groupby(group_keys + ["spell_id"]).cumcount() + 1

    out["entry_price"] = out.groupby(group_keys + ["spell_id"])["current_execution_price"].transform("first")

    out["next_signal_date_same_side"] = out.groupby(group_keys)["signal_date"].shift(-1)
    out["next_execution_date_same_side"] = out.groupby(group_keys)["execution_date"].shift(-1)

    out["will_be_held_next_rebalance"] = (
        out["next_execution_date_same_side"].notna()
        & (out["next_execution_date"] == out["next_execution_date_same_side"])
    )

    out["is_exit_on_next_rebalance"] = ~out["will_be_held_next_rebalance"]

    return out


def classify_exit_type(row: pd.Series) -> str:
    if not bool(row["is_exit_on_next_rebalance"]):
        return ""

    planned_exit_date = row["planned_exit_date"]
    actual_exit_date = row["actual_exit_date"]

    if pd.isna(actual_exit_date):
        return "likely_delisted_or_missing"

    if actual_exit_date == planned_exit_date:
        return "normal"

    if actual_exit_date < planned_exit_date:
        return "fallback_price_used"

    return "unexpected_exit_timing"


def finalize_lifecycle_table(lifecycle: pd.DataFrame) -> pd.DataFrame:
    out = lifecycle.copy()

    out["planned_exit_date"] = out["next_execution_date"]
    out["exit_type"] = out.apply(classify_exit_type, axis=1)

    final_cols = [
        "strategy",
        "signal_date",
        "execution_date",
        "next_execution_date",
        "ticker",
        "side",
        "weight",
        "spell_entry_signal_date",
        "spell_entry_execution_date",
        "entry_price",
        "current_execution_price",
        "position_age_rebalances",
        "was_held_previous_rebalance",
        "is_new_entry",
        "will_be_held_next_rebalance",
        "is_exit_on_next_rebalance",
        "planned_exit_date",
        "actual_exit_date",
        "exit_price",
        "exit_type",
        "asset_return",
        "weighted_return",
    ]

    return out[final_cols].sort_values(
        ["ticker", "side", "execution_date"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


def build_rebalance_snapshot_table(final_audit: pd.DataFrame) -> pd.DataFrame:
    snapshot_cols = [
        "strategy",
        "signal_date",
        "execution_date",
        "next_execution_date",
        "ticker",
        "side",
        "weight",
        "spell_entry_signal_date",
        "spell_entry_execution_date",
        "entry_price",
        "current_execution_price",
        "planned_exit_date",
        "actual_exit_date",
        "exit_price",
        "asset_return",
        "weighted_return",
        "position_age_rebalances",
        "was_held_previous_rebalance",
        "is_new_entry",
        "will_be_held_next_rebalance",
        "is_exit_on_next_rebalance",
        "exit_type",
    ]

    return final_audit[snapshot_cols].sort_values(
        ["signal_date", "side", "ticker"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    weights, asset_returns = load_inputs(input_dir)

    weights = standardize_dates(weights, ["signal_date", "execution_date"])
    asset_returns = standardize_dates(
        asset_returns,
        ["signal_date", "execution_date", "next_execution_date", "actual_exit_date"],
    )

    weights_prepared = prepare_weights_table(weights, strategy=args.strategy)
    asset_returns_prepared = prepare_asset_returns_table(asset_returns)
    merged = merge_core_tables(weights_prepared, asset_returns_prepared)

    lifecycle = add_holding_spell_columns(merged)
    final_audit = finalize_lifecycle_table(lifecycle)

    lifecycle_output_path = output_dir / f"{args.strategy}_position_lifecycle_audit.csv"
    final_audit.to_csv(lifecycle_output_path, index=False)

    rebalance_snapshot = build_rebalance_snapshot_table(final_audit)
    snapshot_output_path = output_dir / f"{args.strategy}_rebalance_snapshot_audit.csv"
    rebalance_snapshot.to_csv(snapshot_output_path, index=False)

    print("Position lifecycle audit built successfully.")
    print(f"merged shape: {merged.shape}")
    print(f"final audit shape: {final_audit.shape}")
    print(f"lifecycle file written to: {lifecycle_output_path}")
    print(f"snapshot file written to: {snapshot_output_path}")

    print("\nExit type counts:")
    print(final_audit["exit_type"].value_counts(dropna=False))

    print("\nFirst 10 rows:")
    print(final_audit.head(10))


if __name__ == "__main__":
    main()