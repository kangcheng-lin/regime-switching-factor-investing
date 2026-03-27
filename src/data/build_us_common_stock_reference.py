from pathlib import Path
import json
import os
import time

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv


# ==============================
# Load environment variables
# ==============================
load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data/raw/fundamentals/fmp_tickers/delisted_tickers_jan_28_2026.csv"
PROMPT_FILE = PROJECT_ROOT / "prompts/delisted_common_stock_classifier_prompt.txt"

AUDIT_FILE = PROJECT_ROOT / "data/processed/universe/delisted_tickers_gpt_audit.csv"
OUTPUT_FILE = PROJECT_ROOT / "data/processed/universe/delisted_us_common_stocks_reference.csv"

TARGET_EXCHANGES = {"NASDAQ", "NYSE", "AMEX"}

REQUEST_SLEEP_SECONDS = 0.25
MAX_RETRIES = 3


def load_prompt() -> str:
    return PROMPT_FILE.read_text(encoding="utf-8").strip()


def normalize_exchange(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def build_user_message(row: pd.Series) -> str:
    record = {
        "symbol": row["symbol"],
        "companyName": row["companyName"],
        "exchange": row["exchange"],
        "ipoDate": row["ipoDate"],
        "delistedDate": row["delistedDate"],
    }

    return (
        "Classify the following delisted security record:\n\n"
        f"{json.dumps(record, ensure_ascii=False, indent=2)}"
    )


def classify_row(client: OpenAI, prompt_text: str, row: pd.Series) -> dict:
    user_message = build_user_message(row)

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": prompt_text},
                    {"role": "user", "content": user_message},
                ],
            )

            content = response.choices[0].message.content
            result = json.loads(content)

            keep = str(result.get("keep", "")).strip().lower()
            security_type = str(result.get("security_type", "")).strip()
            reason = str(result.get("reason", "")).strip()
            confidence = str(result.get("confidence", "")).strip().lower()

            if keep not in {"yes", "no"}:
                raise ValueError(f"Unexpected keep value: {keep}")

            if confidence not in {"high", "medium", "low"}:
                confidence = ""

            return {
                "keep": keep,
                "security_type": security_type,
                "reason": reason,
                "confidence": confidence,
            }

        except Exception as exc:
            last_error = exc
            print(
                f"Retry {attempt}/{MAX_RETRIES} failed for "
                f"{row['symbol']} ({row['companyName']}): {exc}"
            )
            time.sleep(2 * attempt)

    raise RuntimeError(
        f"OpenAI classification failed for {row['symbol']} after {MAX_RETRIES} tries. "
        f"Last error: {last_error}"
    )


def load_existing_audit() -> pd.DataFrame:
    if AUDIT_FILE.exists():
        return pd.read_csv(AUDIT_FILE)
    return pd.DataFrame()


def append_audit_row(row_dict: dict) -> None:
    audit_row = pd.DataFrame([row_dict])

    write_header = not AUDIT_FILE.exists()

    AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    audit_row.to_csv(
        AUDIT_FILE,
        mode="a",
        header=write_header,
        index=False,
    )


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    if not PROMPT_FILE.exists():
        raise FileNotFoundError(f"Prompt file not found: {PROMPT_FILE}")

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    prompt_text = load_prompt()
    client = OpenAI()

    df = pd.read_csv(INPUT_FILE)

    print(f"Initial number of delisted records: {len(df)}")

    df["exchange"] = df["exchange"].map(normalize_exchange)
    df = df[df["exchange"].isin(TARGET_EXCHANGES)].copy()

    print(f"After filtering to NASDAQ/NYSE/AMEX: {len(df)}")

    df = df.reset_index(drop=True)
    df["source_row_id"] = df.index

    existing_audit = load_existing_audit()

    if not existing_audit.empty and "source_row_id" in existing_audit.columns:
        done_ids = set(existing_audit["source_row_id"].tolist())
        to_process = df[~df["source_row_id"].isin(done_ids)].copy()
        print(f"Existing audit rows found: {len(existing_audit)}")
        print(f"Remaining rows to classify: {len(to_process)}")
    else:
        to_process = df.copy()
        print("No existing audit file found. Starting fresh.")

    for i, (_, row) in enumerate(to_process.iterrows(), start=1):
        classification = classify_row(client, prompt_text, row)

        audit_row = {
            "source_row_id": row["source_row_id"],
            "symbol": row["symbol"],
            "companyName": row["companyName"],
            "exchange": row["exchange"],
            "ipoDate": row["ipoDate"],
            "delistedDate": row["delistedDate"],
            "keep": classification["keep"],
            "security_type": classification["security_type"],
            "reason": classification["reason"],
            "confidence": classification["confidence"],
        }

        append_audit_row(audit_row)

        print(
            f"[{i}/{len(to_process)}] "
            f"{row['symbol']} | {row['companyName']} -> {classification['keep']}"
        )

        time.sleep(REQUEST_SLEEP_SECONDS)

    audit_df = pd.read_csv(AUDIT_FILE)
    audit_df = audit_df.sort_values(["symbol", "companyName", "delistedDate"]).reset_index(drop=True)

    keep_df = audit_df[audit_df["keep"].str.lower() == "yes"].copy()
    keep_df = keep_df.sort_values(["symbol", "companyName", "delistedDate"]).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(AUDIT_FILE, index=False)
    keep_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved full audit file to: {AUDIT_FILE}")
    print(f"Saved filtered keep-only file to: {OUTPUT_FILE}")
    print(f"Number of kept securities: {len(keep_df)}")

    print("\nSample of kept securities:")
    print(keep_df.head())


if __name__ == "__main__":
    main()