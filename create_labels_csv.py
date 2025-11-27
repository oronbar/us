"""
Utility to build a labels CSV for HiLoS-Cardio training.

Reads the parquet produced by vvi_xml_preprocess.py, extracts unique short_id
values, and writes a CSV with columns:
  short_id,outcome,event_visit_index

Fields:
- outcome: 0 = no CTRCD during follow-up, 1 = developed CTRCD.
- event_visit_index: optional 0-based index of the first visit where CTRCD
  met clinical criteria; leave blank if unknown.

Example:
  python create_labels_csv.py --parquet path/to/strain_dataset.parquet --out labels.csv

By default, outcome and event_visit_index are left blank for manual entry.
Use --default-outcome 0 to prefill zeros.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


def build_labels(parquet_path: Path, out_csv: Path, default_outcome: Optional[str]) -> None:
    df = pd.read_parquet(parquet_path)
    if "short_id" not in df.columns:
        raise ValueError("Parquet is missing required column 'short_id'.")
    sids = sorted(df["short_id"].dropna().unique().tolist())
    if not sids:
        raise RuntimeError("No short_id values found in parquet.")

    # Prefill outcome/event with blanks unless user supplies default_outcome
    outcome_fill = "" if default_outcome is None else default_outcome
    labels = pd.DataFrame(
        {
            "short_id": sids,
            "outcome": [outcome_fill] * len(sids),
            "event_visit_index": ["" for _ in sids],
        }
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(out_csv, index=False)
    print(f"Wrote labels CSV with {len(sids)} patients to {out_csv}")


def parse_args():
    ap = argparse.ArgumentParser(description="Create labels CSV from strain parquet")
    ap.add_argument("--parquet", type=Path, required=True, help="Path to strain_dataset.parquet")
    ap.add_argument("--out", type=Path, required=True, help="Output CSV path (e.g., labels.csv)")
    ap.add_argument(
        "--default-outcome",
        type=str,
        default=None,
        help="Optional value (e.g., 0) to prefill outcome column; blank by default.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    build_labels(args.parquet, args.out, args.default_outcome)


if __name__ == "__main__":
    main()
