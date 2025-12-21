"""
Combine SZMC and Ichilov VVI Parquet datasets, keep patients with >=3 visits,
and label cardiotoxicity based on GLS trends.

Cardiotoxicity criteria (per patient, using absolute GLS values):
1) Linear trend slope across visits < slope_threshold (visit index vs |GLS|; default -0.9).
2) At least one visit shows a >=15% drop in |GLS| vs any earlier visit
   (|GLS_curr| <= 0.85 * |GLS_previous_best|).
Both conditions must hold to mark cardiotoxicity=True.

The script keeps all original rows (cycles/segments) for the retained patients
and adds a boolean "cardiotoxicity" column. A "patient_key" with source prefix
avoids cross-dataset ID collisions.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# ----------------------------- Logging ------------------------------------
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(stream=sys.stdout),
        logging.FileHandler(r"C:\work\us\combine_vvi_datasets.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("combine_vvi_datasets")


# ----------------------------- Helpers ------------------------------------
def prepare_df(df: pd.DataFrame, source: str, id_col: str) -> pd.DataFrame:
    """Normalize a dataset to have patient_id, patient_key, source, study_datetime, gls."""
    if id_col not in df.columns:
        raise ValueError(f"Expected column '{id_col}' in source {source}")

    out = df.copy()
    out = out.rename(columns={id_col: "patient_id"})
    out["source"] = source
    out["patient_key"] = out["source"] + ":" + out["patient_id"].astype(str)

    # Normalize datetime and GLS
    out["study_datetime"] = pd.to_datetime(out["study_datetime"], errors="coerce")
    out["gls"] = pd.to_numeric(out["gls"], errors="coerce")

    return out


def patients_with_min_visits(df: pd.DataFrame, min_visits: int = 3) -> List[str]:
    counts = df.groupby("patient_key")["study_datetime"].nunique()
    return counts[counts >= min_visits].index.tolist()


def compute_cardiotoxicity_flags(df: pd.DataFrame, slope_threshold: float) -> pd.Series:
    """
    Return a mapping patient_key -> cardiotoxicity (bool) using visit-level GLS means.
    """
    flags = {}

    # Visit-level GLS (mean of absolute GLS per visit)
    visit_gls = (
        df.groupby(["patient_key", "study_datetime"])
        ["gls"]
        .apply(lambda s: np.nanmean(np.abs(s)))
        .dropna()
        .reset_index(name="gls_abs")
    )

    for pk, grp in visit_gls.groupby("patient_key"):
        grp = grp.sort_values("study_datetime")
        vals = grp["gls_abs"].to_numpy()
        if len(vals) < 2:
            flags[pk] = False
            continue

        # Condition 1: slope of trend line < slope_threshold (visits indexed 0..n-1)
        x = np.arange(len(vals), dtype=float)
        try:
            slope = np.polyfit(x, vals, 1)[0]
        except Exception:
            slope = np.nan
        cond_slope = slope < slope_threshold if not np.isnan(slope) else False

        # Condition 2: any >=15% drop vs any earlier visit (not necessarily sequential)
        cond_drop = False
        if len(vals) >= 2:
            prev_max = vals[0]
            for v in vals[1:]:
                if v <= 0.85 * prev_max:  # drop of 15% or more vs an earlier higher value
                    cond_drop = True
                    break
                prev_max = max(prev_max, v)

        flags[pk] = bool(cond_slope and cond_drop)

    return pd.Series(flags, name="cardiotoxicity")


# ----------------------------- Main ---------------------------------------
def main():
    import os

    user_home = os.path.expanduser("~")

    p = argparse.ArgumentParser(description="Combine SZMC and Ichilov VVI datasets with cardiotoxicity labels.")
    p.add_argument(
        "--szmc-parquet",
        type=Path,
        default=Path(user_home)
        / "OneDrive - Technion"
        / "DS"
        / "Tags_SZMC"
        / "VVI"
        / "processed"
        / "strain_dataset.parquet",
        help="Path to SZMC parquet (from vvi_xml_preprocess.py).",
    )
    p.add_argument(
        "--ichilov-parquet",
        type=Path,
        default=Path(user_home)
        / "OneDrive - Technion"
        / "DS"
        / "Tags_Ichilov"
        / "VVI"
        / "processed"
        / "strain_dataset_ichilov.parquet",
        help="Path to Ichilov parquet (from vvi_xml_preprocess_ichilov.py).",
    )
    p.add_argument(
        "--out-parquet",
        type=Path,
        default=Path(user_home)
        / "OneDrive - Technion"
        / "DS"
        / "strain_dataset_combined.parquet",
        help="Output parquet path for the combined dataset.",
    )
    p.add_argument(
        "--slope-threshold",
        type=float,
        default=-0.7,
        help="Slope threshold for GLS trend (visit index vs |GLS|) to flag cardiotoxicity.",
    )
    args = p.parse_args()

    # Load inputs
    if not args.szmc_parquet.is_file():
        raise FileNotFoundError(f"SZMC parquet not found: {args.szmc_parquet}")
    if not args.ichilov_parquet.is_file():
        raise FileNotFoundError(f"Ichilov parquet not found: {args.ichilov_parquet}")

    df_szmc = pd.read_parquet(args.szmc_parquet)
    df_ich = pd.read_parquet(args.ichilov_parquet)

    df_szmc = prepare_df(df_szmc, source="SZMC", id_col="short_id")
    df_ich = prepare_df(df_ich, source="ICHILOV", id_col="patient_id")

    combined = pd.concat([df_szmc, df_ich], ignore_index=True, sort=False)

    # Keep patients with at least 3 visits
    keep_patients = set(patients_with_min_visits(combined, min_visits=3))
    filtered = combined[combined["patient_key"].isin(keep_patients)].copy()

    # Cardiotoxicity per patient_key
    cardio_flags = compute_cardiotoxicity_flags(filtered, slope_threshold=args.slope_threshold)
    filtered = filtered.merge(cardio_flags, left_on="patient_key", right_index=True, how="left")
    filtered["cardiotoxicity"] = filtered["cardiotoxicity"].fillna(False)

    # Save output
    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_parquet(args.out_parquet, index=False)

    logger.info(
        f"Saved combined dataset: {args.out_parquet} "
        f"({len(filtered)} rows, {filtered['patient_key'].nunique()} patients)."
    )


if __name__ == "__main__":
    main()
