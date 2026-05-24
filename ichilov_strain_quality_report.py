"""
Generate quality diagnostics for VVI strain labels and model residuals.

Example:
  python ichilov_strain_quality_report.py ^
    --prepared-parquet "C:\\path\\prepared_dataset.parquet" ^
    --predictions-parquet "C:\\path\\predictions.parquet" ^
    --output-dir "C:\\path\\quality"
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ichilov_strain_curve_utils import dataframe_preview_for_csv, write_json


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("ichilov_strain_quality_report")


def _array(value: object) -> np.ndarray:
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return np.asarray(value.tolist(), dtype=float)
        return value.astype(float, copy=False)
    if isinstance(value, list):
        return np.asarray(value, dtype=float)
    if isinstance(value, str):
        for parser in (json.loads, ast.literal_eval):
            try:
                return np.asarray(parser(value), dtype=float)
            except Exception:
                pass
    try:
        return np.asarray(value, dtype=float)
    except Exception:
        return np.asarray([], dtype=float)


def _add_residuals(prepared: pd.DataFrame, preds: Optional[pd.DataFrame]) -> pd.DataFrame:
    out = prepared.copy()
    if preds is None or preds.empty:
        out["model_curve_mae"] = np.nan
        out["model_peak_gls_error"] = np.nan
        out["model_peak_gls_abs_error"] = np.nan
        out["model_time_to_peak_abs_error"] = np.nan
        return out
    keep_cols = [
        "sample_id",
        "pred_curve",
        "pred_peak_gls",
        "pred_time_to_peak",
        "true_peak_gls",
        "true_time_to_peak",
    ]
    pred_keep = preds[[c for c in keep_cols if c in preds.columns]].copy()
    out = out.merge(pred_keep, on="sample_id", how="left", suffixes=("", "_predfile"))

    curve_mae = []
    for _, row in out.iterrows():
        true_curve = _array(row.get("resampled_strain_curve"))
        pred_curve = _array(row.get("pred_curve"))
        if true_curve.size and pred_curve.size and true_curve.shape == pred_curve.shape:
            curve_mae.append(float(np.nanmean(np.abs(pred_curve - true_curve))))
        else:
            curve_mae.append(np.nan)
    out["model_curve_mae"] = curve_mae
    true_peak = pd.to_numeric(out.get("peak_gls_from_curve"), errors="coerce")
    if "true_peak_gls" in out:
        true_peak = true_peak.fillna(pd.to_numeric(out["true_peak_gls"], errors="coerce"))
    out["model_peak_gls_error"] = pd.to_numeric(out.get("pred_peak_gls"), errors="coerce") - true_peak
    out["model_peak_gls_abs_error"] = out["model_peak_gls_error"].abs()
    true_ttp = pd.to_numeric(out.get("time_to_peak_from_curve"), errors="coerce")
    if "true_time_to_peak" in out:
        true_ttp = true_ttp.fillna(pd.to_numeric(out["true_time_to_peak"], errors="coerce"))
    out["model_time_to_peak_abs_error"] = (
        pd.to_numeric(out.get("pred_time_to_peak"), errors="coerce") - true_ttp
    ).abs()
    return out


def _quality_flags(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    def series(col: str, default: object = np.nan) -> pd.Series:
        if col in out.columns:
            return out[col]
        return pd.Series([default] * len(out), index=out.index)

    out["flag_starts_far_from_zero"] = series("starts_near_zero", True).astype(object).map(lambda x: not bool(x))
    out["flag_invalid_peak"] = series("has_valid_peak", True).astype(object).map(lambda x: not bool(x))
    out["flag_implausible_peak"] = series("peak_in_reasonable_range", True).astype(object).map(lambda x: not bool(x))
    out["flag_noisy_curve"] = pd.to_numeric(series("excessive_noise_score"), errors="coerce") > args.noise_threshold
    out["flag_many_large_peaks"] = pd.to_numeric(series("num_large_peaks"), errors="coerce") > args.large_peak_count_threshold
    out["flag_nan_curve"] = pd.to_numeric(series("curve_nan_fraction"), errors="coerce") > args.nan_fraction_threshold
    out["flag_inter_view_disagreement"] = pd.to_numeric(series("view_gls_disagreement"), errors="coerce") > args.view_disagreement_threshold
    out["flag_model_curve_residual"] = pd.to_numeric(series("model_curve_mae"), errors="coerce") > args.curve_residual_threshold
    out["flag_model_gls_residual"] = pd.to_numeric(series("model_peak_gls_abs_error"), errors="coerce") > args.gls_residual_threshold
    flag_cols = [c for c in out.columns if c.startswith("flag_")]
    out["quality_flag_count"] = out[flag_cols].fillna(False).astype(bool).sum(axis=1)
    out["likely_bad_strain_curve"] = out["quality_flag_count"] >= args.min_flags_for_bad
    return out


def _summary_by_patient_view(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [c for c in ("patient_key", "visit_date", "view") if c in df.columns]
    if not group_cols:
        return pd.DataFrame()
    agg = {
        "sample_id": "count",
        "likely_bad_strain_curve": "sum",
        "quality_flag_count": "mean",
    }
    for col in ("view_gls_disagreement", "model_curve_mae", "model_peak_gls_abs_error", "excessive_noise_score"):
        if col in df.columns:
            agg[col] = "mean"
    out = df.groupby(group_cols, dropna=False).agg(agg).reset_index()
    out = out.rename(columns={"sample_id": "n_samples", "likely_bad_strain_curve": "n_likely_bad"})
    return out


def _write_md(path: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# Strain Quality Report",
        "",
        f"- Samples: {summary['n_samples']}",
        f"- Patients: {summary['n_patients']}",
        f"- Likely bad strain curves: {summary['n_likely_bad']}",
        f"- High inter-view disagreement: {summary['n_high_inter_view_disagreement']}",
        f"- High model residuals: {summary['n_high_model_residuals']}",
        "",
        "## Flag Counts",
    ]
    for key, value in summary["flag_counts"].items():
        lines.append(f"- {key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate strain curve quality report.")
    parser.add_argument("--prepared-parquet", type=Path, required=True)
    parser.add_argument("--predictions-parquet", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--noise-threshold", type=float, default=0.35)
    parser.add_argument("--large-peak-count-threshold", type=int, default=3)
    parser.add_argument("--nan-fraction-threshold", type=float, default=0.05)
    parser.add_argument("--view-disagreement-threshold", type=float, default=5.0)
    parser.add_argument("--curve-residual-threshold", type=float, default=4.0)
    parser.add_argument("--gls-residual-threshold", type=float, default=4.0)
    parser.add_argument("--min-flags-for-bad", type=int, default=1)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepared = pd.read_parquet(args.prepared_parquet)
    preds = None
    if args.predictions_parquet and args.predictions_parquet.exists():
        preds = pd.read_parquet(args.predictions_parquet)
    df = _quality_flags(_add_residuals(prepared, preds), args)

    likely_bad = df[df["likely_bad_strain_curve"]].sort_values("quality_flag_count", ascending=False)
    high_disagreement = df[df["flag_inter_view_disagreement"]].sort_values("view_gls_disagreement", ascending=False)
    high_residual = df[df["flag_model_curve_residual"] | df["flag_model_gls_residual"]].sort_values(
        ["model_peak_gls_abs_error", "model_curve_mae"],
        ascending=False,
    )
    summary_by_patient_view = _summary_by_patient_view(df)

    dataframe_preview_for_csv(df).to_csv(args.output_dir / "quality_all_samples.csv", index=False)
    dataframe_preview_for_csv(likely_bad).to_csv(args.output_dir / "likely_bad_strain_curves.csv", index=False)
    dataframe_preview_for_csv(high_disagreement).to_csv(args.output_dir / "high_inter_view_disagreement.csv", index=False)
    dataframe_preview_for_csv(high_residual).to_csv(args.output_dir / "high_model_residuals.csv", index=False)
    dataframe_preview_for_csv(summary_by_patient_view).to_csv(args.output_dir / "summary_by_patient_view.csv", index=False)

    flag_cols = [c for c in df.columns if c.startswith("flag_")]
    summary = {
        "prepared_parquet": str(args.prepared_parquet),
        "predictions_parquet": str(args.predictions_parquet) if args.predictions_parquet else None,
        "n_samples": int(len(df)),
        "n_patients": int(df["patient_key"].dropna().nunique()) if "patient_key" in df else 0,
        "n_likely_bad": int(len(likely_bad)),
        "n_high_inter_view_disagreement": int(len(high_disagreement)),
        "n_high_model_residuals": int(len(high_residual)),
        "flag_counts": {col: int(df[col].fillna(False).astype(bool).sum()) for col in flag_cols},
        "outputs": {
            "all_samples": str(args.output_dir / "quality_all_samples.csv"),
            "likely_bad": str(args.output_dir / "likely_bad_strain_curves.csv"),
            "high_inter_view_disagreement": str(args.output_dir / "high_inter_view_disagreement.csv"),
            "high_model_residuals": str(args.output_dir / "high_model_residuals.csv"),
            "summary_by_patient_view": str(args.output_dir / "summary_by_patient_view.csv"),
        },
    }
    write_json(args.output_dir / "quality_report.json", summary)
    _write_md(args.output_dir / "quality_report.md", summary)
    logger.info("Saved quality report: %s", args.output_dir / "quality_report.md")


if __name__ == "__main__":
    main()
