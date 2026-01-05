"""
Train baseline models to classify early vs late TomTec strain curves from
the Ichilov GLS+strain report (xlsx).

Input:
  - Report_Ichilov_GLS_and_Strain_oron.xlsx with columns like:
      A2C_STRAIN_CURVES_JSON, A3C_STRAIN_CURVES_JSON, A4C_STRAIN_CURVES_JSON

Output:
  - Metrics JSON/CSV, confusion matrix plots, and trained models.
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from strain_curve_visit_models import (
    collect_results_rows,
    resample_curve,
    run_classification,
    run_classification_torch,
    set_seed,
    sort_nested_results,
    write_json,
    write_results_table,
)


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("TomTec_strain_curves_classification")


VIEW_MAP = {"A2C": "2C", "A3C": "3C", "A4C": "4C"}
VIEW_CURVE_COLS = {
    "A2C": "A2C_STRAIN_CURVES_JSON",
    "A3C": "A3C_STRAIN_CURVES_JSON",
    "A4C": "A4C_STRAIN_CURVES_JSON",
}


def _find_column(df: pd.DataFrame, candidates: Iterable[str], required: bool = True) -> Optional[str]:
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in lower_map:
            return lower_map[key]
    for cand in candidates:
        key = cand.lower()
        for col in df.columns:
            if key in str(col).lower():
                return col
    if required:
        raise ValueError(f"Missing required column. Tried: {list(candidates)}")
    return None


def _parse_curve_blob(val: object) -> Optional[np.ndarray]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, (list, np.ndarray)):
        try:
            return np.asarray(val, dtype=float)
        except Exception:
            return None
    if isinstance(val, str):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(val)
                return np.asarray(parsed, dtype=float)
            except Exception:
                continue
    return None


def _normalize_curve_matrix(arr: np.ndarray, curves_first: Optional[bool]) -> Optional[np.ndarray]:
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2:
        return None
    if curves_first is True:
        return arr
    if curves_first is False:
        return arr.T
    if arr.shape[0] <= arr.shape[1]:
        return arr
    return arr.T


def _build_visit_rows(
    df: pd.DataFrame,
    patient_col: str,
    date_col: str,
    curves_first: Optional[bool],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        pid = row.get(patient_col)
        if pd.isna(pid):
            continue
        dt = pd.to_datetime(row.get(date_col), errors="coerce")
        if pd.isna(dt):
            continue
        for view_key, col in VIEW_CURVE_COLS.items():
            if col not in df.columns:
                continue
            arr = _parse_curve_blob(row.get(col))
            if arr is None:
                continue
            arr = _normalize_curve_matrix(arr, curves_first=curves_first)
            if arr is None or arr.size == 0:
                continue
            curves: List[np.ndarray] = []
            for curve in arr:
                curve_arr = np.asarray(curve, dtype=float)
                if curve_arr.size < 2:
                    continue
                curves.append(curve_arr)
            if not curves:
                continue
            rows.append(
                {
                    patient_col: str(pid).strip(),
                    "study_datetime": dt,
                    "view": VIEW_MAP.get(view_key, view_key),
                    "curves": curves,
                    "n_curves": len(curves),
                }
            )
    return pd.DataFrame(rows)


def _add_visit_index(df: pd.DataFrame, patient_col: str) -> pd.DataFrame:
    out = df.copy()
    out["study_datetime"] = pd.to_datetime(out["study_datetime"], errors="coerce")
    visits = (
        out[[patient_col, "study_datetime"]]
        .dropna(subset=["study_datetime"])
        .drop_duplicates()
        .sort_values([patient_col, "study_datetime"])
    )
    visits["visit_index"] = visits.groupby(patient_col).cumcount()
    out = out.merge(visits, on=[patient_col, "study_datetime"], how="inner")
    return out


def _prepare_labels_first_last(df: pd.DataFrame, patient_col: str) -> pd.DataFrame:
    out = _add_visit_index(df, patient_col)
    visit_counts = out.groupby(patient_col)["study_datetime"].nunique()
    keep_ids = visit_counts[visit_counts >= 2].index
    out = out[out[patient_col].isin(keep_ids)].copy()
    out["min_dt"] = out.groupby(patient_col)["study_datetime"].transform("min")
    out["max_dt"] = out.groupby(patient_col)["study_datetime"].transform("max")
    out = out[(out["study_datetime"] == out["min_dt"]) | (out["study_datetime"] == out["max_dt"])].copy()
    out["label"] = (out["study_datetime"] == out["max_dt"]).astype(int)
    return out


def _prepare_labels_first_n(
    df: pd.DataFrame,
    patient_col: str,
    early_visits: int,
    late_visits: int,
) -> pd.DataFrame:
    out = _add_visit_index(df, patient_col)
    min_visits = early_visits + late_visits
    visit_counts = out.groupby(patient_col)["visit_index"].max().add(1)
    keep_ids = visit_counts[visit_counts >= min_visits].index
    out = out[out[patient_col].isin(keep_ids)].copy()
    out = out[out["visit_index"] < min_visits].copy()
    out["label"] = (out["visit_index"] >= early_visits).astype(int)
    return out


def _infer_expected_curves(df: pd.DataFrame, view_filter: Optional[str]) -> Optional[int]:
    counts = []
    for _, row in df.iterrows():
        if view_filter and row.get("view") != view_filter:
            continue
        n = row.get("n_curves")
        if isinstance(n, (int, np.integer)) and n > 0:
            counts.append(int(n))
    if not counts:
        return None
    most_common = Counter(counts).most_common(1)[0][0]
    return int(most_common)


def _average_curves(curves: Iterable[np.ndarray], T: int) -> Optional[np.ndarray]:
    resampled: List[np.ndarray] = []
    for curve in curves:
        arr = np.asarray(curve, dtype=float)
        if arr.size < 2:
            continue
        res = resample_curve(arr, T)
        res = np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)
        resampled.append(res.astype(np.float32, copy=False))
    if not resampled:
        return None
    return np.mean(np.stack(resampled, axis=0), axis=0)


def _build_samples_curve(
    df: pd.DataFrame,
    patient_col: str,
    T: int,
    view_filter: Optional[str],
    average_segments: bool,
    return_meta: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    features: List[np.ndarray] = []
    targets: List[float] = []
    patients: List[str] = []
    visit_indices: List[int] = []

    for _, row in df.iterrows():
        view = str(row.get("view", "")).strip().upper()
        if view_filter and view != view_filter:
            continue
        label = row.get("label")
        if pd.isna(label):
            continue
        curves = row.get("curves") or []
        if average_segments:
            avg = _average_curves(curves, T)
            if avg is None:
                continue
            features.append(avg[None, :])
            targets.append(float(label))
            patients.append(str(row.get(patient_col)))
            if return_meta:
                visit_indices.append(int(row.get("visit_index", 0)))
        else:
            for curve in curves:
                arr = np.asarray(curve, dtype=float)
                if arr.size < 2:
                    continue
                res = resample_curve(arr, T)
                res = np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)
                features.append(res.astype(np.float32, copy=False)[None, :])
                targets.append(float(label))
                patients.append(str(row.get(patient_col)))
                if return_meta:
                    visit_indices.append(int(row.get("visit_index", 0)))

    if not features:
        X = np.zeros((0, 1, T), dtype=np.float32)
        y = np.array([], dtype=float)
        p = np.array([], dtype=object)
        if return_meta:
            return X, y, p, np.array([], dtype=int)
        return X, y, p, None

    X = np.stack(features, axis=0)
    y = np.asarray(targets, dtype=float)
    p = np.asarray(patients, dtype=object)
    if return_meta:
        return X, y, p, np.asarray(visit_indices, dtype=int)
    return X, y, p, None


def _build_samples_view(
    df: pd.DataFrame,
    patient_col: str,
    T: int,
    view_filter: Optional[str],
    expected_curves: Optional[int],
    average_segments: bool,
    return_meta: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    features: List[np.ndarray] = []
    targets: List[float] = []
    patients: List[str] = []
    visit_indices: List[int] = []

    for _, row in df.iterrows():
        view = str(row.get("view", "")).strip().upper()
        if view_filter and view != view_filter:
            continue
        label = row.get("label")
        if pd.isna(label):
            continue
        curves = row.get("curves") or []
        if not average_segments and expected_curves is not None and len(curves) != expected_curves:
            continue
        if average_segments:
            avg = _average_curves(curves, T)
            if avg is None:
                continue
            features.append(avg[None, :])
        else:
            curve_features: List[np.ndarray] = []
            ok = True
            for curve in curves:
                arr = np.asarray(curve, dtype=float)
                if arr.size < 2:
                    ok = False
                    break
                res = resample_curve(arr, T)
                res = np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)
                curve_features.append(res.astype(np.float32, copy=False))
            if not ok or not curve_features:
                continue
            features.append(np.stack(curve_features, axis=0))
        targets.append(float(label))
        patients.append(str(row.get(patient_col)))
        if return_meta:
            visit_indices.append(int(row.get("visit_index", 0)))

    if not features:
        n_curves = 1 if average_segments else (expected_curves if expected_curves is not None else 1)
        X = np.zeros((0, n_curves, T), dtype=np.float32)
        y = np.array([], dtype=float)
        p = np.array([], dtype=object)
        if return_meta:
            return X, y, p, np.array([], dtype=int)
        return X, y, p, None

    X = np.stack(features, axis=0)
    y = np.asarray(targets, dtype=float)
    p = np.asarray(patients, dtype=object)
    if return_meta:
        return X, y, p, np.asarray(visit_indices, dtype=int)
    return X, y, p, None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train TomTec strain curve classifiers (early vs late).")
    user_home = Path.home()
    ap.add_argument(
        "--input-xlsx",
        type=Path,
        default=user_home / "OneDrive - Technion" / "DS" / "Report_Ichilov_GLS_and_Strain_oron.xlsx",
        help="Path to TomTec report xlsx with strain curves.",
    )
    ap.add_argument(
        "--experiment-name",
        type=str,
        default="TomTec_strain_curves_classification",
        help="Subfolder name appended to --outdir.",
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        default=user_home / "OneDrive - Technion" / "Experiments",
        help="Base output directory for models and metrics; experiment name is appended.",
    )
    ap.add_argument(
        "--view",
        type=str,
        choices=["2C", "3C", "4C", "both"],
        default="both",
        help="View filter.",
    )
    ap.add_argument(
        "--sample-level",
        type=str,
        choices=["curve", "view"],
        default="view",
        help="Sample granularity: curve uses each trace; view uses all traces together.",
    )
    ap.add_argument(
        "--average-segments",
        default=True,
        action="store_true",
        help="Average all segment curves into a single global strain curve per view.",
    )
    ap.add_argument(
        "--classify-mode",
        type=str,
        choices=["first_last", "first_n"],
        default="first_last",
        help="Labeling scheme for early vs late visits.",
    )
    ap.add_argument("--early-visits", type=int, default=2, help="Number of earliest visits labeled early.")
    ap.add_argument("--late-visits", type=int, default=2, help="Number of subsequent visits labeled late.")
    ap.add_argument(
        "--expected-curves",
        type=int,
        default=None,
        help="Expected number of curves per view (only used for sample-level=view).",
    )
    curve_group = ap.add_mutually_exclusive_group()
    curve_group.add_argument(
        "--curves-first",
        action="store_true",
        help="Assume curves are the first axis in the JSON matrix.",
    )
    curve_group.add_argument(
        "--frames-first",
        action="store_true",
        help="Assume frames are the first axis in the JSON matrix.",
    )
    ap.add_argument(
        "--model",
        type=str,
        choices=["linear", "cnn", "transformer", "all"],
        default="linear",
        help="Model type to train.",
    )
    ap.add_argument("--T", type=int, default=64, help="Resample length for each curve.")
    ap.add_argument("--test-size", type=float, default=0.2, help="Validation split fraction by patient.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=20, help="Epochs for CNN/Transformer training.")
    ap.add_argument("--batch", type=int, default=64, help="Batch size for CNN/Transformer training.")
    ap.add_argument("--lr", type=float, default=1e-3, help="Learning rate for CNN/Transformer training.")
    ap.add_argument(
        "--tsne",
        default=True,
        action="store_true",
        help="Enable t-SNE plots using the learned latent representations.",
    )
    ap.add_argument(
        "--tsne-max-samples",
        type=int,
        default=1000,
        help="Maximum number of samples to use for t-SNE.",
    )
    ap.add_argument(
        "--tsne-perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity (will be capped to valid range).",
    )
    ap.add_argument(
        "--tsne-seed",
        type=int,
        default=42,
        help="Random seed for t-SNE subsampling and initialization.",
    )
    ap.add_argument(
        "--logdir",
        type=Path,
        default=None,
        help="Optional TensorBoard log directory. Defaults to <outdir>/<run_stamp>/tensorboard.",
    )
    ap.add_argument(
        "--early-stop-patience",
        type=int,
        default=5,
        help="Stop when train decreases and val increases for this many consecutive epochs.",
    )
    ap.add_argument(
        "--early-stop-delta",
        type=float,
        default=0.0,
        help="Minimum change required to count as divergence.",
    )
    args = ap.parse_args()
    if args.experiment_name:
        args.outdir = args.outdir / args.experiment_name
    return args


def main() -> None:
    args = parse_args()
    use_torch = args.model in ("cnn", "transformer", "all")
    set_seed(args.seed, use_torch=use_torch)

    if not args.input_xlsx.is_file():
        raise FileNotFoundError(f"Input xlsx not found: {args.input_xlsx}")
    df_raw = pd.read_excel(args.input_xlsx, engine="openpyxl")
    df_raw.columns = [str(c).strip() for c in df_raw.columns]

    patient_col = _find_column(df_raw, ["PatientNum", "Patient Num", "Patient Number", "Patient"])
    date_col = _find_column(df_raw, ["Study Date", "StudyDate", "Date"])

    curves_first = None
    if args.curves_first:
        curves_first = True
    elif args.frames_first:
        curves_first = False

    visit_df = _build_visit_rows(df_raw, patient_col=patient_col, date_col=date_col, curves_first=curves_first)
    if visit_df.empty:
        raise ValueError("No TomTec strain curves found in input xlsx.")

    if args.classify_mode == "first_last":
        clf_df = _prepare_labels_first_last(visit_df, patient_col=patient_col)
    else:
        clf_df = _prepare_labels_first_n(
            visit_df,
            patient_col=patient_col,
            early_visits=args.early_visits,
            late_visits=args.late_visits,
        )

    if clf_df.empty:
        raise ValueError("No labeled samples after applying early/late labeling rules.")

    view_filter = None if args.view == "both" else args.view
    expected_curves = args.expected_curves
    if args.sample_level == "view" and not args.average_segments and expected_curves is None:
        expected_curves = _infer_expected_curves(clf_df, view_filter=view_filter)
        if expected_curves is None:
            raise ValueError("Could not infer expected curve count for view-level samples.")

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.outdir / run_stamp
    logdir = args.logdir if args.logdir is not None else run_dir / "tensorboard"

    model_types = ["linear", "cnn", "transformer"] if args.model == "all" else [args.model]

    results: Dict[str, object] = {
        "input_xlsx": str(args.input_xlsx),
        "patient_col": patient_col,
        "view_filter": view_filter or "both",
        "T": args.T,
        "seed": args.seed,
        "run_dir": str(run_dir),
        "model_choice": args.model,
        "model_types": model_types,
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "tensorboard_logdir": str(logdir),
        "early_stop_patience": args.early_stop_patience,
        "early_stop_delta": args.early_stop_delta,
        "classify_mode": args.classify_mode,
        "early_visits": args.early_visits,
        "late_visits": args.late_visits,
        "sample_level": args.sample_level,
        "expected_curves": expected_curves,
        "average_segments": bool(args.average_segments),
        "tsne_enabled": bool(args.tsne),
        "tsne_max_samples": args.tsne_max_samples,
        "tsne_perplexity": args.tsne_perplexity,
        "tsne_seed": args.tsne_seed,
    }

    if args.sample_level == "view":
        X3, y, p, visit_idx = _build_samples_view(
            clf_df,
            patient_col=patient_col,
            T=args.T,
            view_filter=view_filter,
            expected_curves=expected_curves,
            average_segments=args.average_segments,
            return_meta=True,
        )
    else:
        X3, y, p, visit_idx = _build_samples_curve(
            clf_df,
            patient_col=patient_col,
            T=args.T,
            view_filter=view_filter,
            average_segments=args.average_segments,
            return_meta=True,
        )

    if X3.shape[0] == 0:
        raise ValueError("No samples after filtering; check view filter or curve parsing.")

    X_flat = X3.reshape(X3.shape[0], -1)
    label_counts = {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    results["classification"] = {"tomtec": {}}

    avg_tag = "_avgseg" if args.average_segments else ""
    for model_type in model_types:
        tag = (
            f"tomtec_T{args.T}_view{view_filter or 'both'}_sample{args.sample_level}"
            f"{avg_tag}_model{model_type}_taskclass"
        )
        try:
            if model_type == "linear":
                metrics = run_classification(
                    X_flat,
                    y,
                    p,
                    visit_idx,
                    outdir=run_dir,
                    tag=tag,
                    test_size=args.test_size,
                    seed=args.seed,
                    model_type=model_type,
                    tsne_enabled=args.tsne,
                    tsne_max_samples=args.tsne_max_samples,
                    tsne_perplexity=args.tsne_perplexity,
                    tsne_seed=args.tsne_seed,
                )
            else:
                metrics = run_classification_torch(
                    X3,
                    y,
                    p,
                    visit_idx,
                    outdir=run_dir,
                    tag=tag,
                    test_size=args.test_size,
                    seed=args.seed,
                    model_type=model_type,
                    epochs=args.epochs,
                    batch_size=args.batch,
                    lr=args.lr,
                    logdir=logdir,
                    early_stop_patience=args.early_stop_patience,
                    early_stop_delta=args.early_stop_delta,
                    tsne_enabled=args.tsne,
                    tsne_max_samples=args.tsne_max_samples,
                    tsne_perplexity=args.tsne_perplexity,
                    tsne_seed=args.tsne_seed,
                )
            metrics["label_counts"] = label_counts
            results["classification"]["tomtec"][model_type] = metrics
        except Exception as exc:
            results["classification"]["tomtec"][model_type] = {
                "error": str(exc),
                "label_counts": label_counts,
            }

    results["classification"] = sort_nested_results(results["classification"])

    rows = collect_results_rows(results)
    results["results_table"] = rows
    table_path = run_dir / f"metrics_tomtec_{run_stamp}.csv"
    write_results_table(rows, table_path)
    results["results_table_csv"] = str(table_path)

    out_metrics = run_dir / f"metrics_tomtec_{run_stamp}.json"
    write_json(out_metrics, results)
    logger.info("Saved metrics to %s", out_metrics)


if __name__ == "__main__":
    main()
