"""
Train baseline models to classify early vs late visits using DICOM embeddings.

Input:
  - Ichilov_GLS_embeddings_full.parquet with columns like:
      patient_num, study_datetime, view, embedding, embedding_dim

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
logger = logging.getLogger("Embedding_classification")


VIEW_CANON = {
    "2C": "A2C",
    "A2C": "A2C",
    "3C": "A3C",
    "A3C": "A3C",
    "4C": "A4C",
    "A4C": "A4C",
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


def _normalize_view(val: object) -> Optional[str]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip().upper()
    return VIEW_CANON.get(s, s or None)


def _parse_embedding(val: object) -> Optional[np.ndarray]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, np.ndarray):
        return val.astype(float, copy=False)
    if isinstance(val, list):
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
    try:
        return np.asarray(val, dtype=float)
    except Exception:
        return None


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


def _infer_target_dim(
    df: pd.DataFrame,
    embedding_col: str,
    dim_col: Optional[str],
) -> Optional[int]:
    if dim_col and dim_col in df.columns:
        vals = df[dim_col].dropna()
        if not vals.empty:
            try:
                return int(vals.mode().iloc[0])
            except Exception:
                pass
    lengths: List[int] = []
    for val in df[embedding_col].dropna():
        vec = _parse_embedding(val)
        if vec is None or vec.ndim != 1:
            continue
        lengths.append(int(vec.size))
    if not lengths:
        return None
    return int(Counter(lengths).most_common(1)[0][0])


def _build_samples(
    df: pd.DataFrame,
    patient_col: str,
    view_col: str,
    embedding_col: str,
    view_filter: Optional[str],
    target_dim: Optional[int],
    resample_dim: Optional[int],
    return_meta: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[int]]:
    features: List[np.ndarray] = []
    targets: List[float] = []
    patients: List[str] = []
    visit_indices: List[int] = []
    used_dim = resample_dim or target_dim
    skipped_dim = 0

    for _, row in df.iterrows():
        view = _normalize_view(row.get(view_col))
        if view_filter and view != view_filter:
            continue
        label = row.get("label")
        if pd.isna(label):
            continue
        vec = _parse_embedding(row.get(embedding_col))
        if vec is None or vec.ndim != 1 or vec.size < 2:
            continue
        if target_dim is not None and vec.size != target_dim:
            skipped_dim += 1
            continue
        if resample_dim is not None and resample_dim > 0:
            vec = resample_curve(vec, resample_dim)
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        if used_dim is None:
            used_dim = int(vec.size)
        if vec.size != used_dim:
            skipped_dim += 1
            continue
        features.append(vec.astype(np.float32, copy=False)[None, :])
        targets.append(float(label))
        patients.append(str(row.get(patient_col)))
        if return_meta:
            visit_indices.append(int(row.get("visit_index", 0)))

    if skipped_dim:
        logger.info("Skipped %d embeddings due to dimension mismatch.", skipped_dim)

    if not features:
        dim = used_dim if used_dim is not None else 1
        X = np.zeros((0, 1, dim), dtype=np.float32)
        y = np.array([], dtype=float)
        p = np.array([], dtype=object)
        if return_meta:
            return X, y, p, np.array([], dtype=int), used_dim
        return X, y, p, None, used_dim

    X = np.stack(features, axis=0)
    y = np.asarray(targets, dtype=float)
    p = np.asarray(patients, dtype=object)
    if return_meta:
        return X, y, p, np.asarray(visit_indices, dtype=int), used_dim
    return X, y, p, None, used_dim


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train embedding-based early/late classifiers.")
    user_home = Path.home()
    ap.add_argument(
        "--input-parquet",
        type=Path,
        default=user_home / "OneDrive - Technion" / "DS" / "Ichilov_GLS_embeddings_full.parquet",
        help="Path to parquet file containing embeddings.",
    )
    ap.add_argument(
        "--experiment-name",
        "--experiment_name",
        type=str,
        default="Embedding_classification",
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
        choices=["2C", "3C", "4C", "A2C", "A3C", "A4C", "both"],
        default="both",
        help="View filter.",
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
        "--embedding-dim",
        type=int,
        default=None,
        help="Expected embedding dimension; mismatched rows are skipped.",
    )
    ap.add_argument(
        "--T",
        type=int,
        default=None,
        help="Optional resample length for embeddings (default keeps original size).",
    )
    ap.add_argument(
        "--model",
        type=str,
        choices=["linear", "cnn", "transformer", "all"],
        default="cnn",
        help="Model type to train.",
    )
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

    if not args.input_parquet.is_file():
        raise FileNotFoundError(f"Parquet not found: {args.input_parquet}")
    df_raw = pd.read_parquet(args.input_parquet)
    df_raw.columns = [str(c).strip() for c in df_raw.columns]

    patient_col = _find_column(df_raw, ["patient_num", "patient number", "patient", "patient_id", "id"])
    date_col = _find_column(df_raw, ["study_datetime", "study date", "date", "study_date"])
    view_col = _find_column(df_raw, ["view"], required=False) or "view"
    emb_col = _find_column(df_raw, ["embedding", "embed", "vector", "features"])
    dim_col = _find_column(df_raw, ["embedding_dim", "embed_dim", "dim"], required=False)

    if date_col != "study_datetime":
        df_raw = df_raw.rename(columns={date_col: "study_datetime"})
        date_col = "study_datetime"
    if view_col != "view" and view_col in df_raw.columns:
        df_raw = df_raw.rename(columns={view_col: "view"})
        view_col = "view"

    df_raw[view_col] = df_raw[view_col].apply(_normalize_view)
    df_raw[date_col] = pd.to_datetime(df_raw[date_col], errors="coerce")
    df_raw = df_raw.dropna(subset=[date_col])

    if args.classify_mode == "first_last":
        clf_df = _prepare_labels_first_last(df_raw, patient_col=patient_col)
    else:
        clf_df = _prepare_labels_first_n(
            df_raw,
            patient_col=patient_col,
            early_visits=args.early_visits,
            late_visits=args.late_visits,
        )
    if clf_df.empty:
        raise ValueError("No labeled samples after applying early/late labeling rules.")

    view_filter = None if args.view == "both" else _normalize_view(args.view)
    target_dim = args.embedding_dim or _infer_target_dim(clf_df, embedding_col=emb_col, dim_col=dim_col)

    X3, y, p, visit_idx, used_dim = _build_samples(
        clf_df,
        patient_col=patient_col,
        view_col=view_col,
        embedding_col=emb_col,
        view_filter=view_filter,
        target_dim=target_dim,
        resample_dim=args.T,
        return_meta=True,
    )
    if X3.shape[0] == 0 or used_dim is None:
        raise ValueError("No samples after filtering; check view filter or embedding parsing.")

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.outdir / run_stamp
    logdir = args.logdir if args.logdir is not None else run_dir / "tensorboard"
    model_types = ["linear", "cnn", "transformer"] if args.model == "all" else [args.model]

    results: Dict[str, object] = {
        "input_parquet": str(args.input_parquet),
        "patient_col": patient_col,
        "view_filter": view_filter or "both",
        "T": int(used_dim),
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
        "sample_level": "embedding",
        "embedding_dim": int(target_dim) if target_dim is not None else None,
        "resample_dim": args.T,
        "tsne_enabled": bool(args.tsne),
        "tsne_max_samples": args.tsne_max_samples,
        "tsne_perplexity": args.tsne_perplexity,
        "tsne_seed": args.tsne_seed,
        "gls_threshold": None,
        "gls_slope_threshold": None,
    }

    X_flat = X3.reshape(X3.shape[0], -1)
    label_counts = {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    results["classification"] = {"embedding": {}}

    for model_type in model_types:
        tag = f"embedding_T{used_dim}_view{view_filter or 'both'}_model{model_type}_taskclass"
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
            results["classification"]["embedding"][model_type] = metrics
        except Exception as exc:
            results["classification"]["embedding"][model_type] = {
                "error": str(exc),
                "label_counts": label_counts,
            }

    results["classification"] = sort_nested_results(results["classification"])

    rows = collect_results_rows(results)
    results["results_table"] = rows
    table_path = run_dir / f"metrics_embedding_{run_stamp}.csv"
    write_results_table(rows, table_path)
    results["results_table_csv"] = str(table_path)

    out_metrics = run_dir / f"metrics_embedding_{run_stamp}.json"
    write_json(out_metrics, results)
    logger.info("Saved metrics to %s", out_metrics)


if __name__ == "__main__":
    main()
