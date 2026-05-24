"""
Train a temporal model to predict strain curves and peak GLS from phase-frame embeddings.

Example:
  python train_strain_curve_model.py ^
    --input-parquet "C:\\path\\phase_embeddings.parquet" ^
    --output-dir "C:\\path\\train"
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ichilov_strain_curve_utils import dataframe_preview_for_csv, is_missing, write_json
from strain_curve_model import build_strain_curve_model


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("train_strain_curve_model")

VIEW_TO_ID = {"A2C": 0, "A3C": 1, "A4C": 2}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _array(value: object, dtype=np.float32) -> np.ndarray:
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return np.asarray(value.tolist(), dtype=dtype)
        return value.astype(dtype, copy=False)
    if isinstance(value, list):
        try:
            return np.asarray(value, dtype=dtype)
        except Exception:
            return np.asarray([np.asarray(v, dtype=dtype) for v in value], dtype=dtype)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return np.asarray([], dtype=dtype)
        for parser in (json.loads, ast.literal_eval):
            try:
                return np.asarray(parser(text), dtype=dtype)
            except Exception:
                pass
    try:
        return np.asarray(value, dtype=dtype)
    except Exception:
        return np.asarray([], dtype=dtype)


def _finite_float(value: object) -> Optional[float]:
    if is_missing(value):
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if np.isfinite(out) else None


def _clean_training_df(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    skipped = {"missing_embedding": 0, "bad_embedding_shape": 0, "missing_curve": 0, "missing_patient": 0}
    for _, row in df.iterrows():
        emb = _array(row.get("embedding"))
        curve = _array(row.get("resampled_strain_curve"))
        if emb.size == 0:
            skipped["missing_embedding"] += 1
            continue
        if emb.ndim != 2:
            skipped["bad_embedding_shape"] += 1
            continue
        if curve.size < 2:
            skipped["missing_curve"] += 1
            continue
        patient = row.get("patient_key")
        if is_missing(patient):
            skipped["missing_patient"] += 1
            continue
        row_dict = row.to_dict()
        row_dict["_embedding_np"] = emb.astype(np.float32, copy=False)
        row_dict["_curve_np"] = curve.astype(np.float32, copy=False)
        rows.append(row_dict)
    logger.info("Training dataframe: kept %d/%d rows; skipped=%s", len(rows), len(df), skipped)
    return pd.DataFrame(rows)


class StrainEmbeddingDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        emb = row["_embedding_np"]
        mask = _array(row.get("frame_mask"))
        if mask.size != emb.shape[0]:
            mask = np.ones((emb.shape[0],), dtype=np.float32)
        curve = row["_curve_np"]
        peak = _finite_float(row.get("peak_gls_from_report"))
        if peak is None:
            peak = _finite_float(row.get("peak_gls_from_curve"))
        ttp = _finite_float(row.get("time_to_peak_from_curve"))
        view = str(row.get("view") or "")
        return {
            "embedding": torch.from_numpy(emb),
            "frame_mask": torch.from_numpy(mask.astype(np.float32, copy=False)),
            "curve": torch.from_numpy(curve),
            "peak": torch.tensor(np.nan if peak is None else peak, dtype=torch.float32),
            "ttp": torch.tensor(np.nan if ttp is None else ttp, dtype=torch.float32),
            "view_id": torch.tensor(VIEW_TO_ID.get(view, 3), dtype=torch.long),
            "meta_index": idx,
        }


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    max_t = max(item["embedding"].shape[0] for item in batch)
    dim = batch[0]["embedding"].shape[1]
    curve_len = batch[0]["curve"].shape[0]
    embeddings = torch.zeros(len(batch), max_t, dim, dtype=torch.float32)
    masks = torch.zeros(len(batch), max_t, dtype=torch.float32)
    curves = torch.zeros(len(batch), curve_len, dtype=torch.float32)
    peaks = torch.zeros(len(batch), dtype=torch.float32)
    ttps = torch.zeros(len(batch), dtype=torch.float32)
    view_ids = torch.zeros(len(batch), dtype=torch.long)
    meta_indices = torch.zeros(len(batch), dtype=torch.long)
    for i, item in enumerate(batch):
        t = item["embedding"].shape[0]
        embeddings[i, :t] = item["embedding"]
        masks[i, :t] = item["frame_mask"][:t]
        curves[i] = item["curve"]
        peaks[i] = item["peak"]
        ttps[i] = item["ttp"]
        view_ids[i] = item["view_id"]
        meta_indices[i] = int(item["meta_index"])
    return {
        "embedding": embeddings,
        "frame_mask": masks,
        "curve": curves,
        "peak": peaks,
        "ttp": ttps,
        "view_id": view_ids,
        "meta_index": meta_indices,
    }


def _masked_regression_loss(pred: torch.Tensor, target: torch.Tensor, kind: str) -> torch.Tensor:
    mask = torch.isfinite(target)
    if mask.sum() == 0:
        return pred.sum() * 0.0
    pred_m = pred[mask]
    target_m = target[mask]
    if kind == "l1":
        return F.l1_loss(pred_m, target_m)
    if kind == "mse":
        return F.mse_loss(pred_m, target_m)
    return F.huber_loss(pred_m, target_m, delta=1.0)


def _derivative_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape[1] < 2:
        return pred.sum() * 0.0
    return _masked_regression_loss(torch.diff(pred, dim=1), torch.diff(target, dim=1), "l1")


def _smoothness_loss(pred: torch.Tensor) -> torch.Tensor:
    if pred.shape[1] < 3:
        return pred.sum() * 0.0
    return torch.mean(torch.abs(torch.diff(pred, n=2, dim=1)))


def _compute_loss(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], args: argparse.Namespace) -> Tuple[torch.Tensor, Dict[str, float]]:
    curve_loss = _masked_regression_loss(outputs["pred_curve"], batch["curve"], args.curve_loss)
    peak_loss = _masked_regression_loss(outputs["pred_peak_gls"], batch["peak"], "l1")
    ttp_loss = _masked_regression_loss(outputs["pred_time_to_peak"], batch["ttp"], "l1")
    derivative = _derivative_loss(outputs["pred_curve"], batch["curve"])
    smoothness = _smoothness_loss(outputs["pred_curve"])
    total = (
        args.curve_loss_weight * curve_loss
        + args.peak_loss_weight * peak_loss
        + args.ttp_loss_weight * ttp_loss
        + args.derivative_loss_weight * derivative
        + args.smoothness_loss_weight * smoothness
    )
    parts = {
        "loss": float(total.detach().cpu()),
        "curve_loss": float(curve_loss.detach().cpu()),
        "peak_loss": float(peak_loss.detach().cpu()),
        "ttp_loss": float(ttp_loss.detach().cpu()),
        "derivative_loss": float(derivative.detach().cpu()),
        "smoothness_loss": float(smoothness.detach().cpu()),
    }
    return total, parts


def _pearson(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return None
    aa = a[mask]
    bb = b[mask]
    if np.std(aa) == 0 or np.std(bb) == 0:
        return None
    return float(np.corrcoef(aa, bb)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    try:
        from scipy.stats import spearmanr

        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 2:
            return None
        corr = spearmanr(a[mask], b[mask]).correlation
        return float(corr) if np.isfinite(corr) else None
    except Exception:
        return None


def compute_metrics(preds: pd.DataFrame) -> Dict[str, Any]:
    if preds.empty:
        return {}
    true_curves = np.stack(preds["true_curve"].map(lambda x: _array(x, dtype=np.float32)).to_numpy())
    pred_curves = np.stack(preds["pred_curve"].map(lambda x: _array(x, dtype=np.float32)).to_numpy())
    diff = pred_curves - true_curves
    true_peak = pd.to_numeric(preds["true_peak_gls"], errors="coerce").to_numpy(dtype=float)
    pred_peak = pd.to_numeric(preds["pred_peak_gls"], errors="coerce").to_numpy(dtype=float)
    true_ttp = pd.to_numeric(preds["true_time_to_peak"], errors="coerce").to_numpy(dtype=float)
    pred_ttp = pd.to_numeric(preds["pred_time_to_peak"], errors="coerce").to_numpy(dtype=float)
    metrics: Dict[str, Any] = {
        "n": int(len(preds)),
        "curve_mae": float(np.nanmean(np.abs(diff))),
        "curve_rmse": float(np.sqrt(np.nanmean(diff**2))),
        "derivative_mae": float(np.nanmean(np.abs(np.diff(pred_curves, axis=1) - np.diff(true_curves, axis=1)))),
        "peak_gls_mae": float(np.nanmean(np.abs(pred_peak - true_peak))),
        "peak_gls_rmse": float(np.sqrt(np.nanmean((pred_peak - true_peak) ** 2))),
        "peak_gls_pearson": _pearson(true_peak, pred_peak),
        "peak_gls_spearman": _spearman(true_peak, pred_peak),
        "time_to_peak_mae": float(np.nanmean(np.abs(pred_ttp - true_ttp))),
    }
    view_metrics = {}
    if "view" in preds.columns:
        for view, sub in preds.groupby("view"):
            view_metrics[str(view)] = compute_metrics(sub.drop(columns=["view"], errors="ignore"))
    metrics["by_view"] = view_metrics
    if "peak_in_reasonable_range" in preds.columns:
        metrics["by_quality_peak_reasonable"] = {}
        for quality, sub in preds.groupby("peak_in_reasonable_range", dropna=False):
            metrics["by_quality_peak_reasonable"][str(quality)] = compute_metrics(sub.drop(columns=["peak_in_reasonable_range"], errors="ignore"))
    return metrics


def _evaluate(model: torch.nn.Module, loader: DataLoader, df_meta: pd.DataFrame, device: torch.device) -> pd.DataFrame:
    model.eval()
    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            batch_device = {
                k: v.to(device) if isinstance(v, torch.Tensor) and k != "meta_index" else v
                for k, v in batch.items()
            }
            outputs = model(batch_device["embedding"], batch_device["frame_mask"], batch_device["view_id"])
            pred_curve = outputs["pred_curve"].detach().cpu().numpy()
            pred_peak = outputs["pred_peak_gls"].detach().cpu().numpy()
            pred_ttp = outputs["pred_time_to_peak"].detach().cpu().numpy()
            true_curve = batch["curve"].detach().cpu().numpy()
            true_peak = batch["peak"].detach().cpu().numpy()
            true_ttp = batch["ttp"].detach().cpu().numpy()
            for i, meta_idx in enumerate(batch["meta_index"].tolist()):
                meta = df_meta.iloc[int(meta_idx)].to_dict()
                rows.append(
                    {
                        "sample_id": meta.get("sample_id"),
                        "patient_key": meta.get("patient_key"),
                        "dicom_path": meta.get("dicom_path"),
                        "view": meta.get("view"),
                        "true_curve": true_curve[i].astype(float).tolist(),
                        "pred_curve": pred_curve[i].astype(float).tolist(),
                        "true_peak_gls": float(true_peak[i]) if np.isfinite(true_peak[i]) else np.nan,
                        "pred_peak_gls": float(pred_peak[i]),
                        "true_time_to_peak": float(true_ttp[i]) if np.isfinite(true_ttp[i]) else np.nan,
                        "pred_time_to_peak": float(pred_ttp[i]),
                        "starts_near_zero": meta.get("starts_near_zero"),
                        "has_valid_peak": meta.get("has_valid_peak"),
                        "peak_in_reasonable_range": meta.get("peak_in_reasonable_range"),
                        "excessive_noise_score": meta.get("excessive_noise_score"),
                        "num_large_peaks": meta.get("num_large_peaks"),
                        "curve_nan_fraction": meta.get("curve_nan_fraction"),
                        "view_gls_disagreement": meta.get("view_gls_disagreement"),
                    }
                )
    return pd.DataFrame(rows)


def _plot_curve(true_curve: Iterable[float], pred_curve: Iterable[float], title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(list(true_curve), label="true", linewidth=2)
    ax.plot(list(pred_curve), label="pred", linewidth=2)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("normalized curve index")
    ax.set_ylabel("strain")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_scatter(preds: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.scatter(preds["true_peak_gls"], preds["pred_peak_gls"], alpha=0.75)
    vals = pd.to_numeric(pd.concat([preds["true_peak_gls"], preds["pred_peak_gls"]]), errors="coerce").dropna()
    if len(vals):
        lo, hi = float(vals.min()), float(vals.max())
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
    ax.set_xlabel("true peak GLS")
    ax.set_ylabel("predicted peak GLS")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_bland_altman(preds: pd.DataFrame, out_path: Path) -> None:
    true = pd.to_numeric(preds["true_peak_gls"], errors="coerce").to_numpy(dtype=float)
    pred = pd.to_numeric(preds["pred_peak_gls"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(true) & np.isfinite(pred)
    mean = (true[mask] + pred[mask]) / 2.0
    diff = pred[mask] - true[mask]
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.scatter(mean, diff, alpha=0.75)
    if diff.size:
        md = float(np.mean(diff))
        sd = float(np.std(diff))
        ax.axhline(md, color="black", linewidth=1, label="bias")
        ax.axhline(md + 1.96 * sd, color="gray", linestyle="--", linewidth=1)
        ax.axhline(md - 1.96 * sd, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("mean GLS")
    ax.set_ylabel("pred - true GLS")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_ttp(preds: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.scatter(preds["true_time_to_peak"], preds["pred_time_to_peak"], alpha=0.75)
    ax.plot([0, 1], [0, 1], color="black", linewidth=1)
    ax.set_xlabel("true time-to-peak")
    ax.set_ylabel("predicted time-to-peak")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_diagnostics(preds: pd.DataFrame, output_dir: Path, max_samples: int) -> None:
    plot_dir = output_dir / "plots"
    if preds.empty:
        return
    curve_mae = preds.apply(
        lambda r: float(np.nanmean(np.abs(_array(r["pred_curve"]) - _array(r["true_curve"])))),
        axis=1,
    )
    preds = preds.copy()
    preds["curve_mae"] = curve_mae
    preds["peak_abs_error"] = np.abs(
        pd.to_numeric(preds["pred_peak_gls"], errors="coerce")
        - pd.to_numeric(preds["true_peak_gls"], errors="coerce")
    )
    for _, row in preds.sort_values("curve_mae", ascending=False).head(max_samples).iterrows():
        sample_id = str(row.get("sample_id") or "sample")
        safe_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in sample_id)
        _plot_curve(row["true_curve"], row["pred_curve"], f"{row.get('view')} {sample_id}", plot_dir / "curves" / f"{safe_id}.png")
    _plot_scatter(preds, plot_dir / "peak_gls_scatter.png")
    _plot_bland_altman(preds, plot_dir / "peak_gls_bland_altman.png")
    _plot_ttp(preds, plot_dir / "time_to_peak_scatter.png")
    dataframe_preview_for_csv(preds.sort_values("curve_mae", ascending=False).head(50)).to_csv(
        output_dir / "worst_curve_mae_cases.csv",
        index=False,
    )
    dataframe_preview_for_csv(preds.sort_values("peak_abs_error", ascending=False).head(50)).to_csv(
        output_dir / "worst_gls_error_cases.csv",
        index=False,
    )


def _metrics_md(metrics: Dict[str, Any]) -> str:
    lines = ["# Strain Curve Training Metrics", ""]
    for key in ("n", "curve_mae", "curve_rmse", "peak_gls_mae", "peak_gls_pearson", "peak_gls_spearman", "time_to_peak_mae", "derivative_mae"):
        lines.append(f"- {key}: {metrics.get(key)}")
    if metrics.get("by_view"):
        lines.extend(["", "## By View"])
        for view, vals in metrics["by_view"].items():
            lines.append(f"- {view}: peak_gls_mae={vals.get('peak_gls_mae')}, curve_mae={vals.get('curve_mae')}, n={vals.get('n')}")
    return "\n".join(lines) + "\n"


def _split_by_patient(df: pd.DataFrame, val_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    groups = df["patient_key"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        raise ValueError("Need at least two patient groups for train/validation split.")
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(splitter.split(df, groups=groups))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


def train_once(df: pd.DataFrame, args: argparse.Namespace, output_dir: Path, seed: int) -> Dict[str, Any]:
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df, val_df = _split_by_patient(df, args.val_ratio, seed)
    logger.info(
        "Patient split: train samples=%d patients=%d; val samples=%d patients=%d",
        len(train_df),
        train_df["patient_key"].nunique(),
        len(val_df),
        val_df["patient_key"].nunique(),
    )

    first_emb = train_df.iloc[0]["_embedding_np"]
    first_curve = train_df.iloc[0]["_curve_np"]
    model_config = {
        "input_dim": int(first_emb.shape[1]),
        "curve_length": int(first_curve.shape[0]),
        "model_type": args.model_type,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
        "derive_peak_from_curve": args.derive_predicted_peak_from_curve,
        "use_view_embedding": True,
    }
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    model = build_strain_curve_model(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=bool(args.use_amp and device.type == "cuda"))
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=bool(args.use_amp and device.type == "cuda"))

    train_loader = DataLoader(
        StrainEmbeddingDataset(train_df),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        StrainEmbeddingDataset(val_df),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )

    history: List[Dict[str, Any]] = []
    best_metric = float("inf")
    best_path = output_dir / "strain_curve_model_best.pt"
    latest_path = output_dir / "strain_curve_model_latest.pt"
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: List[Dict[str, float]] = []
        for batch in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            batch_device = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=bool(args.use_amp and device.type == "cuda")):
                outputs = model(batch_device["embedding"], batch_device["frame_mask"], batch_device["view_id"])
                loss, parts = _compute_loss(outputs, batch_device, args)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            losses.append(parts)

        val_preds = _evaluate(model, val_loader, val_df, device)
        val_metrics = compute_metrics(val_preds)
        train_loss = float(np.mean([x["loss"] for x in losses])) if losses else np.nan
        epoch_record = {"epoch": epoch, "train_loss": train_loss, "val": val_metrics}
        history.append(epoch_record)
        metric = float(val_metrics.get(args.early_stopping_metric, val_metrics.get("peak_gls_mae", np.inf)))
        logger.info(
            "epoch=%d train_loss=%.4f val_peak_mae=%.4f val_curve_mae=%.4f",
            epoch,
            train_loss,
            float(val_metrics.get("peak_gls_mae", np.nan)),
            float(val_metrics.get("curve_mae", np.nan)),
        )

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "model_config": model_config,
            "train_config": vars(args),
            "best_metric": min(best_metric, metric),
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint, latest_path)
        if metric < best_metric:
            best_metric = metric
            no_improve = 0
            torch.save(checkpoint, best_path)
            val_preds.to_parquet(output_dir / "validation_predictions_best.parquet", index=False)
        else:
            no_improve += 1
            if no_improve >= args.early_stopping_patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    history_path = output_dir / "history.json"
    write_json(history_path, {"history": history})
    try:
        best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
    except TypeError:
        best_ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    val_preds = _evaluate(model, val_loader, val_df, device)
    metrics = compute_metrics(val_preds)
    val_preds.to_parquet(output_dir / "validation_predictions.parquet", index=False)
    dataframe_preview_for_csv(val_preds).to_csv(output_dir / "validation_predictions.csv", index=False)
    write_json(output_dir / "metrics.json", metrics)
    (output_dir / "metrics.md").write_text(_metrics_md(metrics), encoding="utf-8")
    save_diagnostics(val_preds, output_dir, args.diagnostic_samples)

    split_info = {
        "train_patients": sorted(train_df["patient_key"].astype(str).unique().tolist()),
        "val_patients": sorted(val_df["patient_key"].astype(str).unique().tolist()),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
    }
    write_json(output_dir / "split.json", split_info)
    return {
        "best_checkpoint": str(best_path),
        "latest_checkpoint": str(latest_path),
        "metrics": metrics,
        "history": str(history_path),
        "validation_predictions": str(output_dir / "validation_predictions.parquet"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train strain curve temporal model.")
    parser.add_argument("--input-parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-type", type=str, default="temporal_transformer", choices=["temporal_mean_pool", "temporal_transformer", "gru"])
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-metric", type=str, default="peak_gls_mae")
    parser.add_argument("--curve-loss", type=str, default="huber", choices=["huber", "l1", "mse"])
    parser.add_argument("--curve-loss-weight", type=float, default=1.0)
    parser.add_argument("--peak-loss-weight", type=float, default=0.5)
    parser.add_argument("--ttp-loss-weight", type=float, default=0.1)
    parser.add_argument("--derivative-loss-weight", type=float, default=0.2)
    parser.add_argument("--smoothness-loss-weight", type=float, default=0.01)
    parser.add_argument("--derive-predicted-peak-from-curve", action="store_true")
    parser.add_argument("--use-amp", dest="use_amp", action="store_true", default=True)
    parser.add_argument("--no-use-amp", dest="use_amp", action="store_false")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--diagnostic-samples", type=int, default=40)
    parser.add_argument("--cv-run", action="store_true", help="Run repeated patient-grouped shuffle validation.")
    parser.add_argument("--cv-seeds", type=str, default="")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df_raw = pd.read_parquet(args.input_parquet)
    df = _clean_training_df(df_raw)
    if df.empty:
        raise ValueError("No trainable samples found in phase embeddings parquet.")

    if args.cv_run:
        seeds = [int(s.strip()) for s in args.cv_seeds.split(",") if s.strip()] or [args.seed]
        runs = []
        for seed in seeds:
            fold_dir = args.output_dir / f"cv_seed_{seed}"
            runs.append(train_once(df, args, fold_dir, seed))
        write_json(args.output_dir / "cv_summary.json", {"runs": runs})
        logger.info("Saved CV summary: %s", args.output_dir / "cv_summary.json")
    else:
        result = train_once(df, args, args.output_dir, args.seed)
        write_json(args.output_dir / "train_summary.json", result)
        logger.info("Best checkpoint: %s", result["best_checkpoint"])


if __name__ == "__main__":
    main()
