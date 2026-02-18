"""
Train Ichilov pipeline3 end-to-end longitudinal model.

This script is intentionally aligned with the pipeline2 training ecosystem:
  - YAML-first configuration
  - patient-level split
  - run-level checkpoints, history JSON/CSV, and prediction exports
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

if __package__ is None or __package__ == "":  # pragma: no cover - script execution path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Ichilov_pipeline3.datasets.visit_dataset import (
    VisitDataset,
    split_patient_indices,
    visit_collate_fn,
)
from Ichilov_pipeline3.losses import DeltaGLSLoss, PairwiseRankingLoss, SmoothnessLoss
from Ichilov_pipeline3.models.full_model import IchilovPipeline3Model

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional dependency
    SummaryWriter = None  # type: ignore

try:
    import yaml
except Exception as exc:  # pragma: no cover - runtime dependency check
    raise RuntimeError(
        "PyYAML is required. Install with: .venv\\Scripts\\python -m pip install pyyaml"
    ) from exc

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("ichilov_pipeline3.train")

CONFIG_ENV = "ICHILOV_PIPELINE3_TRAIN_CONFIG"
DEFAULT_CONFIG_NAME = "config_pipeline3.yaml"


@dataclass
class ResolvedConfig:
    run_name: str
    input_xlsx: Path
    echo_root: Path
    cropped_root: Optional[Path]
    output_dir: Path
    log_dir: Optional[Path]
    output_parquet: Optional[Path]
    views: str
    sampling_mode: str
    t_frames: int
    clip_stride: int
    include_last_window: bool
    max_visits: int
    min_visits: int
    num_workers: int
    val_ratio: float
    test_ratio: float
    seed: int
    batch_size: int
    epochs: int
    device: str
    lr: float
    weight_decay: float
    scheduler: str
    scheduler_patience: int
    scheduler_factor: float
    scheduler_min_lr: float
    backbone_name: str
    backbone_pretrained: bool
    backbone_freeze: bool
    unfreeze_last_blocks: int
    temporal_layers: int
    temporal_heads: int
    temporal_dropout: float
    longitudinal_model: str
    longitudinal_hidden: int
    longitudinal_layers: int
    longitudinal_heads: int
    longitudinal_dropout: float
    use_time_encoding: bool
    lambda_delta: float
    lambda_rank: float
    lambda_smooth: float
    huber_beta: float
    risk_delta_threshold: float
    save_every: int


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _expand_path(value: Optional[Any]) -> Optional[Path]:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"none", "null", "auto"}:
        return None
    return Path(s).expanduser()


def _resolve_config_path(cli_value: Optional[Path]) -> Path:
    if cli_value is not None:
        return cli_value
    env_raw = None
    try:
        import os

        env_raw = os.environ.get(CONFIG_ENV)
    except Exception:
        env_raw = None
    if env_raw:
        import os

        return Path(os.path.expandvars(env_raw)).expanduser()
    return Path(__file__).resolve().with_name(DEFAULT_CONFIG_NAME)


def _load_config(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def _parse_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(int(value))
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _extract_step_args(config: Dict[str, Any]) -> Dict[str, Any]:
    steps = config.get("steps", {})
    if isinstance(steps, dict):
        long_cfg = steps.get("longitudinal_train", {})
        if isinstance(long_cfg, dict):
            args = long_cfg.get("args", {})
            if isinstance(args, dict):
                return deepcopy(args)
    args = config.get("args", {})
    if isinstance(args, dict):
        return deepcopy(args)
    return {}


def _merge_resolved_config(
    cli_args: argparse.Namespace,
    yaml_cfg: Dict[str, Any],
) -> ResolvedConfig:
    step_args = _extract_step_args(yaml_cfg)
    paths_cfg = yaml_cfg.get("paths", {}) if isinstance(yaml_cfg.get("paths"), dict) else {}
    pipe_cfg = yaml_cfg.get("pipeline", {}) if isinstance(yaml_cfg.get("pipeline"), dict) else {}

    merged: Dict[str, Any] = {}
    merged.update(paths_cfg)
    merged.update(step_args)
    merged_lower = {str(k).lower(): v for k, v in merged.items()}

    def pick(name: str, default: Any, aliases: Tuple[str, ...] = ()) -> Any:
        val = getattr(cli_args, name, None)
        if val is not None:
            return val
        for key in (name,) + aliases:
            if key in merged:
                return merged[key]
            key_l = str(key).lower()
            if key_l in merged_lower:
                return merged_lower[key_l]
        return default

    run_name_raw = pick("run_name", str(pipe_cfg.get("run_name") or "").strip())
    run_name = str(run_name_raw).strip() or f"pipeline3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    input_xlsx = _expand_path(pick("input_xlsx", None))
    echo_root = _expand_path(pick("echo_root", Path(r"D:\\")))
    cropped_root = _expand_path(pick("cropped_root", None))
    output_dir = _expand_path(pick("output_dir", None))
    log_dir = _expand_path(pick("log_dir", None))
    output_parquet = _expand_path(pick("output_parquet", None))
    if input_xlsx is None:
        raise ValueError("input_xlsx is required (CLI or YAML steps.longitudinal_train.args.input_xlsx).")
    if echo_root is None:
        raise ValueError("echo_root is required.")
    if output_dir is None:
        raise ValueError("output_dir is required.")

    return ResolvedConfig(
        run_name=run_name,
        input_xlsx=input_xlsx,
        echo_root=echo_root,
        cropped_root=cropped_root,
        output_dir=output_dir,
        log_dir=log_dir,
        output_parquet=output_parquet,
        views=str(pick("views", "") or ""),
        sampling_mode=str(pick("sampling_mode", "uniform") or "uniform"),
        t_frames=int(pick("t_frames", 16, aliases=("T_frames",))),
        clip_stride=int(pick("clip_stride", 1)),
        include_last_window=_parse_bool(pick("include_last_window", True), True),
        max_visits=int(pick("max_visits", 5)),
        min_visits=int(pick("min_visits", 2)),
        num_workers=int(pick("num_workers", 0)),
        val_ratio=float(pick("val_ratio", 0.2)),
        test_ratio=float(pick("test_ratio", 0.1)),
        seed=int(pick("seed", 42)),
        batch_size=int(pick("batch_size", 2)),
        epochs=int(pick("epochs", 30)),
        device=str(pick("device", "auto")),
        lr=float(pick("lr", 1e-4)),
        weight_decay=float(pick("weight_decay", 0.01)),
        scheduler=str(pick("scheduler", "cosine")),
        scheduler_patience=int(pick("scheduler_patience", 5)),
        scheduler_factor=float(pick("scheduler_factor", 0.5)),
        scheduler_min_lr=float(pick("scheduler_min_lr", 1e-6)),
        backbone_name=str(pick("backbone_name", "vit_small_patch16_dinov2.lvd142m")),
        backbone_pretrained=_parse_bool(pick("backbone_pretrained", True), True),
        backbone_freeze=_parse_bool(pick("backbone_freeze", True), True),
        unfreeze_last_blocks=int(pick("unfreeze_last_blocks", 0)),
        temporal_layers=int(pick("temporal_layers", 2)),
        temporal_heads=int(pick("temporal_heads", 6)),
        temporal_dropout=float(pick("temporal_dropout", 0.1)),
        longitudinal_model=str(pick("longitudinal_model", "gru")),
        longitudinal_hidden=int(pick("longitudinal_hidden", 256)),
        longitudinal_layers=int(pick("longitudinal_layers", 1)),
        longitudinal_heads=int(pick("longitudinal_heads", 4)),
        longitudinal_dropout=float(pick("longitudinal_dropout", 0.1)),
        use_time_encoding=_parse_bool(pick("use_time_encoding", True), True),
        lambda_delta=float(pick("lambda_delta", 1.0)),
        lambda_rank=float(pick("lambda_rank", 1.0)),
        lambda_smooth=float(pick("lambda_smooth", 0.1)),
        huber_beta=float(pick("huber_beta", 1.0)),
        risk_delta_threshold=float(pick("risk_delta_threshold", 2.0)),
        save_every=max(1, int(pick("save_every", 10))),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Ichilov pipeline3 longitudinal model.")
    parser.add_argument("--config", type=Path, default=None, help="YAML config path.")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--input-xlsx", type=Path, default=None)
    parser.add_argument("--echo-root", type=Path, default=None)
    parser.add_argument("--cropped-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--output-parquet", type=Path, default=None)
    parser.add_argument("--views", type=str, default=None)
    parser.add_argument("--sampling-mode", type=str, choices=["uniform", "sliding_window"], default=None)
    parser.add_argument("--t-frames", type=int, default=None)
    parser.add_argument("--clip-stride", type=int, default=None)
    parser.add_argument("--include-last-window", dest="include_last_window", action="store_true")
    parser.add_argument("--no-include-last-window", dest="include_last_window", action="store_false")
    parser.set_defaults(include_last_window=None)
    parser.add_argument("--max-visits", type=int, default=None)
    parser.add_argument("--min-visits", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--test-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--scheduler", type=str, choices=["none", "cosine", "plateau"], default=None)
    parser.add_argument("--scheduler-patience", type=int, default=None)
    parser.add_argument("--scheduler-factor", type=float, default=None)
    parser.add_argument("--scheduler-min-lr", type=float, default=None)
    parser.add_argument("--backbone-name", type=str, default=None)
    parser.add_argument("--backbone-pretrained", dest="backbone_pretrained", action="store_true")
    parser.add_argument("--no-backbone-pretrained", dest="backbone_pretrained", action="store_false")
    parser.set_defaults(backbone_pretrained=None)
    parser.add_argument("--backbone-freeze", dest="backbone_freeze", action="store_true")
    parser.add_argument("--no-backbone-freeze", dest="backbone_freeze", action="store_false")
    parser.set_defaults(backbone_freeze=None)
    parser.add_argument("--unfreeze-last-blocks", type=int, default=None)
    parser.add_argument("--temporal-layers", type=int, default=None)
    parser.add_argument("--temporal-heads", type=int, default=None)
    parser.add_argument("--temporal-dropout", type=float, default=None)
    parser.add_argument("--longitudinal-model", type=str, choices=["gru", "transformer"], default=None)
    parser.add_argument("--longitudinal-hidden", type=int, default=None)
    parser.add_argument("--longitudinal-layers", type=int, default=None)
    parser.add_argument("--longitudinal-heads", type=int, default=None)
    parser.add_argument("--longitudinal-dropout", type=float, default=None)
    parser.add_argument("--use-time-encoding", dest="use_time_encoding", action="store_true")
    parser.add_argument("--no-use-time-encoding", dest="use_time_encoding", action="store_false")
    parser.set_defaults(use_time_encoding=None)
    parser.add_argument("--lambda-delta", type=float, default=None)
    parser.add_argument("--lambda-rank", type=float, default=None)
    parser.add_argument("--lambda-smooth", type=float, default=None)
    parser.add_argument("--huber-beta", type=float, default=None)
    parser.add_argument("--risk-delta-threshold", type=float, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    return parser


def _to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    out: Dict[str, object] = {"patient_id": batch["patient_id"]}
    out["frames_by_view"] = {
        k: v.to(device, non_blocking=True) for k, v in batch["frames_by_view"].items()
    }
    out["frame_masks_by_view"] = {
        k: v.to(device, non_blocking=True) for k, v in batch["frame_masks_by_view"].items()
    }
    for key in (
        "view_mask",
        "visit_mask",
        "visit_times",
        "gls",
        "gls_mask",
        "delta_gls_target",
        "delta_gls_mask",
        "risk_label",
        "risk_mask",
    ):
        out[key] = batch[key].to(device, non_blocking=True)
    return out


def _compute_losses(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, object],
    delta_loss_fn: DeltaGLSLoss,
    rank_loss_fn: PairwiseRankingLoss,
    smooth_loss_fn: SmoothnessLoss,
    cfg: ResolvedConfig,
) -> Dict[str, torch.Tensor]:
    delta_pred = outputs["delta_gls"]
    severity_pred = outputs["severity_score"]

    delta_target = batch["delta_gls_target"]
    delta_mask = batch["delta_gls_mask"]
    gls_target = batch["gls"]
    gls_mask = batch["gls_mask"] & batch["visit_mask"]

    loss_delta = delta_loss_fn(delta_pred, delta_target, delta_mask)
    loss_rank = rank_loss_fn(severity_pred, gls_target, gls_mask)
    loss_smooth = smooth_loss_fn(severity_pred, batch["visit_mask"])
    total = (
        cfg.lambda_delta * loss_delta
        + cfg.lambda_rank * loss_rank
        + cfg.lambda_smooth * loss_smooth
    )

    with torch.no_grad():
        if delta_mask.any():
            delta_mae = torch.abs(delta_pred[delta_mask] - delta_target[delta_mask]).mean()
        else:
            delta_mae = torch.tensor(0.0, device=delta_pred.device, dtype=delta_pred.dtype)
        risk_prob = outputs["risk_prob"]
        risk_mask = batch["risk_mask"]
        if risk_mask.any():
            risk_pred = (risk_prob[risk_mask] >= 0.5).float()
            risk_true = batch["risk_label"][risk_mask]
            risk_acc = (risk_pred == risk_true).float().mean()
        else:
            risk_acc = torch.tensor(0.0, device=delta_pred.device, dtype=delta_pred.dtype)

    return {
        "loss": total,
        "loss_delta": loss_delta.detach(),
        "loss_rank": loss_rank.detach(),
        "loss_smooth": loss_smooth.detach(),
        "delta_mae": delta_mae.detach(),
        "risk_acc": risk_acc.detach(),
    }


def _run_epoch(
    model: IchilovPipeline3Model,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    delta_loss_fn: DeltaGLSLoss,
    rank_loss_fn: PairwiseRankingLoss,
    smooth_loss_fn: SmoothnessLoss,
    cfg: ResolvedConfig,
    train: bool,
) -> Dict[str, float]:
    if train:
        model.train()
    else:
        model.eval()

    totals: Dict[str, float] = {
        "loss": 0.0,
        "loss_delta": 0.0,
        "loss_rank": 0.0,
        "loss_smooth": 0.0,
        "delta_mae": 0.0,
        "risk_acc": 0.0,
    }
    n_batches = 0

    it = tqdm(loader, desc="Train" if train else "Val", leave=False)
    for batch_cpu in it:
        batch = _to_device(batch_cpu, device)
        with torch.set_grad_enabled(train):
            outputs = model(
                frames_by_view=batch["frames_by_view"],
                frame_masks_by_view=batch["frame_masks_by_view"],
                visit_mask=batch["visit_mask"],
                visit_times=batch["visit_times"],
            )
            metrics = _compute_losses(
                outputs,
                batch,
                delta_loss_fn=delta_loss_fn,
                rank_loss_fn=rank_loss_fn,
                smooth_loss_fn=smooth_loss_fn,
                cfg=cfg,
            )
            loss = metrics["loss"]
            if train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        n_batches += 1
        for k in totals:
            val = metrics[k]
            totals[k] += float(val.item())
        it.set_postfix(loss=f"{totals['loss'] / max(1, n_batches):.4f}")

    if n_batches == 0:
        return {k: 0.0 for k in totals}
    return {k: v / n_batches for k, v in totals.items()}


def _predict(
    model: IchilovPipeline3Model,
    loader: Optional[DataLoader],
    device: torch.device,
    split_name: str,
) -> pd.DataFrame:
    if loader is None:
        return pd.DataFrame()
    model.eval()
    rows: List[dict] = []
    with torch.no_grad():
        for batch_cpu in tqdm(loader, desc=f"Predict {split_name}", leave=False):
            batch = _to_device(batch_cpu, device)
            outputs = model(
                frames_by_view=batch["frames_by_view"],
                frame_masks_by_view=batch["frame_masks_by_view"],
                visit_mask=batch["visit_mask"],
                visit_times=batch["visit_times"],
            )
            delta_pred = outputs["delta_gls"].detach().cpu().numpy()
            risk_prob = outputs["risk_prob"].detach().cpu().numpy()
            severity = outputs["severity_score"].detach().cpu().numpy()
            visit_mask = batch["visit_mask"].detach().cpu().numpy().astype(bool)
            visit_times = batch["visit_times"].detach().cpu().numpy()
            gls = batch["gls"].detach().cpu().numpy()
            gls_mask = batch["gls_mask"].detach().cpu().numpy().astype(bool)
            delta_target = batch["delta_gls_target"].detach().cpu().numpy()
            delta_mask = batch["delta_gls_mask"].detach().cpu().numpy().astype(bool)
            risk_label = batch["risk_label"].detach().cpu().numpy()
            risk_mask = batch["risk_mask"].detach().cpu().numpy().astype(bool)

            for i, patient_id in enumerate(batch["patient_id"]):
                valid_vis = visit_mask[i]
                rows.append(
                    {
                        "split": split_name,
                        "patient_id": str(patient_id),
                        "delta_gls_pred": float(delta_pred[i]),
                        "delta_gls_target": float(delta_target[i]) if delta_mask[i] else None,
                        "risk_prob": float(risk_prob[i]),
                        "risk_label": float(risk_label[i]) if risk_mask[i] else None,
                        "severity_score": severity[i, valid_vis].tolist(),
                        "gls_target": gls[i, gls_mask[i]].tolist(),
                        "visit_times": visit_times[i, valid_vis].tolist(),
                    }
                )
    return pd.DataFrame(rows)


def _make_dataloader(
    dataset: Optional[Subset],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> Optional[DataLoader]:
    if dataset is None or len(dataset) == 0:
        return None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=max(0, int(num_workers)),
        pin_memory=torch.cuda.is_available(),
        collate_fn=visit_collate_fn,
    )


def _resolve_device(device_raw: str) -> torch.device:
    if device_raw.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_raw)


def main() -> None:
    parser = _build_parser()
    cli_args = parser.parse_args()

    config_path = _resolve_config_path(cli_args.config)
    yaml_cfg = _load_config(config_path if config_path.exists() else None)
    cfg = _merge_resolved_config(cli_args, yaml_cfg)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    if cfg.log_dir is None:
        cfg.log_dir = cfg.output_dir / "tensorboard"
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    _set_seed(cfg.seed)
    device = _resolve_device(cfg.device)
    logger.info("Training on device: %s", device)

    dataset = VisitDataset(
        input_xlsx=cfg.input_xlsx,
        echo_root=cfg.echo_root,
        cropped_root=cfg.cropped_root,
        views=cfg.views,
        t_frames=cfg.t_frames,
        sampling_mode=cfg.sampling_mode,
        clip_stride=cfg.clip_stride,
        include_last_window=cfg.include_last_window,
        max_visits=cfg.max_visits,
        min_visits=cfg.min_visits,
        risk_delta_threshold=cfg.risk_delta_threshold,
        random_view_sampling=False,  # TODO: add epoch-wise randomized view sampling if needed.
    )

    train_idx, val_idx, test_idx = split_patient_indices(
        n_patients=len(dataset),
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        seed=cfg.seed,
    )
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx) if val_idx else None
    test_ds = Subset(dataset, test_idx) if test_idx else None

    train_loader = _make_dataloader(train_ds, cfg.batch_size, True, cfg.num_workers)
    val_loader = _make_dataloader(val_ds, cfg.batch_size, False, cfg.num_workers)
    test_loader = _make_dataloader(test_ds, cfg.batch_size, False, cfg.num_workers)
    if train_loader is None:
        raise RuntimeError("No training samples available after patient split.")

    model = IchilovPipeline3Model(
        backbone_name=cfg.backbone_name,
        backbone_pretrained=cfg.backbone_pretrained,
        backbone_freeze=cfg.backbone_freeze,
        unfreeze_last_blocks=cfg.unfreeze_last_blocks,
        temporal_layers=cfg.temporal_layers,
        temporal_heads=cfg.temporal_heads,
        temporal_dropout=cfg.temporal_dropout,
        longitudinal_model_type=cfg.longitudinal_model,
        longitudinal_hidden=cfg.longitudinal_hidden,
        longitudinal_layers=cfg.longitudinal_layers,
        longitudinal_heads=cfg.longitudinal_heads,
        longitudinal_dropout=cfg.longitudinal_dropout,
        use_time_encoding=cfg.use_time_encoding,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    if cfg.scheduler == "cosine":
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, cfg.epochs),
            eta_min=cfg.scheduler_min_lr,
        )
        plateau_scheduler = None
    elif cfg.scheduler == "plateau":
        scheduler = None
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg.scheduler_factor,
            patience=cfg.scheduler_patience,
            min_lr=cfg.scheduler_min_lr,
        )
    else:
        scheduler = None
        plateau_scheduler = None

    delta_loss_fn = DeltaGLSLoss(beta=cfg.huber_beta)
    rank_loss_fn = PairwiseRankingLoss()
    smooth_loss_fn = SmoothnessLoss()

    writer = SummaryWriter(log_dir=str(cfg.log_dir)) if SummaryWriter is not None else None
    best_val = float("inf")
    best_path = cfg.output_dir / f"{cfg.run_name}_best.pt"
    latest_path = cfg.output_dir / f"{cfg.run_name}_latest.pt"
    history: List[dict] = []

    for epoch in range(1, cfg.epochs + 1):
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            delta_loss_fn=delta_loss_fn,
            rank_loss_fn=rank_loss_fn,
            smooth_loss_fn=smooth_loss_fn,
            cfg=cfg,
            train=True,
        )
        val_metrics = _run_epoch(
            model=model,
            loader=val_loader if val_loader is not None else train_loader,
            optimizer=None,
            device=device,
            delta_loss_fn=delta_loss_fn,
            rank_loss_fn=rank_loss_fn,
            smooth_loss_fn=smooth_loss_fn,
            cfg=cfg,
            train=False,
        )

        if scheduler is not None:
            scheduler.step()
        if plateau_scheduler is not None:
            plateau_scheduler.step(val_metrics["loss"])

        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": train_metrics["loss"],
            "train_delta": train_metrics["loss_delta"],
            "train_rank": train_metrics["loss_rank"],
            "train_smooth": train_metrics["loss_smooth"],
            "train_delta_mae": train_metrics["delta_mae"],
            "train_risk_acc": train_metrics["risk_acc"],
            "val_loss": val_metrics["loss"],
            "val_delta": val_metrics["loss_delta"],
            "val_rank": val_metrics["loss_rank"],
            "val_smooth": val_metrics["loss_smooth"],
            "val_delta_mae": val_metrics["delta_mae"],
            "val_risk_acc": val_metrics["risk_acc"],
        }
        history.append(row)
        logger.info(
            "Epoch %d/%d | train=%.4f val=%.4f delta=%.4f rank=%.4f smooth=%.4f",
            epoch,
            cfg.epochs,
            row["train_loss"],
            row["val_loss"],
            row["val_delta"],
            row["val_rank"],
            row["val_smooth"],
        )

        if writer is not None:
            writer.add_scalars("loss/total", {"train": row["train_loss"], "val": row["val_loss"]}, epoch)
            writer.add_scalars("loss/delta", {"train": row["train_delta"], "val": row["val_delta"]}, epoch)
            writer.add_scalars("loss/rank", {"train": row["train_rank"], "val": row["val_rank"]}, epoch)
            writer.add_scalars("loss/smooth", {"train": row["train_smooth"], "val": row["val_smooth"]}, epoch)
            writer.add_scalars(
                "metric/delta_mae",
                {"train": row["train_delta_mae"], "val": row["val_delta_mae"]},
                epoch,
            )
            writer.add_scalars(
                "metric/risk_acc",
                {"train": row["train_risk_acc"], "val": row["val_risk_acc"]},
                epoch,
            )

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": vars(cfg),
            "history_tail": history[-5:],
        }
        torch.save(checkpoint, latest_path)

        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            torch.save(checkpoint, best_path)

        if epoch % cfg.save_every == 0:
            torch.save(checkpoint, cfg.output_dir / f"{cfg.run_name}_epoch{epoch:03d}.pt")

    if writer is not None:
        writer.close()

    history_df = pd.DataFrame(history)
    history_json = cfg.output_dir / f"{cfg.run_name}_history.json"
    history_csv = cfg.output_dir / f"{cfg.run_name}_history.csv"
    history_json.write_text(json.dumps(history, indent=2), encoding="utf-8")
    history_df.to_csv(history_csv, index=False)

    cfg_json = cfg.output_dir / f"{cfg.run_name}_config.json"
    cfg_json.write_text(json.dumps(vars(cfg), indent=2, default=str), encoding="utf-8")

    pred_frames: List[pd.DataFrame] = []
    pred_frames.append(_predict(model, train_loader, device, split_name="train"))
    pred_frames.append(_predict(model, val_loader, device, split_name="val"))
    pred_frames.append(_predict(model, test_loader, device, split_name="test"))
    pred_df = pd.concat([df for df in pred_frames if not df.empty], axis=0, ignore_index=True)
    pred_path = cfg.output_parquet or (cfg.output_dir / f"{cfg.run_name}_predictions.parquet")
    if pred_path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pred_path = pred_path.with_name(f"{pred_path.stem}_{stamp}{pred_path.suffix}")
        logger.warning("Prediction output exists; writing to %s", pred_path)
    pred_df.to_parquet(pred_path, index=False)

    logger.info("Best checkpoint: %s", best_path)
    logger.info("Latest checkpoint: %s", latest_path)
    logger.info("History JSON: %s", history_json)
    logger.info("Predictions: %s", pred_path)


if __name__ == "__main__":
    main()
