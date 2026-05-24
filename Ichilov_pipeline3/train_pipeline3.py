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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

if __package__ is None or __package__ == "":  # pragma: no cover - script execution path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Ichilov_pipeline3.datasets.visit_dataset import (
    DELTA_GLS_CONVENTION,
    FrameEmbeddingVisitDataset,
    VisitDataset,
    embedding_visit_collate_fn,
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

try:
    from sklearn.linear_model import Ridge, SGDClassifier
    from sklearn.metrics import roc_auc_score
except Exception:  # pragma: no cover - optional diagnostics dependency
    Ridge = None  # type: ignore
    SGDClassifier = None  # type: ignore
    roc_auc_score = None  # type: ignore

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("ichilov_pipeline3.train")

CONFIG_ENV = "ICHILOV_PIPELINE3_TRAIN_CONFIG"
DEFAULT_CONFIG_NAME = "config_pipeline3.yaml"


@dataclass
class ResolvedConfig:
    run_name: str
    input_xlsx: Optional[Path]
    input_embeddings: Optional[Path]
    echo_root: Optional[Path]
    cropped_root: Optional[Path]
    output_dir: Path
    log_dir: Optional[Path]
    output_parquet: Optional[Path]
    report_xlsx: Optional[Path]
    views: str
    sampling_mode: str
    t_frames: int
    clip_stride: int
    include_last_window: bool
    phase_aligned_include_midpoints: bool
    phase_aligned_strategy: str
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
    temporal_mode: str
    view_fusion_mode: str
    longitudinal_ablation_mode: str
    single_best_view: str
    time_encoding_mode: str
    enable_diagnostics: bool
    diagnostics_batches: int
    diagnostics_topk: int
    diagnostics_every: int
    run_patient_id_probe: bool
    run_delta_gls_probe: bool
    delta_gls_convention: str
    debug_delta: bool
    debug_delta_samples: int
    split_json: Optional[Path]
    summary_json: Optional[Path]


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
    for nested_key in ("model", "ablation", "diagnostics"):
        nested = step_args.get(nested_key)
        if isinstance(nested, dict):
            merged.update(nested)
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
    input_embeddings = _expand_path(pick("input_embeddings", None))
    echo_root = _expand_path(pick("echo_root", Path(r"D:\\")))
    cropped_root = _expand_path(pick("cropped_root", None))
    output_dir = _expand_path(pick("output_dir", None))
    log_dir = _expand_path(pick("log_dir", None))
    output_parquet = _expand_path(pick("output_parquet", None))
    report_xlsx = _expand_path(pick("report_xlsx", input_xlsx))
    if input_embeddings is None and input_xlsx is None:
        raise ValueError("Provide either input_embeddings or input_xlsx for training.")
    if input_embeddings is None and echo_root is None:
        raise ValueError("echo_root is required when training from raw frames.")
    if output_dir is None:
        raise ValueError("output_dir is required.")

    time_encoding_mode = str(pick("time_encoding_mode", "inherit")).strip().lower()
    if time_encoding_mode not in {"inherit", "on", "off"}:
        raise ValueError("time_encoding_mode must be one of: inherit, on, off")
    use_time_encoding = _parse_bool(pick("use_time_encoding", True), True)
    if time_encoding_mode == "on":
        use_time_encoding = True
    elif time_encoding_mode == "off":
        use_time_encoding = False

    sampling_mode = str(pick("sampling_mode", "uniform") or "uniform").strip().lower()
    if sampling_mode not in {"uniform", "sliding_window", "phase_aligned"}:
        raise ValueError("sampling_mode must be one of: uniform, sliding_window, phase_aligned")
    phase_aligned_strategy = str(pick("phase_aligned_strategy", "cycle") or "cycle").strip().lower()
    if phase_aligned_strategy not in {"cycle", "segment"}:
        raise ValueError("phase_aligned_strategy must be one of: cycle, segment")
    delta_gls_convention = str(
        pick("delta_gls_convention", DELTA_GLS_CONVENTION) or DELTA_GLS_CONVENTION
    ).strip().lower()
    if delta_gls_convention not in {"latest_minus_earliest", "earliest_minus_latest"}:
        raise ValueError("delta_gls_convention must be latest_minus_earliest or earliest_minus_latest")

    return ResolvedConfig(
        run_name=run_name,
        input_xlsx=input_xlsx,
        input_embeddings=input_embeddings,
        echo_root=echo_root,
        cropped_root=cropped_root,
        output_dir=output_dir,
        log_dir=log_dir,
        output_parquet=output_parquet,
        report_xlsx=report_xlsx,
        views=str(pick("views", "") or ""),
        sampling_mode=sampling_mode,
        t_frames=int(pick("t_frames", 16, aliases=("T_frames",))),
        clip_stride=int(pick("clip_stride", 1)),
        include_last_window=_parse_bool(pick("include_last_window", True), True),
        phase_aligned_include_midpoints=_parse_bool(
            pick("phase_aligned_include_midpoints", True),
            True,
        ),
        phase_aligned_strategy=phase_aligned_strategy,
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
        use_time_encoding=use_time_encoding,
        lambda_delta=float(pick("lambda_delta", 1.0)),
        lambda_rank=float(pick("lambda_rank", 1.0)),
        lambda_smooth=float(pick("lambda_smooth", 0.1)),
        huber_beta=float(pick("huber_beta", 1.0)),
        risk_delta_threshold=float(pick("risk_delta_threshold", 2.0)),
        save_every=max(1, int(pick("save_every", 10))),
        temporal_mode=str(pick("temporal_mode", pick("temporal_pooling_mode", "attn_pool"))).lower(),
        view_fusion_mode=str(pick("view_fusion_mode", "attn")).lower(),
        longitudinal_ablation_mode=str(
            pick("longitudinal_ablation_mode", pick("longitudinal_mode", "inherit"))
        ).lower(),
        single_best_view=str(pick("single_best_view", "A4C")).upper(),
        time_encoding_mode=time_encoding_mode,
        enable_diagnostics=_parse_bool(pick("enable_diagnostics", True), True),
        diagnostics_batches=max(1, int(pick("diagnostics_batches", 2))),
        diagnostics_topk=max(1, int(pick("diagnostics_topk", 5))),
        diagnostics_every=max(1, int(pick("diagnostics_every", 1))),
        run_patient_id_probe=_parse_bool(pick("run_patient_id_probe", True), True),
        run_delta_gls_probe=_parse_bool(pick("run_delta_gls_probe", True), True),
        delta_gls_convention=delta_gls_convention,
        debug_delta=_parse_bool(pick("debug_delta", False), False),
        debug_delta_samples=max(1, int(pick("debug_delta_samples", 2))),
        split_json=_expand_path(pick("split_json", None)),
        summary_json=_expand_path(pick("summary_json", None)),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Ichilov pipeline3 longitudinal model.")
    parser.add_argument("--config", type=Path, default=None, help="YAML config path.")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--input-xlsx", type=Path, default=None)
    parser.add_argument("--input-embeddings", type=Path, default=None)
    parser.add_argument("--echo-root", type=Path, default=None)
    parser.add_argument("--cropped-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--output-parquet", type=Path, default=None)
    parser.add_argument("--report-xlsx", type=Path, default=None)
    parser.add_argument("--views", type=str, default=None)
    parser.add_argument(
        "--sampling-mode",
        type=str,
        choices=["uniform", "sliding_window", "phase_aligned"],
        default=None,
    )
    parser.add_argument("--t-frames", type=int, default=None)
    parser.add_argument("--clip-stride", type=int, default=None)
    parser.add_argument("--include-last-window", dest="include_last_window", action="store_true")
    parser.add_argument("--no-include-last-window", dest="include_last_window", action="store_false")
    parser.set_defaults(include_last_window=None)
    parser.add_argument(
        "--phase-aligned-include-midpoints",
        dest="phase_aligned_include_midpoints",
        action="store_true",
    )
    parser.add_argument(
        "--no-phase-aligned-include-midpoints",
        dest="phase_aligned_include_midpoints",
        action="store_false",
    )
    parser.set_defaults(phase_aligned_include_midpoints=None)
    parser.add_argument("--phase-aligned-strategy", type=str, choices=["cycle", "segment"], default=None)
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
    parser.add_argument(
        "--temporal-mode",
        type=str,
        choices=["attn_pool", "mean_pool", "max_pool", "first_frame", "last_frame"],
        default=None,
    )
    parser.add_argument(
        "--view-fusion-mode",
        type=str,
        choices=["attn", "mean", "max", "single_best_view", "concat_then_linear"],
        default=None,
    )
    parser.add_argument("--single-best-view", type=str, choices=["A2C", "A3C", "A4C"], default=None)
    parser.add_argument(
        "--longitudinal-model",
        type=str,
        choices=[
            "gru",
            "transformer",
            "last_visit_linear",
            "mean_visit_linear",
            "delta_only_linear",
        ],
        default=None,
    )
    parser.add_argument(
        "--longitudinal-ablation-mode",
        type=str,
        choices=["inherit", "gru", "transformer", "last_visit_linear", "mean_visit_linear", "delta_only_linear"],
        default=None,
    )
    parser.add_argument("--longitudinal-hidden", type=int, default=None)
    parser.add_argument("--longitudinal-layers", type=int, default=None)
    parser.add_argument("--longitudinal-heads", type=int, default=None)
    parser.add_argument("--longitudinal-dropout", type=float, default=None)
    parser.add_argument("--use-time-encoding", dest="use_time_encoding", action="store_true")
    parser.add_argument("--no-use-time-encoding", dest="use_time_encoding", action="store_false")
    parser.set_defaults(use_time_encoding=None)
    parser.add_argument("--time-encoding-mode", type=str, choices=["inherit", "on", "off"], default=None)
    parser.add_argument("--lambda-delta", type=float, default=None)
    parser.add_argument("--lambda-rank", type=float, default=None)
    parser.add_argument("--lambda-smooth", type=float, default=None)
    parser.add_argument("--huber-beta", type=float, default=None)
    parser.add_argument("--risk-delta-threshold", type=float, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--enable-diagnostics", dest="enable_diagnostics", action="store_true")
    parser.add_argument("--no-enable-diagnostics", dest="enable_diagnostics", action="store_false")
    parser.set_defaults(enable_diagnostics=None)
    parser.add_argument("--diagnostics-batches", type=int, default=None)
    parser.add_argument("--diagnostics-topk", type=int, default=None)
    parser.add_argument("--diagnostics-every", type=int, default=None)
    parser.add_argument("--run-patient-id-probe", dest="run_patient_id_probe", action="store_true")
    parser.add_argument("--no-run-patient-id-probe", dest="run_patient_id_probe", action="store_false")
    parser.set_defaults(run_patient_id_probe=None)
    parser.add_argument("--run-delta-gls-probe", dest="run_delta_gls_probe", action="store_true")
    parser.add_argument("--no-run-delta-gls-probe", dest="run_delta_gls_probe", action="store_false")
    parser.set_defaults(run_delta_gls_probe=None)
    parser.add_argument(
        "--delta-gls-convention",
        type=str,
        choices=["latest_minus_earliest", "earliest_minus_latest"],
        default=None,
    )
    parser.add_argument("--debug-delta", dest="debug_delta", action="store_true")
    parser.add_argument("--no-debug-delta", dest="debug_delta", action="store_false")
    parser.set_defaults(debug_delta=None)
    parser.add_argument("--debug-delta-samples", type=int, default=None)
    parser.add_argument("--split-json", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    return parser


def _to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    out: Dict[str, object] = {"patient_id": batch["patient_id"]}
    if "frames_by_view" in batch:
        out["frames_by_view"] = {
            k: v.to(device, non_blocking=True) for k, v in batch["frames_by_view"].items()
        }
    if "embeddings_by_view" in batch:
        out["embeddings_by_view"] = {
            k: v.to(device, non_blocking=True) for k, v in batch["embeddings_by_view"].items()
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
        "delta_first_visit_index",
        "delta_last_visit_index",
        "delta_first_visit_time",
        "delta_last_visit_time",
        "delta_first_gls",
        "delta_last_gls",
        "risk_label",
        "risk_mask",
    ):
        if key in batch:
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
    visit_mask = batch["visit_mask"]
    invalid_visits = ~visit_mask
    if torch.any(invalid_visits):
        invalid_abs = torch.abs(severity_pred[invalid_visits])
        if torch.any(invalid_abs > 1e-5):
            raise RuntimeError("severity_score contains non-zero values on masked visits.")

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
    use_precomputed_embeddings: bool,
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
            if use_precomputed_embeddings:
                outputs = model.forward_from_frame_embeddings(
                    embeddings_by_view=batch["embeddings_by_view"],
                    frame_masks_by_view=batch["frame_masks_by_view"],
                    visit_mask=batch["visit_mask"],
                    visit_times=batch["visit_times"],
                )
            else:
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
    use_precomputed_embeddings: bool,
) -> pd.DataFrame:
    if loader is None:
        return pd.DataFrame()
    model.eval()
    rows: List[dict] = []
    with torch.no_grad():
        for batch_cpu in tqdm(loader, desc=f"Predict {split_name}", leave=False):
            batch = _to_device(batch_cpu, device)
            if use_precomputed_embeddings:
                outputs = model.forward_from_frame_embeddings(
                    embeddings_by_view=batch["embeddings_by_view"],
                    frame_masks_by_view=batch["frame_masks_by_view"],
                    visit_mask=batch["visit_mask"],
                    visit_times=batch["visit_times"],
                )
            else:
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
            first_t = batch["delta_first_visit_time"].detach().cpu().numpy() if "delta_first_visit_time" in batch else np.zeros_like(delta_target)
            last_t = batch["delta_last_visit_time"].detach().cpu().numpy() if "delta_last_visit_time" in batch else np.zeros_like(delta_target)
            first_gls = batch["delta_first_gls"].detach().cpu().numpy() if "delta_first_gls" in batch else np.zeros_like(delta_target)
            last_gls = batch["delta_last_gls"].detach().cpu().numpy() if "delta_last_gls" in batch else np.zeros_like(delta_target)

            for i, patient_id in enumerate(batch["patient_id"]):
                valid_vis = visit_mask[i]
                rows.append(
                    {
                        "split": split_name,
                        "patient_id": str(patient_id),
                        "delta_gls_pred": float(delta_pred[i]),
                        "delta_gls_target": float(delta_target[i]) if delta_mask[i] else None,
                        "delta_gls_mask": bool(delta_mask[i]),
                        "risk_prob": float(risk_prob[i]),
                        "risk_label": float(risk_label[i]) if risk_mask[i] else None,
                        "risk_mask": bool(risk_mask[i]),
                        "delta_first_visit_time": float(first_t[i]) if delta_mask[i] else None,
                        "delta_last_visit_time": float(last_t[i]) if delta_mask[i] else None,
                        "delta_first_gls": float(first_gls[i]) if delta_mask[i] else None,
                        "delta_last_gls": float(last_gls[i]) if delta_mask[i] else None,
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
    collate_fn: Any,
) -> Optional[DataLoader]:
    if dataset is None or len(dataset) == 0:
        return None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=max(0, int(num_workers)),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )


def _resolve_split_indices(
    dataset: Any,
    cfg: ResolvedConfig,
) -> Tuple[List[int], List[int], List[int]]:
    if cfg.split_json is None:
        return split_patient_indices(
            n_patients=len(dataset),
            val_ratio=cfg.val_ratio,
            test_ratio=cfg.test_ratio,
            seed=cfg.seed,
        )
    split_path = cfg.split_json
    if split_path is None or not split_path.exists():
        raise FileNotFoundError(f"split_json not found: {split_path}")
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("split_json must be a JSON object")
    idx_keys = ("train_indices", "val_indices", "test_indices")
    if all(k in payload for k in idx_keys):
        train_idx = [int(x) for x in payload.get("train_indices", [])]
        val_idx = [int(x) for x in payload.get("val_indices", [])]
        test_idx = [int(x) for x in payload.get("test_indices", [])]
    else:
        if not hasattr(dataset, "patient_records"):
            raise ValueError("split_json with patient IDs requires dataset.patient_records")
        id_to_idx = {
            str(rec.patient_id): i for i, rec in enumerate(getattr(dataset, "patient_records"))
        }
        train_idx = [id_to_idx[str(pid)] for pid in payload.get("train_patient_ids", []) if str(pid) in id_to_idx]
        val_idx = [id_to_idx[str(pid)] for pid in payload.get("val_patient_ids", []) if str(pid) in id_to_idx]
        test_idx = [id_to_idx[str(pid)] for pid in payload.get("test_patient_ids", []) if str(pid) in id_to_idx]
    used = set()
    for name, idxs in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        clean: List[int] = []
        for i in idxs:
            if i < 0 or i >= len(dataset):
                raise ValueError(f"{name} index out of range: {i}")
            if i in used:
                raise ValueError(f"Duplicate index across splits: {i}")
            used.add(i)
            clean.append(i)
        if name == "train":
            train_idx = clean
        elif name == "val":
            val_idx = clean
        else:
            test_idx = clean
    if not train_idx:
        raise ValueError("split_json produced empty train split.")
    return train_idx, val_idx, test_idx


def _resolve_device(device_raw: str) -> torch.device:
    if device_raw.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_raw)


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    x = x[mask]
    y = y[mask]
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x ** 2).sum() * (y ** 2).sum())
    if denom <= 0:
        return float("nan")
    return float((x * y).sum() / denom)


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    xr = pd.Series(x[mask]).rank(method="average").to_numpy(dtype=np.float64)
    yr = pd.Series(y[mask]).rank(method="average").to_numpy(dtype=np.float64)
    return _safe_pearson(xr, yr)


def _entropy_from_weights(weights: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    p = torch.where(mask, weights, torch.zeros_like(weights))
    p = p.clamp(min=1e-12)
    ent = -(p * torch.log(p)).sum(dim=dim)
    valid = mask.any(dim=dim)
    ent = torch.where(valid, ent, torch.zeros_like(ent))
    return ent


def _plot_hist(values: np.ndarray, out_path: Path, title: str, xlabel: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values, bins=30, color="#1f77b4", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_scatter(x: np.ndarray, y: np.ndarray, out_path: Path, title: str, xlabel: str, ylabel: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, s=14, alpha=0.7, color="#ff7f0e")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _collect_diagnostics(
    model: IchilovPipeline3Model,
    loader: Optional[DataLoader],
    device: torch.device,
    use_precomputed_embeddings: bool,
    max_batches: int,
    topk: int,
    out_dir: Optional[Path] = None,
    tag: str = "latest",
) -> Dict[str, Any]:
    if loader is None:
        return {}
    model.eval()
    frame_valid_weights: List[float] = []
    frame_masked_max_vals: List[float] = []
    frame_topk_rows: List[Dict[str, Any]] = []
    view_valid_weights: List[float] = []
    view_masked_max_vals: List[float] = []
    view_dom_counts = {v: 0 for v in IchilovPipeline3Model.VIEW_ORDER}
    within_cos: List[float] = []
    within_dt: List[float] = []
    across_cos: List[float] = []
    frame_entropy_vals: List[float] = []
    view_entropy_vals: List[float] = []

    with torch.no_grad():
        for batch_idx, batch_cpu in enumerate(loader):
            if batch_idx >= max_batches:
                break
            batch = _to_device(batch_cpu, device)
            if use_precomputed_embeddings:
                outputs = model.forward_from_frame_embeddings(
                    embeddings_by_view=batch["embeddings_by_view"],
                    frame_masks_by_view=batch["frame_masks_by_view"],
                    visit_mask=batch["visit_mask"],
                    visit_times=batch["visit_times"],
                )
            else:
                outputs = model(
                    frames_by_view=batch["frames_by_view"],
                    frame_masks_by_view=batch["frame_masks_by_view"],
                    visit_mask=batch["visit_mask"],
                    visit_times=batch["visit_times"],
                )
            diag = outputs.get("diagnostics", {})
            visit_emb = outputs["visit_embedding"]
            visit_mask = batch["visit_mask"].bool()
            visit_times = batch["visit_times"].float()
            patient_ids = [str(p) for p in batch["patient_id"]]

            frame_attn_by_view = diag.get("frame_attn_by_view", {})
            frame_mask_by_view = diag.get("frame_mask_by_view", {})
            for view in IchilovPipeline3Model.VIEW_ORDER:
                attn = frame_attn_by_view.get(view)
                msk = frame_mask_by_view.get(view)
                if attn is None or msk is None:
                    continue
                attn = attn.detach().cpu()
                msk = msk.detach().cpu().bool()
                valid_vals = attn[msk]
                if valid_vals.numel() > 0:
                    frame_valid_weights.extend(valid_vals.numpy().tolist())
                invalid_vals = attn[~msk]
                if invalid_vals.numel() > 0:
                    frame_masked_max_vals.append(float(invalid_vals.max().item()))
                frame_ent = _entropy_from_weights(attn, msk, dim=-1)
                valid_seq = msk.any(dim=-1)
                if valid_seq.any():
                    frame_entropy_vals.extend(frame_ent[valid_seq].numpy().tolist())

                bsz, n_visits, n_frames = attn.shape
                k = max(1, min(int(topk), n_frames))
                for bi in range(bsz):
                    for vi in range(n_visits):
                        if not bool(msk[bi, vi].any()):
                            continue
                        top_idx = torch.topk(attn[bi, vi], k=k).indices.tolist()
                        frame_topk_rows.append(
                            {
                                "batch_index": int(batch_idx),
                                "patient_id": patient_ids[bi],
                                "view": view,
                                "visit_index": int(vi),
                                "topk_frame_idx": [int(x) for x in top_idx],
                            }
                        )
                        if len(frame_topk_rows) >= 300:
                            break
                    if len(frame_topk_rows) >= 300:
                        break

            view_attn = diag.get("view_attn")
            view_attn_mask = diag.get("view_attn_mask")
            if view_attn is not None and view_attn_mask is not None:
                vattn = view_attn.detach().cpu()
                vmask = view_attn_mask.detach().cpu().bool()
                valid = vattn[vmask]
                if valid.numel() > 0:
                    view_valid_weights.extend(valid.numpy().tolist())
                invalid = vattn[~vmask]
                if invalid.numel() > 0:
                    view_masked_max_vals.append(float(invalid.max().item()))
                vent = _entropy_from_weights(vattn, vmask, dim=-1)
                valid_visit = vmask.any(dim=-1)
                if valid_visit.any():
                    view_entropy_vals.extend(vent[valid_visit].numpy().tolist())
                    top_view = vattn.argmax(dim=-1)
                    for view_idx, view_name in enumerate(IchilovPipeline3Model.VIEW_ORDER):
                        view_dom_counts[view_name] += int(
                            ((top_view == view_idx) & valid_visit).sum().item()
                        )

            visit_emb_cpu = visit_emb.detach().cpu()
            visit_mask_cpu = visit_mask.detach().cpu()
            visit_times_cpu = visit_times.detach().cpu()
            bsz, n_visits, _ = visit_emb_cpu.shape
            for bi in range(bsz):
                valid_idx = torch.where(visit_mask_cpu[bi])[0].tolist()
                for ii in range(len(valid_idx) - 1):
                    a = visit_emb_cpu[bi, valid_idx[ii]]
                    b = visit_emb_cpu[bi, valid_idx[ii + 1]]
                    cos = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1).item()
                    dt = float(visit_times_cpu[bi, valid_idx[ii + 1]] - visit_times_cpu[bi, valid_idx[ii]])
                    within_cos.append(float(cos))
                    within_dt.append(dt)
            for bi in range(bsz):
                for bj in range(bi + 1, bsz):
                    vi = torch.where(visit_mask_cpu[bi])[0]
                    vj = torch.where(visit_mask_cpu[bj])[0]
                    if len(vi) == 0 or len(vj) == 0:
                        continue
                    ei = visit_emb_cpu[bi, int(vi[0].item())]
                    ej = visit_emb_cpu[bj, int(vj[0].item())]
                    cos = torch.nn.functional.cosine_similarity(ei.unsqueeze(0), ej.unsqueeze(0), dim=1).item()
                    across_cos.append(float(cos))

    frame_entropy_mean = float(np.mean(frame_entropy_vals)) if frame_entropy_vals else float("nan")
    view_entropy_mean = float(np.mean(view_entropy_vals)) if view_entropy_vals else float("nan")
    diag_summary: Dict[str, Any] = {
        "frame_attention": {
            "n_valid_weights": int(len(frame_valid_weights)),
            "masked_attention_max": float(np.max(frame_masked_max_vals)) if frame_masked_max_vals else 0.0,
            "entropy_mean": frame_entropy_mean,
            "topk_examples": frame_topk_rows[:200],
        },
        "view_attention": {
            "n_valid_weights": int(len(view_valid_weights)),
            "masked_attention_max": float(np.max(view_masked_max_vals)) if view_masked_max_vals else 0.0,
            "entropy_mean": view_entropy_mean,
            "dominance_counts": view_dom_counts,
        },
        "visit_embedding": {
            "within_patient_pairs": int(len(within_cos)),
            "across_patient_pairs": int(len(across_cos)),
            "within_next_cos_mean": float(np.mean(within_cos)) if within_cos else float("nan"),
            "across_cos_mean": float(np.mean(across_cos)) if across_cos else float("nan"),
        },
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        if frame_valid_weights:
            _plot_hist(
                np.asarray(frame_valid_weights),
                out_dir / f"{tag}_frame_attention_hist.png",
                "Frame attention weights",
                "weight",
            )
        if view_valid_weights:
            _plot_hist(
                np.asarray(view_valid_weights),
                out_dir / f"{tag}_view_attention_hist.png",
                "View attention weights",
                "weight",
            )
        if within_cos and within_dt:
            _plot_scatter(
                np.asarray(within_dt),
                np.asarray(within_cos),
                out_dir / f"{tag}_within_cos_vs_delta_time.png",
                "Within-patient visit cosine vs delta time",
                "delta time (months)",
                "cosine similarity",
            )
        if across_cos:
            _plot_hist(
                np.asarray(across_cos),
                out_dir / f"{tag}_across_patient_cosine_hist.png",
                "Across-patient visit cosine",
                "cosine similarity",
            )
        (out_dir / f"{tag}_diagnostics.json").write_text(
            json.dumps(diag_summary, indent=2),
            encoding="utf-8",
        )
    return diag_summary


def _compute_split_metrics(pred_df: pd.DataFrame, split: str) -> Dict[str, float]:
    sub = pred_df[pred_df["split"] == split].copy()
    out = {
        "n_rows": float(len(sub)),
        "delta_pearson": float("nan"),
        "delta_spearman": float("nan"),
        "delta_pearson_negpred": float("nan"),
        "delta_spearman_negpred": float("nan"),
        "corr_signflip_advantage": float("nan"),
        "delta_mae": float("nan"),
        "risk_auc": float("nan"),
    }
    if sub.empty:
        return out
    mask_delta = sub["delta_gls_mask"].fillna(False).astype(bool).to_numpy()
    if mask_delta.any():
        y_true = sub.loc[mask_delta, "delta_gls_target"].astype(float).to_numpy()
        y_pred = sub.loc[mask_delta, "delta_gls_pred"].astype(float).to_numpy()
        finite = np.isfinite(y_true) & np.isfinite(y_pred)
        if finite.any():
            y_true_f = y_true[finite]
            y_pred_f = y_pred[finite]
            out["delta_pearson"] = _safe_pearson(y_true_f, y_pred_f)
            out["delta_spearman"] = _safe_spearman(y_true_f, y_pred_f)
            out["delta_pearson_negpred"] = _safe_pearson(y_true_f, -y_pred_f)
            out["delta_spearman_negpred"] = _safe_spearman(y_true_f, -y_pred_f)
            if np.isfinite(out["delta_pearson"]) and np.isfinite(out["delta_pearson_negpred"]):
                out["corr_signflip_advantage"] = float(
                    out["delta_pearson_negpred"] - out["delta_pearson"]
                )
            out["delta_mae"] = float(np.mean(np.abs(y_pred_f - y_true_f)))
    mask_risk = sub["risk_mask"].fillna(False).astype(bool).to_numpy()
    if roc_auc_score is not None and mask_risk.any():
        y_true_risk = sub.loc[mask_risk, "risk_label"].astype(float).to_numpy()
        y_prob_risk = sub.loc[mask_risk, "risk_prob"].astype(float).to_numpy()
        finite = np.isfinite(y_true_risk) & np.isfinite(y_prob_risk)
        if finite.any():
            y_true_risk_f = y_true_risk[finite]
            y_prob_risk_f = y_prob_risk[finite]
            if len(np.unique(y_true_risk_f)) >= 2:
                out["risk_auc"] = float(roc_auc_score(y_true_risk_f, y_prob_risk_f))
    return out


def _run_delta_debug(
    model: IchilovPipeline3Model,
    loader: Optional[DataLoader],
    device: torch.device,
    use_precomputed_embeddings: bool,
    max_samples: int = 2,
) -> Dict[str, Any]:
    if loader is None:
        return {}
    model.eval()
    y_true: List[float] = []
    y_pred: List[float] = []
    sample_rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for batch_cpu in loader:
            batch = _to_device(batch_cpu, device)
            if use_precomputed_embeddings:
                outputs = model.forward_from_frame_embeddings(
                    embeddings_by_view=batch["embeddings_by_view"],
                    frame_masks_by_view=batch["frame_masks_by_view"],
                    visit_mask=batch["visit_mask"],
                    visit_times=batch["visit_times"],
                )
            else:
                outputs = model(
                    frames_by_view=batch["frames_by_view"],
                    frame_masks_by_view=batch["frame_masks_by_view"],
                    visit_mask=batch["visit_mask"],
                    visit_times=batch["visit_times"],
                )
            pred = outputs["delta_gls"].detach().cpu().numpy()
            target = batch["delta_gls_target"].detach().cpu().numpy()
            mask = batch["delta_gls_mask"].detach().cpu().numpy().astype(bool)
            first_t = batch.get("delta_first_visit_time")
            last_t = batch.get("delta_last_visit_time")
            first_gls = batch.get("delta_first_gls")
            last_gls = batch.get("delta_last_gls")
            if first_t is not None:
                first_t_np = first_t.detach().cpu().numpy()
                last_t_np = last_t.detach().cpu().numpy() if last_t is not None else np.zeros_like(first_t_np)
                first_gls_np = first_gls.detach().cpu().numpy() if first_gls is not None else np.zeros_like(first_t_np)
                last_gls_np = last_gls.detach().cpu().numpy() if last_gls is not None else np.zeros_like(first_t_np)
            else:
                first_t_np = np.zeros_like(target)
                last_t_np = np.zeros_like(target)
                first_gls_np = np.zeros_like(target)
                last_gls_np = np.zeros_like(target)
            for i, ok in enumerate(mask):
                if not ok:
                    continue
                y_true.append(float(target[i]))
                y_pred.append(float(pred[i]))
                if len(sample_rows) < max_samples:
                    sample_rows.append(
                        {
                            "patient_id": str(batch["patient_id"][i]),
                            "t_first": float(first_t_np[i]),
                            "t_last": float(last_t_np[i]),
                            "gls_first": float(first_gls_np[i]),
                            "gls_last": float(last_gls_np[i]),
                            "target_delta": float(target[i]),
                            "pred_delta": float(pred[i]),
                        }
                    )
    y_true_np = np.asarray(y_true, dtype=np.float64)
    y_pred_np = np.asarray(y_pred, dtype=np.float64)
    corr = _safe_pearson(y_true_np, y_pred_np)
    corr_flip = _safe_pearson(y_true_np, -y_pred_np)
    return {
        "n": int(len(y_true)),
        "corr_pred": float(corr),
        "corr_negpred": float(corr_flip),
        "corr_signflip_advantage": float(corr_flip - corr)
        if np.isfinite(corr_flip) and np.isfinite(corr)
        else float("nan"),
        "samples": sample_rows,
    }


def _collect_probe_features(
    model: IchilovPipeline3Model,
    loader: Optional[DataLoader],
    device: torch.device,
    split_name: str,
    use_precomputed_embeddings: bool,
) -> pd.DataFrame:
    if loader is None:
        return pd.DataFrame()
    model.eval()
    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for batch_cpu in tqdm(loader, desc=f"Probe {split_name}", leave=False):
            batch = _to_device(batch_cpu, device)
            if use_precomputed_embeddings:
                outputs = model.forward_from_frame_embeddings(
                    embeddings_by_view=batch["embeddings_by_view"],
                    frame_masks_by_view=batch["frame_masks_by_view"],
                    visit_mask=batch["visit_mask"],
                    visit_times=batch["visit_times"],
                )
            else:
                outputs = model(
                    frames_by_view=batch["frames_by_view"],
                    frame_masks_by_view=batch["frame_masks_by_view"],
                    visit_mask=batch["visit_mask"],
                    visit_times=batch["visit_times"],
                )
            visit_emb = outputs["visit_embedding"].detach().cpu().numpy()
            visit_mask = batch["visit_mask"].detach().cpu().numpy().astype(bool)
            delta_pred = outputs["delta_gls"].detach().cpu().numpy()
            delta_target = batch["delta_gls_target"].detach().cpu().numpy()
            delta_mask = batch["delta_gls_mask"].detach().cpu().numpy().astype(bool)
            risk_prob = outputs["risk_prob"].detach().cpu().numpy()
            for i, pid in enumerate(batch["patient_id"]):
                valid_idx = np.where(visit_mask[i])[0]
                if len(valid_idx) == 0:
                    continue
                first = visit_emb[i, valid_idx[0]]
                last = visit_emb[i, valid_idx[-1]]
                mean = visit_emb[i, valid_idx].mean(axis=0)
                rows.append(
                    {
                        "split": split_name,
                        "patient_id": str(pid),
                        "first_emb": first,
                        "last_emb": last,
                        "mean_emb": mean,
                        "delta_emb": last - first,
                        "delta_pred": float(delta_pred[i]),
                        "risk_prob": float(risk_prob[i]),
                        "delta_target": float(delta_target[i]) if delta_mask[i] else np.nan,
                        "delta_mask": bool(delta_mask[i]),
                    }
                )
    return pd.DataFrame(rows)


def _run_patient_id_probe(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict[str, float]:
    if SGDClassifier is None:
        return {"train_acc": float("nan"), "val_acc": float("nan"), "known_val_fraction": float("nan")}
    if train_df.empty:
        return {"train_acc": float("nan"), "val_acc": float("nan"), "known_val_fraction": float("nan")}
    X_train = np.stack(train_df["mean_emb"].to_list()).astype(np.float32)
    y_train = train_df["patient_id"].astype(str).to_numpy()
    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        max_iter=2000,
        tol=1e-4,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    train_acc = float((clf.predict(X_train) == y_train).mean())
    if val_df.empty:
        return {"train_acc": train_acc, "val_acc": float("nan"), "known_val_fraction": float("nan")}
    X_val = np.stack(val_df["mean_emb"].to_list()).astype(np.float32)
    y_val = val_df["patient_id"].astype(str).to_numpy()
    classes = set(y_train.tolist())
    known = np.asarray([yy in classes for yy in y_val], dtype=bool)
    if not known.any():
        return {"train_acc": train_acc, "val_acc": float("nan"), "known_val_fraction": 0.0}
    y_hat = clf.predict(X_val[known])
    val_acc = float((y_hat == y_val[known]).mean())
    return {
        "train_acc": train_acc,
        "val_acc": val_acc,
        "known_val_fraction": float(known.mean()),
    }


def _run_delta_gls_probe(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict[str, float]:
    out = {
        "baseline_pearson": float("nan"),
        "baseline_spearman": float("nan"),
        "baseline_mae": float("nan"),
        "model_pearson": float("nan"),
        "model_spearman": float("nan"),
        "model_mae": float("nan"),
    }
    if train_df.empty or val_df.empty:
        return out
    train_mask = train_df["delta_mask"].fillna(False).astype(bool).to_numpy()
    val_mask = val_df["delta_mask"].fillna(False).astype(bool).to_numpy()
    if not train_mask.any() or not val_mask.any():
        return out
    y_train = train_df.loc[train_mask, "delta_target"].astype(float).to_numpy()
    y_val = val_df.loc[val_mask, "delta_target"].astype(float).to_numpy()
    y_val_model = val_df.loc[val_mask, "delta_pred"].astype(float).to_numpy()
    out["model_pearson"] = _safe_pearson(y_val, y_val_model)
    out["model_spearman"] = _safe_spearman(y_val, y_val_model)
    out["model_mae"] = float(np.mean(np.abs(y_val_model - y_val)))
    if Ridge is None:
        return out
    X_train = np.stack(train_df.loc[train_mask, "delta_emb"].to_list()).astype(np.float32)
    X_val = np.stack(val_df.loc[val_mask, "delta_emb"].to_list()).astype(np.float32)
    reg = Ridge(alpha=1.0)
    reg.fit(X_train, y_train)
    y_val_base = reg.predict(X_val)
    out["baseline_pearson"] = _safe_pearson(y_val, y_val_base)
    out["baseline_spearman"] = _safe_spearman(y_val, y_val_base)
    out["baseline_mae"] = float(np.mean(np.abs(y_val_base - y_val)))
    return out


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
    logger.info(
        "Delta GLS convention: %s (latest_minus_earliest => GLS_latest - GLS_earliest).",
        cfg.delta_gls_convention,
    )

    if cfg.input_embeddings is not None:
        dataset = FrameEmbeddingVisitDataset(
            input_embeddings=cfg.input_embeddings,
            views=cfg.views,
            t_frames=cfg.t_frames,
            sampling_mode=cfg.sampling_mode,
            clip_stride=cfg.clip_stride,
            include_last_window=cfg.include_last_window,
            phase_aligned_include_midpoints=cfg.phase_aligned_include_midpoints,
            phase_aligned_strategy=cfg.phase_aligned_strategy,
            max_visits=cfg.max_visits,
            min_visits=cfg.min_visits,
            risk_delta_threshold=cfg.risk_delta_threshold,
            delta_gls_convention=cfg.delta_gls_convention,
            report_xlsx=cfg.report_xlsx,
        )
        collate = embedding_visit_collate_fn
        use_precomputed_embeddings = True
    else:
        dataset = VisitDataset(
            input_xlsx=cfg.input_xlsx,
            echo_root=cfg.echo_root,
            cropped_root=cfg.cropped_root,
            views=cfg.views,
            t_frames=cfg.t_frames,
            sampling_mode=cfg.sampling_mode,
            clip_stride=cfg.clip_stride,
            include_last_window=cfg.include_last_window,
            phase_aligned_include_midpoints=cfg.phase_aligned_include_midpoints,
            phase_aligned_strategy=cfg.phase_aligned_strategy,
            max_visits=cfg.max_visits,
            min_visits=cfg.min_visits,
            risk_delta_threshold=cfg.risk_delta_threshold,
            delta_gls_convention=cfg.delta_gls_convention,
            random_view_sampling=False,  # TODO: add epoch-wise randomized view sampling if needed.
        )
        collate = visit_collate_fn
        use_precomputed_embeddings = False

    train_idx, val_idx, test_idx = _resolve_split_indices(dataset=dataset, cfg=cfg)
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx) if val_idx else None
    test_ds = Subset(dataset, test_idx) if test_idx else None

    train_loader = _make_dataloader(train_ds, cfg.batch_size, True, cfg.num_workers, collate)
    val_loader = _make_dataloader(val_ds, cfg.batch_size, False, cfg.num_workers, collate)
    test_loader = _make_dataloader(test_ds, cfg.batch_size, False, cfg.num_workers, collate)
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
        temporal_mode=cfg.temporal_mode,
        view_fusion_mode=cfg.view_fusion_mode,
        single_best_view=cfg.single_best_view,
        longitudinal_model_type=(
            cfg.longitudinal_model
            if cfg.longitudinal_ablation_mode in {"inherit", "", "none"}
            else cfg.longitudinal_ablation_mode
        ),
        longitudinal_hidden=cfg.longitudinal_hidden,
        longitudinal_layers=cfg.longitudinal_layers,
        longitudinal_heads=cfg.longitudinal_heads,
        longitudinal_dropout=cfg.longitudinal_dropout,
        use_time_encoding=cfg.use_time_encoding,
    ).to(device)
    if use_precomputed_embeddings:
        for p in model.frame_encoder.parameters():
            p.requires_grad = False

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
    diagnostics_dir = cfg.output_dir / "diagnostics"
    latest_diag_summary: Dict[str, Any] = {}

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
            use_precomputed_embeddings=use_precomputed_embeddings,
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
            use_precomputed_embeddings=use_precomputed_embeddings,
        )

        if scheduler is not None:
            scheduler.step()
        if plateau_scheduler is not None:
            plateau_scheduler.step(val_metrics["loss"])

        diag_epoch: Dict[str, Any] = {}
        if cfg.enable_diagnostics and (epoch % cfg.diagnostics_every == 0 or epoch == cfg.epochs):
            diag_epoch = _collect_diagnostics(
                model=model,
                loader=val_loader if val_loader is not None else train_loader,
                device=device,
                use_precomputed_embeddings=use_precomputed_embeddings,
                max_batches=cfg.diagnostics_batches,
                topk=cfg.diagnostics_topk,
                out_dir=None,
                tag=f"epoch_{epoch:03d}",
            )
            latest_diag_summary = diag_epoch

        delta_debug = {}
        if cfg.debug_delta:
            delta_debug = _run_delta_debug(
                model=model,
                loader=val_loader if val_loader is not None else train_loader,
                device=device,
                use_precomputed_embeddings=use_precomputed_embeddings,
                max_samples=cfg.debug_delta_samples,
            )
            if delta_debug:
                logger.info(
                    "Delta debug epoch %d: n=%d corr(pred,target)=%.4f corr(-pred,target)=%.4f",
                    epoch,
                    int(delta_debug.get("n", 0)),
                    float(delta_debug.get("corr_pred", float("nan"))),
                    float(delta_debug.get("corr_negpred", float("nan"))),
                )
                for s in delta_debug.get("samples", []):
                    logger.info(
                        "Delta sample pid=%s t=(%.3f,%.3f) gls=(%.3f,%.3f) target=%.3f pred=%.3f",
                        s.get("patient_id"),
                        float(s.get("t_first", 0.0)),
                        float(s.get("t_last", 0.0)),
                        float(s.get("gls_first", 0.0)),
                        float(s.get("gls_last", 0.0)),
                        float(s.get("target_delta", 0.0)),
                        float(s.get("pred_delta", 0.0)),
                    )
                adv = float(delta_debug.get("corr_signflip_advantage", float("nan")))
                if np.isfinite(adv) and adv > 0.2:
                    logger.warning(
                        "Possible delta_gls sign mismatch detected. corr(pred)=%.4f corr(-pred)=%.4f",
                        float(delta_debug.get("corr_pred", float("nan"))),
                        float(delta_debug.get("corr_negpred", float("nan"))),
                    )

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
            "frame_attn_entropy": float(
                diag_epoch.get("frame_attention", {}).get("entropy_mean", float("nan"))
            ),
            "view_attn_entropy": float(
                diag_epoch.get("view_attention", {}).get("entropy_mean", float("nan"))
            ),
            "delta_debug_corr_pred": float(delta_debug.get("corr_pred", float("nan")))
            if delta_debug
            else float("nan"),
            "delta_debug_corr_negpred": float(delta_debug.get("corr_negpred", float("nan")))
            if delta_debug
            else float("nan"),
            "delta_debug_signflip_advantage": float(
                delta_debug.get("corr_signflip_advantage", float("nan"))
            )
            if delta_debug
            else float("nan"),
        }
        history.append(row)
        logger.info(
            "Epoch %d/%d | train=%.4f val=%.4f delta=%.4f rank=%.4f smooth=%.4f frame_H=%.4f view_H=%.4f",
            epoch,
            cfg.epochs,
            row["train_loss"],
            row["val_loss"],
            row["val_delta"],
            row["val_rank"],
            row["val_smooth"],
            row["frame_attn_entropy"] if np.isfinite(row["frame_attn_entropy"]) else float("nan"),
            row["view_attn_entropy"] if np.isfinite(row["view_attn_entropy"]) else float("nan"),
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
            if np.isfinite(row["frame_attn_entropy"]):
                writer.add_scalar("diag/frame_attention_entropy", row["frame_attn_entropy"], epoch)
            if np.isfinite(row["view_attn_entropy"]):
                writer.add_scalar("diag/view_attention_entropy", row["view_attn_entropy"], epoch)
            if np.isfinite(row["delta_debug_corr_pred"]):
                writer.add_scalar("diag/delta_corr_pred", row["delta_debug_corr_pred"], epoch)
            if np.isfinite(row["delta_debug_corr_negpred"]):
                writer.add_scalar("diag/delta_corr_negpred", row["delta_debug_corr_negpred"], epoch)
            if np.isfinite(row["delta_debug_signflip_advantage"]):
                writer.add_scalar(
                    "diag/delta_corr_signflip_advantage",
                    row["delta_debug_signflip_advantage"],
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

    final_diag = {}
    if cfg.enable_diagnostics:
        final_diag = _collect_diagnostics(
            model=model,
            loader=val_loader if val_loader is not None else train_loader,
            device=device,
            use_precomputed_embeddings=use_precomputed_embeddings,
            max_batches=max(cfg.diagnostics_batches * 2, 2),
            topk=cfg.diagnostics_topk,
            out_dir=diagnostics_dir,
            tag="final",
        )
        if not final_diag:
            final_diag = latest_diag_summary
        masked_frame = final_diag.get("frame_attention", {}).get("masked_attention_max")
        masked_view = final_diag.get("view_attention", {}).get("masked_attention_max")
        if masked_frame is not None and float(masked_frame) > 1e-5:
            logger.warning("Frame attention leakage into masked frames: max=%.6f", float(masked_frame))
        if masked_view is not None and float(masked_view) > 1e-5:
            logger.warning("View attention leakage into masked views: max=%.6f", float(masked_view))

    pred_frames: List[pd.DataFrame] = []
    pred_frames.append(_predict(model, train_loader, device, split_name="train", use_precomputed_embeddings=use_precomputed_embeddings))
    pred_frames.append(_predict(model, val_loader, device, split_name="val", use_precomputed_embeddings=use_precomputed_embeddings))
    pred_frames.append(_predict(model, test_loader, device, split_name="test", use_precomputed_embeddings=use_precomputed_embeddings))
    pred_df = pd.concat([df for df in pred_frames if not df.empty], axis=0, ignore_index=True)
    pred_path = cfg.output_parquet or (cfg.output_dir / f"{cfg.run_name}_predictions.parquet")
    if pred_path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pred_path = pred_path.with_name(f"{pred_path.stem}_{stamp}{pred_path.suffix}")
        logger.warning("Prediction output exists; writing to %s", pred_path)
    pred_df.to_parquet(pred_path, index=False)

    split_metrics = {
        "train": _compute_split_metrics(pred_df, "train"),
        "val": _compute_split_metrics(pred_df, "val"),
        "test": _compute_split_metrics(pred_df, "test"),
    }
    val_adv = split_metrics.get("val", {}).get("corr_signflip_advantage", float("nan"))
    if np.isfinite(val_adv) and float(val_adv) > 0.2:
        logger.warning(
            "Possible delta_gls sign mismatch detected from predictions. "
            "corr_signflip_advantage=%.4f (corr(-pred,target) > corr(pred,target)).",
            float(val_adv),
        )

    probe_summary: Dict[str, Any] = {}
    if cfg.run_patient_id_probe or cfg.run_delta_gls_probe:
        probe_train = _collect_probe_features(
            model=model,
            loader=train_loader,
            device=device,
            split_name="train",
            use_precomputed_embeddings=use_precomputed_embeddings,
        )
        probe_val = _collect_probe_features(
            model=model,
            loader=val_loader if val_loader is not None else train_loader,
            device=device,
            split_name="val",
            use_precomputed_embeddings=use_precomputed_embeddings,
        )
        if cfg.run_patient_id_probe:
            pid_probe = _run_patient_id_probe(probe_train, probe_val)
            probe_summary["patient_id_probe"] = pid_probe
            if np.isfinite(pid_probe.get("train_acc", float("nan"))) and pid_probe["train_acc"] > 0.9:
                logger.warning(
                    "Patient-ID probe train accuracy is high (%.4f). Potential identity leakage.",
                    pid_probe["train_acc"],
                )
        if cfg.run_delta_gls_probe:
            probe_summary["delta_gls_probe"] = _run_delta_gls_probe(probe_train, probe_val)

    history_best = history_df.sort_values("val_loss", ascending=True).iloc[0].to_dict()
    summary = {
        "run_name": cfg.run_name,
        "output_dir": str(cfg.output_dir),
        "predictions_path": str(pred_path),
        "best_checkpoint": str(best_path),
        "latest_checkpoint": str(latest_path),
        "best_epoch": int(history_best["epoch"]),
        "best_val_loss": float(history_best["val_loss"]),
        "best_val_delta_mae": float(history_best["val_delta_mae"]),
        "split_metrics": split_metrics,
        "ablation": {
            "temporal_mode": cfg.temporal_mode,
            "view_fusion_mode": cfg.view_fusion_mode,
            "longitudinal_model": (
                cfg.longitudinal_model
                if cfg.longitudinal_ablation_mode in {"inherit", "", "none"}
                else cfg.longitudinal_ablation_mode
            ),
            "time_encoding": bool(cfg.use_time_encoding),
            "sampling_mode": cfg.sampling_mode,
            "phase_aligned_strategy": cfg.phase_aligned_strategy,
        },
        "delta_gls_convention": cfg.delta_gls_convention,
        "split_json": str(cfg.split_json) if cfg.split_json is not None else None,
        "diagnostics": final_diag,
        "probes": probe_summary,
    }
    summary_path = cfg.summary_json or (cfg.output_dir / f"{cfg.run_name}_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    logger.info("Best checkpoint: %s", best_path)
    logger.info("Latest checkpoint: %s", latest_path)
    logger.info("History JSON: %s", history_json)
    logger.info("Predictions: %s", pred_path)
    logger.info("Summary: %s", summary_path)


if __name__ == "__main__":
    main()
