"""
Frame-level SimCLR pretraining with pipeline2-compatible CLI.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ichilov_pipeline2_utils import (
    VIEW_KEYS,
    collect_dicoms_from_report,
    configure_pydicom_handlers,
    load_cropped_frames,
    parse_views,
    resize_tensor,
    to_cropped_path,
    to_tensor,
)

try:
    from sklearn.model_selection import GroupShuffleSplit
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "scikit-learn is required. Install with: .venv\\Scripts\\python -m pip install scikit-learn"
    ) from exc

from Ichilov_pipeline3.models.frame_encoder import FrameEncoder

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("ichilov_pretrain_frame_simclr")


@dataclass
class FrameEntry:
    cropped_path: Path
    frame_index: int
    view: Optional[str]
    patient_key: str


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _read_frame_count(dicom_path: Path) -> Optional[int]:
    try:
        import pydicom

        ds = pydicom.dcmread(str(dicom_path), force=True, stop_before_pixels=True)
    except Exception:
        return None
    n_frames = getattr(ds, "NumberOfFrames", None)
    if n_frames is None:
        return 1
    try:
        return int(n_frames)
    except Exception:
        return None


def _infer_view_from_path(path: Path) -> Optional[str]:
    name = path.as_posix().upper()
    for v in VIEW_KEYS:
        if v in name:
            return v
    return None


def _build_entries(
    cropped_root: Path,
    input_xlsx: Optional[Path],
    echo_root: Optional[Path],
    views: Optional[set],
    frame_stride: int,
    data_ratio: float,
    max_samples: int,
) -> List[FrameEntry]:
    entries: List[FrameEntry] = []
    if input_xlsx and input_xlsx.exists() and echo_root is not None:
        df = pd.read_excel(input_xlsx, engine="openpyxl")
        df.columns = [str(c).strip() for c in df.columns]
        dicoms = collect_dicoms_from_report(df, echo_root, views=views)
        for entry in dicoms:
            cropped = to_cropped_path(entry.path, echo_root, cropped_root)
            if cropped is None:
                continue
            n_frames = _read_frame_count(cropped) or 0
            if n_frames <= 0:
                continue
            patient_key = entry.patient_num or entry.patient_id or cropped.parts[0]
            for idx in range(0, n_frames, max(int(frame_stride), 1)):
                entries.append(
                    FrameEntry(
                        cropped_path=cropped,
                        frame_index=int(idx),
                        view=entry.view,
                        patient_key=str(patient_key),
                    )
                )
    else:
        for path in cropped_root.rglob("*.dcm"):
            n_frames = _read_frame_count(path) or 0
            if n_frames <= 0:
                continue
            view = _infer_view_from_path(path)
            patient_key = path.relative_to(cropped_root).parts[0] if cropped_root in path.parents else path.stem
            for idx in range(0, n_frames, max(int(frame_stride), 1)):
                entries.append(
                    FrameEntry(
                        cropped_path=path,
                        frame_index=int(idx),
                        view=view,
                        patient_key=str(patient_key),
                    )
                )

    if data_ratio < 1.0 and entries:
        rng = np.random.default_rng(42)
        keep = max(1, int(round(len(entries) * data_ratio)))
        entries = list(rng.choice(entries, size=keep, replace=False))

    if max_samples > 0 and len(entries) > max_samples:
        rng = np.random.default_rng(42)
        entries = list(rng.choice(entries, size=max_samples, replace=False))

    return entries


class FramePairDataset(Dataset):
    def __init__(
        self,
        entries: Sequence[FrameEntry],
        image_size: int = 224,
        aug_brightness: float = 0.1,
        aug_contrast: float = 0.1,
        aug_speckle: float = 0.0,
    ) -> None:
        self.entries = list(entries)
        self.image_size = int(image_size)
        self.aug_brightness = float(aug_brightness)
        self.aug_contrast = float(aug_contrast)
        self.aug_speckle = float(aug_speckle)

    def __len__(self) -> int:
        return len(self.entries)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        if self.aug_contrast > 0:
            alpha = 1.0 + random.uniform(-self.aug_contrast, self.aug_contrast)
            mean = x.mean(dim=(2, 3), keepdim=True)
            x = (x - mean) * alpha + mean
        if self.aug_brightness > 0:
            beta = 1.0 + random.uniform(-self.aug_brightness, self.aug_brightness)
            x = x * beta
        if self.aug_speckle > 0:
            x = x + torch.randn_like(x) * self.aug_speckle
        return x.clamp(0.0, 1.0)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        frames = load_cropped_frames(entry.cropped_path)
        if frames is None or entry.frame_index >= frames.shape[0]:
            return None
        frame = frames[entry.frame_index : entry.frame_index + 1]
        tensor = to_tensor(frame)  # [1,C,H,W]
        tensor = resize_tensor(tensor, size=self.image_size)
        x1 = self._augment(tensor.clone()).squeeze(0)
        x2 = self._augment(tensor.clone()).squeeze(0)
        view_idx = VIEW_KEYS.index(entry.view) if entry.view in VIEW_KEYS else -1
        return x1, x2, view_idx


def _collate(batch: list):
    items = [b for b in batch if b is not None]
    if not items:
        return None
    x1, x2, view_idx = zip(*items)
    return (
        torch.stack(x1, dim=0),
        torch.stack(x2, dim=0),
        torch.tensor(view_idx, dtype=torch.long),
    )


class DINOv2PretrainModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        pretrained: bool,
        train_encoder_blocks: int,
        freeze_backbone: bool,
    ) -> None:
        super().__init__()
        self.frame_encoder = FrameEncoder(
            backbone_name=backbone_name,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            unfreeze_last_blocks=max(0, train_encoder_blocks),
        )
        d = int(self.frame_encoder.output_dim)
        self.projector = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, 128),
        )
        self.view_head = nn.Linear(d, len(VIEW_KEYS))
        self.recon_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, 3 * 56 * 56),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.frame_encoder(x)

    def reconstruct_from_features(self, feats: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
        recon = self.recon_head(feats).reshape(feats.shape[0], 3, 56, 56)
        recon = F.interpolate(recon, size=out_hw, mode="bilinear", align_corners=False)
        return recon.clamp(0.0, 1.0)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encode(x)
        return self.reconstruct_from_features(feats, (x.shape[-2], x.shape[-1]))


def _nt_xent(z1: torch.Tensor, z2: torch.Tensor, temp: float = 0.2) -> torch.Tensor:
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = torch.mm(z1, z2.t()) / temp
    labels = torch.arange(z1.size(0), device=z1.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def _train_or_eval(
    model: DINOv2PretrainModel,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    view_head: bool,
    view_loss_weight: float,
    recon_loss_weight: float,
) -> Tuple[float, float, float]:
    train = optimizer is not None
    model.train() if train else model.eval()
    total = 0.0
    total_view = 0.0
    total_recon = 0.0
    n = 0
    grad_ctx = torch.enable_grad() if train else torch.no_grad()
    with grad_ctx:
        for batch in tqdm(loader, desc="Train" if train else "Val", leave=False):
            if batch is None:
                continue
            x1, x2, view_idx = batch
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            view_idx = view_idx.to(device, non_blocking=True)

            f1 = model.encode(x1)
            f2 = model.encode(x2)
            z1 = model.projector(f1)
            z2 = model.projector(f2)
            loss = _nt_xent(z1, z2)
            view_loss = torch.tensor(0.0, device=device)
            recon_loss = torch.tensor(0.0, device=device)
            if recon_loss_weight > 0:
                recon = model.reconstruct_from_features(f1, (x1.shape[-2], x1.shape[-1]))
                recon_loss = F.mse_loss(recon, x1)
                loss = loss + recon_loss_weight * recon_loss
            if view_head:
                mask = view_idx >= 0
                if mask.any():
                    logits = model.view_head(f1)
                    view_loss = F.cross_entropy(logits[mask], view_idx[mask])
                    loss = loss + view_loss_weight * view_loss

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            bsz = x1.size(0)
            total += float(loss.item()) * bsz
            total_view += float(view_loss.item()) * bsz
            total_recon += float(recon_loss.item()) * bsz
            n += bsz
    return total / max(1, n), total_view / max(1, n), total_recon / max(1, n)


def _save_recon_grid(
    model: DINOv2PretrainModel,
    loader: Optional[DataLoader],
    device: torch.device,
    out_path: Path,
    n_samples: int = 9,
) -> None:
    if loader is None:
        return
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            x1, _, _ = batch
            x1 = x1.to(device, non_blocking=True)
            recon = model.reconstruct(x1)
            count = min(int(n_samples), x1.shape[0])
            if count <= 0:
                return
            orig = x1[:count].detach().cpu().permute(0, 2, 3, 1).numpy()
            rec = recon[:count].detach().cpu().permute(0, 2, 3, 1).numpy()

            rows = int(np.ceil(count / 3))
            fig, axes = plt.subplots(rows, 6, figsize=(12, 2.6 * rows))
            axes = np.array(axes).reshape(rows, 6)
            for i in range(rows * 3):
                r = i // 3
                c0 = (i % 3) * 2
                ax_o = axes[r, c0]
                ax_r = axes[r, c0 + 1]
                if i < count:
                    ax_o.imshow(np.clip(orig[i], 0.0, 1.0))
                    ax_o.set_title("Original", fontsize=9)
                    ax_r.imshow(np.clip(rec[i], 0.0, 1.0))
                    ax_r.set_title("Reconstruction", fontsize=9)
                ax_o.axis("off")
                ax_r.axis("off")
            fig.tight_layout()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=170)
            plt.close(fig)
            return


def _load_checkpoint_if_any(model: DINOv2PretrainModel, weights: Optional[Path]) -> None:
    if weights is None or not weights.exists():
        return
    ckpt = torch.load(weights, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state" in ckpt and isinstance(ckpt["model_state"], dict):
        missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
        logger.info("Loaded checkpoint %s (missing=%d unexpected=%d)", weights, len(missing), len(unexpected))
        return
    if isinstance(ckpt, dict) and "backbone_state" in ckpt and isinstance(ckpt["backbone_state"], dict):
        model.frame_encoder.backbone.load_state_dict(ckpt["backbone_state"], strict=False)
        logger.info("Loaded backbone checkpoint %s", weights)
        return
    if isinstance(ckpt, dict):
        model.frame_encoder.backbone.load_state_dict(ckpt, strict=False)
        logger.info("Loaded raw state_dict checkpoint %s", weights)


def _latest_resume_checkpoint(output_dir: Path, latest_only: bool = False) -> Optional[Path]:
    if not output_dir.exists():
        return None
    latest = sorted(output_dir.glob("*_latest.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if latest:
        return latest[0]
    if latest_only:
        return None
    best = sorted(output_dir.glob("*_best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if best:
        return best[0]
    any_pt = sorted(output_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return any_pt[0] if any_pt else None


def _resume_run_name_from_checkpoint(path: Path) -> str:
    stem = path.stem
    for suffix in ("_latest", "_best"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _load_resume_checkpoint(path: Path, model: DINOv2PretrainModel) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unsupported checkpoint format for resume: {path}")

    if "model_state" in ckpt and isinstance(ckpt["model_state"], dict):
        model.load_state_dict(ckpt["model_state"], strict=False)
    elif "backbone_state" in ckpt and isinstance(ckpt["backbone_state"], dict):
        model.frame_encoder.backbone.load_state_dict(ckpt["backbone_state"], strict=False)
    else:
        raise RuntimeError(f"Checkpoint does not include model_state/backbone_state: {path}")
    logger.info("Resumed model from checkpoint: %s", path)
    return ckpt


def _load_existing_history(path: Path) -> List[dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to parse existing history file %s: %s", path, exc)
        return []
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return []


def _merge_history(base: List[dict], incoming: List[dict]) -> List[dict]:
    by_epoch: Dict[int, dict] = {}
    ordered_epochs: List[int] = []
    for item in base + incoming:
        if not isinstance(item, dict):
            continue
        try:
            epoch = int(item.get("epoch"))
        except Exception:
            continue
        if epoch not in by_epoch:
            ordered_epochs.append(epoch)
        by_epoch[epoch] = item
    return [by_epoch[e] for e in sorted(set(ordered_epochs))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Frame-level SimCLR pretraining.")
    parser.add_argument("--cropped-root", type=Path, required=True, help="Root of cropped DICOMs.")
    parser.add_argument("--weights", type=Path, default=None, help="Optional initial checkpoint.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for checkpoints.")
    parser.add_argument("--input-xlsx", type=Path, default=None, help="Optional Excel registry for view filtering.")
    parser.add_argument("--echo-root", type=Path, default=None, help="Root of original DICOMs.")
    parser.add_argument("--views", type=str, default="", help="Comma/space-separated views to include.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Stride between frames sampled from each cine.")
    parser.add_argument("--mask-ratio", type=float, default=0.75, help="Unused (kept for pipeline2 compatibility).")
    parser.add_argument("--train-encoder-blocks", type=int, default=2, help="Number of last encoder blocks to train.")
    parser.add_argument("--freeze-decoder", action="store_true", help="Unused for SimCLR (kept for compatibility).")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="Data loader workers.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="Weight decay.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio by patient.")
    parser.add_argument("--data-ratio", type=float, default=1.0, help="Fraction of frames to use.")
    parser.add_argument("--max-samples", type=int, default=0, help="Max number of frame samples.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto|cuda|cpu.")
    parser.add_argument("--run-name", type=str, default="", help="Run name.")
    parser.add_argument("--view-head", action="store_true", help="Enable view classification head.")
    parser.add_argument("--view-loss-weight", type=float, default=0.1, help="View loss weight.")
    parser.add_argument("--aug-brightness", type=float, default=0.1, help="Max brightness delta.")
    parser.add_argument("--aug-contrast", type=float, default=0.1, help="Max contrast delta.")
    parser.add_argument("--aug-rotate-deg", type=float, default=5.0, help="Unused (kept for compatibility).")
    parser.add_argument("--aug-blur", type=float, default=0.0, help="Unused (kept for compatibility).")
    parser.add_argument("--aug-speckle", type=float, default=0.0, help="Speckle noise std.")
    parser.add_argument("--aug-random-erase", type=float, default=0.0, help="Unused (kept for compatibility).")
    parser.add_argument("--recon-samples", type=int, default=9, help="Unused (kept for compatibility).")
    parser.add_argument("--recon-seed", type=int, default=123, help="Unused (kept for compatibility).")
    parser.add_argument("--recon-mask-ratio", type=float, default=None, help="Unused (kept for compatibility).")
    parser.add_argument("--recon-loss-weight", type=float, default=1.0, help="Weight for reconstruction loss.")
    parser.add_argument("--safe-decode", action="store_true", help="Disable GDCM/pylibjpeg decoders.")
    parser.add_argument("--backbone-name", type=str, default="vit_small_patch14_dinov2.lvd142m")
    parser.add_argument("--resume-latest", action="store_true", help="Resume from the newest checkpoint in output-dir.")
    parser.add_argument("--resume-path", type=Path, default=None, help="Explicit checkpoint path to resume from.")
    parser.add_argument(
        "--continue-session",
        action="store_true",
        help="Resume exactly from <run_name>_latest.pt in output-dir and treat --epochs as additional epochs.",
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.freeze_decoder:
        logger.info("--freeze-decoder is ignored for SimCLR pretraining.")
    configure_pydicom_handlers(args.safe_decode)
    _set_seed(args.seed)

    views = parse_views(args.views)
    entries = _build_entries(
        args.cropped_root,
        args.input_xlsx,
        args.echo_root,
        views,
        args.frame_stride,
        args.data_ratio,
        args.max_samples,
    )
    if not entries:
        raise RuntimeError("No frame samples found. Check inputs and filters.")

    if args.val_ratio > 0 and len(entries) > 1:
        groups = [e.patient_key for e in entries]
        splitter = GroupShuffleSplit(n_splits=1, test_size=args.val_ratio, random_state=args.seed)
        train_idx, val_idx = next(splitter.split(entries, groups=groups))
        train_entries = [entries[i] for i in train_idx]
        val_entries = [entries[i] for i in val_idx]
    else:
        train_entries = entries
        val_entries = []

    train_ds = FramePairDataset(
        train_entries,
        image_size=518,
        aug_brightness=args.aug_brightness,
        aug_contrast=args.aug_contrast,
        aug_speckle=args.aug_speckle,
    )
    val_ds = (
        FramePairDataset(
            val_entries,
            image_size=518,
            aug_brightness=0.0,
            aug_contrast=0.0,
            aug_speckle=0.0,
        )
        if val_entries
        else None
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=_collate,
        )
        if val_ds is not None
        else None
    )

    model = DINOv2PretrainModel(
        backbone_name=args.backbone_name,
        pretrained=True,
        train_encoder_blocks=max(0, args.train_encoder_blocks),
        freeze_backbone=True,
    ).to(device)

    resume_path: Optional[Path] = None
    if args.resume_path is not None:
        resume_path = args.resume_path
        if not resume_path.exists():
            raise FileNotFoundError(f"resume-path not found: {resume_path}")
    elif args.continue_session:
        resume_path = _latest_resume_checkpoint(args.output_dir, latest_only=True)
        if resume_path is None:
            raise FileNotFoundError(
                f"--continue-session requires an existing *_latest.pt checkpoint in: {args.output_dir}"
            )
    elif args.resume_latest:
        resume_path = _latest_resume_checkpoint(args.output_dir)
        if resume_path is None:
            logger.warning("--resume-latest requested but no checkpoint found in %s", args.output_dir)

    resume_ckpt: Optional[Dict[str, Any]] = None
    if resume_path is not None:
        resume_ckpt = _load_resume_checkpoint(resume_path, model)
        if not args.run_name:
            args.run_name = _resume_run_name_from_checkpoint(resume_path)
    else:
        _load_checkpoint_if_any(model, args.weights)

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters.")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 1
    best_loss = float("inf")
    history: List[dict] = []
    prev_epoch = 0
    if resume_ckpt is not None:
        if "optimizer_state" in resume_ckpt and isinstance(resume_ckpt["optimizer_state"], dict):
            try:
                optimizer.load_state_dict(resume_ckpt["optimizer_state"])
                logger.info("Loaded optimizer state from resume checkpoint.")
            except Exception as exc:
                logger.warning("Failed to load optimizer state from resume checkpoint: %s", exc)
        prev_epoch = int(resume_ckpt.get("epoch", 0))
        start_epoch = max(1, prev_epoch + 1)
        best_loss = float(resume_ckpt.get("best_loss", best_loss))
        if isinstance(resume_ckpt.get("history"), list):
            history = list(resume_ckpt["history"])

    run_name = args.run_name or f"frame_dinov2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / f"{run_name}_best.pt"
    latest_path = args.output_dir / f"{run_name}_latest.pt"
    history_path = args.output_dir / f"{run_name}_history.json"
    config_path = args.output_dir / f"{run_name}_config.json"
    recon_dir = args.output_dir / f"{run_name}_reconstructions"
    recon_dir.mkdir(parents=True, exist_ok=True)
    if resume_ckpt is not None:
        existing_history = _load_existing_history(history_path)
        history = _merge_history(existing_history, history)

    target_epoch = int(args.epochs)
    if args.continue_session and resume_ckpt is not None:
        target_epoch = prev_epoch + max(0, int(args.epochs))
        logger.info(
            "Continue-session enabled: previous epoch=%d, additional epochs=%d, target epoch=%d",
            prev_epoch,
            int(args.epochs),
            target_epoch,
        )
    writer = (
        SummaryWriter(
            log_dir=str(args.output_dir / "tensorboard" / run_name),
            purge_step=(start_epoch if resume_ckpt is not None else None),
        )
        if SummaryWriter is not None
        else None
    )

    if start_epoch > target_epoch:
        logger.warning(
            "Resume checkpoint epoch (%d) is already >= requested epochs (%d). Nothing to train.",
            start_epoch - 1,
            target_epoch,
        )

    for epoch in range(start_epoch, target_epoch + 1):
        train_loss, train_view, train_recon = _train_or_eval(
            model, train_loader, optimizer, device, args.view_head, args.view_loss_weight, args.recon_loss_weight
        )
        val_loss, val_view, val_recon = (0.0, 0.0, 0.0)
        if val_loader is not None:
            val_loss, val_view, val_recon = _train_or_eval(
                model, val_loader, None, device, args.view_head, args.view_loss_weight, args.recon_loss_weight
            )
            _save_recon_grid(
                model=model,
                loader=val_loader,
                device=device,
                out_path=recon_dir / f"epoch_{epoch:03d}.png",
                n_samples=max(1, int(args.recon_samples)),
            )
        metric = val_loss if val_loader is not None else train_loss
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss if val_loader is not None else None,
                "train_view_loss": train_view,
                "val_view_loss": val_view if val_loader is not None else None,
                "train_recon_loss": train_recon,
                "val_recon_loss": val_recon if val_loader is not None else None,
            }
        )
        logger.info(
            "Epoch %d/%d | train=%.4f val=%.4f recon_val=%.4f",
            epoch,
            target_epoch,
            train_loss,
            val_loss if val_loader is not None else train_loss,
            val_recon if val_loader is not None else train_recon,
        )
        if writer is not None:
            writer.add_scalars("loss/total", {"train": train_loss, "val": val_loss if val_loader is not None else train_loss}, epoch)
            writer.add_scalars("loss/recon", {"train": train_recon, "val": val_recon if val_loader is not None else train_recon}, epoch)
            writer.add_scalars("loss/view", {"train": train_view, "val": val_view if val_loader is not None else train_view}, epoch)
        torch.save(
            {
                "epoch": epoch,
                "best_loss": best_loss,
                "model_state": model.state_dict(),
                "backbone_state": model.frame_encoder.backbone.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "history": history,
                "config": vars(args),
                "target_epoch": target_epoch,
            },
            latest_path,
        )
        if metric < best_loss:
            best_loss = metric
            torch.save(
                {
                    "epoch": epoch,
                    "best_loss": best_loss,
                    "model_state": model.state_dict(),
                    "backbone_state": model.frame_encoder.backbone.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "history": history,
                    "config": vars(args),
                    "target_epoch": target_epoch,
                },
                best_path,
            )

    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    config_payload = dict(vars(args))
    config_payload["resolved_run_name"] = run_name
    config_payload["start_epoch"] = start_epoch
    config_payload["target_epoch"] = target_epoch
    config_payload["resume_checkpoint"] = str(resume_path) if resume_path is not None else None
    config_payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    config_path.write_text(json.dumps(config_payload, indent=2, default=str), encoding="utf-8")
    if writer is not None:
        writer.close()
    logger.info("Best checkpoint: %s", best_path)
    logger.info("Latest checkpoint: %s", latest_path)


if __name__ == "__main__":
    main()


