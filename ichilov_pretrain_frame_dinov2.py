"""
Frame-level DINOv2 pretraining with pipeline2-compatible CLI.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
import logging
import math
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
logger = logging.getLogger("ichilov_pretrain_frame_dinov2")


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

class DINOMultiCropDataset(Dataset):
    def __init__(
        self,
        entries: Sequence[FrameEntry],
        image_size: int = 518,
        local_crops: int = 1,
        aug_brightness: float = 0.1,
        aug_contrast: float = 0.1,
        aug_speckle: float = 0.0,
    ) -> None:
        self.entries = list(entries)
        self.image_size = int(image_size)
        self.local_crops = max(0, int(local_crops))
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

    def _random_resized_crop(self, x: torch.Tensor, scale_min: float, scale_max: float) -> torch.Tensor:
        _, _, h, w = x.shape
        area = float(h * w)
        target_area = random.uniform(float(scale_min), float(scale_max)) * area
        side = max(1, int(round(math.sqrt(target_area))))
        crop_h = max(1, min(h, side))
        crop_w = max(1, min(w, side))
        top = random.randint(0, max(0, h - crop_h))
        left = random.randint(0, max(0, w - crop_w))
        crop = x[:, :, top : top + crop_h, left : left + crop_w]
        if crop.shape[-2:] != (self.image_size, self.image_size):
            crop = F.interpolate(crop, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return crop

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        frames = load_cropped_frames(entry.cropped_path)
        if frames is None or entry.frame_index >= frames.shape[0]:
            return None
        frame = frames[entry.frame_index : entry.frame_index + 1]
        tensor = to_tensor(frame)  # [1,C,H,W]
        tensor = resize_tensor(tensor, size=self.image_size)

        global_crops: List[torch.Tensor] = []
        for _ in range(2):
            crop = self._random_resized_crop(tensor, 0.8, 1.0)
            crop = self._augment(crop).squeeze(0)
            global_crops.append(crop)

        local_crops: List[torch.Tensor] = []
        for _ in range(self.local_crops):
            crop = self._random_resized_crop(tensor, 0.4, 0.6)
            crop = self._augment(crop).squeeze(0)
            local_crops.append(crop)

        recon_target = global_crops[0].clone()
        return global_crops, local_crops, recon_target


def _collate(batch: list):
    items = [b for b in batch if b is not None]
    if not items:
        return None

    n_globals = len(items[0][0])
    n_locals = len(items[0][1])

    global_batches = [torch.stack([it[0][i] for it in items], dim=0) for i in range(n_globals)]
    local_batches = [torch.stack([it[1][i] for it in items], dim=0) for i in range(n_locals)]
    recon_targets = torch.stack([it[2] for it in items], dim=0)
    return global_batches, local_batches, recon_targets


class DINOProjector(nn.Module):
    def __init__(self, dim: int, out_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, out_dim),
        )
        self.out_dim = int(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, dim=1)


class DINOv2PretrainModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        pretrained: bool,
        train_encoder_blocks: int,
        freeze_backbone: bool,
    ) -> None:
        super().__init__()
        self.student_encoder = FrameEncoder(
            backbone_name=backbone_name,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            unfreeze_last_blocks=max(0, train_encoder_blocks),
        )
        d = int(self.student_encoder.output_dim)
        self.student_projector = DINOProjector(d, out_dim=256)

        self.teacher_encoder = copy.deepcopy(self.student_encoder)
        self.teacher_projector = copy.deepcopy(self.student_projector)

        self.recon_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, 3 * 56 * 56),
        )
        self._set_teacher_trainable(False)

    def _set_teacher_trainable(self, trainable: bool) -> None:
        for p in self.teacher_encoder.parameters():
            p.requires_grad = bool(trainable)
        for p in self.teacher_projector.parameters():
            p.requires_grad = bool(trainable)

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher_encoder.eval()
        self.teacher_projector.eval()
        return self

    def student_encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.student_encoder(x)

    @torch.no_grad()
    def teacher_encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.teacher_encoder(x)

    def student_project(self, feats: torch.Tensor) -> torch.Tensor:
        return self.student_projector(feats)

    @torch.no_grad()
    def teacher_project(self, feats: torch.Tensor) -> torch.Tensor:
        return self.teacher_projector(feats)

    def reconstruct_from_features(self, feats: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
        recon = self.recon_head(feats).reshape(feats.shape[0], 3, 56, 56)
        recon = F.interpolate(recon, size=out_hw, mode="bilinear", align_corners=False)
        return recon.clamp(0.0, 1.0)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.student_encode(x)
        return self.reconstruct_from_features(feats, (x.shape[-2], x.shape[-1]))


@torch.no_grad()
def _sync_teacher_from_student(model: DINOv2PretrainModel) -> None:
    model.teacher_encoder.load_state_dict(model.student_encoder.state_dict(), strict=False)
    model.teacher_projector.load_state_dict(model.student_projector.state_dict(), strict=False)


@torch.no_grad()
def _ema_update(student: nn.Module, teacher: nn.Module, momentum: float) -> None:
    for p_s, p_t in zip(student.parameters(), teacher.parameters()):
        p_t.data.mul_(momentum).add_(p_s.data, alpha=1.0 - momentum)


@torch.no_grad()
def _update_teacher(model: DINOv2PretrainModel, momentum: float) -> None:
    _ema_update(model.student_encoder, model.teacher_encoder, momentum)
    _ema_update(model.student_projector, model.teacher_projector, momentum)

def _teacher_momentum_at_step(step: int, total_steps: int, start: float, end: float = 0.9995) -> float:
    if total_steps <= 0:
        return float(end)
    step = min(max(int(step), 0), int(total_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * float(step) / float(total_steps)))
    return float(end - (end - start) * cosine)


def _dino_loss(
    student_outs: Sequence[torch.Tensor],
    teacher_outs: Sequence[torch.Tensor],
    center: torch.Tensor,
    student_temp: float,
    teacher_temp: float,
) -> torch.Tensor:
    teacher_probs = [
        F.softmax((t.detach() - center) / float(teacher_temp), dim=-1) for t in teacher_outs
    ]
    student_log_probs = [
        F.log_softmax(s / float(student_temp), dim=-1) for s in student_outs
    ]

    total = 0.0
    terms = 0
    for t_prob in teacher_probs:
        for s_log in student_log_probs:
            total = total + torch.sum(-t_prob * s_log, dim=-1).mean()
            terms += 1
    if terms <= 0:
        raise RuntimeError("DINO loss received zero view pairs.")
    return total / float(terms)


@torch.no_grad()
def _update_center(center: torch.Tensor, teacher_outs: Sequence[torch.Tensor], momentum: float) -> None:
    if not teacher_outs:
        return
    batch_center = torch.cat([t.detach() for t in teacher_outs], dim=0).mean(dim=0, keepdim=True)
    center.mul_(float(momentum)).add_(batch_center, alpha=1.0 - float(momentum))


def _amp_autocast(device_type: str, enabled: bool):
    if not enabled:
        return contextlib.nullcontext()

    amp_mod = getattr(torch, "amp", None)
    if amp_mod is not None and hasattr(amp_mod, "autocast"):
        return amp_mod.autocast(device_type=device_type, enabled=enabled)

    cuda_amp_mod = getattr(torch.cuda, "amp", None)
    if cuda_amp_mod is not None and hasattr(cuda_amp_mod, "autocast"):
        return cuda_amp_mod.autocast(enabled=enabled)

    return contextlib.nullcontext()


def _make_grad_scaler(device_type: str, enabled: bool):
    amp_mod = getattr(torch, "amp", None)
    if amp_mod is not None and hasattr(amp_mod, "GradScaler"):
        try:
            return amp_mod.GradScaler(device=device_type, enabled=enabled)
        except TypeError:
            try:
                return amp_mod.GradScaler(device_type, enabled=enabled)
            except TypeError:
                pass

    cuda_amp_mod = getattr(torch.cuda, "amp", None)
    if cuda_amp_mod is not None and hasattr(cuda_amp_mod, "GradScaler"):
        return cuda_amp_mod.GradScaler(enabled=enabled)

    return None


def _train_or_eval(
    model: DINOv2PretrainModel,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[Any],
    device: torch.device,
    center: torch.Tensor,
    student_temp: float,
    teacher_temp: float,
    center_momentum: float,
    recon_loss_weight: float,
    use_amp: bool,
    global_step: int,
    total_steps: int,
    teacher_momentum_start: float,
    teacher_momentum_final: float,
) -> Tuple[float, float, int]:
    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss = 0.0
    total_recon = 0.0
    n = 0

    amp_enabled = bool(use_amp and device.type == "cuda")
    grad_ctx = torch.enable_grad() if train else torch.no_grad()

    with grad_ctx:
        for batch in tqdm(loader, desc="Train" if train else "Val", leave=False):
            if batch is None:
                continue
            global_crops, local_crops, recon_target = batch
            global_crops = [crop.to(device, non_blocking=True) for crop in global_crops]
            local_crops = [crop.to(device, non_blocking=True) for crop in local_crops]
            recon_target = recon_target.to(device, non_blocking=True)
            student_inputs = list(global_crops) + list(local_crops)

            amp_ctx = _amp_autocast(device_type=device.type, enabled=amp_enabled)
            with amp_ctx:
                student_feats = [model.student_encode(crop) for crop in student_inputs]
                student_outs = [model.student_project(feat) for feat in student_feats]

                with torch.no_grad():
                    teacher_feats = [model.teacher_encode(crop) for crop in global_crops]
                    teacher_outs = [model.teacher_project(feat) for feat in teacher_feats]

                dino_loss = _dino_loss(
                    student_outs=student_outs,
                    teacher_outs=teacher_outs,
                    center=center,
                    student_temp=student_temp,
                    teacher_temp=teacher_temp,
                )
                recon_loss = torch.tensor(0.0, device=device)
                if recon_loss_weight > 0:
                    recon = model.reconstruct_from_features(
                        student_feats[0],
                        out_hw=(recon_target.shape[-2], recon_target.shape[-1]),
                    )
                    recon_loss = F.mse_loss(recon, recon_target)

                loss = dino_loss + float(recon_loss_weight) * recon_loss

            if train:
                optimizer.zero_grad(set_to_none=True)
                if amp_enabled and scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                momentum = _teacher_momentum_at_step(
                    step=global_step,
                    total_steps=total_steps,
                    start=teacher_momentum_start,
                    end=teacher_momentum_final,
                )
                _update_teacher(model, momentum)
                _update_center(center, teacher_outs, center_momentum)
                global_step += 1

            bsz = global_crops[0].size(0)
            total_loss += float(loss.item()) * bsz
            total_recon += float(recon_loss.item()) * bsz
            n += bsz

    return total_loss / max(1, n), total_recon / max(1, n), global_step


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
            global_crops, _, _ = batch
            x = global_crops[0].to(device, non_blocking=True)
            recon = model.reconstruct(x)
            count = min(int(n_samples), x.shape[0])
            if count <= 0:
                return
            orig = x[:count].detach().cpu().permute(0, 2, 3, 1).numpy()
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

def _extract_backbone_state_from_model_state(model_state: Dict[str, Any]) -> Dict[str, Any]:
    backbone_sd: Dict[str, Any] = {}
    for k, v in model_state.items():
        if k.startswith("student_encoder.backbone."):
            backbone_sd[k[len("student_encoder.backbone.") :]] = v
        elif k.startswith("frame_encoder.backbone."):
            backbone_sd[k[len("frame_encoder.backbone.") :]] = v
        elif k.startswith("backbone."):
            backbone_sd[k[len("backbone.") :]] = v
    return backbone_sd


def _load_checkpoint_if_any(model: DINOv2PretrainModel, weights: Optional[Path], center: torch.Tensor) -> None:
    if weights is None or not weights.exists():
        return
    ckpt = torch.load(weights, map_location="cpu")

    if isinstance(ckpt, dict) and "model_state" in ckpt and isinstance(ckpt["model_state"], dict):
        model_state = ckpt["model_state"]
        dino_keys = any(k.startswith("student_encoder.") for k in model_state.keys())
        if dino_keys:
            missing, unexpected = model.load_state_dict(model_state, strict=False)
            logger.info("Loaded DINO checkpoint %s (missing=%d unexpected=%d)", weights, len(missing), len(unexpected))
            if isinstance(ckpt.get("center"), torch.Tensor):
                c = ckpt["center"]
                if c.ndim == 1:
                    c = c.unsqueeze(0)
                if c.shape == center.shape:
                    center.copy_(c)
            return

        backbone_sd = _extract_backbone_state_from_model_state(model_state)
        if backbone_sd:
            model.student_encoder.backbone.load_state_dict(backbone_sd, strict=False)
            _sync_teacher_from_student(model)
            logger.info("Loaded backbone from model_state checkpoint %s", weights)
            return

    if isinstance(ckpt, dict) and "backbone_state" in ckpt and isinstance(ckpt["backbone_state"], dict):
        model.student_encoder.backbone.load_state_dict(ckpt["backbone_state"], strict=False)
        _sync_teacher_from_student(model)
        logger.info("Loaded backbone checkpoint %s", weights)
        return

    if isinstance(ckpt, dict):
        model.student_encoder.backbone.load_state_dict(ckpt, strict=False)
        _sync_teacher_from_student(model)
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


def _load_resume_checkpoint(path: Path, model: DINOv2PretrainModel, center: torch.Tensor) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unsupported checkpoint format for resume: {path}")

    if "model_state" in ckpt and isinstance(ckpt["model_state"], dict):
        model.load_state_dict(ckpt["model_state"], strict=False)
    else:
        raise RuntimeError(f"Checkpoint does not include model_state: {path}")

    if isinstance(ckpt.get("center"), torch.Tensor):
        c = ckpt["center"]
        if c.ndim == 1:
            c = c.unsqueeze(0)
        if c.shape == center.shape:
            center.copy_(c)

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
    parser = argparse.ArgumentParser(description="Frame-level DINOv2 pretraining.")
    parser.add_argument("--cropped-root", type=Path, required=True, help="Root of cropped DICOMs.")
    parser.add_argument("--weights", type=Path, default=None, help="Optional initial checkpoint.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for checkpoints.")
    parser.add_argument("--input-xlsx", type=Path, default=None, help="Optional Excel registry for view filtering.")
    parser.add_argument("--echo-root", type=Path, default=None, help="Root of original DICOMs.")
    parser.add_argument("--views", type=str, default="", help="Comma/space-separated views to include.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Stride between frames sampled from each cine.")
    parser.add_argument("--mask-ratio", type=float, default=0.75, help="Unused (kept for pipeline2 compatibility).")
    parser.add_argument("--train-encoder-blocks", type=int, default=2, help="Number of last encoder blocks to train.")
    parser.add_argument("--freeze-decoder", action="store_true", help="Unused for DINOv2 (kept for compatibility).")
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
    parser.add_argument("--view-head", action="store_true", help="Unused (kept for pipeline2 compatibility).")
    parser.add_argument("--view-loss-weight", type=float, default=0.1, help="Unused (kept for pipeline2 compatibility).")
    parser.add_argument("--aug-brightness", type=float, default=0.1, help="Max brightness delta.")
    parser.add_argument("--aug-contrast", type=float, default=0.1, help="Max contrast delta.")
    parser.add_argument("--aug-rotate-deg", type=float, default=5.0, help="Unused (kept for pipeline2 compatibility).")
    parser.add_argument("--aug-blur", type=float, default=0.0, help="Unused (kept for pipeline2 compatibility).")
    parser.add_argument("--aug-speckle", type=float, default=0.0, help="Speckle noise std.")
    parser.add_argument("--aug-random-erase", type=float, default=0.0, help="Unused (kept for pipeline2 compatibility).")
    parser.add_argument("--recon-samples", type=int, default=9, help="Number of validation samples to save per epoch.")
    parser.add_argument("--recon-seed", type=int, default=123, help="Unused (kept for pipeline2 compatibility).")
    parser.add_argument("--recon-mask-ratio", type=float, default=None, help="Unused (kept for pipeline2 compatibility).")
    parser.add_argument("--recon-loss-weight", type=float, default=0.0, help="Weight for optional reconstruction loss.")
    parser.add_argument("--safe-decode", action="store_true", help="Disable GDCM/pylibjpeg decoders.")
    parser.add_argument("--backbone-name", type=str, default="vit_small_patch14_dinov2.lvd142m")
    parser.add_argument("--resume-latest", action="store_true", help="Resume from the newest checkpoint in output-dir.")
    parser.add_argument("--resume-path", type=Path, default=None, help="Explicit checkpoint path to resume from.")
    parser.add_argument(
        "--continue-session",
        action="store_true",
        help="Resume exactly from <run_name>_latest.pt in output-dir and treat --epochs as additional epochs.",
    )
    parser.add_argument("--teacher-momentum", type=float, default=0.996, help="Initial EMA momentum for teacher update.")
    parser.add_argument("--teacher-momentum-final", type=float, default=0.9995, help="Final EMA momentum at the end of cosine schedule.")
    parser.add_argument("--student-temp", type=float, default=0.1, help="Student temperature.")
    parser.add_argument("--teacher-temp", type=float, default=0.04, help="Teacher temperature.")
    parser.add_argument("--local-crops", type=int, default=1, help="Number of local crops per frame.")
    parser.add_argument("--center-momentum", type=float, default=0.9, help="Center update momentum.")
    parser.add_argument("--use-amp", action="store_true", help="Enable torch.cuda.amp mixed precision training.")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.freeze_decoder:
        logger.info("--freeze-decoder is ignored for DINOv2 pretraining.")
    if args.view_head:
        logger.info("--view-head is ignored for DINOv2 pretraining.")
    if args.view_loss_weight != 0.1:
        logger.info("--view-loss-weight is ignored for DINOv2 pretraining.")

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

    train_ds = DINOMultiCropDataset(
        train_entries,
        image_size=518,
        local_crops=max(0, int(args.local_crops)),
        aug_brightness=args.aug_brightness,
        aug_contrast=args.aug_contrast,
        aug_speckle=args.aug_speckle,
    )
    val_ds = (
        DINOMultiCropDataset(
            val_entries,
            image_size=518,
            local_crops=max(0, int(args.local_crops)),
            aug_brightness=0.0,
            aug_contrast=0.0,
            aug_speckle=0.0,
        )
        if val_entries
        else None
    )

    loader_kwargs: Dict[str, Any] = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": False,
        "collate_fn": _collate,
    }
    if args.num_workers > 0:
        loader_kwargs.update(
            {
                "persistent_workers": True,
                "prefetch_factor": 1,
            }
        )

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = (
        DataLoader(
            val_ds,
            shuffle=False,
            **loader_kwargs,
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
    _sync_teacher_from_student(model)

    center = torch.zeros(1, model.student_projector.out_dim, device=device)

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
        resume_ckpt = _load_resume_checkpoint(resume_path, model, center)
        if not args.run_name:
            args.run_name = _resume_run_name_from_checkpoint(resume_path)
    else:
        _load_checkpoint_if_any(model, args.weights, center)

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters.")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    amp_enabled = bool(args.use_amp and device.type == "cuda")
    scaler = _make_grad_scaler(device_type=device.type, enabled=amp_enabled)

    start_epoch = 1
    best_loss = float("inf")
    history: List[dict] = []
    prev_epoch = 0
    global_step = 0
    if resume_ckpt is not None:
        if "optimizer_state" in resume_ckpt and isinstance(resume_ckpt["optimizer_state"], dict):
            try:
                optimizer.load_state_dict(resume_ckpt["optimizer_state"])
                logger.info("Loaded optimizer state from resume checkpoint.")
            except Exception as exc:
                logger.warning("Failed to load optimizer state from resume checkpoint: %s", exc)
        if "scaler_state" in resume_ckpt and isinstance(resume_ckpt["scaler_state"], dict):
            try:
                scaler.load_state_dict(resume_ckpt["scaler_state"])
                logger.info("Loaded AMP scaler state from resume checkpoint.")
            except Exception as exc:
                logger.warning("Failed to load AMP scaler state from resume checkpoint: %s", exc)
        prev_epoch = int(resume_ckpt.get("epoch", 0))
        start_epoch = max(1, prev_epoch + 1)
        best_loss = float(resume_ckpt.get("best_loss", best_loss))
        global_step = int(resume_ckpt.get("global_step", max(0, (start_epoch - 1) * max(1, len(train_loader)))))
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

    total_steps = max(1, target_epoch * max(1, len(train_loader)))

    for epoch in range(start_epoch, target_epoch + 1):
        train_loss, train_recon, global_step = _train_or_eval(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            center=center,
            student_temp=float(args.student_temp),
            teacher_temp=float(args.teacher_temp),
            center_momentum=float(args.center_momentum),
            recon_loss_weight=float(args.recon_loss_weight),
            use_amp=amp_enabled,
            global_step=global_step,
            total_steps=total_steps,
            teacher_momentum_start=float(args.teacher_momentum),
            teacher_momentum_final=float(args.teacher_momentum_final),
        )

        val_loss, val_recon = (0.0, 0.0)
        if val_loader is not None:
            val_loss, val_recon, _ = _train_or_eval(
                model=model,
                loader=val_loader,
                optimizer=None,
                scaler=None,
                device=device,
                center=center,
                student_temp=float(args.student_temp),
                teacher_temp=float(args.teacher_temp),
                center_momentum=float(args.center_momentum),
                recon_loss_weight=float(args.recon_loss_weight),
                use_amp=amp_enabled,
                global_step=global_step,
                total_steps=total_steps,
                teacher_momentum_start=float(args.teacher_momentum),
                teacher_momentum_final=float(args.teacher_momentum_final),
            )
            if args.recon_loss_weight > 0:
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
            writer.add_scalars(
                "loss/total",
                {"train": train_loss, "val": val_loss if val_loader is not None else train_loss},
                epoch,
            )
            writer.add_scalars(
                "loss/recon",
                {"train": train_recon, "val": val_recon if val_loader is not None else train_recon},
                epoch,
            )

        torch.save(
            {
                "epoch": epoch,
                "best_loss": best_loss,
                "global_step": global_step,
                "center": center.detach().cpu(),
                "model_state": model.state_dict(),
                "backbone_state": model.student_encoder.backbone.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict() if amp_enabled else None,
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
                    "global_step": global_step,
                    "center": center.detach().cpu(),
                    "model_state": model.state_dict(),
                    "backbone_state": model.student_encoder.backbone.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scaler_state": scaler.state_dict() if amp_enabled else None,
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
