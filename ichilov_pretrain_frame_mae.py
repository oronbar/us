"""
Frame-level MAE pretraining using USF-MAE (ViT-MAE, 2D frames).
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
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
except Exception as exc:  # pragma: no cover - runtime check
    raise RuntimeError(
        "scikit-learn is required. Install with: .venv\\Scripts\\python -m pip install scikit-learn"
    ) from exc

from usf_mae_model import MaskedAutoencoderViT, mae_vit_base_patch16_dec512d8b

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("ichilov_pretrain_frame_mae")


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


def _normalize_state_dict(checkpoint: object) -> dict:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Unsupported checkpoint format.")

    state_dict = checkpoint
    for prefix in ("module.", "model."):
        if state_dict and all(k.startswith(prefix) for k in state_dict.keys()):
            state_dict = {k[len(prefix) :]: v for k, v in state_dict.items()}
    return state_dict


def _load_pretraining_model(weights_path: Optional[Path]) -> MaskedAutoencoderViT:
    model = mae_vit_base_patch16_dec512d8b()
    if weights_path is None:
        return model
    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = _normalize_state_dict(checkpoint)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to load USF-MAE weights. Ensure the checkpoint matches mae_vit_base_patch16_dec512d8b."
        ) from exc
    return model


def _set_trainable_layers(model: MaskedAutoencoderViT, train_encoder_blocks: int, train_decoder: bool) -> None:
    for param in model.parameters():
        param.requires_grad = False

    encoder_layers = model.blocks
    max_blocks = len(encoder_layers)
    if train_encoder_blocks > max_blocks:
        logger.warning("Requested %d encoder blocks, clamping to %d.", train_encoder_blocks, max_blocks)
        train_encoder_blocks = max_blocks
    if train_encoder_blocks > 0:
        for layer in encoder_layers[-train_encoder_blocks:]:
            for param in layer.parameters():
                param.requires_grad = True

    if train_decoder:
        for param in model.decoder_embed.parameters():
            param.requires_grad = True
        for param in model.decoder_blocks.parameters():
            param.requires_grad = True
        for param in model.decoder_norm.parameters():
            param.requires_grad = True
        for param in model.decoder_pred.parameters():
            param.requires_grad = True
        model.mask_token.requires_grad = True


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


class FrameDataset(Dataset):
    def __init__(
        self,
        entries: Sequence[FrameEntry],
        image_size: int = 224,
        aug_brightness: float = 0.0,
        aug_contrast: float = 0.0,
        aug_rotate_deg: float = 0.0,
        aug_blur: float = 0.0,
        aug_speckle: float = 0.0,
        aug_random_erase: float = 0.0,
    ) -> None:
        self.entries = list(entries)
        self.image_size = int(image_size)
        self.aug_brightness = float(aug_brightness)
        self.aug_contrast = float(aug_contrast)
        self.aug_rotate_deg = float(aug_rotate_deg)
        self.aug_blur = float(aug_blur)
        self.aug_speckle = float(aug_speckle)
        self.aug_random_erase = float(aug_random_erase)

        self._tv = None
        try:
            import torchvision.transforms.functional as TF
            from torchvision.transforms import GaussianBlur, RandomErasing
            self._tv = (TF, GaussianBlur, RandomErasing)
        except Exception:
            if any(x > 0 for x in [self.aug_rotate_deg, self.aug_blur, self.aug_random_erase]):
                logger.warning("torchvision not available; rotation/blur/erase augmentations disabled.")

    def __len__(self) -> int:
        return len(self.entries)

    def _apply_basic_aug(self, tensor: torch.Tensor) -> torch.Tensor:
        # tensor shape: [1,C,H,W]
        out = tensor
        if self.aug_contrast > 0:
            contrast = 1.0 + random.uniform(-self.aug_contrast, self.aug_contrast)
            mean = out.mean(dim=(2, 3), keepdim=True)
            out = (out - mean) * contrast + mean
        if self.aug_brightness > 0:
            brightness = 1.0 + random.uniform(-self.aug_brightness, self.aug_brightness)
            out = out * brightness
        if self.aug_speckle > 0:
            noise = torch.randn_like(out) * self.aug_speckle
            out = out + noise
        return out.clamp(0.0, 1.0)

    def _apply_tv_aug(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._tv is None:
            return tensor
        TF, GaussianBlur, RandomErasing = self._tv
        out = tensor
        if self.aug_rotate_deg > 0:
            angle = random.uniform(-self.aug_rotate_deg, self.aug_rotate_deg)
            out = TF.rotate(out, angle=angle)
        if self.aug_blur > 0:
            blur = GaussianBlur(kernel_size=3, sigma=self.aug_blur)
            out = blur(out)
        if self.aug_random_erase > 0:
            eraser = RandomErasing(p=0.5, scale=(0.02, min(0.4, self.aug_random_erase)))
            out = eraser(out)
        return out

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        frames = load_cropped_frames(entry.cropped_path)
        if frames is None:
            return None
        if entry.frame_index >= frames.shape[0]:
            return None
        frame = frames[entry.frame_index : entry.frame_index + 1]
        tensor = to_tensor(frame)  # [1,C,H,W]
        tensor = resize_tensor(tensor, size=self.image_size)
        tensor = self._apply_basic_aug(tensor)
        tensor = self._apply_tv_aug(tensor)
        tensor = tensor.squeeze(0)
        return tensor, entry.view


def _collate(batch: list):
    items = [b for b in batch if b is not None]
    if not items:
        return None
    tensors, views = zip(*items)
    view_idx = []
    for v in views:
        if v in VIEW_KEYS:
            view_idx.append(VIEW_KEYS.index(v))
        else:
            view_idx.append(-1)
    return torch.stack(tensors, dim=0), torch.tensor(view_idx, dtype=torch.long)


def _train_one_epoch(
    model: MaskedAutoencoderViT,
    view_head: Optional[torch.nn.Module],
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mask_ratio: float,
    view_loss_weight: float,
) -> Tuple[float, float]:
    model.train()
    if view_head is not None:
        view_head.train()
    total_loss = 0.0
    total_view = 0.0
    total_count = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        if batch is None:
            continue
        pixel_values, view_idx = batch
        pixel_values = pixel_values.to(device, non_blocking=True)
        view_idx = view_idx.to(device, non_blocking=True)
        loss, _, _ = model(pixel_values, mask_ratio=mask_ratio)
        view_loss = torch.tensor(0.0, device=device)

        if view_head is not None:
            with torch.no_grad():
                enc, _, _ = model.forward_encoder(pixel_values, mask_ratio=0.0)
            cls_tok = enc[:, 0, :]
            logits = view_head(cls_tok)
            mask = view_idx >= 0
            if mask.any():
                view_loss = F.cross_entropy(logits[mask], view_idx[mask])
                loss = loss + view_loss_weight * view_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * batch_size
        total_view += float(view_loss.item()) * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1), total_view / max(total_count, 1)


def _evaluate(
    model: MaskedAutoencoderViT,
    view_head: Optional[torch.nn.Module],
    loader: DataLoader,
    device: torch.device,
    mask_ratio: float,
    view_loss_weight: float,
) -> Tuple[float, float]:
    model.eval()
    if view_head is not None:
        view_head.eval()
    total_loss = 0.0
    total_view = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Val", leave=False):
            if batch is None:
                continue
            pixel_values, view_idx = batch
            pixel_values = pixel_values.to(device, non_blocking=True)
            view_idx = view_idx.to(device, non_blocking=True)
            loss, _, _ = model(pixel_values, mask_ratio=mask_ratio)
            view_loss = torch.tensor(0.0, device=device)

            if view_head is not None:
                enc, _, _ = model.forward_encoder(pixel_values, mask_ratio=0.0)
                cls_tok = enc[:, 0, :]
                logits = view_head(cls_tok)
                mask = view_idx >= 0
                if mask.any():
                    view_loss = F.cross_entropy(logits[mask], view_idx[mask])
                    loss = loss + view_loss_weight * view_loss

            total_loss += float(loss.item()) * batch_size
            total_view += float(view_loss.item()) * batch_size
            total_count += batch_size

    return total_loss / max(total_count, 1), total_view / max(total_count, 1)

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
        dicoms = list(cropped_root.rglob("*.dcm"))
        for path in dicoms:
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

    if data_ratio < 1.0:
        rng = np.random.default_rng(42)
        keep = int(round(len(entries) * data_ratio))
        entries = list(rng.choice(entries, size=max(1, keep), replace=False))

    if max_samples > 0 and len(entries) > max_samples:
        rng = np.random.default_rng(42)
        entries = list(rng.choice(entries, size=max_samples, replace=False))

    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Frame-level MAE pretraining (USF-MAE ViT backbone).")
    parser.add_argument("--cropped-root", type=Path, required=True, help="Root of cropped DICOMs.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path.home()
        / "OneDrive - Technion"
        / "models"
        / "Encoder_weights"
        / "USF-MAE_full_pretrain_43dataset_100epochs.pt",
        help="Optional initial weights checkpoint (USF-MAE).",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for checkpoints.")
    parser.add_argument("--input-xlsx", type=Path, default=None, help="Optional Excel registry for view filtering.")
    parser.add_argument("--echo-root", type=Path, default=None, help="Root of original DICOMs.")
    parser.add_argument("--views", type=str, default="", help="Comma/space-separated views to include.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Stride between frames sampled from each cine.")
    parser.add_argument("--mask-ratio", type=float, default=0.75, help="Fraction of tokens to mask.")
    parser.add_argument("--train-encoder-blocks", type=int, default=2, help="Number of last encoder blocks to train.")
    parser.add_argument("--freeze-decoder", action="store_true", help="Freeze decoder (train encoder only).")
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
    parser.add_argument("--aug-rotate-deg", type=float, default=5.0, help="Max rotation degrees.")
    parser.add_argument("--aug-blur", type=float, default=0.0, help="Gaussian blur sigma.")
    parser.add_argument("--aug-speckle", type=float, default=0.0, help="Speckle noise std.")
    parser.add_argument("--aug-random-erase", type=float, default=0.0, help="Random erase max area fraction.")
    parser.add_argument("--safe-decode", action="store_true", help="Disable GDCM/pylibjpeg decoders.")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

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

    train_ds = FrameDataset(
        train_entries,
        aug_brightness=args.aug_brightness,
        aug_contrast=args.aug_contrast,
        aug_rotate_deg=args.aug_rotate_deg,
        aug_blur=args.aug_blur,
        aug_speckle=args.aug_speckle,
        aug_random_erase=args.aug_random_erase,
    )
    val_ds = FrameDataset(
        val_entries,
        aug_brightness=0.0,
        aug_contrast=0.0,
        aug_rotate_deg=0.0,
        aug_blur=0.0,
        aug_speckle=0.0,
        aug_random_erase=0.0,
    ) if val_entries else None

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

    logger.info("Loading USF-MAE on %s", device)
    model = _load_pretraining_model(args.weights)
    model.to(device)

    _set_trainable_layers(
        model,
        train_encoder_blocks=max(0, args.train_encoder_blocks),
        train_decoder=not bool(args.freeze_decoder),
    )

    view_head = None
    if args.view_head:
        view_head = torch.nn.Linear(model.embed_dim, len(VIEW_KEYS)).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    if view_head is not None:
        trainable += [p for p in view_head.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters.")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    history: List[dict] = []
    best_loss = float("inf")
    run_name = args.run_name or f"frame_mae_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / f"{run_name}_best.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_view = _train_one_epoch(
            model,
            view_head,
            train_loader,
            optimizer,
            device,
            args.mask_ratio,
            args.view_loss_weight,
        )
        val_loss, val_view = (0.0, 0.0)
        if val_loader is not None:
            val_loss, val_view = _evaluate(
                model,
                view_head,
                val_loader,
                device,
                args.mask_ratio,
                args.view_loss_weight,
            )
        metric = val_loss if val_loader is not None else train_loss
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss if val_loader is not None else None,
                "train_view_loss": train_view,
                "val_view_loss": val_view if val_loader is not None else None,
            }
        )
        logger.info(
            "Epoch %d/%d | train=%.4f val=%.4f",
            epoch,
            args.epochs,
            train_loss,
            val_loss if val_loader is not None else train_loss,
        )
        if metric < best_loss:
            best_loss = metric
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "view_head_state": view_head.state_dict() if view_head is not None else None,
                    "config": {
                        "mask_ratio": args.mask_ratio,
                        "view_head": args.view_head,
                    },
                },
                best_path,
            )

    (args.output_dir / f"{run_name}_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    (args.output_dir / f"{run_name}_config.json").write_text(
        json.dumps(vars(args), indent=2, default=str), encoding="utf-8"
    )
    logger.info("Best checkpoint: %s", best_path)


if __name__ == "__main__":
    main()
