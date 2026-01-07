"""
Fine-tune the EchoVisionFM encoder-decoder (MAE reconstruction) on cropped DICOM clips.
Trains only the last N encoder blocks (and optionally the decoder) with no GLS head.

Usage (PowerShell):
  .venv\\Scripts\\python ichilov_train_echovisionfm_last_layer.py ^
    --cropped-root "F:\\OneDrive - Technion\\DS\\Ichilov_cropped" ^
    --weights "F:\\OneDrive - Technion\\models\\Encoder_weights\\pytorch_model.bin" ^
    --output-dir "F:\\OneDrive - Technion\\DS\\Ichilov_EchoVisionFM_pretrain_last_layers"
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import pydicom
except Exception as exc:  # pragma: no cover - runtime check
    raise RuntimeError(
        "pydicom is required. Install with: .venv\\Scripts\\python -m pip install pydicom"
    ) from exc

try:
    from sklearn.model_selection import GroupShuffleSplit
except Exception as exc:  # pragma: no cover - runtime check
    raise RuntimeError(
        "scikit-learn is required. Install with: .venv\\Scripts\\python -m pip install scikit-learn"
    ) from exc

try:
    from transformers import VideoMAEConfig, VideoMAEForPreTraining
except Exception as exc:  # pragma: no cover - runtime check
    raise RuntimeError(
        "transformers is required. Install with: .venv\\Scripts\\python -m pip install transformers"
    ) from exc


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("ichilov_train_echovisionfm_last_layer")

USER_HOME = Path.home()
if USER_HOME.name == "oronbar.RF":
    USER_HOME = Path("F:\\")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _sample_indices(n_frames: int, target: int = 16) -> np.ndarray:
    if n_frames <= 0:
        return np.array([], dtype=int)
    if n_frames == target:
        return np.arange(n_frames, dtype=int)
    idx = np.linspace(0, n_frames - 1, target)
    return np.clip(np.round(idx).astype(int), 0, n_frames - 1)


def _to_tensor(frames: np.ndarray) -> torch.Tensor:
    # frames: T,H,W,C with C in {1,3}
    if frames.ndim != 4:
        raise ValueError(f"Expected 4D frames, got {frames.shape}")
    if frames.shape[-1] == 1:
        frames = np.repeat(frames, 3, axis=-1)
    arr = frames.astype(np.float32)
    max_val = float(arr.max()) if arr.size else 0.0
    if max_val > 1.0:
        if max_val > 255.0:
            arr = arr / max_val * 255.0
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
    return tensor


def _load_cropped_dicom(dicom_path: Path, target_frames: int = 16) -> Optional[torch.Tensor]:
    try:
        ds = pydicom.dcmread(str(dicom_path), force=True)
    except Exception as exc:
        logger.warning(f"Failed to read {dicom_path}: {exc}")
        return None
    try:
        arr = ds.pixel_array
    except Exception as exc:
        logger.warning(f"Failed to decode pixel data {dicom_path}: {exc}")
        return None

    if arr.ndim == 2:
        frames = arr[None, ..., None]
    elif arr.ndim == 3:
        if int(getattr(ds, "SamplesPerPixel", 1)) > 1:
            frames = arr[None, ...]
        else:
            frames = arr[..., None]
    elif arr.ndim == 4:
        frames = arr
    else:
        logger.warning(f"Unexpected pixel array shape {arr.shape} for {dicom_path}")
        return None

    n_frames = frames.shape[0]
    idx = _sample_indices(n_frames, target=target_frames)
    if idx.size == 0:
        return None
    frames = frames[idx]
    return _to_tensor(frames)


class CroppedDicomDataset(Dataset):
    def __init__(self, dicom_paths: Sequence[Path], target_frames: int = 16) -> None:
        self._paths = list(dicom_paths)
        self._target_frames = int(target_frames)

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int) -> Optional[torch.Tensor]:
        dicom_path = self._paths[idx]
        return _load_cropped_dicom(dicom_path, target_frames=self._target_frames)


def _collate_batch(batch: Sequence[Optional[torch.Tensor]]) -> Optional[torch.Tensor]:
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return torch.stack(batch, dim=0)


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


def _load_pretraining_model(weights_path: Path, frames: int) -> VideoMAEForPreTraining:
    config = VideoMAEConfig()
    config.image_size = 224
    config.num_frames = int(frames)
    model = VideoMAEForPreTraining(config)
    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = _normalize_state_dict(checkpoint)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to load weights. Ensure the checkpoint matches VideoMAEForPreTraining "
            "and uses 16 frames."
        ) from exc
    return model


def _set_trainable_layers(model: VideoMAEForPreTraining, train_encoder_blocks: int, train_decoder: bool) -> None:
    for param in model.parameters():
        param.requires_grad = False

    encoder_layers = model.videomae.encoder.layer
    max_blocks = len(encoder_layers)
    if train_encoder_blocks > max_blocks:
        logger.warning(f"Requested {train_encoder_blocks} encoder blocks, clamping to {max_blocks}.")
        train_encoder_blocks = max_blocks
    if train_encoder_blocks > 0:
        for layer in encoder_layers[-train_encoder_blocks:]:
            for param in layer.parameters():
                param.requires_grad = True

    if train_decoder:
        for param in model.encoder_to_decoder.parameters():
            param.requires_grad = True
        for param in model.decoder.parameters():
            param.requires_grad = True
        model.mask_token.requires_grad = True


def _list_dicom_paths(
    cropped_root: Path,
    data_ratio: float,
    max_samples: int,
    seed: int,
) -> List[Path]:
    paths = sorted(cropped_root.rglob("*.dcm"))
    if not paths:
        raise RuntimeError(f"No DICOMs found under {cropped_root}")
    rng = random.Random(seed)
    if 0 < data_ratio < 1.0:
        keep = max(1, int(len(paths) * data_ratio))
        paths = rng.sample(paths, keep)
    if max_samples > 0 and len(paths) > max_samples:
        paths = rng.sample(paths, max_samples)
    return sorted(paths)


def _patient_key(path: Path, cropped_root: Path) -> str:
    try:
        rel = path.resolve().relative_to(cropped_root.resolve())
        if rel.parts:
            return rel.parts[0]
    except Exception:
        pass
    return path.parent.name


def _split_paths(
    paths: Sequence[Path],
    cropped_root: Path,
    val_ratio: float,
    seed: int,
) -> Tuple[List[Path], List[Path]]:
    if val_ratio <= 0.0 or len(paths) < 2:
        return list(paths), []
    groups = [_patient_key(p, cropped_root) for p in paths]
    if len(set(groups)) < 2:
        return list(paths), []
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(splitter.split(paths, groups=groups))
    train = [paths[i] for i in train_idx]
    val = [paths[i] for i in val_idx]
    return train, val


def _seq_length(config: VideoMAEConfig, frames: int) -> int:
    if frames % config.tubelet_size != 0:
        raise ValueError("frames must be divisible by tubelet_size.")
    num_patches_per_frame = (config.image_size // config.patch_size) ** 2
    return (frames // config.tubelet_size) * num_patches_per_frame


def _build_mask(batch_size: int, seq_length: int, mask_ratio: float, device: torch.device) -> torch.Tensor:
    num_mask = int(round(seq_length * mask_ratio))
    num_mask = max(1, min(seq_length - 1, num_mask))
    rand = torch.rand((batch_size, seq_length), device=device)
    ids = rand.argsort(dim=1)
    mask = torch.zeros((batch_size, seq_length), dtype=torch.bool, device=device)
    mask.scatter_(1, ids[:, :num_mask], True)
    return mask


def _train_one_epoch(
    model: VideoMAEForPreTraining,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    seq_length: int,
    mask_ratio: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        if batch is None:
            continue
        pixel_values = batch.to(device, non_blocking=True)
        batch_size = pixel_values.shape[0]
        bool_masked_pos = _build_mask(batch_size, seq_length, mask_ratio, device)

        outputs = model(pixel_values=pixel_values, bool_masked_pos=bool_masked_pos)
        loss = outputs.loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


def _evaluate(
    model: VideoMAEForPreTraining,
    loader: DataLoader,
    device: torch.device,
    seq_length: int,
    mask_ratio: float,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Val", leave=False):
            if batch is None:
                continue
            pixel_values = batch.to(device, non_blocking=True)
            batch_size = pixel_values.shape[0]
            bool_masked_pos = _build_mask(batch_size, seq_length, mask_ratio, device)

            outputs = model(pixel_values=pixel_values, bool_masked_pos=bool_masked_pos)
            loss = outputs.loss

            total_loss += float(loss.item()) * batch_size
            total_count += batch_size

    return total_loss / max(total_count, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune EchoVisionFM MAE on cropped DICOMs.")
    parser.add_argument(
        "--cropped-root",
        type=Path,
        default=USER_HOME / "OneDrive - Technion" / "DS" / "Ichilov_cropped",
        help="Root of cropped DICOM tree.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=USER_HOME / "OneDrive - Technion" / "models" / "Encoder_weights" / "pytorch_model.bin",
        help="Path to EchoVisionFM pretraining weights.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=USER_HOME / "OneDrive - Technion" / "DS" / "Ichilov_EchoVisionFM_pretrain_last_layers",
        help="Output directory for checkpoints and metrics.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=16,
        help="Number of frames per clip (must match pretraining weights).",
    )
    parser.add_argument(
        "--mask-ratio",
        type=float,
        default=0.9,
        help="Fraction of tokens to mask for MAE reconstruction.",
    )
    parser.add_argument(
        "--train-encoder-blocks",
        type=int,
        default=2,
        help="Number of last encoder blocks to train.",
    )
    parser.add_argument(
        "--freeze-decoder",
        action="store_true",
        help="Freeze the decoder and encoder-to-decoder projection.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.05,
        help="Weight decay.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio (by patient folder).",
    )
    parser.add_argument(
        "--data-ratio",
        type=float,
        default=1.0,
        help="Fraction of files to use from the cropped root.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional hard limit for number of DICOMs (0 disables).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run folder name (defaults to timestamp).",
    )
    args = parser.parse_args()

    _set_seed(args.seed)

    if not args.cropped_root.exists():
        raise FileNotFoundError(f"Cropped root not found: {args.cropped_root}")
    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    if not (0.0 < args.mask_ratio <= 1.0):
        raise ValueError("--mask-ratio must be in (0, 1].")
    if not (0.0 < args.data_ratio <= 1.0):
        raise ValueError("--data-ratio must be in (0, 1].")

    run_name = args.run_name or datetime.now().strftime("echovisionfm_mae_%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Listing cropped DICOMs...")
    paths = _list_dicom_paths(args.cropped_root, args.data_ratio, args.max_samples, args.seed)
    train_paths, val_paths = _split_paths(paths, args.cropped_root, args.val_ratio, args.seed)
    logger.info(f"Train DICOMs: {len(train_paths)}, Val DICOMs: {len(val_paths)}")

    train_ds = CroppedDicomDataset(train_paths, target_frames=args.frames)
    val_ds = CroppedDicomDataset(val_paths, target_frames=args.frames) if val_paths else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate_batch,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=_collate_batch,
        )
        if val_ds is not None
        else None
    )

    device = torch.device(args.device)
    logger.info(f"Loading VideoMAE on {device}")
    model = _load_pretraining_model(args.weights, frames=args.frames)
    model.to(device)

    _set_trainable_layers(
        model,
        train_encoder_blocks=max(0, args.train_encoder_blocks),
        train_decoder=not bool(args.freeze_decoder),
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters. Check --train-encoder-blocks/--train-decoder.")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    seq_length = _seq_length(model.config, args.frames)
    history: List[dict] = []
    best_loss = float("inf")
    best_path = run_dir / "best_mae.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = _train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            seq_length,
            args.mask_ratio,
        )
        if val_loader is not None:
            val_loss = _evaluate(model, val_loader, device, seq_length, args.mask_ratio)
        else:
            val_loss = float("nan")

        logger.info(
            f"Epoch {epoch}/{args.epochs} - train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )

        metric = val_loss if val_loader is not None else train_loss
        if metric < best_loss:
            best_loss = metric
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_loss": best_loss,
                    "epoch": epoch,
                },
                best_path,
            )

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    logger.info(f"Saved metrics to {run_dir / 'metrics.json'}")
    logger.info(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
