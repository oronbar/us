"""
Plot random frame reconstructions using a trained USF-MAE checkpoint.

Example:
  python plot_usf_mae_reconstructions.py \
    --weights "C:\\Users\\Oron\\OneDrive - Technion\\Experiments\\NewPipelineRun_8\\frame_pretrain\\frame_mae_NewPipelineRun_8_best.pt" \
    --cropped-root "D:\\DS\\Ichilov_cropped\\NewPipelineRun_4" \
    --output "C:\\Users\\Oron\\OneDrive - Technion\\Experiments\\NewPipelineRun_8\\frame_recon_grid.png"
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    import pydicom
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "pydicom is required. Install with: .venv\\Scripts\\python -m pip install pydicom"
    ) from exc

from usf_mae_model import mae_vit_base_patch16_dec512d8b


@dataclass
class FrameRef:
    dicom_path: Path
    frame_index: int


def _normalize_state_dict(checkpoint: object) -> dict:
    if isinstance(checkpoint, dict):
        for key in ("model_state", "model", "state_dict", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Unsupported checkpoint format.")
    state_dict = checkpoint
    for prefix in ("module.", "model."):
        if state_dict and all(k.startswith(prefix) for k in state_dict.keys()):
            state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def _load_model(weights_path: Path, device: torch.device) -> torch.nn.Module:
    model = mae_vit_base_patch16_dec512d8b()
    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = _normalize_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def _load_frames(dicom_path: Path) -> Optional[np.ndarray]:
    try:
        ds = pydicom.dcmread(str(dicom_path), force=True)
        arr = ds.pixel_array
    except Exception:
        return None

    if arr.ndim == 2:
        return arr[None, ..., None]
    if arr.ndim == 3:
        if int(getattr(ds, "SamplesPerPixel", 1)) > 1:
            return arr[None, ...]
        return arr[..., None]
    if arr.ndim == 4:
        return arr
    return None


def _frame_to_tensor(frame: np.ndarray) -> torch.Tensor:
    # frame shape: H,W,C where C in {1,3}
    if frame.ndim != 3:
        raise ValueError(f"Expected 3D frame, got {frame.shape}")
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    arr = frame.astype(np.float32)
    max_val = float(arr.max()) if arr.size else 0.0
    if max_val > 1.0:
        if max_val > 255.0:
            arr = arr / max_val * 255.0
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
    tensor = torch.nn.functional.interpolate(
        tensor,
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    )
    return tensor


def _collect_random_frame_refs(cropped_root: Path, n: int, seed: int) -> List[FrameRef]:
    rng = random.Random(seed)
    dicoms = list(cropped_root.rglob("*.dcm"))
    if not dicoms:
        return []

    refs: List[FrameRef] = []
    tries = 0
    max_tries = max(2000, n * 200)

    while len(refs) < n and tries < max_tries:
        tries += 1
        dcm = rng.choice(dicoms)
        frames = _load_frames(dcm)
        if frames is None or frames.shape[0] == 0:
            continue
        frame_idx = rng.randrange(0, int(frames.shape[0]))
        refs.append(FrameRef(dicom_path=dcm, frame_index=frame_idx))

    return refs


def _reconstruct_frame(model: torch.nn.Module, frame_tensor: torch.Tensor, mask_ratio: float) -> torch.Tensor:
    # frame_tensor: [1,3,224,224]
    with torch.no_grad():
        _, pred, _ = model(frame_tensor, mask_ratio=mask_ratio)
        recon = model.unpatchify(pred)
    return recon.clamp(0.0, 1.0)


def _tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    # tensor: [1,3,H,W]
    arr = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    return np.clip(arr, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot random frames and USF-MAE reconstructions.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path(r"C:\Users\Oron\OneDrive - Technion\Experiments\NewPipelineRun_8\frame_pretrain\frame_mae_NewPipelineRun_8_best.pt"),
        help="Path to trained model checkpoint.",
    )
    parser.add_argument(
        "--cropped-root",
        type=Path,
        default=Path(r"D:\DS\Ichilov_cropped\NewPipelineRun_4"),
        help="Root of cropped DICOMs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(r"C:\Users\Oron\OneDrive - Technion\Experiments\NewPipelineRun_8\frame_recon_grid.png"),
        help="Output PNG path.",
    )
    parser.add_argument("--num-samples", type=int, default=9, help="Number of random frames to plot.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--mask-ratio", type=float, default=0.75, help="MAE masking ratio for reconstruction.")
    parser.add_argument("--device", type=str, default="auto", help="auto|cuda|cpu")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if not args.weights.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.weights}")
    if not args.cropped_root.exists():
        raise FileNotFoundError(f"Cropped root not found: {args.cropped_root}")

    model = _load_model(args.weights, device)

    refs = _collect_random_frame_refs(args.cropped_root, args.num_samples, args.seed)
    if len(refs) < args.num_samples:
        raise RuntimeError(
            f"Could only sample {len(refs)} frames from {args.cropped_root}; requested {args.num_samples}."
        )

    pairs: List[Tuple[np.ndarray, np.ndarray, str]] = []
    for ref in refs:
        frames = _load_frames(ref.dicom_path)
        if frames is None or ref.frame_index >= frames.shape[0]:
            continue
        frame = frames[ref.frame_index]
        frame_tensor = _frame_to_tensor(frame).to(device)
        recon_tensor = _reconstruct_frame(model, frame_tensor, mask_ratio=args.mask_ratio)
        orig_img = _tensor_to_image(frame_tensor)
        recon_img = _tensor_to_image(recon_tensor)
        title = f"{ref.dicom_path.name} | f={ref.frame_index}"
        pairs.append((orig_img, recon_img, title))

    if len(pairs) < args.num_samples:
        raise RuntimeError(
            f"Only reconstructed {len(pairs)} frames; expected {args.num_samples}."
        )

    rows = int(np.ceil(args.num_samples / 3))
    cols = 6
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.6))
    axes = np.array(axes).reshape(rows, cols)

    for i in range(rows * 3):
        r = i // 3
        c0 = (i % 3) * 2
        ax_orig = axes[r, c0]
        ax_recon = axes[r, c0 + 1]
        if i < len(pairs):
            orig, recon, title = pairs[i]
            ax_orig.imshow(orig)
            ax_orig.set_title("Original", fontsize=9)
            ax_orig.set_xlabel(title, fontsize=7)
            ax_recon.imshow(recon)
            ax_recon.set_title("Reconstruction", fontsize=9)
        ax_orig.axis("off")
        ax_recon.axis("off")

    fig.suptitle("USF-MAE: Random Frame Reconstructions", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(f"Saved reconstruction grid to: {args.output}")


if __name__ == "__main__":
    main()
