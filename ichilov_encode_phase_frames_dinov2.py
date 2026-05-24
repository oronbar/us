"""
Encode phase-aligned DICOM frames with a frozen DINOv2 frame encoder.

Example:
  python ichilov_encode_phase_frames_dinov2.py ^
    --input-parquet "C:\\path\\prepared_dataset.parquet" ^
    --output-parquet "C:\\path\\phase_embeddings.parquet"
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from Ichilov_pipeline3.models.frame_encoder import FrameEncoder
from ichilov_pipeline2_utils import (
    configure_pydicom_handlers,
    load_cropped_frames,
    resize_tensor,
    sample_indices,
    to_tensor,
)
from ichilov_strain_curve_utils import clean_string, dataframe_preview_for_csv, is_missing, to_int, write_json


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("ichilov_encode_phase_frames_dinov2")


def _load_encoder(
    weights_path: Optional[Path],
    backbone_name: str,
    image_size: int,
    freeze_backbone: bool,
) -> FrameEncoder:
    model = FrameEncoder(
        backbone_name=backbone_name,
        pretrained=True,
        freeze_backbone=freeze_backbone,
        unfreeze_last_blocks=0,
        input_size=image_size,
    )
    if weights_path is None or not weights_path.exists():
        model.eval()
        return model

    checkpoint = torch.load(weights_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "backbone_state" in checkpoint and isinstance(checkpoint["backbone_state"], dict):
        model.backbone.load_state_dict(checkpoint["backbone_state"], strict=False)
    elif isinstance(checkpoint, dict) and "model_state" in checkpoint and isinstance(checkpoint["model_state"], dict):
        sd = checkpoint["model_state"]
        backbone_sd = {}
        for k, v in sd.items():
            if k.startswith("student_encoder.backbone."):
                backbone_sd[k[len("student_encoder.backbone.") :]] = v
            elif k.startswith("frame_encoder.backbone."):
                backbone_sd[k[len("frame_encoder.backbone.") :]] = v
            elif k.startswith("backbone."):
                backbone_sd[k[len("backbone.") :]] = v
        model.backbone.load_state_dict(backbone_sd or sd, strict=False)
    elif isinstance(checkpoint, dict):
        model.backbone.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model


def _ed_es_only_indices(n_frames: int, t_frames: int, ed: Optional[int], es: Optional[int]) -> np.ndarray:
    if n_frames <= 0 or t_frames <= 0:
        return np.asarray([], dtype=int)
    ed_i = 0 if ed is None else int(np.clip(ed, 0, n_frames - 1))
    es_i = ed_i if es is None else int(np.clip(es, 0, n_frames - 1))
    anchors = np.asarray([ed_i, es_i], dtype=int)
    if t_frames == 1:
        return anchors[:1]
    if t_frames == 2:
        return anchors
    pos = np.linspace(0, len(anchors) - 1, t_frames)
    idx = np.clip(np.round(pos).astype(int), 0, len(anchors) - 1)
    return anchors[idx]


def _arc_indices_zero_based(start: int, end: int, n_frames: int) -> List[int]:
    out = [int(start)]
    cur = int(start)
    for _ in range(max(1, n_frames * 2)):
        if cur == int(end):
            break
        cur = (cur + 1) % n_frames
        out.append(cur)
        if cur == int(end):
            break
    return out


def _sample_path(path: List[int], n: int) -> np.ndarray:
    if n <= 0 or not path:
        return np.asarray([], dtype=int)
    if len(path) == 1:
        return np.full((n,), int(path[0]), dtype=int)
    pos = np.linspace(0, len(path) - 1, n)
    idx = np.clip(np.round(pos).astype(int), 0, len(path) - 1)
    return np.asarray([int(path[i]) for i in idx], dtype=int)


def _phase_indices_zero_based(
    n_frames: int,
    t_frames: int,
    ed: Optional[int],
    es: Optional[int],
    strategy: str,
) -> np.ndarray:
    if n_frames <= 0 or t_frames <= 0:
        return np.asarray([], dtype=int)
    if ed is None or es is None:
        return sample_indices(n_frames, target=t_frames)
    ed_i = int(np.clip(ed, 0, n_frames - 1))
    es_i = int(np.clip(es, 0, n_frames - 1))
    if ed_i == es_i:
        return sample_indices(n_frames, target=t_frames)
    if strategy == "segment":
        return _sample_path(_arc_indices_zero_based(ed_i, es_i, n_frames), t_frames)
    if t_frames == 1:
        return np.asarray([ed_i], dtype=int)
    if t_frames == 2:
        return np.asarray([ed_i, es_i], dtype=int)
    arc1 = _arc_indices_zero_based(ed_i, es_i, n_frames)
    arc2 = _arc_indices_zero_based(es_i, ed_i, n_frames)
    n1 = max(2, int(np.ceil(float(t_frames) / 2.0)))
    n2 = max(2, t_frames - n1 + 1)
    seg1 = _sample_path(arc1, n1)
    seg2 = _sample_path(arc2, n2)
    return np.concatenate([seg1, seg2[1:]])[:t_frames].astype(int)


def _selected_indices(row: pd.Series, n_frames: int, t_frames: int, mode: str) -> np.ndarray:
    ed = to_int(row.get("ed_index"))
    es = to_int(row.get("es_index"))
    mode = mode.strip().lower()
    if mode == "uniform" or ed is None or es is None:
        return sample_indices(n_frames, target=t_frames)
    if mode == "ed_es_cycle":
        return _phase_indices_zero_based(n_frames, t_frames, ed, es, strategy="cycle")
    if mode == "ed_es_segment":
        return _phase_indices_zero_based(n_frames, t_frames, ed, es, strategy="segment")
    if mode == "ed_es_only":
        return _ed_es_only_indices(n_frames, t_frames, ed, es)
    raise ValueError(f"Unknown phase_sampling_mode: {mode}")


def _encode_frames(
    model: FrameEncoder,
    tensor: torch.Tensor,
    device: torch.device,
    batch_size: int,
    use_amp: bool,
) -> np.ndarray:
    outputs: List[np.ndarray] = []
    amp_enabled = bool(use_amp and device.type == "cuda")
    for start in range(0, tensor.shape[0], batch_size):
        batch = tensor[start : start + batch_size].to(device)
        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                emb = model(batch)
        outputs.append(emb.detach().float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _save_frame_grid(
    frames: np.ndarray,
    indices: np.ndarray,
    ed: Optional[int],
    es: Optional[int],
    out_path: Path,
    max_cols: int = 8,
) -> None:
    if frames.size == 0:
        return
    n = len(indices)
    cols = min(max_cols, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.1 * cols, 2.2 * rows))
    axes_arr = np.asarray(axes).reshape(-1)
    for ax in axes_arr:
        ax.axis("off")
    for i, idx in enumerate(indices):
        ax = axes_arr[i]
        img = frames[int(idx)]
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img[..., 0]
        ax.imshow(img, cmap="gray")
        tags = []
        if ed is not None and int(idx) == int(ed):
            tags.append("ED")
        if es is not None and int(idx) == int(es):
            tags.append("ES")
        title = f"{int(idx)}" + (f" ({'/'.join(tags)})" if tags else "")
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _save_indices_plot(indices: np.ndarray, n_frames: int, ed: Optional[int], es: Optional[int], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 2.4))
    ax.plot(np.arange(len(indices)), indices, marker="o", linewidth=1.5)
    if ed is not None:
        ax.axhline(ed, color="#1f77b4", linestyle="--", linewidth=1, label="ED")
    if es is not None:
        ax.axhline(es, color="#d62728", linestyle="--", linewidth=1, label="ES")
    ax.set_xlabel("sample position")
    ax.set_ylabel("frame index")
    ax.set_ylim(-1, max(n_frames, 1))
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _input_path(row: pd.Series) -> Optional[Path]:
    for col in ("cropped_path", "dicom_path"):
        value = clean_string(row.get(col))
        if value and Path(value).is_file():
            return Path(value)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode phase-aligned frames with frozen DINOv2.")
    parser.add_argument("--input-parquet", type=Path, required=True)
    parser.add_argument("--output-parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--echo-root", type=Path, default=None)
    parser.add_argument("--cropped-root", type=Path, default=None)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--backbone-name", type=str, default="vit_small_patch14_dinov2.lvd142m")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--t-frames", type=int, default=16)
    parser.add_argument("--phase-sampling-mode", type=str, default="ed_es_cycle", choices=["ed_es_cycle", "ed_es_segment", "uniform", "ed_es_only"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0, help="Reserved for future DataLoader implementation.")
    parser.add_argument("--use-amp", dest="use_amp", action="store_true", default=True)
    parser.add_argument("--no-use-amp", dest="use_amp", action="store_false")
    parser.add_argument("--freeze-backbone", dest="freeze_backbone", action="store_true", default=True)
    parser.add_argument("--no-freeze-backbone", dest="freeze_backbone", action="store_false")
    parser.add_argument("--safe-decode", action="store_true")
    parser.add_argument("--diagnostic-samples", type=int, default=12)
    args = parser.parse_args()

    configure_pydicom_handlers(args.safe_decode)
    output_dir = args.output_dir or args.output_parquet.parent
    diagnostics_dir = output_dir / "phase_frame_diagnostics"
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input_parquet)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading DINOv2 encoder on %s", device)
    model = _load_encoder(args.weights, args.backbone_name, args.image_size, args.freeze_backbone).to(device)
    model.eval()

    rows: List[Dict[str, object]] = []
    skipped = {"missing_input_path": 0, "decode_failed": 0, "empty_indices": 0, "encode_failed": 0, "missing_curve": 0}
    diag_count = 0
    embedding_dim = None

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Encoding phase frames"):
        curve = row.get("resampled_strain_curve")
        if is_missing(curve):
            skipped["missing_curve"] += 1
            continue
        input_path = _input_path(row)
        if input_path is None:
            skipped["missing_input_path"] += 1
            continue
        frames = load_cropped_frames(input_path)
        if frames is None or len(frames) == 0:
            skipped["decode_failed"] += 1
            continue
        indices = _selected_indices(row, int(frames.shape[0]), args.t_frames, args.phase_sampling_mode)
        if indices.size == 0:
            skipped["empty_indices"] += 1
            continue
        indices = np.clip(indices.astype(int), 0, int(frames.shape[0]) - 1)
        selected_frames = frames[indices]
        tensor = resize_tensor(to_tensor(selected_frames), size=args.image_size)
        frame_mask = np.ones((len(indices),), dtype=np.float32)
        try:
            embeddings = _encode_frames(model, tensor, device, args.batch_size, args.use_amp)
        except Exception as exc:
            logger.warning("Encoding failed for %s: %s", input_path, exc)
            skipped["encode_failed"] += 1
            continue
        embedding_dim = int(embeddings.shape[1])

        ed = to_int(row.get("ed_index"))
        es = to_int(row.get("es_index"))
        if diag_count < args.diagnostic_samples:
            sample_id = clean_string(row.get("sample_id")) or f"sample_{diag_count:03d}"
            _save_frame_grid(
                frames,
                indices,
                ed,
                es,
                diagnostics_dir / f"{sample_id}_frames.png",
            )
            _save_indices_plot(
                indices,
                int(frames.shape[0]),
                ed,
                es,
                diagnostics_dir / f"{sample_id}_indices.png",
            )
            diag_count += 1

        out = row.to_dict()
        out.update(
            {
                "input_dicom_for_encoding": str(input_path),
                "phase_sampling_mode": args.phase_sampling_mode,
                "selected_frame_indices": [int(i) for i in indices.tolist()],
                "frame_mask": [float(x) for x in frame_mask.tolist()],
                "embedding": embeddings.astype(np.float32).tolist(),
                "embedding_shape": [int(embeddings.shape[0]), int(embeddings.shape[1])],
                "embedding_dim": int(embeddings.shape[1]),
            }
        )
        rows.append(out)

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(args.output_parquet, index=False)
    dataframe_preview_for_csv(out_df.drop(columns=["embedding"], errors="ignore")).to_csv(
        output_dir / "phase_embeddings_preview.csv",
        index=False,
    )
    summary = {
        "input_parquet": str(args.input_parquet),
        "output_parquet": str(args.output_parquet),
        "n_input_samples": int(len(df)),
        "n_encoded_samples": int(len(out_df)),
        "n_patients": int(out_df["patient_key"].dropna().nunique()) if "patient_key" in out_df else 0,
        "embedding_dim": embedding_dim,
        "t_frames": int(args.t_frames),
        "phase_sampling_mode": args.phase_sampling_mode,
        "skipped": skipped,
        "diagnostics_dir": str(diagnostics_dir),
    }
    write_json(output_dir / "phase_embeddings_summary.json", summary)
    logger.info("Saved phase embeddings: %s (%d samples)", args.output_parquet, len(out_df))
    logger.info("Skipped samples: %s", skipped)


if __name__ == "__main__":
    main()
