"""
Encode cropped DICOMs into per-frame embeddings using USF-MAE (ViT-MAE) encoder.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ichilov_pipeline2_utils import (
    VIEW_KEYS,
    collect_dicoms_from_report,
    configure_pydicom_handlers,
    iter_cropped_frames,
    parse_views,
    resize_tensor,
    to_cropped_path,
)
from usf_mae_model import MaskedAutoencoderViT, mae_vit_base_patch16_dec512d8b

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("ichilov_encode_frames")

DEFAULT_USF_WEIGHTS = Path(r"D:\us\USF-MAE\USF-MAE_full_pretrain_43dataset_100epochs.pt")


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


def _load_encoder(weights_path: Path) -> MaskedAutoencoderViT:
    model = mae_vit_base_patch16_dec512d8b()
    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = _normalize_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def _encode_tensor(model: MaskedAutoencoderViT, tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    pixel_values = tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        latent, _, _ = model.forward_encoder(pixel_values, mask_ratio=0.0)
        cls_tok = latent[:, 0, :]
    return cls_tok.squeeze(0).cpu().numpy()


def _infer_view_from_path(path: Path) -> Optional[str]:
    name = path.as_posix().upper()
    for v in VIEW_KEYS:
        if v in name:
            return v
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode cropped DICOMs into per-frame embeddings.")
    parser.add_argument("--input-xlsx", type=Path, default=None, help="Optional Excel registry.")
    parser.add_argument("--echo-root", type=Path, default=None, help="Root of original DICOMs.")
    parser.add_argument("--cropped-root", type=Path, required=True, help="Root of cropped DICOMs.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_USF_WEIGHTS,
        help="Path to USF-MAE weights.",
    )
    parser.add_argument("--output-parquet", type=Path, required=True, help="Output parquet path.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output parquet.")
    parser.add_argument("--sampling-mode", type=str, choices=["uniform", "sliding_window"], default="uniform")
    parser.add_argument("--clip-length", type=int, default=1, help="Frames per sample (1 for frame-level).")
    parser.add_argument("--clip-stride", type=int, default=1, help="Stride between frames (sliding_window).")
    parser.add_argument("--include-last-window", dest="include_last_window", action="store_true", default=True)
    parser.add_argument("--no-include-last-window", dest="include_last_window", action="store_false")
    parser.add_argument("--views", type=str, default="", help="Comma/space-separated views to include.")
    parser.add_argument("--safe-decode", action="store_true", help="Disable GDCM/pylibjpeg decoders.")
    args = parser.parse_args()

    configure_pydicom_handlers(args.safe_decode)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading encoder on %s", device)
    encoder = _load_encoder(args.weights)
    encoder.to(device)

    selected_views = parse_views(args.views)
    rows: List[dict] = []

    if args.input_xlsx and args.input_xlsx.exists() and args.echo_root is not None:
        df = pd.read_excel(args.input_xlsx, engine="openpyxl")
        df.columns = [str(c).strip() for c in df.columns]
        dicoms = collect_dicoms_from_report(df, args.echo_root, views=selected_views)
        entries = []
        for entry in dicoms:
            cropped = to_cropped_path(entry.path, args.echo_root, args.cropped_root)
            if cropped is None:
                continue
            entries.append((entry, cropped))
    else:
        entries = []
        for path in args.cropped_root.rglob("*.dcm"):
            view = _infer_view_from_path(path)
            if selected_views and view not in selected_views:
                continue
            dummy = type("E", (), {"view": view, "patient_num": None, "patient_id": None, "study_datetime": None, "end_diastole": None, "end_systole": None, "path": path})
            entries.append((dummy, path))

    for entry, cropped_path in tqdm(entries, desc="Encoding"):
        if entry.view and selected_views and entry.view not in selected_views:
            continue
        frame_items = iter_cropped_frames(
            cropped_path,
            sampling_mode=args.sampling_mode,
            clip_length=args.clip_length,
            clip_stride=args.clip_stride,
            include_last=args.include_last_window,
        )
        if not frame_items:
            continue
        for item in frame_items:
            tensor = item["tensor"]
            tensor = resize_tensor(tensor, size=224)
            emb = _encode_tensor(encoder, tensor, device)
            rows.append(
                {
                    "patient_num": getattr(entry, "patient_num", None),
                    "patient_id": getattr(entry, "patient_id", None),
                    "study_datetime": getattr(entry, "study_datetime", None),
                    "view": getattr(entry, "view", None),
                    "source_dicom": str(getattr(entry, "path", cropped_path)),
                    "cropped_dicom": str(cropped_path),
                    "end_diastole": getattr(entry, "end_diastole", None),
                    "end_systole": getattr(entry, "end_systole", None),
                    "sampling_mode": args.sampling_mode,
                    "frame_index": int(item["frame_index"]),
                    "frame_time": float(item["frame_time"]),
                    "frame_count": int(item["frame_count"]),
                    "embedding": emb.tolist(),
                    "embedding_dim": int(emb.shape[0]),
                }
            )

    out_df = pd.DataFrame(rows)
    output_path = args.output_parquet
    if output_path.exists() and not args.overwrite:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_path.with_name(f"{output_path.stem}_{stamp}{output_path.suffix}")
        logger.warning("Output exists; writing embeddings to %s", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path, index=False)
    logger.info("Saved embeddings: %s (%d rows)", output_path, len(out_df))


if __name__ == "__main__":
    main()
