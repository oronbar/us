"""
Encode cropped DICOMs into per-frame embeddings using DINOv2 encoder.
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

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
from Ichilov_pipeline3.models.frame_encoder import FrameEncoder

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("ichilov_encode_frames_dinov2")


def _infer_view_from_path(path: Path) -> Optional[str]:
    name = path.as_posix().upper()
    for v in VIEW_KEYS:
        if v in name:
            return v
    return None


def _load_encoder(weights_path: Optional[Path], backbone_name: str) -> FrameEncoder:
    model = FrameEncoder(
        backbone_name=backbone_name,
        pretrained=True,
        freeze_backbone=False,
        unfreeze_last_blocks=0,
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
            if k.startswith("frame_encoder.backbone."):
                backbone_sd[k[len("frame_encoder.backbone.") :]] = v
            elif k.startswith("backbone."):
                backbone_sd[k[len("backbone.") :]] = v
        if backbone_sd:
            model.backbone.load_state_dict(backbone_sd, strict=False)
        else:
            model.backbone.load_state_dict(sd, strict=False)
    elif isinstance(checkpoint, dict):
        model.backbone.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model


def _encode_tensor(model: FrameEncoder, tensor: torch.Tensor, device: torch.device):
    with torch.no_grad():
        emb = model(tensor.unsqueeze(0).to(device))
    return emb.squeeze(0).cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode cropped DICOMs into per-frame embeddings.")
    parser.add_argument("--input-xlsx", type=Path, default=None, help="Optional Excel registry.")
    parser.add_argument("--echo-root", type=Path, default=None, help="Root of original DICOMs.")
    parser.add_argument("--cropped-root", type=Path, required=True, help="Root of cropped DICOMs.")
    parser.add_argument("--weights", type=Path, default=None, help="Path to DINOv2 checkpoint.")
    parser.add_argument("--output-parquet", type=Path, required=True, help="Output parquet path.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output parquet.")
    parser.add_argument("--sampling-mode", type=str, choices=["uniform", "sliding_window"], default="uniform")
    parser.add_argument("--clip-length", type=int, default=1, help="Frames per sample (1 for frame-level).")
    parser.add_argument("--clip-stride", type=int, default=1, help="Stride between frames (sliding_window).")
    parser.add_argument("--include-last-window", dest="include_last_window", action="store_true", default=True)
    parser.add_argument("--no-include-last-window", dest="include_last_window", action="store_false")
    parser.add_argument("--views", type=str, default="", help="Comma/space-separated views to include.")
    parser.add_argument("--safe-decode", action="store_true", help="Disable GDCM/pylibjpeg decoders.")
    parser.add_argument("--backbone-name", type=str, default="vit_small_patch14_dinov2.lvd142m")
    args = parser.parse_args()

    configure_pydicom_handlers(args.safe_decode)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading DINOv2 encoder on %s", device)
    encoder = _load_encoder(args.weights, backbone_name=args.backbone_name).to(device)

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
            dummy = type(
                "E",
                (),
                {
                    "view": view,
                    "patient_num": None,
                    "patient_id": None,
                    "study_datetime": None,
                    "end_diastole": None,
                    "end_systole": None,
                    "path": path,
                },
            )
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
            tensor = resize_tensor(item["tensor"], size=224).squeeze(0)
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
    logger.info("Saved DINOv2 embeddings: %s (%d rows)", output_path, len(out_df))


if __name__ == "__main__":
    main()

