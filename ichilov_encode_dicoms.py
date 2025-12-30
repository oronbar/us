"""
Encode cropped Ichilov DICOM clips into MAE embeddings.

Input:
  - Excel registry with GLS source DICOM paths
  - Cropped DICOMs under D:\\DS\\Ichilov_cropped (same relative tree)

Output:
  - Parquet file with one row per (patient, view, dicom) and embedding vector

Usage (PowerShell):
  .venv\\Scripts\\python ichilov_encode_dicoms.py ^
    --input-xlsx "$env:USERPROFILE\\OneDrive - Technion\\DS\\Report_Ichilov_GLS_oron.xlsx" ^
    --echo-root "D:\\DS\\Ichilov" ^
    --cropped-root "D:\\DS\\Ichilov_cropped" ^
    --weights "C:\\work\\us\\Echo-Vison-FM\\weights\\pytorch_model.bin" ^
    --output-parquet "$env:USERPROFILE\\OneDrive - Technion\\DS\\Ichilov_GLS_embeddings.parquet"
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

try:
    import pydicom
except Exception as exc:  # pragma: no cover - runtime check
    raise RuntimeError(
        "pydicom is required. Install with: .venv\\Scripts\\python -m pip install pydicom"
    ) from exc

try:
    from transformers import VideoMAEConfig, VideoMAEForPreTraining
except Exception as exc:  # pragma: no cover - runtime check
    raise RuntimeError(
        "transformers is required. Install with: .venv\\Scripts\\python -m pip install transformers"
    ) from exc


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("ichilov_encode_dicoms")

VIEW_KEYS = ("A2C", "A4C")
PROJECT_ROOT = Path(__file__).resolve().parent
ECHO_VISION_ROOT = (PROJECT_ROOT / "Echo-Vison-FM").resolve()
if ECHO_VISION_ROOT.exists():
    sys.path.append(str(ECHO_VISION_ROOT))
USER_HOME = Path.home()


def _find_column(df: pd.DataFrame, candidates: Sequence[str], required: bool = False) -> Optional[str]:
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in lower_map:
            return lower_map[key]
    for cand in candidates:
        key = cand.lower()
        for col in df.columns:
            if key in str(col).lower():
                return col
    if required:
        raise ValueError(f"Missing required column. Tried: {candidates}")
    return None


def _parse_datetime(val: object) -> Optional[pd.Timestamp]:
    if pd.isna(val):
        return None
    if isinstance(val, pd.Timestamp):
        return val
    s = str(val).strip()
    if not s:
        return None
    s = s.replace("_", "-").replace("\\", "-").replace("/", "-")
    dt = pd.to_datetime(s, errors="coerce")
    return None if pd.isna(dt) else dt


def _split_dicom_list(val: object) -> List[str]:
    if pd.isna(val):
        return []
    s = str(val).strip()
    if not s:
        return []
    parts = [p.strip() for p in re.split(r"[;,\s]+", s) if p.strip()]
    out: List[str] = []
    for p in parts:
        if p.lower().endswith(".dcm"):
            out.append(p[:-4])
        else:
            out.append(p)
    return out


def _candidate_patient_keys(patient_num: str) -> List[str]:
    s = str(patient_num).strip()
    return list(dict.fromkeys([s, s.zfill(2), s.zfill(3)]))


def _find_patient_dir(echo_root: Path, patient_num: str) -> Optional[Path]:
    for cand in _candidate_patient_keys(patient_num):
        p = echo_root / cand
        if p.is_dir():
            return p
    for child in sorted(echo_root.iterdir()):
        if child.is_dir() and child.name.startswith(str(patient_num)):
            return child
    return None


def _date_folder_candidates(dt: Optional[pd.Timestamp]) -> List[str]:
    if dt is None or pd.isna(dt):
        return []
    try:
        return [dt.strftime("%Y-%m-%d"), dt.strftime("%Y_%m_%d")]
    except Exception:
        return []


def _study_dirs(patient_dir: Path, study_dt: Optional[pd.Timestamp]) -> List[Path]:
    date_candidates = _date_folder_candidates(study_dt)
    dirs: List[Path] = []
    for sub in patient_dir.iterdir():
        if not sub.is_dir():
            continue
        if any(sub.name.startswith(dc) for dc in date_candidates):
            dirs.append(sub)
    return dirs if dirs else [patient_dir]


def _locate_dicom_in_dirs(dirs: Iterable[Path], dicom_basename: str) -> Optional[Path]:
    targets = {f"{dicom_basename}.dcm".lower(), dicom_basename.lower()}
    for d in dirs:
        for child in d.iterdir():
            if child.is_file() and child.name.lower() in targets:
                return child
    for d in dirs:
        for child in d.rglob("*"):
            if child.is_file() and child.name.lower() in targets:
                return child
    return None


def _resolve_dicom_path(
    value: object,
    patient_num: Optional[str],
    study_dt: Optional[pd.Timestamp],
    echo_root: Path,
) -> Optional[Path]:
    if pd.isna(value):
        return None
    s = str(value).strip()
    if not s:
        return None
    p = Path(s)
    if p.is_file():
        return p
    if patient_num is None:
        return None
    patient_dir = _find_patient_dir(echo_root, patient_num)
    if patient_dir is None:
        return None
    study_dirs = _study_dirs(patient_dir, study_dt)
    base = p.stem if p.suffix else p.name
    return _locate_dicom_in_dirs(study_dirs, base)


def _to_cropped_path(src_path: Path, echo_root: Path, cropped_root: Path) -> Optional[Path]:
    try:
        rel = src_path.resolve().relative_to(echo_root.resolve())
        cand = cropped_root / rel
        if cand.is_file():
            return cand
    except Exception:
        pass
    cand = cropped_root / src_path.name
    if cand.is_file():
        return cand
    hits = list(cropped_root.rglob(src_path.name))
    return hits[0] if hits else None


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


def _load_cropped_dicom(dicom_path: Path, target_frames: int = 16) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
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
    temporal_indices = torch.as_tensor(idx, dtype=torch.int32)
    return _to_tensor(frames), temporal_indices


def _load_encoder(weights_path: Path, device: torch.device) -> torch.nn.Module:
    config = VideoMAEConfig()
    config.image_size = 224
    config.num_frames = 16
    model = VideoMAEForPreTraining.from_pretrained("MCG-NJU/videomae-base", config=config)
    checkpoint = torch.load(weights_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if isinstance(checkpoint, dict) and all(k.startswith("module.") for k in checkpoint.keys()):
        checkpoint = {k[len("module.") :]: v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint)
    encoder = model.videomae
    encoder.eval()
    encoder.to(device)
    return encoder


def _load_stf_net(device: torch.device) -> torch.nn.Module:
    try:
        from modeling.stff_net import SpatioTemporalFeatureFusionNet
    except Exception as exc:  # pragma: no cover - runtime check
        raise RuntimeError(
            "Requested STF fusion but failed to import it. Make sure the Echo-Vison-FM submodule is present "
            "and install its dependency: python -m pip install positional-encodings"
        ) from exc

    stf_net = SpatioTemporalFeatureFusionNet(feat_dim=768, size_feat_map=(8, 14, 14))
    stf_net.eval()
    stf_net.to(device)
    return stf_net


def _encode_tensor(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    device: torch.device,
    stf_net: Optional[torch.nn.Module] = None,
    temporal_indices: Optional[torch.Tensor] = None,
) -> np.ndarray:
    # tensor shape: T,C,H,W
    pixel_values = tensor.unsqueeze(0).to(device)
    temp_idx = temporal_indices
    if temp_idx is not None:
        temp_idx = temp_idx.to(device)
        if temp_idx.ndim == 1:
            temp_idx = temp_idx.unsqueeze(0)
    with torch.no_grad():
        out = model(pixel_values=pixel_values).last_hidden_state
        if stf_net is not None:
            if temp_idx is None:
                raise ValueError("temporal_indices must be provided when using STF net")
            emb = stf_net(out, temp_idx)
        else:
            emb = torch.max(out, dim=1)[0]
    return emb.squeeze(0).cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode cropped DICOMs into MAE embeddings.")
    parser.add_argument(
        "--input-xlsx",
        type=Path,
        default=USER_HOME / "OneDrive - Technion" / "DS" / "Report_Ichilov_GLS_oron.xlsx",
        help="Input Excel with GLS source DICOM paths",
    )
    parser.add_argument(
        "--echo-root",
        type=Path,
        default=Path(r"D:\DS\Ichilov"),
        help="Root of original DICOM tree",
    )
    parser.add_argument(
        "--cropped-root",
        type=Path,
        default=Path(r"D:\DS\Ichilov_cropped"),
        help="Root of cropped DICOM tree",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=USER_HOME / "OneDrive - Technion" / "models" / "Ichilov_GLS_models" / "Encoder_weights" / "pytorch_model.bin",
        help="Path to MAE weights",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=USER_HOME / "OneDrive - Technion" / "DS" / "Ichilov_GLS_embeddings_full.parquet",
        help="Output parquet path",
    )
    parser.add_argument(
        "--use-stf",
        action="store_true",
        default=True,
        help="Use EchoVisionFM STF fusion head (outputs 1536-dim instead of 768-dim).",
    )
    args = parser.parse_args()

    df = pd.read_excel(args.input_xlsx, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]

    col_patient = _find_column(df, ["PatientNum", "Patient Num", "Patient Number"], required=False)
    col_id = _find_column(df, ["ID", "Patient ID", "Id"], required=False)
    col_date = _find_column(df, ["Study Date", "StudyDate", "Date"], required=False)
    col_src = {v: _find_column(df, [f"{v}_GLS_SOURCE_DICOM"], required=False) for v in VIEW_KEYS}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading encoder on {device}")
    model = _load_encoder(args.weights, device=device)
    stf_net = _load_stf_net(device=device) if args.use_stf else None
    if args.use_stf:
        logger.info("STF fusion enabled (EchoVisionFM)")

    cache: Dict[str, np.ndarray] = {}
    rows: List[dict] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Encoding"):
        patient_num = str(row[col_patient]).strip() if col_patient and not pd.isna(row[col_patient]) else None
        patient_id = str(row[col_id]).strip() if col_id and not pd.isna(row[col_id]) else None
        study_dt = _parse_datetime(row[col_date]) if col_date else None

        for view in VIEW_KEYS:
            col = col_src.get(view)
            if not col:
                logger.debug(f"Skipping view {view} - no source column")
                continue
            src_path = _resolve_dicom_path(row[col], patient_num, study_dt, args.echo_root)
            if src_path is None:
               logger.warning(f"Failed to resolve source DICOM path for view {view}")
               continue
            cropped_path = _to_cropped_path(src_path, args.echo_root, args.cropped_root)
            if cropped_path is None:
                logger.warning(f"Cropped DICOM not found for {src_path}")
                continue

            key = str(cropped_path)
            if key in cache:
                emb = cache[key]
            else:
                loaded = _load_cropped_dicom(cropped_path, target_frames=16)
                if loaded is None:
                    logger.warning(f"Failed to load cropped DICOM {cropped_path}")
                    continue
                tensor, temporal_indices = loaded
                emb = _encode_tensor(
                    model,
                    tensor,
                    device=device,
                    stf_net=stf_net,
                    temporal_indices=temporal_indices,
                )
                cache[key] = emb

            rows.append(
                {
                    "patient_num": patient_num,
                    "patient_id": patient_id,
                    "study_datetime": study_dt,
                    "view": view,
                    "source_dicom": str(src_path),
                    "cropped_dicom": str(cropped_path),
                    "embedding": emb.tolist(),
                    "embedding_dim": int(emb.shape[0]),
                }
            )

    out_df = pd.DataFrame(rows)
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.output_parquet, index=False)
    logger.info(f"Saved embeddings: {args.output_parquet} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
