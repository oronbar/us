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
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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

VIEW_KEYS = ("A2C", "A3C", "A4C")
PROJECT_ROOT = Path(__file__).resolve().parent
ECHO_VISION_ROOT = (PROJECT_ROOT / "Echo-Vison-FM").resolve()
if ECHO_VISION_ROOT.exists():
    sys.path.append(str(ECHO_VISION_ROOT))
USER_HOME = Path.home()
if USER_HOME.name == "oronbar.RF":
        USER_HOME = Path("F:\\")


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


def _sample_aug_factor(rng: np.random.Generator, max_delta: float) -> float:
    if max_delta <= 0:
        return 1.0
    return 1.0 + float(rng.uniform(-max_delta, max_delta))


def _build_patient_augmentations(
    rng: np.random.Generator,
    count: int,
    brightness_max: float,
    contrast_max: float,
    rotate_max_deg: float,
    stretch_max: float,
) -> List[Dict[str, float]]:
    augments: List[Dict[str, float]] = []
    for _ in range(count):
        brightness = _sample_aug_factor(rng, brightness_max)
        contrast = _sample_aug_factor(rng, contrast_max)
        rotation = float(rng.uniform(-rotate_max_deg, rotate_max_deg)) if rotate_max_deg > 0 else 0.0
        scale_x = 1.0
        scale_y = 1.0
        if stretch_max > 0:
            if rng.random() < 0.5:
                scale_x = _sample_aug_factor(rng, stretch_max)
            else:
                scale_y = _sample_aug_factor(rng, stretch_max)
        augments.append(
            {
                "brightness": brightness,
                "contrast": contrast,
                "rotation_deg": rotation,
                "scale_x": scale_x,
                "scale_y": scale_y,
            }
        )
    return augments


def _format_aug_key(aug: Optional[Dict[str, float]]) -> str:
    if aug is None:
        return "orig"
    return (
        f"b{aug['brightness']:.3f}_"
        f"c{aug['contrast']:.3f}_"
        f"r{aug['rotation_deg']:.2f}_"
        f"sx{aug['scale_x']:.3f}_"
        f"sy{aug['scale_y']:.3f}"
    )


def _apply_patient_augmentation(tensor: torch.Tensor, aug: Dict[str, float]) -> torch.Tensor:
    out = tensor
    angle = aug.get("rotation_deg", 0.0)
    scale_x = aug.get("scale_x", 1.0)
    scale_y = aug.get("scale_y", 1.0)
    if angle != 0.0 or scale_x != 1.0 or scale_y != 1.0:
        rad = math.radians(angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        theta = torch.tensor(
            [
                [scale_x * cos_a, -scale_y * sin_a, 0.0],
                [scale_x * sin_a, scale_y * cos_a, 0.0],
            ],
            dtype=out.dtype,
            device=out.device,
        )
        theta = theta.unsqueeze(0).repeat(out.shape[0], 1, 1)
        grid = F.affine_grid(theta, out.size(), align_corners=False)
        out = F.grid_sample(out, grid, mode="bilinear", padding_mode="border", align_corners=False)
    contrast = aug.get("contrast", 1.0)
    if contrast != 1.0:
        mean = out.mean(dim=(0, 2, 3), keepdim=True)
        out = (out - mean) * contrast + mean
    brightness = aug.get("brightness", 1.0)
    if brightness != 1.0:
        out = out * brightness
    return out.clamp(0.0, 1.0)


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
    suffix = weights_path.suffix.lower()
    if suffix not in {".pt", ".bin"}:
        raise ValueError(f"Unsupported weights extension {weights_path.suffix!r}; expected .pt or .bin")
    try:
        checkpoint = torch.load(str(weights_path), map_location="cpu")
    except OSError as exc:
        fallback = weights_path.with_suffix(".bin") if suffix == ".pt" else None
        if fallback is not None and fallback.exists():
            logger.warning(
                "Failed to load %s (%s). Falling back to %s", weights_path, exc, fallback
            )
            checkpoint = torch.load(str(fallback), map_location="cpu")
        else:
            raise OSError(
                f"Failed to read weights at {weights_path}. If this is a cloud/OneDrive file, "
                "make sure it is available offline or provide a .bin checkpoint."
            ) from exc
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
            checkpoint = checkpoint["model"]
    if isinstance(checkpoint, dict) and all(k.startswith("module.") for k in checkpoint.keys()):
        checkpoint = {k[len("module.") :]: v for k, v in checkpoint.items()}
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unexpected checkpoint format in {weights_path}")
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
        default=USER_HOME / "OneDrive - Technion" / "models" / "Encoder_weights" / "01_best_mae.pt",
        help="Path to MAE weights",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=USER_HOME / "OneDrive - Technion" / "DS" / "Ichilov_GLS_embeddings_full_trained_aug.parquet",
        help="Output parquet path",
    )
    parser.add_argument(
        "--use-stf",
        action="store_true",
        default=True,
        help="Use EchoVisionFM STF fusion head (outputs 1536-dim instead of 768-dim).",
    )
    parser.add_argument(
        "--aug-per-patient",
        type=int,
        default=2,
        help="Number of augmented patient copies to generate per patient (0 disables).",
    )
    parser.add_argument(
        "--aug-seed",
        type=int,
        default=42,
        help="Random seed for patient-level augmentation sampling (default: random).",
    )
    parser.add_argument(
        "--aug-brightness",
        type=float,
        default=0.1,
        help="Max absolute brightness factor delta (e.g., 0.1 -> [0.9, 1.1]).",
    )
    parser.add_argument(
        "--aug-contrast",
        type=float,
        default=0.1,
        help="Max absolute contrast factor delta (e.g., 0.1 -> [0.9, 1.1]).",
    )
    parser.add_argument(
        "--aug-rotate-deg",
        type=float,
        default=5.0,
        help="Max absolute rotation in degrees.",
    )
    parser.add_argument(
        "--aug-stretch",
        type=float,
        default=0.05,
        help="Max absolute stretch factor delta along x or y axis.",
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

    aug_per_patient = max(int(args.aug_per_patient), 0)
    rng = np.random.default_rng(args.aug_seed) if args.aug_seed is not None else np.random.default_rng()
    patient_augments: Dict[str, List[Dict[str, float]]] = {}
    cache: Dict[Tuple[str, str], np.ndarray] = {}
    rows: List[dict] = []

    for row_idx, row in tqdm(df.iterrows(), total=len(df), desc="Encoding"):
        patient_num = str(row[col_patient]).strip() if col_patient and not pd.isna(row[col_patient]) else None
        patient_id = str(row[col_id]).strip() if col_id and not pd.isna(row[col_id]) else None
        study_dt = _parse_datetime(row[col_date]) if col_date else None

        patient_key = patient_num or patient_id or f"row_{row_idx}"
        if aug_per_patient > 0 and patient_key not in patient_augments:
            patient_augments[patient_key] = _build_patient_augmentations(
                rng,
                aug_per_patient,
                brightness_max=args.aug_brightness,
                contrast_max=args.aug_contrast,
                rotate_max_deg=args.aug_rotate_deg,
                stretch_max=args.aug_stretch,
            )
        augments = patient_augments.get(patient_key, [])
        variants = [
            {
                "aug_id": 0,
                "label": "orig",
                "params": None,
                "patient_num": patient_num,
                "patient_id": patient_id,
            }
        ]
        for idx, aug in enumerate(augments, start=1):
            variants.append(
                {
                    "aug_id": idx,
                    "label": f"aug{idx}",
                    "params": aug,
                    "patient_num": f"{patient_num}_aug{idx}" if patient_num is not None else None,
                    "patient_id": f"{patient_id}_aug{idx}" if patient_id is not None else None,
                }
            )

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

            cropped_key = str(cropped_path)
            variant_keys = [(v, _format_aug_key(v["params"])) for v in variants]
            need_load = any((cropped_key, aug_key) not in cache for _, aug_key in variant_keys)
            base_tensor: Optional[torch.Tensor] = None
            temporal_indices: Optional[torch.Tensor] = None
            if need_load:
                loaded = _load_cropped_dicom(cropped_path, target_frames=16)
                if loaded is None:
                    logger.warning(f"Failed to load cropped DICOM {cropped_path}")
                    continue
                base_tensor, temporal_indices = loaded

            for variant, aug_key in variant_keys:
                cache_key = (cropped_key, aug_key)
                if cache_key in cache:
                    emb = cache[cache_key]
                else:
                    if base_tensor is None or temporal_indices is None:
                        logger.warning(f"Missing base tensor for {cropped_path} ({aug_key})")
                        continue
                    tensor = base_tensor
                    if variant["params"] is not None:
                        tensor = _apply_patient_augmentation(base_tensor, variant["params"])
                    emb = _encode_tensor(
                        model,
                        tensor,
                        device=device,
                        stf_net=stf_net,
                        temporal_indices=temporal_indices,
                    )
                    cache[cache_key] = emb

                aug_params = variant["params"]
                rows.append(
                    {
                        "patient_num": variant["patient_num"],
                        "patient_id": variant["patient_id"],
                        "patient_num_base": patient_num,
                        "patient_id_base": patient_id,
                        "augmentation": variant["label"],
                        "augmentation_id": int(variant["aug_id"]),
                        "augmentation_key": aug_key,
                        "augmented": aug_params is not None,
                        "aug_brightness": float(aug_params["brightness"]) if aug_params else 1.0,
                        "aug_contrast": float(aug_params["contrast"]) if aug_params else 1.0,
                        "aug_rotation_deg": float(aug_params["rotation_deg"]) if aug_params else 0.0,
                        "aug_scale_x": float(aug_params["scale_x"]) if aug_params else 1.0,
                        "aug_scale_y": float(aug_params["scale_y"]) if aug_params else 1.0,
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
