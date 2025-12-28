"""
Crop Ichilov DICOM cine clips to fixed temporal/spatial sizes.

Workflow:
  - Load Excel registry (Report_Ichilov_GLS_oron.xlsx by default)
  - Collect 2C/4C source DICOM paths
  - For each DICOM:
      * pick a 1-second window based on FrameTime
      * sample 16 frames uniformly inside the window
      * compute a mask per sampled frame (RGB > 5), keep only largest blob
      * average blob masks and use it for the crop box (height == blob height)
      * x center is the mean of the top-20 tallest columns in y
      * resize to 224x224
      * save to D:\\DS\\Ichilov_cropped with the same folder structure

Usage (PowerShell):
  .venv\\Scripts\\python ichilov_crop_dicoms.py ^
    --input-xlsx "C:\\Users\\oronbarazani\\OneDrive - Technion\\DS\\Report_Ichilov_GLS_oron.xlsx" ^
    --echo-root "D:\\DS\\Ichilov" ^
    --output-root "D:\\DS\\Ichilov_cropped"
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from tqdm import tqdm

try:
    import pydicom
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
except Exception as exc:  # pragma: no cover - runtime check
    raise RuntimeError(
        "pydicom is required. Install with: .venv\\Scripts\\python -m pip install pydicom"
    ) from exc


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("ichilov_crop_dicoms")


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


def _normalize_frames(arr: np.ndarray, samples_per_pixel: int) -> Tuple[np.ndarray, int, int]:
    if arr.ndim == 2:
        frames = arr[None, ...]
        return frames[..., None], 1, 1
    if arr.ndim == 3:
        if samples_per_pixel > 1:
            frames = arr[None, ...]
            return frames, 1, samples_per_pixel
        return arr[..., None], arr.shape[0], 1
    if arr.ndim == 4:
        return arr, arr.shape[0], arr.shape[-1]
    raise ValueError(f"Unexpected pixel array shape: {arr.shape}")


def _largest_blob_mask(mask: np.ndarray) -> np.ndarray:
    labeled, num = ndi.label(mask)
    if num == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = ndi.sum(mask, labeled, index=list(range(1, num + 1)))
    largest = int(np.argmax(sizes)) + 1
    return labeled == largest


def _mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    h, w = mask.shape
    if not np.any(mask):
        return 0, h, 0, w
    ys = np.where(mask.any(axis=1))[0]
    xs = np.where(mask.any(axis=0))[0]
    y0, y1 = int(ys[0]), int(ys[-1]) + 1
    x0, x1 = int(xs[0]), int(xs[-1]) + 1
    return y0, y1, x0, x1


def _crop_square(frame: np.ndarray, bbox: Tuple[int, int, int, int], x_center: Optional[float] = None) -> np.ndarray:
    y0, y1, x0, x1 = bbox
    if frame.ndim == 2:
        frame = frame[..., None]
    h, w, c = frame.shape
    height = max(1, y1 - y0)
    cx = x_center if x_center is not None else 0.5 * (x0 + x1)
    x_start = int(round(cx - height / 2.0))
    x_end = x_start + height
    out = np.zeros((height, height, c), dtype=frame.dtype)

    src_y0 = max(y0, 0)
    src_y1 = min(y1, h)
    src_x0 = max(x_start, 0)
    src_x1 = min(x_end, w)

    dst_y0 = src_y0 - y0
    dst_x0 = src_x0 - x_start
    out[dst_y0 : dst_y0 + (src_y1 - src_y0), dst_x0 : dst_x0 + (src_x1 - src_x0), :] = frame[
        src_y0:src_y1, src_x0:src_x1, :
    ]
    return out


def _mask_longest_y_center(mask: np.ndarray) -> Optional[float]:
    if not np.any(mask):
        return None
    heights = np.zeros(mask.shape[1], dtype=int)
    for x in range(mask.shape[1]):
        col = mask[:, x]
        if not np.any(col):
            continue
        y_idx = np.where(col)[0]
        heights[x] = int(y_idx[-1] - y_idx[0] + 1)
    if heights.max() <= 0:
        return None
    top_n = min(20, int(np.count_nonzero(heights)))
    if top_n <= 0:
        return None
    top_idx = np.argsort(heights)[-top_n:]
    return float(np.mean(top_idx))


def _pad_or_crop(arr: np.ndarray, size: int) -> np.ndarray:
    h, w = arr.shape[:2]
    if h == size and w == size:
        return arr
    out = np.zeros((size, size) + arr.shape[2:], dtype=arr.dtype)
    y0 = max(0, (size - h) // 2)
    x0 = max(0, (size - w) // 2)
    src_y0 = max(0, (h - size) // 2)
    src_x0 = max(0, (w - size) // 2)
    y_len = min(size, h)
    x_len = min(size, w)
    out[y0 : y0 + y_len, x0 : x0 + x_len, ...] = arr[src_y0 : src_y0 + y_len, src_x0 : src_x0 + x_len, ...]
    return out


def _resize_frame(frame: np.ndarray, out_size: int) -> np.ndarray:
    if frame.ndim == 2:
        frame = frame[..., None]
    dtype = frame.dtype
    frame_f = frame.astype(np.float32)
    zoom = (out_size / frame.shape[0], out_size / frame.shape[1], 1)
    resized = ndi.zoom(frame_f, zoom, order=1)
    resized = _pad_or_crop(resized, out_size)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        resized = np.clip(resized, info.min, info.max)
    return resized.astype(dtype)


def _sample_indices(n_frames: int, frame_time_ms: float, target: int = 16, window_sec: float = 1.0) -> np.ndarray:
    if n_frames <= 0:
        return np.array([], dtype=int)
    if frame_time_ms <= 0:
        return np.arange(min(n_frames, target), dtype=int)
    frames_in_window = int(round((window_sec * 1000.0) / frame_time_ms))
    frames_in_window = max(1, frames_in_window)
    if n_frames < frames_in_window:
        start = 0
        frames_in_window = n_frames
    else:
        start = max(0, (n_frames - frames_in_window) // 2)
    if frames_in_window <= 1:
        return np.zeros((target,), dtype=int)
    idx = np.linspace(start, start + frames_in_window - 1, target)
    idx = np.clip(np.round(idx).astype(int), 0, n_frames - 1)
    return idx


def _process_dicom(
    dicom_path: Path,
    echo_root: Path,
    output_root: Path,
    out_size: int = 224,
    target_frames: int = 16,
    window_sec: float = 1.0,
    overwrite: bool = False,
) -> Optional[Path]:
    try:
        ds = pydicom.dcmread(str(dicom_path), force=True)
    except Exception as exc:
        logger.warning(f"Failed to read {dicom_path}: {exc}")
        return None

    frame_time = getattr(ds, "FrameTime", None)
    frame_time_ms = float(frame_time) if frame_time is not None else 0.0
    try:
        arr = ds.pixel_array
    except Exception as exc:
        logger.warning(f"Failed to decode pixel data {dicom_path}: {exc}")
        return None

    samples_per_pixel = int(getattr(ds, "SamplesPerPixel", 1))
    frames, n_frames, channels = _normalize_frames(arr, samples_per_pixel)

    if n_frames <= 0:
        logger.warning(f"No frames in {dicom_path}")
        return None

    idx = _sample_indices(n_frames, frame_time_ms, target=target_frames, window_sec=window_sec)
    if idx.size == 0:
        logger.warning(f"No sample indices for {dicom_path}")
        return None

    acc = np.zeros(frames.shape[1:3], dtype=np.float32)
    for i in idx:
        frame = frames[int(i)]
        if channels > 1:
            raw_mask = np.any(frame[..., :3] > 5, axis=-1)
        else:
            raw_mask = frame[..., 0] > 5
        acc += _largest_blob_mask(raw_mask).astype(np.float32)

    avg_mask = acc / float(len(idx))
    avg_bin = avg_mask > 0.5
    if not np.any(avg_bin):
        avg_bin = avg_mask > 0
    bbox = _mask_bbox(avg_bin)
    x_center = _mask_longest_y_center(avg_bin)

    out_frames: List[np.ndarray] = []
    for i in idx:
        frame = frames[int(i)]
        if channels > 1:
            raw_mask = np.any(frame[..., :3] > 5, axis=-1)
        else:
            raw_mask = frame[..., 0] > 5
        keep_mask = _largest_blob_mask(raw_mask)
        drop_mask = raw_mask & ~keep_mask
        if np.any(drop_mask):
            frame = frame.copy()
            if frame.ndim == 2:
                frame[drop_mask] = 0
            else:
                frame[drop_mask, :] = 0
        cropped = _crop_square(frame, bbox, x_center=x_center)
        resized = _resize_frame(cropped, out_size)
        out_frames.append(resized)

    stack = np.stack(out_frames, axis=0)
    if channels == 1:
        stack = stack[..., 0]

    try:
        rel = dicom_path.relative_to(echo_root)
        out_path = output_root / rel
    except ValueError:
        out_path = output_root / dicom_path.name

    if out_path.exists() and not overwrite:
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds_out = ds.copy()
    ds_out.Rows = out_size
    ds_out.Columns = out_size
    ds_out.NumberOfFrames = int(stack.shape[0])
    if channels > 1:
        ds_out.SamplesPerPixel = channels
        ds_out.PlanarConfiguration = 0
        ds_out.PhotometricInterpretation = "RGB"
    else:
        ds_out.SamplesPerPixel = 1
        if "PlanarConfiguration" in ds_out:
            del ds_out.PlanarConfiguration
        ds_out.PhotometricInterpretation = "MONOCHROME2"
    bits = int(stack.dtype.itemsize * 8)
    ds_out.BitsAllocated = bits
    ds_out.BitsStored = bits
    ds_out.HighBit = bits - 1
    ds_out.PixelRepresentation = 1 if stack.dtype.kind == "i" else 0
    ds_out.PixelData = stack.tobytes()
    ds_out.SOPInstanceUID = generate_uid()
    if hasattr(ds_out, "file_meta"):
        ds_out.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds_out.is_little_endian = True
    ds_out.is_implicit_VR = False
    ds_out.save_as(str(out_path), write_like_original=False)
    return out_path


def _collect_dicoms(df: pd.DataFrame, echo_root: Path) -> List[Path]:
    col_patient = _find_column(df, ["PatientNum", "Patient Num", "Patient Number"], required=False)
    col_date = _find_column(df, ["Study Date", "StudyDate", "Date"], required=False)
    col_a2c = _find_column(df, ["A2C_GLS_SOURCE_DICOM"], required=False)
    col_a4c = _find_column(df, ["A4C_GLS_SOURCE_DICOM"], required=False)
    col_2c = _find_column(df, ["2-Chambers", "2C", "2 Chambers"], required=False)
    col_4c = _find_column(df, ["4-Chambers", "4C", "4 Chambers"], required=False)

    dicoms: Dict[str, Path] = {}
    for _, row in df.iterrows():
        patient_num = str(row[col_patient]).strip() if col_patient and not pd.isna(row[col_patient]) else None
        study_dt = _parse_datetime(row[col_date]) if col_date else None

        for col in (col_a2c, col_a4c):
            if not col:
                continue
            p = _resolve_dicom_path(row[col], patient_num, study_dt, echo_root)
            if p and p.is_file():
                dicoms[str(p)] = p

        if col_2c:
            for base in _split_dicom_list(row[col_2c]):
                p = _resolve_dicom_path(base, patient_num, study_dt, echo_root)
                if p and p.is_file():
                    dicoms[str(p)] = p
        if col_4c:
            for base in _split_dicom_list(row[col_4c]):
                p = _resolve_dicom_path(base, patient_num, study_dt, echo_root)
                if p and p.is_file():
                    dicoms[str(p)] = p

    return list(dicoms.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Crop Ichilov DICOMs to 16x224x224.")
    parser.add_argument(
        "--input-xlsx",
        type=Path,
        default=Path(r"C:\Users\oronbarazani\OneDrive - Technion\DS\Report_Ichilov_GLS_oron.xlsx"),
        help="Input Excel with GLS source DICOM paths",
    )
    parser.add_argument(
        "--echo-root",
        type=Path,
        default=Path(r"D:\DS\Ichilov"),
        help="Root directory of original DICOMs",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"D:\DS\Ichilov_cropped"),
        help="Root directory to save cropped DICOMs",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    args = parser.parse_args()

    df = pd.read_excel(args.input_xlsx, engine="openpyxl")
    dicoms = _collect_dicoms(df, args.echo_root)
    logger.info(f"Found {len(dicoms)} DICOMs to process")

    for dicom_path in tqdm(dicoms, desc="Cropping DICOMs"):
        _process_dicom(
            dicom_path=dicom_path,
            echo_root=args.echo_root,
            output_root=args.output_root,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
