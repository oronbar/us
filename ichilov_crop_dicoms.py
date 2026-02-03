"""
Crop Ichilov DICOM cine clips to fixed temporal/spatial sizes.

Workflow:
  - Load Excel registry (Report_Ichilov_GLS_and_Strain_oron.xlsx by default)
  - Collect 2C/3C/4C source DICOM paths
  - For each DICOM:
      * pick a 1-second window based on FrameTime (window mode)
      * OR pick an ED-anchored window that pulls ES toward the samples (adjusting_window mode)
      * OR sample 16 frames between ED/ES using phase interpolation (phase mode)
      * compute a mask per sampled frame (RGB > 5), keep only largest blob
      * average blob masks and use it for the crop box (height == blob height)
      * x center is the mean of the top-20 tallest columns in y
      * resize to 224x224
      * save to D:\\DS\\Ichilov_cropped with the same folder structure

Usage (PowerShell):
  .venv\\Scripts\\python ichilov_crop_dicoms.py ^
    --input-xlsx "$env:USERPROFILE\\OneDrive - Technion\\DS\\Report_Ichilov_GLS_and_Strain_oron.xlsx" ^
    --echo-root "D:\\DS\\Ichilov" ^
    --output-root "D:\\DS\\Ichilov_cropped" ^
    --sampling-mode phase

  Sliding-window mode keeps all frames (spatially cropped) so the encoder can
  run overlapping 16-frame clips later.
"""
from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from os import cpu_count

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from tqdm import tqdm

try:
    import pydicom
    from pydicom.dataset import FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
except Exception as exc:  # pragma: no cover - runtime check
    raise RuntimeError(
        "pydicom is required. Install with: .venv\\Scripts\\python -m pip install pydicom"
    ) from exc


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("ichilov_crop_dicoms")
USER_HOME = Path.home()
fallback_drive = Path("F:\\")
if not (USER_HOME / "OneDrive - Technion").exists() and (fallback_drive / "OneDrive - Technion").exists():
    USER_HOME = fallback_drive


@dataclass
class DicomEntry:
    path: Path
    view: Optional[str]
    end_diastole: Optional[int]
    end_systole: Optional[int]


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


def _safe_int(val: object) -> Optional[int]:
    if pd.isna(val):
        return None
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, float):
        if np.isnan(val):
            return None
        return int(round(val))
    s = str(val).strip()
    if not s:
        return None
    try:
        return int(round(float(s)))
    except ValueError:
        return None


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


def _sample_indices_window(
    n_frames: int, frame_time_ms: float, target: int = 16, window_sec: float = 1.0
) -> np.ndarray:
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


def _frame_count_in_window(frame_time_ms: float, n_frames: int, window_sec: float, target: int) -> int:
    if n_frames <= 0:
        return 0
    if frame_time_ms <= 0:
        return min(n_frames, target)
    frames_in_window = int(round((window_sec * 1000.0) / frame_time_ms))
    frames_in_window = max(1, frames_in_window)
    return min(frames_in_window, n_frames)


def _best_window_start(n_frames: int, frames_in_window: int, end_diastole: Optional[int], end_systole: Optional[int]) -> int:
    if n_frames <= 0 or frames_in_window <= 0:
        return 0
    frames_in_window = min(frames_in_window, n_frames)
    if end_diastole is None:
        return max(0, (n_frames - frames_in_window) // 2)

    ed = max(0, min(int(end_diastole), n_frames - 1))
    start_lo = max(0, ed - frames_in_window + 1)
    start_hi = min(ed, n_frames - frames_in_window)
    if start_lo > start_hi:
        start_lo, start_hi = start_hi, start_lo
    start_candidates = range(start_lo, start_hi + 1)

    if end_systole is None:
        desired = ed - (frames_in_window - 1) / 2.0
        start = int(round(desired))
        return min(max(start, start_lo), start_hi)

    es = max(0, min(int(end_systole), n_frames - 1))
    feasible = [s for s in start_candidates if s <= es <= s + frames_in_window - 1]
    if feasible:
        mid_target = 0.5 * (ed + es)
        return min(feasible, key=lambda s: (abs((s + (frames_in_window - 1) / 2.0) - mid_target), s))

    def es_distance(s: int) -> Tuple[float, float, int]:
        left, right = s, s + frames_in_window - 1
        if es < left:
            dist = left - es
        elif es > right:
            dist = es - right
        else:
            dist = 0.0
        center = s + (frames_in_window - 1) / 2.0
        return dist, abs(center - es), s

    return min(start_candidates, key=es_distance)


def _sample_indices_adjusting_window(
    n_frames: int,
    frame_time_ms: float,
    target: int,
    window_sec: float,
    end_diastole: Optional[int],
    end_systole: Optional[int],
) -> np.ndarray:
    frames_in_window = _frame_count_in_window(frame_time_ms, n_frames, window_sec, target)
    start = _best_window_start(n_frames, frames_in_window, end_diastole, end_systole)
    if frames_in_window <= 1 or n_frames <= 0:
        idx = np.full((target,), max(0, min(start, max(0, n_frames - 1))), dtype=int)
    else:
        end = start + frames_in_window - 1
        idx = np.linspace(start, end, target)
        idx = np.clip(np.round(idx).astype(int), 0, n_frames - 1)

    def _adjust_es(idx_arr: np.ndarray) -> np.ndarray:
        if end_systole is None or n_frames <= 0:
            return idx_arr
        es = max(0, min(int(end_systole), n_frames - 1))
        dist = np.abs(idx_arr - es)
        nearest = int(np.argmin(dist))
        if dist[nearest] <= 1:
            idx_arr[nearest] = es
        else:
            step = 1 if es > idx_arr[nearest] else -1
            idx_arr[nearest] = np.clip(idx_arr[nearest] + step, 0, n_frames - 1)
        return idx_arr

    def _ensure_ed(idx_arr: np.ndarray) -> np.ndarray:
        if end_diastole is None or n_frames <= 0:
            return idx_arr
        ed = max(0, min(int(end_diastole), n_frames - 1))
        if ed in idx_arr:
            return idx_arr
        order = list(range(len(idx_arr)))
        order.sort(key=lambda i: (abs(idx_arr[i] - ed), idx_arr[i] == end_systole, i))
        idx_arr[order[0]] = ed
        return idx_arr

    idx = _adjust_es(idx)
    idx = _ensure_ed(idx)
    return idx


def _sample_indices_phase(start_idx: int, end_idx: int, target: int = 16) -> np.ndarray:
    if target <= 0:
        return np.array([], dtype=np.float32)
    return np.linspace(float(start_idx), float(end_idx), target, dtype=np.float32)


def _sample_frames_from_indices(frames: np.ndarray, indices: np.ndarray) -> List[np.ndarray]:
    if indices.size == 0:
        return []
    if indices.dtype.kind in "iu":
        return [frames[int(i)] for i in indices]
    n_frames = frames.shape[0]
    dtype = frames.dtype
    out: List[np.ndarray] = []
    for idx in indices:
        i0 = int(np.floor(idx))
        i1 = int(np.ceil(idx))
        i0 = max(0, min(i0, n_frames - 1))
        i1 = max(0, min(i1, n_frames - 1))
        if i0 == i1:
            out.append(frames[i0])
            continue
        alpha = float(idx - i0)
        frame = (1.0 - alpha) * frames[i0].astype(np.float32) + alpha * frames[i1].astype(np.float32)
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            frame = np.clip(np.round(frame), info.min, info.max)
        out.append(frame.astype(dtype))
    return out


def _sampled_frames(
    frames: np.ndarray,
    frame_time_ms: float,
    target_frames: int,
    window_sec: float,
    sampling_mode: str,
    end_diastole: Optional[int],
    end_systole: Optional[int],
    dicom_path: Path,
    view: Optional[str],
) -> List[np.ndarray]:
    n_frames = frames.shape[0]
    if n_frames <= 0:
        return []
    label = f" ({view})" if view else ""
    if sampling_mode == "phase":
        if end_diastole is None or end_systole is None:
            logger.warning(f"Missing ED/ES for {dicom_path}{label}; falling back to window sampling.")
            indices = _sample_indices_window(n_frames, frame_time_ms, target=target_frames, window_sec=window_sec)
            return _sample_frames_from_indices(frames, indices)
        ed = int(end_diastole)
        es = int(end_systole)
        if ed < 0 or es < 0:
            logger.warning(f"Invalid ED/ES for {dicom_path}{label}; falling back to window sampling.")
            indices = _sample_indices_window(n_frames, frame_time_ms, target=target_frames, window_sec=window_sec)
            return _sample_frames_from_indices(frames, indices)
        if ed > es:
            logger.warning(f"ED > ES for {dicom_path}{label}; swapping indices.")
            ed, es = es, ed
        ed = max(0, min(ed, n_frames - 1))
        es = max(0, min(es, n_frames - 1))
        indices = _sample_indices_phase(ed, es, target=target_frames)
        return _sample_frames_from_indices(frames, indices)
    if sampling_mode == "window":
        indices = _sample_indices_window(n_frames, frame_time_ms, target=target_frames, window_sec=window_sec)
        return _sample_frames_from_indices(frames, indices)
    if sampling_mode == "adjusting_window":
        indices = _sample_indices_adjusting_window(
            n_frames=n_frames,
            frame_time_ms=frame_time_ms,
            target=target_frames,
            window_sec=window_sec,
            end_diastole=end_diastole,
            end_systole=end_systole,
        )
        return _sample_frames_from_indices(frames, indices)
    if sampling_mode == "sliding_window":
        return [frames[i] for i in range(n_frames)]
    raise ValueError(f"Unknown sampling mode: {sampling_mode}")


def _process_dicom(
    dicom_path: Path,
    echo_root: Path,
    output_root: Path,
    out_size: int = 224,
    target_frames: int = 16,
    window_sec: float = 1.0,
    overwrite: bool = False,
    sampling_mode: str = "window",
    end_diastole: Optional[int] = None,
    end_systole: Optional[int] = None,
    view: Optional[str] = None,
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

    sampled_frames = _sampled_frames(
        frames=frames,
        frame_time_ms=frame_time_ms,
        target_frames=target_frames,
        window_sec=window_sec,
        sampling_mode=sampling_mode,
        end_diastole=end_diastole,
        end_systole=end_systole,
        dicom_path=dicom_path,
        view=view,
    )
    if not sampled_frames:
        logger.warning(f"No sampled frames for {dicom_path}")
        return None

    acc = np.zeros(frames.shape[1:3], dtype=np.float32)
    for frame in sampled_frames:
        if channels > 1:
            raw_mask = np.any(frame[..., :3] > 5, axis=-1)
        else:
            raw_mask = frame[..., 0] > 5
        acc += _largest_blob_mask(raw_mask).astype(np.float32)

    avg_mask = acc / float(len(sampled_frames))
    avg_bin = avg_mask > 0.5
    if not np.any(avg_bin):
        avg_bin = avg_mask > 0
    bbox = _mask_bbox(avg_bin)
    x_center = _mask_longest_y_center(avg_bin)

    out_frames: List[np.ndarray] = []
    for frame in sampled_frames:
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
    if not hasattr(ds_out, "file_meta") or ds_out.file_meta is None:
        ds_out.file_meta = FileMetaDataset()
    ds_out.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds_out.save_as(str(out_path), enforce_file_format=True)
    return out_path


def _collect_dicoms(df: pd.DataFrame, echo_root: Path) -> List[DicomEntry]:
    col_patient = _find_column(df, ["PatientNum", "Patient Num", "Patient Number"], required=False)
    col_date = _find_column(df, ["Study Date", "StudyDate", "Date"], required=False)
    col_a2c = _find_column(df, ["A2C_GLS_SOURCE_DICOM"], required=False)
    col_a3c = _find_column(df, ["A3C_GLS_SOURCE_DICOM"], required=False)
    col_a4c = _find_column(df, ["A4C_GLS_SOURCE_DICOM"], required=False)
    col_2c = _find_column(df, ["2-Chambers", "2C", "2 Chambers"], required=False)
    col_3c = _find_column(df, ["3-Chambers", "3C", "3 Chambers"], required=False)
    col_4c = _find_column(df, ["4-Chambers", "4C", "4 Chambers"], required=False)
    col_ed = {
        "A2C": _find_column(df, ["A2C_END_DIASTOLE_FRAME"], required=False),
        "A3C": _find_column(df, ["A3C_END_DIASTOLE_FRAME"], required=False),
        "A4C": _find_column(df, ["A4C_END_DIASTOLE_FRAME"], required=False),
    }
    col_es = {
        "A2C": _find_column(df, ["A2C_END_SYSTOLE_FRAME"], required=False),
        "A3C": _find_column(df, ["A3C_END_SYSTOLE_FRAME"], required=False),
        "A4C": _find_column(df, ["A4C_END_SYSTOLE_FRAME"], required=False),
    }

    dicoms: Dict[str, DicomEntry] = {}

    def add_entry(
        p: Optional[Path],
        view: Optional[str],
        end_diastole: Optional[int],
        end_systole: Optional[int],
    ) -> None:
        if not p or not p.is_file():
            return
        key = str(p)
        entry = dicoms.get(key)
        if entry is None:
            dicoms[key] = DicomEntry(
                path=p,
                view=view,
                end_diastole=end_diastole,
                end_systole=end_systole,
            )
            return
        if entry.view is None and view is not None:
            entry.view = view
        if entry.end_diastole is None and end_diastole is not None:
            entry.end_diastole = end_diastole
        if entry.end_systole is None and end_systole is not None:
            entry.end_systole = end_systole

    for _, row in df.iterrows():
        patient_num = str(row[col_patient]).strip() if col_patient and not pd.isna(row[col_patient]) else None
        study_dt = _parse_datetime(row[col_date]) if col_date else None

        view_frames = {}
        for view in ("A2C", "A3C", "A4C"):
            ed = _safe_int(row[col_ed[view]]) if col_ed[view] else None
            es = _safe_int(row[col_es[view]]) if col_es[view] else None
            view_frames[view] = (ed, es)

        if col_a2c:
            p = _resolve_dicom_path(row[col_a2c], patient_num, study_dt, echo_root)
            ed, es = view_frames["A2C"]
            add_entry(p, "A2C", ed, es)
        if col_a3c:
            p = _resolve_dicom_path(row[col_a3c], patient_num, study_dt, echo_root)
            ed, es = view_frames["A3C"]
            add_entry(p, "A3C", ed, es)
        if col_a4c:
            p = _resolve_dicom_path(row[col_a4c], patient_num, study_dt, echo_root)
            ed, es = view_frames["A4C"]
            add_entry(p, "A4C", ed, es)

        if col_2c:
            ed, es = view_frames["A2C"]
            for base in _split_dicom_list(row[col_2c]):
                p = _resolve_dicom_path(base, patient_num, study_dt, echo_root)
                add_entry(p, "A2C", ed, es)
        if col_3c:
            ed, es = view_frames["A3C"]
            for base in _split_dicom_list(row[col_3c]):
                p = _resolve_dicom_path(base, patient_num, study_dt, echo_root)
                add_entry(p, "A3C", ed, es)
        if col_4c:
            ed, es = view_frames["A4C"]
            for base in _split_dicom_list(row[col_4c]):
                p = _resolve_dicom_path(base, patient_num, study_dt, echo_root)
                add_entry(p, "A4C", ed, es)

    return list(dicoms.values())


def _process_entry(
    entry: DicomEntry,
    echo_root: Path,
    output_root: Path,
    overwrite: bool,
    sampling_mode: str,
    target_frames: int,
    window_sec: float,
) -> Optional[Path]:
    return _process_dicom(
        dicom_path=entry.path,
        echo_root=echo_root,
        output_root=output_root,
        overwrite=overwrite,
        sampling_mode=sampling_mode,
        target_frames=target_frames,
        window_sec=window_sec,
        end_diastole=entry.end_diastole,
        end_systole=entry.end_systole,
        view=entry.view,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Crop Ichilov DICOMs to 224x224 with optional 16-frame sampling.")
    parser.add_argument(
        "--input-xlsx",
        type=Path,
        default=USER_HOME / "OneDrive - Technion" / "DS" / "Report_Ichilov_GLS_and_Strain_oron.xlsx",
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
    parser.add_argument(
        "--clip-length",
        type=int,
        default=16,
        help="Frames per clip for window/phase sampling (sliding_window keeps full length).",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=0.6,
        help="Temporal window length in seconds for window/adjusting_window sampling.",
    )
    parser.add_argument(
        "--sampling-mode",
        choices=("window", "adjusting_window", "phase", "sliding_window"),
        default="window",
        help="Sampling mode: 'window' uses a fixed window; 'adjusting_window' keeps ED, nudges ES into samples; "
             "'phase' interpolates ED->ES frames; 'sliding_window' keeps all frames for clip-level MIL.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of parallel workers (ProcessPool). 0 lets the script pick a sensible default; "
             "set to 1 to disable parallelism.",
    )
    args = parser.parse_args()

    df = pd.read_excel(args.input_xlsx, engine="openpyxl")
    dicoms = _collect_dicoms(df, args.echo_root)
    logger.info(f"Found {len(dicoms)} DICOMs to process")

    worker_count = args.workers
    if worker_count <= 0:
        cpu = cpu_count() or 1
        worker_count = max(1, min(cpu, cpu - 1))

    if worker_count == 1:
        for entry in tqdm(dicoms, desc="Cropping DICOMs"):
            _process_entry(
                entry,
                echo_root=args.echo_root,
                output_root=args.output_root,
                overwrite=args.overwrite,
                sampling_mode=args.sampling_mode,
                target_frames=args.clip_length,
                window_sec=args.window_sec,
            )
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            list(
                tqdm(
                    executor.map(
                        _process_entry,
                        dicoms,
                        repeat(args.echo_root),
                        repeat(args.output_root),
                        repeat(args.overwrite),
                        repeat(args.sampling_mode),
                        repeat(args.clip_length),
                        repeat(args.window_sec),
                    ),
                    total=len(dicoms),
                    desc=f"Cropping DICOMs (workers={worker_count})",
                )
            )


if __name__ == "__main__":
    main()
