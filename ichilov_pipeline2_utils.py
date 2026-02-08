"""
Shared helpers for Ichilov pipeline2 stage scripts.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

try:
    import pydicom
except Exception as exc:  # pragma: no cover - runtime check
    raise RuntimeError(
        "pydicom is required. Install with: .venv\\Scripts\\python -m pip install pydicom"
    ) from exc

logger = logging.getLogger("ichilov_pipeline2_utils")

VIEW_KEYS = ("A2C", "A3C", "A4C")
PATIENT_COL_CANDIDATES = (
    "patient_id",
    "patient_num",
    "patient",
    "short_id",
    "id",
    "patient_number",
    "patient_num_base",
    "patient_id_base",
)
TIME_COL_CANDIDATES = (
    "study_datetime",
    "study_date",
    "visit_date",
    "exam_date",
    "acquisition_date",
    "acquisition_time",
    "scan_date",
    "visit_time",
    "date",
    "datetime",
    "visit_index",
    "visit_num",
    "visit_number",
    "timepoint",
    "time_point",
)
VISIT_COL_CANDIDATES = (
    "visit_id",
    "visit",
    "visit_index",
    "visit_num",
    "visit_number",
    "study_id",
    "study_uid",
    "study_instance_uid",
    "accession",
    "accession_number",
    "exam_id",
) + TIME_COL_CANDIDATES


@dataclass
class DicomEntry:
    path: Path
    view: Optional[str] = None
    patient_num: Optional[str] = None
    patient_id: Optional[str] = None
    study_datetime: Optional[pd.Timestamp] = None
    end_diastole: Optional[int] = None
    end_systole: Optional[int] = None


def parse_views(view_str: Optional[str]) -> Optional[set]:
    if view_str is None:
        return None
    raw = str(view_str).strip()
    if not raw:
        return None
    parts = [p.strip().upper() for p in re.split(r"[;,\s]+", raw) if p.strip()]
    if not parts:
        return None
    mapped = []
    for p in parts:
        token = p.replace("APICAL", "").replace("CHAMBER", "").replace("CH", "").replace("-", "")
        token = token.replace("A", "")
        if token in {"2C", "2"}:
            mapped.append("A2C")
        elif token in {"3C", "3"}:
            mapped.append("A3C")
        elif token in {"4C", "4"}:
            mapped.append("A4C")
        else:
            mapped.append(p)
    return set(mapped)


def find_column(df: pd.DataFrame, candidates: Sequence[str], required: bool = False) -> Optional[str]:
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


def parse_datetime(val: object) -> Optional[pd.Timestamp]:
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


def split_dicom_list(val: object) -> List[str]:
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


def candidate_patient_keys(patient_num: str) -> List[str]:
    s = str(patient_num).strip()
    return list(dict.fromkeys([s, s.zfill(2), s.zfill(3)]))


def find_patient_dir(echo_root: Path, patient_num: str) -> Optional[Path]:
    for cand in candidate_patient_keys(patient_num):
        p = echo_root / cand
        if p.is_dir():
            return p
    for child in sorted(echo_root.iterdir()):
        if child.is_dir() and child.name.startswith(str(patient_num)):
            return child
    return None


def date_folder_candidates(dt: Optional[pd.Timestamp]) -> List[str]:
    if dt is None or pd.isna(dt):
        return []
    try:
        return [dt.strftime("%Y-%m-%d"), dt.strftime("%Y_%m_%d")]
    except Exception:
        return []


def study_dirs(patient_dir: Path, study_dt: Optional[pd.Timestamp]) -> List[Path]:
    date_candidates = date_folder_candidates(study_dt)
    dirs: List[Path] = []
    for sub in patient_dir.iterdir():
        if not sub.is_dir():
            continue
        if any(sub.name.startswith(dc) for dc in date_candidates):
            dirs.append(sub)
    return dirs if dirs else [patient_dir]


def locate_dicom_in_dirs(dirs: Iterable[Path], dicom_basename: str) -> Optional[Path]:
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


def resolve_dicom_path(
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
    patient_dir = find_patient_dir(echo_root, patient_num)
    if patient_dir is None:
        return None
    study_candidates = study_dirs(patient_dir, study_dt)
    base = p.stem if p.suffix else p.name
    return locate_dicom_in_dirs(study_candidates, base)


def to_cropped_path(src_path: Path, echo_root: Path, cropped_root: Path) -> Optional[Path]:
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


def normalize_path(s: str) -> str:
    return str(Path(s)).lower().replace("\\", "/")


def add_gls_from_report(df: pd.DataFrame, report_xlsx: Path) -> pd.DataFrame:
    rep = pd.read_excel(report_xlsx, engine="openpyxl")
    rep.columns = [str(c).strip() for c in rep.columns]

    mapping_full: Dict[str, float] = {}
    mapping_base: Dict[str, float] = {}

    for view in VIEW_KEYS:
        gls_col = find_column(rep, [f"{view}_GLS"], required=False)
        src_col = find_column(rep, [f"{view}_GLS_SOURCE_DICOM"], required=False)
        if not gls_col or not src_col:
            continue
        for _, row in rep.iterrows():
            src = row.get(src_col)
            gls = row.get(gls_col)
            if pd.isna(src) or pd.isna(gls):
                continue
            try:
                gls_val = float(gls)
            except Exception:
                continue
            src_str = str(src)
            mapping_full[normalize_path(src_str)] = gls_val
            mapping_base[Path(src_str).name.lower()] = gls_val

    if "source_dicom" not in df.columns:
        raise ValueError("Embeddings dataframe missing 'source_dicom' column needed for GLS join.")

    gls_vals: List[Optional[float]] = []
    for _, row in df.iterrows():
        src = row.get("source_dicom")
        if pd.isna(src):
            gls_vals.append(None)
            continue
        src_str = str(src)
        key_full = normalize_path(src_str)
        gls_val = mapping_full.get(key_full)
        if gls_val is None:
            gls_val = mapping_base.get(Path(src_str).name.lower())
        gls_vals.append(gls_val)

    df = df.copy()
    df["gls"] = gls_vals
    return df


def resolve_patient_column(df: pd.DataFrame) -> Optional[str]:
    return find_column(df, PATIENT_COL_CANDIDATES, required=False)


def resolve_visit_column(df: pd.DataFrame, override: Optional[str] = None) -> Optional[str]:
    if override:
        if override in df.columns:
            return override
        raise ValueError(f"Visit column '{override}' not found in dataframe")
    return find_column(df, VISIT_COL_CANDIDATES, required=False)


def resolve_time_column(df: pd.DataFrame, override: Optional[str] = None) -> Optional[str]:
    if override:
        if override in df.columns:
            return override
        raise ValueError(f"Time column '{override}' not found in dataframe")
    return find_column(df, TIME_COL_CANDIDATES, required=False)


def coerce_time_values(series: pd.Series) -> Optional[pd.Series]:
    if series is None:
        return None
    if pd.api.types.is_numeric_dtype(series):
        return series
    try:
        out = pd.to_datetime(series, errors="coerce")
        return out
    except Exception:
        return None


def load_cropped_frames(dicom_path: Path) -> Optional[np.ndarray]:
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
        return arr[None, ..., None]
    if arr.ndim == 3:
        if int(getattr(ds, "SamplesPerPixel", 1)) > 1:
            return arr[None, ...]
        return arr[..., None]
    if arr.ndim == 4:
        return arr
    logger.warning(f"Unexpected pixel array shape {arr.shape} for {dicom_path}")
    return None


def sample_indices(n_frames: int, target: int = 16) -> np.ndarray:
    if n_frames <= 0:
        return np.array([], dtype=int)
    if n_frames == target:
        return np.arange(n_frames, dtype=int)
    idx = np.linspace(0, n_frames - 1, target)
    return np.clip(np.round(idx).astype(int), 0, n_frames - 1)


def clip_indices_uniform(n_frames: int, clip_length: int) -> List[np.ndarray]:
    idx = sample_indices(n_frames, target=clip_length)
    if idx.size == 0:
        return []
    return [idx]


def clip_indices_sliding(n_frames: int, clip_length: int, stride: int, include_last: bool) -> List[np.ndarray]:
    if n_frames <= 0:
        return []
    if n_frames < clip_length:
        return clip_indices_uniform(n_frames, clip_length)
    stride = max(int(stride), 1)
    last_start = n_frames - clip_length
    starts = list(range(0, last_start + 1, stride))
    if include_last and starts and starts[-1] != last_start:
        starts.append(last_start)
    return [np.arange(start, start + clip_length, dtype=int) for start in starts]


def to_tensor(frames: np.ndarray) -> torch.Tensor:
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


def resize_tensor(tensor: torch.Tensor, size: int = 224) -> torch.Tensor:
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.shape[-1] == size and tensor.shape[-2] == size:
        return tensor
    return torch.nn.functional.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)


def iter_cropped_frames(
    dicom_path: Path,
    sampling_mode: str,
    clip_length: int,
    clip_stride: int,
    include_last: bool,
) -> Optional[List[Dict[str, object]]]:
    frames = load_cropped_frames(dicom_path)
    if frames is None:
        return None
    n_frames = int(frames.shape[0])
    if sampling_mode == "uniform":
        indices_list = clip_indices_uniform(n_frames, clip_length)
    elif sampling_mode == "sliding_window":
        indices_list = clip_indices_sliding(n_frames, clip_length, clip_stride, include_last)
    else:
        raise ValueError(f"Unknown sampling mode: {sampling_mode}")
    if not indices_list:
        return None
    items: List[Dict[str, object]] = []
    for indices in indices_list:
        for idx in indices:
            idx_int = int(idx)
            frame = frames[idx_int : idx_int + 1]
            tensor = to_tensor(frame)
            time_norm = float(idx_int / (n_frames - 1)) if n_frames > 1 else 0.0
            items.append(
                {
                    "tensor": tensor,
                    "frame_index": idx_int,
                    "frame_count": int(n_frames),
                    "frame_time": time_norm,
                }
            )
    return items


def configure_pydicom_handlers(safe_decode: bool) -> None:
    if not safe_decode:
        return
    try:
        from pydicom import config
        if hasattr(config, "use_gdcm"):
            config.use_gdcm = False
        if hasattr(config, "use_pylibjpeg"):
            config.use_pylibjpeg = False
    except Exception:
        return


def collect_dicoms_from_report(
    df: pd.DataFrame,
    echo_root: Path,
    views: Optional[set] = None,
) -> List[DicomEntry]:
    col_patient = find_column(df, ["PatientNum", "Patient Num", "Patient Number"], required=False)
    col_id = find_column(df, ["ID", "Patient ID", "Id"], required=False)
    col_date = find_column(df, ["Study Date", "StudyDate", "Date"], required=False)
    col_a2c = find_column(df, ["A2C_GLS_SOURCE_DICOM"], required=False)
    col_a3c = find_column(df, ["A3C_GLS_SOURCE_DICOM"], required=False)
    col_a4c = find_column(df, ["A4C_GLS_SOURCE_DICOM"], required=False)
    col_2c = find_column(df, ["2-Chambers", "2C", "2 Chambers"], required=False)
    col_3c = find_column(df, ["3-Chambers", "3C", "3 Chambers"], required=False)
    col_4c = find_column(df, ["4-Chambers", "4C", "4 Chambers"], required=False)
    col_ed = {
        "A2C": find_column(df, ["A2C_END_DIASTOLE_FRAME"], required=False),
        "A3C": find_column(df, ["A3C_END_DIASTOLE_FRAME"], required=False),
        "A4C": find_column(df, ["A4C_END_DIASTOLE_FRAME"], required=False),
    }
    col_es = {
        "A2C": find_column(df, ["A2C_END_SYSTOLE_FRAME"], required=False),
        "A3C": find_column(df, ["A3C_END_SYSTOLE_FRAME"], required=False),
        "A4C": find_column(df, ["A4C_END_SYSTOLE_FRAME"], required=False),
    }

    dicoms: Dict[str, DicomEntry] = {}

    def add_entry(
        p: Optional[Path],
        view: Optional[str],
        patient_num: Optional[str],
        patient_id: Optional[str],
        study_dt: Optional[pd.Timestamp],
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
                patient_num=patient_num,
                patient_id=patient_id,
                study_datetime=study_dt,
                end_diastole=end_diastole,
                end_systole=end_systole,
            )
            return
        if entry.view is None and view is not None:
            entry.view = view
        if entry.patient_num is None and patient_num is not None:
            entry.patient_num = patient_num
        if entry.patient_id is None and patient_id is not None:
            entry.patient_id = patient_id
        if entry.study_datetime is None and study_dt is not None:
            entry.study_datetime = study_dt
        if entry.end_diastole is None and end_diastole is not None:
            entry.end_diastole = end_diastole
        if entry.end_systole is None and end_systole is not None:
            entry.end_systole = end_systole

    for _, row in df.iterrows():
        patient_num = str(row[col_patient]).strip() if col_patient and not pd.isna(row[col_patient]) else None
        patient_id = str(row[col_id]).strip() if col_id and not pd.isna(row[col_id]) else None
        study_dt = parse_datetime(row[col_date]) if col_date else None

        view_frames = {}
        for view in ("A2C", "A3C", "A4C"):
            ed = int(row[col_ed[view]]) if col_ed[view] and not pd.isna(row[col_ed[view]]) else None
            es = int(row[col_es[view]]) if col_es[view] and not pd.isna(row[col_es[view]]) else None
            view_frames[view] = (ed, es)

        if col_a2c and (views is None or "A2C" in views):
            p = resolve_dicom_path(row[col_a2c], patient_num, study_dt, echo_root)
            ed, es = view_frames["A2C"]
            add_entry(p, "A2C", patient_num, patient_id, study_dt, ed, es)
        if col_a3c and (views is None or "A3C" in views):
            p = resolve_dicom_path(row[col_a3c], patient_num, study_dt, echo_root)
            ed, es = view_frames["A3C"]
            add_entry(p, "A3C", patient_num, patient_id, study_dt, ed, es)
        if col_a4c and (views is None or "A4C" in views):
            p = resolve_dicom_path(row[col_a4c], patient_num, study_dt, echo_root)
            ed, es = view_frames["A4C"]
            add_entry(p, "A4C", patient_num, patient_id, study_dt, ed, es)

        if col_2c and (views is None or "A2C" in views):
            ed, es = view_frames["A2C"]
            for base in split_dicom_list(row[col_2c]):
                p = resolve_dicom_path(base, patient_num, study_dt, echo_root)
                add_entry(p, "A2C", patient_num, patient_id, study_dt, ed, es)
        if col_3c and (views is None or "A3C" in views):
            ed, es = view_frames["A3C"]
            for base in split_dicom_list(row[col_3c]):
                p = resolve_dicom_path(base, patient_num, study_dt, echo_root)
                add_entry(p, "A3C", patient_num, patient_id, study_dt, ed, es)
        if col_4c and (views is None or "A4C" in views):
            ed, es = view_frames["A4C"]
            for base in split_dicom_list(row[col_4c]):
                p = resolve_dicom_path(base, patient_num, study_dt, echo_root)
                add_entry(p, "A4C", patient_num, patient_id, study_dt, ed, es)

    return list(dicoms.values())
