"""
Shared helpers for Ichilov pipeline5 strain-curve prediction.

The report exports are not assumed to be stable. Helpers here infer column
names from common variants, normalize wide per-view report columns into
DICOM/view rows, parse strain curve blobs, and compute lightweight quality
heuristics.
"""
from __future__ import annotations

import ast
import json
import logging
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ichilov_pipeline2_utils import (
    VIEW_KEYS,
    parse_datetime,
    parse_views,
    resolve_dicom_path,
    to_cropped_path,
)

logger = logging.getLogger("ichilov_strain_curve_utils")


PATIENT_ID_CANDIDATES = (
    "ID",
    "PatientID",
    "Patient ID",
    "patient_id",
    "patient id",
    "MRN",
)
PATIENT_NUM_CANDIDATES = (
    "PatientNum",
    "Patient Num",
    "Patient Number",
    "patient_num",
    "patient number",
)
DATE_CANDIDATES = (
    "Study Date",
    "StudyDate",
    "Visit Date",
    "visit_date",
    "Exam Date",
    "Date",
)
VIEW_CANDIDATES = ("View", "view", "Projection", "Apical View")
DICOM_CANDIDATES = (
    "DICOM",
    "DICOM Path",
    "dicom_path",
    "source_dicom",
    "Source DICOM",
    "GLS_SOURCE_DICOM",
)
ED_CANDIDATES = (
    "ED_frame",
    "ED Frame",
    "ED",
    "end_diastole",
    "end_diastole_frame",
    "END_DIASTOLE_FRAME",
)
ES_CANDIDATES = (
    "ES_frame",
    "ES Frame",
    "ES",
    "end_systole",
    "end_systole_frame",
    "END_SYSTOLE_FRAME",
)
GLS_CANDIDATES = (
    "GLS",
    "gls",
    "peak_gls",
    "gls_peak",
    "Global Longitudinal Strain",
)
STRAIN_CURVE_CANDIDATES = (
    "STRAIN_CURVES_JSON",
    "strain_curve",
    "strain_curve_json",
    "curve",
    "global_strain_curve",
)


def is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def clean_string(value: object) -> Optional[str]:
    if is_missing(value):
        return None
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and math.isfinite(float(value)):
        f = float(value)
        if f.is_integer():
            return str(int(f))
    s = str(value).strip()
    return s or None


def to_float(value: object) -> Optional[float]:
    if is_missing(value):
        return None
    if isinstance(value, (int, float, np.number)):
        if math.isfinite(float(value)):
            return float(value)
        return None
    s = str(value).strip().replace(",", ".")
    try:
        out = float(s)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def to_int(value: object) -> Optional[int]:
    f = to_float(value)
    if f is None:
        return None
    return int(round(f))


def normalize_column_name(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def find_column(
    df: pd.DataFrame,
    candidates: Sequence[str],
    configured: Optional[str] = None,
) -> Optional[str]:
    if configured:
        configured = str(configured).strip()
        if configured in df.columns:
            return configured
        lower = {str(c).strip().lower(): c for c in df.columns}
        if configured.lower() in lower:
            return lower[configured.lower()]
        logger.warning("Configured column '%s' not found.", configured)

    exact = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in exact:
            return exact[key]

    norm_map = {normalize_column_name(c): c for c in df.columns}
    for cand in candidates:
        key = normalize_column_name(cand)
        if key in norm_map:
            return norm_map[key]

    for cand in candidates:
        key = normalize_column_name(cand)
        if len(key) < 3:
            continue
        for col in df.columns:
            if key and key in normalize_column_name(col):
                return col
    return None


def parse_views_list(value: Optional[str]) -> List[str]:
    parsed = parse_views(value)
    if not parsed:
        return list(VIEW_KEYS)
    return [v for v in VIEW_KEYS if v in parsed]


def canonical_view(value: object) -> Optional[str]:
    s = clean_string(value)
    if not s:
        return None
    u = s.upper().replace("-", "").replace("_", "").replace(" ", "")
    if u in {"A2C", "2C", "2CH", "2CHAMBER", "2CHAMBERS"}:
        return "A2C"
    if u in {"A3C", "3C", "3CH", "3CHAMBER", "3CHAMBERS"}:
        return "A3C"
    if u in {"A4C", "4C", "4CH", "4CHAMBER", "4CHAMBERS"}:
        return "A4C"
    for view in VIEW_KEYS:
        if view in u:
            return view
    return None


def normalize_path_key(value: object) -> Optional[str]:
    s = clean_string(value)
    if not s:
        return None
    s = os.path.expandvars(s)
    return str(Path(s)).replace("\\", "/").lower()


def basename_key(value: object) -> Optional[str]:
    s = clean_string(value)
    if not s:
        return None
    return Path(s).name.lower()


def stem_key(value: object) -> Optional[str]:
    s = clean_string(value)
    if not s:
        return None
    return Path(s).stem.lower()


def visit_key(value: object) -> Optional[str]:
    if is_missing(value):
        return None
    dt = parse_datetime(value)
    if dt is not None:
        return dt.strftime("%Y-%m-%d")
    return clean_string(value)


def patient_key(patient_num: object = None, patient_id: object = None) -> Optional[str]:
    pn = clean_string(patient_num)
    if pn:
        return pn
    pid = clean_string(patient_id)
    if pid:
        return pid
    return None


def read_excel_first_sheet(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _configured(column_map: Mapping[str, Any], key: str) -> Optional[str]:
    val = column_map.get(key)
    if val is None:
        return None
    s = str(val).strip()
    return s or None


def infer_report_columns(
    ed_df: Optional[pd.DataFrame] = None,
    strain_df: Optional[pd.DataFrame] = None,
    column_map: Optional[Mapping[str, Any]] = None,
    views: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Infer report column mapping while preserving configured overrides."""
    cm = dict(column_map or {})
    selected_views = list(views or VIEW_KEYS)
    resolved: Dict[str, Any] = {
        "input_column_map": cm,
        "ed_es": {},
        "strain": {"format": "unknown", "views": {}},
        "common": {},
    }

    for label, df in (("ed_es", ed_df), ("strain", strain_df)):
        if df is None:
            continue
        resolved[label]["patient_id"] = find_column(
            df, PATIENT_ID_CANDIDATES, _configured(cm, "patient_id")
        )
        resolved[label]["patient_num"] = find_column(
            df, PATIENT_NUM_CANDIDATES, _configured(cm, "patient_num")
        )
        resolved[label]["visit_date"] = find_column(
            df, DATE_CANDIDATES, _configured(cm, "visit_date")
        )
        resolved[label]["view"] = find_column(
            df, VIEW_CANDIDATES, _configured(cm, "view")
        )

    if ed_df is not None:
        resolved["ed_es"].update(
            {
                "dicom_path": find_column(ed_df, DICOM_CANDIDATES, _configured(cm, "dicom_path")),
                "dicom_name": find_column(ed_df, ("DICOM", "DICOM Name", "dicom_name"), _configured(cm, "dicom_name")),
                "ed_index": find_column(ed_df, ED_CANDIDATES, _configured(cm, "ed_index")),
                "es_index": find_column(ed_df, ES_CANDIDATES, _configured(cm, "es_index")),
                "num_frames": find_column(ed_df, ("NumFrames", "NumberOfFrames", "n_frames", "FrameCount")),
                "frame_time_ms": find_column(ed_df, ("FrameTime_ms", "FrameTime", "frame_time_ms")),
            }
        )
        sample_cols = [c for c in ed_df.columns if re.match(r"(?i)^sample[_\s-]*\d+", str(c).strip())]
        resolved["ed_es"]["sample_index_columns"] = sample_cols

    if strain_df is not None:
        view_specific_hits = 0
        for view in selected_views:
            view_map = {
                "dicom_path": find_column(
                    strain_df,
                    (
                        f"{view}_GLS_SOURCE_DICOM",
                        f"{view}_SOURCE_DICOM",
                        f"{view}_DICOM",
                        f"{view}_GLS_ANALYSIS_DICOM",
                    ),
                ),
                "gls_peak": find_column(
                    strain_df,
                    (f"{view}_GLS", f"{view}_PEAK_GLS", f"{view}_GLS_PEAK"),
                ),
                "strain_curve": find_column(
                    strain_df,
                    (
                        f"{view}_STRAIN_CURVES_JSON",
                        f"{view}_STRAIN_CURVE_JSON",
                        f"{view}_STRAIN_CURVE",
                        f"{view}_CURVE",
                    ),
                ),
                "ed_index": find_column(
                    strain_df,
                    (
                        f"{view}_END_DIASTOLE_FRAME",
                        f"{view}_ED_FRAME",
                        f"{view}_ED",
                    ),
                ),
                "es_index": find_column(
                    strain_df,
                    (
                        f"{view}_END_SYSTOLE_FRAME",
                        f"{view}_ES_FRAME",
                        f"{view}_ES",
                    ),
                ),
            }
            view_specific_hits += sum(1 for v in view_map.values() if v)
            resolved["strain"]["views"][view] = view_map

        generic_curve = find_column(
            strain_df,
            STRAIN_CURVE_CANDIDATES,
            _configured(cm, "strain_curve"),
        )
        generic_gls = find_column(strain_df, GLS_CANDIDATES, _configured(cm, "gls_peak"))
        generic_dicom = find_column(
            strain_df,
            DICOM_CANDIDATES,
            _configured(cm, "dicom_path") or _configured(cm, "dicom_name"),
        )
        resolved["strain"].update(
            {
                "dicom_path": generic_dicom,
                "gls_peak": generic_gls,
                "strain_curve": generic_curve,
                "ed_index": find_column(strain_df, ED_CANDIDATES, _configured(cm, "ed_index")),
                "es_index": find_column(strain_df, ES_CANDIDATES, _configured(cm, "es_index")),
            }
        )
        resolved["strain"]["format"] = "wide_by_view" if view_specific_hits else "long"

        if generic_curve is None:
            time_cols = []
            for col in strain_df.columns:
                name = str(col).strip().lower()
                if re.match(r"^(strain|curve|sample|time)[_\s-]*\d+$", name):
                    time_cols.append(col)
            resolved["strain"]["timepoint_columns"] = time_cols
        else:
            resolved["strain"]["timepoint_columns"] = []

    common: Dict[str, Any] = {}
    for key in ("patient_id", "patient_num", "visit_date"):
        common[key] = (
            (resolved.get("strain") or {}).get(key)
            or (resolved.get("ed_es") or {}).get(key)
        )
    resolved["common"] = common
    return resolved


def _resolve_path(
    value: object,
    patient_num_value: object,
    visit_date_value: object,
    echo_root: Optional[Path],
) -> Optional[str]:
    s = clean_string(value)
    if not s:
        return None
    p = Path(os.path.expandvars(s)).expanduser()
    if p.is_file():
        return str(p)
    if echo_root is not None:
        dt = parse_datetime(visit_date_value)
        resolved = resolve_dicom_path(s, clean_string(patient_num_value), dt, echo_root)
        if resolved is not None:
            return str(resolved)
    return s


def _curve_from_numeric_columns(row: pd.Series, cols: Sequence[str]) -> Optional[List[float]]:
    vals = [to_float(row.get(c)) for c in cols]
    vals = [v for v in vals if v is not None]
    return vals if len(vals) >= 2 else None


def strain_report_to_view_rows(
    df: pd.DataFrame,
    mapping: Mapping[str, Any],
    echo_root: Optional[Path] = None,
    views: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    selected_views = list(views or VIEW_KEYS)
    strain_map = mapping.get("strain", {})
    fmt = strain_map.get("format")
    rows: List[Dict[str, Any]] = []
    patient_num_col = strain_map.get("patient_num")
    patient_id_col = strain_map.get("patient_id")
    visit_col = strain_map.get("visit_date")

    if fmt == "wide_by_view":
        for idx, row in df.iterrows():
            pn = row.get(patient_num_col) if patient_num_col else None
            pid = row.get(patient_id_col) if patient_id_col else None
            visit = row.get(visit_col) if visit_col else None
            pkey = patient_key(pn, pid)
            vkey = visit_key(visit)
            for view in selected_views:
                vm = strain_map.get("views", {}).get(view, {})
                curve_col = vm.get("strain_curve")
                gls_col = vm.get("gls_peak")
                dicom_col = vm.get("dicom_path")
                ed_col = vm.get("ed_index")
                es_col = vm.get("es_index")
                curve_raw = row.get(curve_col) if curve_col else None
                gls = to_float(row.get(gls_col)) if gls_col else None
                dicom_raw = row.get(dicom_col) if dicom_col else None
                if is_missing(curve_raw) and gls is None and is_missing(dicom_raw):
                    continue
                dicom_path = _resolve_path(dicom_raw, pn, visit, echo_root)
                rows.append(
                    {
                        "source_row_index": int(idx),
                        "patient_num": clean_string(pn),
                        "patient_id": clean_string(pid),
                        "patient_key": pkey,
                        "visit_date": vkey,
                        "view": view,
                        "dicom_path": dicom_path,
                        "dicom_name": basename_key(dicom_path or dicom_raw),
                        "peak_gls_from_report": gls,
                        "strain_curve_raw": curve_raw,
                        "strain_ed_index": to_int(row.get(ed_col)) if ed_col else None,
                        "strain_es_index": to_int(row.get(es_col)) if es_col else None,
                    }
                )
        return pd.DataFrame(rows)

    view_col = strain_map.get("view")
    dicom_col = strain_map.get("dicom_path")
    curve_col = strain_map.get("strain_curve")
    gls_col = strain_map.get("gls_peak")
    ed_col = strain_map.get("ed_index")
    es_col = strain_map.get("es_index")
    time_cols = strain_map.get("timepoint_columns") or []
    for idx, row in df.iterrows():
        view = canonical_view(row.get(view_col)) if view_col else None
        if view is None and selected_views:
            continue
        if view not in selected_views:
            continue
        pn = row.get(patient_num_col) if patient_num_col else None
        pid = row.get(patient_id_col) if patient_id_col else None
        visit = row.get(visit_col) if visit_col else None
        curve_raw = row.get(curve_col) if curve_col else _curve_from_numeric_columns(row, time_cols)
        dicom_raw = row.get(dicom_col) if dicom_col else None
        dicom_path = _resolve_path(dicom_raw, pn, visit, echo_root)
        rows.append(
            {
                "source_row_index": int(idx),
                "patient_num": clean_string(pn),
                "patient_id": clean_string(pid),
                "patient_key": patient_key(pn, pid),
                "visit_date": visit_key(visit),
                "view": view,
                "dicom_path": dicom_path,
                "dicom_name": basename_key(dicom_path or dicom_raw),
                "peak_gls_from_report": to_float(row.get(gls_col)) if gls_col else None,
                "strain_curve_raw": curve_raw,
                "strain_ed_index": to_int(row.get(ed_col)) if ed_col else None,
                "strain_es_index": to_int(row.get(es_col)) if es_col else None,
            }
        )
    return pd.DataFrame(rows)


def ed_es_report_to_rows(
    df: pd.DataFrame,
    mapping: Mapping[str, Any],
    echo_root: Optional[Path] = None,
    views: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    selected_views = set(views or VIEW_KEYS)
    ed_map = mapping.get("ed_es", {})
    rows: List[Dict[str, Any]] = []
    patient_num_col = ed_map.get("patient_num")
    patient_id_col = ed_map.get("patient_id")
    visit_col = ed_map.get("visit_date")
    view_col = ed_map.get("view")
    dicom_col = ed_map.get("dicom_path") or ed_map.get("dicom_name")
    ed_col = ed_map.get("ed_index")
    es_col = ed_map.get("es_index")
    nf_col = ed_map.get("num_frames")
    ft_col = ed_map.get("frame_time_ms")
    sample_cols = ed_map.get("sample_index_columns") or []

    for idx, row in df.iterrows():
        view = canonical_view(row.get(view_col)) if view_col else None
        if view and view not in selected_views:
            continue
        pn = row.get(patient_num_col) if patient_num_col else None
        pid = row.get(patient_id_col) if patient_id_col else None
        visit = row.get(visit_col) if visit_col else None
        dicom_raw = row.get(dicom_col) if dicom_col else None
        dicom_path = _resolve_path(dicom_raw, pn, visit, echo_root)
        sample_indices = [to_int(row.get(c)) for c in sample_cols]
        sample_indices = [int(v) for v in sample_indices if v is not None]
        rows.append(
            {
                "ed_es_row_index": int(idx),
                "patient_num_ed": clean_string(pn),
                "patient_id_ed": clean_string(pid),
                "patient_key_ed": patient_key(pn, pid),
                "visit_date_ed": visit_key(visit),
                "view_ed": view,
                "dicom_path_ed": dicom_path,
                "dicom_name_ed": basename_key(dicom_path or dicom_raw),
                "ed_index": to_int(row.get(ed_col)) if ed_col else None,
                "es_index": to_int(row.get(es_col)) if es_col else None,
                "n_frames_report": to_int(row.get(nf_col)) if nf_col else None,
                "frame_time_ms_report": to_float(row.get(ft_col)) if ft_col else None,
                "phase_sample_indices_report": sample_indices,
            }
        )
    return pd.DataFrame(rows)


def _extract_curves_from_obj(obj: object) -> List[np.ndarray]:
    if obj is None:
        return []
    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return []
        if obj.ndim == 1:
            return [obj.astype(float, copy=False)]
        if obj.ndim == 2 and obj.shape[1] == 2 and obj.shape[0] > 2:
            return [obj[:, 1].astype(float, copy=False)]
        return [np.asarray(x, dtype=float).reshape(-1) for x in obj]
    if isinstance(obj, (list, tuple)):
        if not obj:
            return []
        try:
            arr = np.asarray(obj, dtype=float)
            return _extract_curves_from_obj(arr)
        except Exception:
            curves: List[np.ndarray] = []
            for item in obj:
                curves.extend(_extract_curves_from_obj(item))
            return curves
    if isinstance(obj, dict):
        for key in ("curves", "curve", "values", "value", "strain", "data"):
            if key in obj:
                out = _extract_curves_from_obj(obj[key])
                if out:
                    return out
        curves = []
        for val in obj.values():
            curves.extend(_extract_curves_from_obj(val))
        return curves
    if isinstance(obj, (int, float, np.number)):
        return [np.asarray([float(obj)], dtype=float)]
    return []


def parse_curve_blob(value: object) -> List[np.ndarray]:
    if is_missing(value):
        return []
    if isinstance(value, (list, tuple, dict, np.ndarray)):
        return _extract_curves_from_obj(value)
    text = str(value).strip()
    if not text:
        return []
    parsed: object
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            return _extract_curves_from_obj(parsed)
        except Exception:
            pass
    numbers = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", text)
    if len(numbers) < 2:
        return []
    return [np.asarray([float(x) for x in numbers], dtype=float)]


def resample_curve(curve: np.ndarray, length: int) -> np.ndarray:
    arr = np.asarray(curve, dtype=float).reshape(-1)
    if length <= 0:
        raise ValueError("curve length must be positive")
    if arr.size == 0:
        return np.full(length, np.nan, dtype=float)
    finite = np.isfinite(arr)
    if finite.sum() == 0:
        return np.full(length, np.nan, dtype=float)
    if finite.sum() < arr.size:
        x = np.arange(arr.size)
        arr = np.interp(x, x[finite], arr[finite])
    if arr.size == length:
        return arr.astype(float, copy=True)
    if arr.size == 1:
        return np.full(length, float(arr[0]), dtype=float)
    x_old = np.linspace(0.0, 1.0, arr.size)
    x_new = np.linspace(0.0, 1.0, length)
    return np.interp(x_new, x_old, arr).astype(float)


def reduce_global_curve(curves: Sequence[np.ndarray]) -> Optional[np.ndarray]:
    valid = []
    lengths = []
    for curve in curves:
        arr = np.asarray(curve, dtype=float).reshape(-1)
        if arr.size < 2:
            continue
        if np.isfinite(arr).sum() < 2:
            continue
        valid.append(arr)
        lengths.append(arr.size)
    if not valid:
        return None
    target = int(round(float(np.median(lengths))))
    target = max(target, 2)
    stacked = np.stack([resample_curve(c, target) for c in valid], axis=0)
    return np.nanmean(stacked, axis=0)


def parse_global_curve(value: object) -> Tuple[Optional[np.ndarray], List[List[float]]]:
    curves = parse_curve_blob(value)
    serializable = []
    for c in curves:
        arr = np.asarray(c, dtype=float).reshape(-1)
        if arr.size >= 2:
            serializable.append([float(x) if np.isfinite(x) else None for x in arr])
    return reduce_global_curve(curves), serializable


def curve_peak_and_time(curve: object) -> Tuple[Optional[float], Optional[float], Optional[int]]:
    arr = np.asarray(curve, dtype=float).reshape(-1)
    finite = np.isfinite(arr)
    if arr.size == 0 or finite.sum() == 0:
        return None, None, None
    masked = arr.copy()
    masked[~finite] = np.inf
    idx = int(np.argmin(masked))
    peak = float(masked[idx])
    ttp = float(idx / (arr.size - 1)) if arr.size > 1 else 0.0
    return peak, ttp, idx


def local_minima_count(curve: np.ndarray, threshold: float = -5.0) -> int:
    arr = np.asarray(curve, dtype=float).reshape(-1)
    if arr.size < 3:
        return 0
    count = 0
    for i in range(1, arr.size - 1):
        if not np.isfinite(arr[i - 1 : i + 2]).all():
            continue
        if arr[i] <= arr[i - 1] and arr[i] <= arr[i + 1] and arr[i] <= threshold:
            count += 1
    return count


def curve_quality_heuristics(curve: object) -> Dict[str, Any]:
    arr = np.asarray(curve, dtype=float).reshape(-1)
    if arr.size == 0:
        return {
            "starts_near_zero": False,
            "has_valid_peak": False,
            "peak_in_reasonable_range": False,
            "excessive_noise_score": None,
            "num_large_peaks": 0,
            "curve_nan_fraction": 1.0,
        }
    finite = np.isfinite(arr)
    nan_fraction = float(1.0 - finite.mean()) if arr.size else 1.0
    if finite.sum() == 0:
        return {
            "starts_near_zero": False,
            "has_valid_peak": False,
            "peak_in_reasonable_range": False,
            "excessive_noise_score": None,
            "num_large_peaks": 0,
            "curve_nan_fraction": nan_fraction,
        }
    filled = arr.copy()
    if finite.sum() < arr.size:
        x = np.arange(arr.size)
        filled = np.interp(x, x[finite], arr[finite])
    peak, _, _ = curve_peak_and_time(filled)
    first = float(filled[0])
    curve_range = float(np.nanmax(filled) - np.nanmin(filled))
    if filled.size >= 3:
        noise = float(np.mean(np.abs(np.diff(filled, n=2))) / max(curve_range, 1.0))
    else:
        noise = 0.0
    has_valid_peak = peak is not None and peak < -1.0
    peak_reasonable = peak is not None and -35.0 <= peak <= -5.0
    return {
        "starts_near_zero": bool(abs(first) <= 3.0),
        "has_valid_peak": bool(has_valid_peak),
        "peak_in_reasonable_range": bool(peak_reasonable),
        "excessive_noise_score": noise,
        "num_large_peaks": int(local_minima_count(filled, threshold=-5.0)),
        "curve_nan_fraction": nan_fraction,
    }


def read_dicom_basic_metadata(path: object) -> Dict[str, Any]:
    p = clean_string(path)
    out = {"n_frames": None, "frame_time_ms": None, "heart_rate": None}
    if not p:
        return out
    try:
        import pydicom

        ds = pydicom.dcmread(p, stop_before_pixels=True, force=True)
    except Exception as exc:
        logger.warning("Failed to read DICOM metadata %s: %s", p, exc)
        return out
    out["n_frames"] = to_int(getattr(ds, "NumberOfFrames", None))
    out["frame_time_ms"] = to_float(getattr(ds, "FrameTime", None))
    if out["frame_time_ms"] is None:
        cine_rate = to_float(getattr(ds, "CineRate", None))
        if cine_rate and cine_rate > 0:
            out["frame_time_ms"] = 1000.0 / cine_rate
    out["heart_rate"] = to_float(getattr(ds, "HeartRate", None))
    return out


def _counter_duplicates(keys: Iterable[Tuple[Any, ...]]) -> Dict[str, int]:
    counts = Counter(k for k in keys if all(x is not None for x in k))
    return {"|".join(map(str, k)): int(v) for k, v in counts.items() if v > 1}


def add_match_keys(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    out = df.copy()
    dicom_col = f"{prefix}dicom_path" if f"{prefix}dicom_path" in out.columns else "dicom_path"
    view_col = f"{prefix}view" if f"{prefix}view" in out.columns else "view"
    patient_col = f"{prefix}patient_key" if f"{prefix}patient_key" in out.columns else "patient_key"
    visit_col = f"{prefix}visit_date" if f"{prefix}visit_date" in out.columns else "visit_date"
    out["path_key"] = out[dicom_col].map(normalize_path_key) if dicom_col in out else None
    out["basename_key"] = out[dicom_col].map(basename_key) if dicom_col in out else None
    out["stem_key"] = out[dicom_col].map(stem_key) if dicom_col in out else None
    if view_col in out:
        out["view_key"] = out[view_col].map(canonical_view)
    else:
        out["view_key"] = None
    out["patient_visit_view_key"] = [
        (
            row.get(patient_col),
            row.get(visit_col),
            row.get("view_key"),
        )
        for _, row in out.iterrows()
    ]
    out["path_view_key"] = [
        (row.get("path_key"), row.get("view_key")) for _, row in out.iterrows()
    ]
    out["basename_patient_visit_view_key"] = [
        (
            row.get("basename_key"),
            row.get(patient_col),
            row.get(visit_col),
            row.get("view_key"),
        )
        for _, row in out.iterrows()
    ]
    return out


def duplicate_key_summary(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    if df.empty:
        return {}
    keyed = add_match_keys(df)
    return {
        "path_view": _counter_duplicates(keyed["path_view_key"]),
        "basename_patient_visit_view": _counter_duplicates(keyed["basename_patient_visit_view_key"]),
        "patient_visit_view": _counter_duplicates(keyed["patient_visit_view_key"]),
    }


def attach_ed_es(strain_rows: pd.DataFrame, ed_rows: pd.DataFrame) -> pd.DataFrame:
    strain = add_match_keys(strain_rows)
    ed = add_match_keys(
        ed_rows.rename(
            columns={
                "dicom_path_ed": "dicom_path",
                "view_ed": "view",
                "patient_key_ed": "patient_key",
                "visit_date_ed": "visit_date",
            }
        )
    )
    ed_records = ed.to_dict("records")

    def unique_map(key_name: str) -> Dict[Tuple[Any, ...], Dict[str, Any]]:
        grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
        for rec in ed_records:
            key = rec.get(key_name)
            if isinstance(key, tuple) and all(x is not None for x in key):
                grouped[key].append(rec)
        return {k: v[0] for k, v in grouped.items() if len(v) == 1}

    by_path = unique_map("path_view_key")
    by_base = unique_map("basename_patient_visit_view_key")
    by_patient_visit = unique_map("patient_visit_view_key")

    merged: List[Dict[str, Any]] = []
    for _, row in strain.iterrows():
        base = row.to_dict()
        match = None
        method = "none"
        for method_name, key_name, lookup in (
            ("path_view", "path_view_key", by_path),
            ("basename_patient_visit_view", "basename_patient_visit_view_key", by_base),
            ("patient_visit_view", "patient_visit_view_key", by_patient_visit),
        ):
            key = base.get(key_name)
            if isinstance(key, tuple) and key in lookup:
                match = lookup[key]
                method = method_name
                break
        if match:
            for key, value in match.items():
                if key in {
                    "path_key",
                    "basename_key",
                    "stem_key",
                    "view_key",
                    "path_view_key",
                    "basename_patient_visit_view_key",
                    "patient_visit_view_key",
                    "dicom_path",
                    "view",
                    "patient_key",
                    "visit_date",
                }:
                    continue
                base[key] = value
            if not base.get("dicom_path") and match.get("dicom_path"):
                base["dicom_path"] = match.get("dicom_path")
        base["ed_es_match_method"] = method
        merged.append(base)
    out = pd.DataFrame(merged)
    drop_cols = [
        "path_key",
        "basename_key",
        "stem_key",
        "view_key",
        "path_view_key",
        "basename_patient_visit_view_key",
        "patient_visit_view_key",
    ]
    return out.drop(columns=[c for c in drop_cols if c in out.columns])


def add_view_disagreement(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["view_gls_disagreement"] = np.nan
    if out.empty:
        return out
    value_col = "peak_gls_from_report"
    if value_col not in out.columns:
        value_col = "peak_gls_from_curve"
    group_cols = [c for c in ("patient_key", "visit_date") if c in out.columns]
    if not group_cols or value_col not in out.columns:
        return out
    for _, idxs in out.groupby(group_cols, dropna=False).groups.items():
        vals = pd.to_numeric(out.loc[idxs, value_col], errors="coerce").dropna().to_numpy()
        if vals.size >= 2:
            out.loc[idxs, "view_gls_disagreement"] = float(np.nanmax(vals) - np.nanmin(vals))
    return out


def json_dumps_compact(value: object) -> str:
    def default(obj: object) -> object:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        if pd.isna(obj):
            return None
        return str(obj)

    return json.dumps(value, default=default, ensure_ascii=False)


def make_sample_id(row: Mapping[str, Any], index: int) -> str:
    parts = [
        clean_string(row.get("patient_key")) or "patient",
        clean_string(row.get("visit_date")) or "visit",
        clean_string(row.get("view")) or "view",
        stem_key(row.get("dicom_path")) or f"row{index}",
    ]
    safe = [re.sub(r"[^A-Za-z0-9._-]+", "-", p) for p in parts]
    return "__".join(safe)


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))


def dataframe_preview_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].map(lambda x: isinstance(x, (list, dict, tuple, np.ndarray))).any():
            out[col] = out[col].map(lambda x: json_dumps_compact(x) if not is_missing(x) else "")
    return out


def summarize_samples(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty:
        return {"n_samples": 0, "n_patients": 0}
    summary = {"n_samples": int(len(df))}
    if "patient_key" in df.columns:
        summary["n_patients"] = int(df["patient_key"].dropna().nunique())
    if "view" in df.columns:
        summary["views"] = {str(k): int(v) for k, v in df["view"].value_counts(dropna=False).items()}
    for col in ("ed_index", "es_index", "resampled_strain_curve", "peak_gls_from_report", "peak_gls_from_curve"):
        if col in df.columns:
            summary[f"missing_{col}"] = int(df[col].map(is_missing).sum())
    return summary
