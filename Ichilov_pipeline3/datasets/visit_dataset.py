"""
Visit-level dataset for pipeline3, built on top of pipeline2 utilities.
"""
from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ichilov_pipeline2_utils import (
    VIEW_KEYS,
    add_gls_from_report,
    clip_indices_sliding,
    collect_dicoms_from_report,
    iter_cropped_frames,
    parse_views,
    phase_aligned_indices,
    resize_tensor,
    sample_indices,
    to_cropped_path,
)

logger = logging.getLogger("ichilov_pipeline3.visit_dataset")

DELTA_GLS_CONVENTION = "latest_minus_earliest"
DELTA_GLS_CONVENTION_CHOICES = {"latest_minus_earliest", "earliest_minus_latest"}
PHASE_ALIGNED_STRATEGY_CHOICES = {"cycle", "segment"}


@dataclass
class VisitRecord:
    visit_id: str
    visit_time_months: float
    study_datetime: Optional[pd.Timestamp]
    gls: Optional[float]
    view_dicoms: Dict[str, List[Path]]
    view_phase: Dict[str, Tuple[Optional[int], Optional[int]]] = field(default_factory=dict)


@dataclass
class PatientRecord:
    patient_id: str
    visits: List[VisitRecord]


def _coerce_optional_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return int(round(float(value)))
    except Exception:
        return None


def _validate_monotonic_visit_times(patient_id: str, visit_times: Sequence[float]) -> None:
    if len(visit_times) < 2:
        return
    diffs = np.diff(np.asarray(visit_times, dtype=np.float64))
    if np.any(diffs < -1e-8):
        raise AssertionError(
            f"Non-monotonic visit_time_months for patient {patient_id}. "
            "Visits must be sorted earliest->latest."
        )
    if np.any(np.isclose(diffs, 0.0, atol=1e-8)):
        logger.warning(
            "Patient %s has repeated visit timestamps (equal month offsets).",
            patient_id,
        )


def _compute_delta_target(
    gls: torch.Tensor,
    gls_mask: torch.Tensor,
    n_visits: int,
    convention: str,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[int], Optional[int]]:
    valid_gls_idx = torch.where(gls_mask[:n_visits])[0]
    delta_gls_target = torch.tensor(0.0, dtype=torch.float32)
    delta_gls_mask = torch.tensor(False, dtype=torch.bool)
    first: Optional[int] = None
    last: Optional[int] = None
    if valid_gls_idx.numel() >= 2:
        first = int(valid_gls_idx[0].item())
        last = int(valid_gls_idx[-1].item())
        if convention == "latest_minus_earliest":
            delta_gls_target = gls[last] - gls[first]
        elif convention == "earliest_minus_latest":
            delta_gls_target = gls[first] - gls[last]
        else:
            raise ValueError(
                f"Unsupported delta_gls convention '{convention}'. "
                f"Expected one of: {sorted(DELTA_GLS_CONVENTION_CHOICES)}"
            )
        delta_gls_mask = torch.tensor(True, dtype=torch.bool)
    return delta_gls_target, delta_gls_mask, first, last


def _sample_embedding_positions(
    n_frames: int,
    t_frames: int,
    sampling_mode: str,
    clip_stride: int,
    include_last_window: bool,
    end_diastole: Optional[int],
    end_systole: Optional[int],
    phase_aligned_include_midpoints: bool,
    phase_aligned_strategy: str,
) -> np.ndarray:
    if n_frames <= 0:
        return np.array([], dtype=np.int64)
    if sampling_mode == "uniform":
        return sample_indices(n_frames, target=t_frames).astype(np.int64)
    if sampling_mode == "sliding_window":
        windows = clip_indices_sliding(
            n_frames=n_frames,
            clip_length=t_frames,
            stride=clip_stride,
            include_last=include_last_window,
        )
        if not windows:
            return np.array([], dtype=np.int64)
        return windows[0].astype(np.int64)
    if sampling_mode == "phase_aligned":
        return phase_aligned_indices(
            n_frames=n_frames,
            clip_length=t_frames,
            end_diastole=end_diastole,
            end_systole=end_systole,
            include_midpoints=phase_aligned_include_midpoints,
            strategy=phase_aligned_strategy,
        ).astype(np.int64)
    raise ValueError(f"Unknown sampling_mode: {sampling_mode}")


def _patient_key(entry: object) -> str:
    pid = getattr(entry, "patient_id", None)
    if pid is not None and str(pid).strip():
        return str(pid).strip()
    pnum = getattr(entry, "patient_num", None)
    if pnum is not None and str(pnum).strip():
        return str(pnum).strip()
    src = getattr(entry, "path", None)
    if src is not None:
        try:
            return str(Path(src).parts[-3])
        except Exception:
            pass
    return "unknown_patient"


def _visit_key(entry: object, fallback_idx: int) -> str:
    study_dt = getattr(entry, "study_datetime", None)
    if isinstance(study_dt, pd.Timestamp) and not pd.isna(study_dt):
        return study_dt.isoformat()
    if study_dt is not None:
        try:
            dt = pd.to_datetime(study_dt, errors="coerce")
            if not pd.isna(dt):
                return dt.isoformat()
        except Exception:
            pass
    src = getattr(entry, "path", None)
    if src is not None:
        parent = Path(src).parent.name
        if parent:
            return parent
    return f"visit_{fallback_idx:04d}"


def _to_months(sorted_dt: Sequence[Optional[pd.Timestamp]]) -> List[float]:
    valid = [dt for dt in sorted_dt if isinstance(dt, pd.Timestamp) and not pd.isna(dt)]
    if not valid:
        return [float(i) for i in range(len(sorted_dt))]
    base = valid[0]
    out: List[float] = []
    for idx, dt in enumerate(sorted_dt):
        if isinstance(dt, pd.Timestamp) and not pd.isna(dt):
            delta_days = (dt - base).total_seconds() / 86400.0
            out.append(float(delta_days / 30.4375))
        else:
            out.append(float(idx))
    return out


def _numeric_hint(val: str) -> float:
    if val is None:
        return float("inf")
    m = re.search(r"[-+]?\d+(\.\d+)?", str(val))
    if not m:
        return float("inf")
    try:
        return float(m.group(0))
    except Exception:
        return float("inf")


def _parse_embedding_vector(value: object) -> Optional[np.ndarray]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, np.ndarray):
        return value.astype(np.float32)
    if isinstance(value, list):
        return np.asarray(value, dtype=np.float32)
    if isinstance(value, str):
        try:
            parsed = eval(value, {"__builtins__": {}})
            return np.asarray(parsed, dtype=np.float32)
        except Exception:
            return None
    return None


class VisitDataset(Dataset):
    """
    Dataset item is one patient with up to max_visits visits.
    """

    def __init__(
        self,
        input_xlsx: Path,
        echo_root: Path,
        cropped_root: Optional[Path],
        views: str = "",
        t_frames: int = 16,
        sampling_mode: str = "uniform",
        clip_stride: int = 1,
        include_last_window: bool = True,
        max_visits: int = 5,
        min_visits: int = 2,
        image_size: int = 224,
        risk_delta_threshold: float = 2.0,
        patient_filter: Optional[Sequence[str]] = None,
        random_view_sampling: bool = False,
        phase_aligned_include_midpoints: bool = True,
        phase_aligned_strategy: str = "cycle",
        delta_gls_convention: str = DELTA_GLS_CONVENTION,
    ) -> None:
        super().__init__()
        self.input_xlsx = Path(input_xlsx)
        self.echo_root = Path(echo_root)
        self.cropped_root = Path(cropped_root) if cropped_root is not None else None
        self.selected_views = parse_views(views) or set(VIEW_KEYS)
        self.t_frames = int(t_frames)
        self.sampling_mode = str(sampling_mode)
        self.clip_stride = int(clip_stride)
        self.include_last_window = bool(include_last_window)
        self.phase_aligned_include_midpoints = bool(phase_aligned_include_midpoints)
        self.phase_aligned_strategy = str(phase_aligned_strategy).strip().lower()
        self.max_visits = int(max_visits)
        self.min_visits = int(min_visits)
        self.image_size = int(image_size)
        self.risk_delta_threshold = float(risk_delta_threshold)
        self.delta_gls_convention = str(delta_gls_convention).strip().lower()
        self.patient_filter = set(str(x) for x in patient_filter) if patient_filter else None
        self.random_view_sampling = bool(random_view_sampling)

        if self.sampling_mode not in {"uniform", "sliding_window", "phase_aligned"}:
            raise ValueError("sampling_mode must be one of: uniform, sliding_window, phase_aligned.")
        if self.phase_aligned_strategy not in PHASE_ALIGNED_STRATEGY_CHOICES:
            raise ValueError(
                f"phase_aligned_strategy must be one of: {sorted(PHASE_ALIGNED_STRATEGY_CHOICES)}"
            )
        if self.delta_gls_convention not in DELTA_GLS_CONVENTION_CHOICES:
            raise ValueError(
                f"delta_gls_convention must be one of: {sorted(DELTA_GLS_CONVENTION_CHOICES)}"
            )

        self.patient_records = self._build_patient_records()
        if not self.patient_records:
            raise RuntimeError("No valid patient records found for VisitDataset.")

    def _build_patient_records(self) -> List[PatientRecord]:
        report_df = pd.read_excel(self.input_xlsx, engine="openpyxl")
        report_df.columns = [str(c).strip() for c in report_df.columns]

        entries = collect_dicoms_from_report(
            report_df,
            self.echo_root,
            views=self.selected_views,
        )
        logger.info("Collected %d source DICOM entries from report.", len(entries))

        rows: List[dict] = []
        for idx, entry in enumerate(entries):
            source_dicom = Path(entry.path)
            if self.cropped_root is not None:
                cropped = to_cropped_path(source_dicom, self.echo_root, self.cropped_root)
                if cropped is None or not cropped.exists():
                    continue
            else:
                cropped = source_dicom
            if not cropped.exists():
                continue

            pid = _patient_key(entry)
            if self.patient_filter is not None and pid not in self.patient_filter:
                continue
            visit_id = _visit_key(entry, fallback_idx=idx)
            study_dt = getattr(entry, "study_datetime", None)
            if study_dt is not None and not isinstance(study_dt, pd.Timestamp):
                study_dt = pd.to_datetime(study_dt, errors="coerce")
            rows.append(
                {
                    "patient_id": pid,
                    "visit_id": visit_id,
                    "study_datetime": study_dt,
                    "view": getattr(entry, "view", None),
                    "source_dicom": str(source_dicom),
                    "cropped_dicom": str(cropped),
                    "end_diastole": _coerce_optional_int(getattr(entry, "end_diastole", None)),
                    "end_systole": _coerce_optional_int(getattr(entry, "end_systole", None)),
                }
            )

        if not rows:
            return []

        meta_df = pd.DataFrame(rows)
        meta_df = add_gls_from_report(meta_df, self.input_xlsx)
        meta_df["study_datetime"] = pd.to_datetime(meta_df["study_datetime"], errors="coerce")

        patient_map: Dict[str, List[VisitRecord]] = {}
        for (pid, visit_id), grp in meta_df.groupby(["patient_id", "visit_id"], dropna=False):
            pid_str = str(pid)
            visit_str = str(visit_id)

            dt_vals = grp["study_datetime"].dropna()
            study_dt = dt_vals.iloc[0] if len(dt_vals) else None

            gls_vals = pd.to_numeric(grp.get("gls"), errors="coerce").dropna().values
            gls = float(np.mean(gls_vals)) if len(gls_vals) else None

            view_dicoms: Dict[str, List[Path]] = {}
            view_phase: Dict[str, Tuple[Optional[int], Optional[int]]] = {}
            for view in VIEW_KEYS:
                view_grp = grp[grp["view"] == view]
                dicoms = [
                    Path(p)
                    for p in dict.fromkeys(view_grp["cropped_dicom"].tolist())
                    if isinstance(p, str) and p
                ]
                if dicoms:
                    view_dicoms[view] = dicoms
                    ed_vals = [_coerce_optional_int(v) for v in view_grp.get("end_diastole", pd.Series([]))]
                    es_vals = [_coerce_optional_int(v) for v in view_grp.get("end_systole", pd.Series([]))]
                    ed = next((v for v in ed_vals if v is not None), None)
                    es = next((v for v in es_vals if v is not None), None)
                    view_phase[view] = (ed, es)

            if not view_dicoms:
                continue

            patient_map.setdefault(pid_str, []).append(
                VisitRecord(
                    visit_id=visit_str,
                    visit_time_months=0.0,  # filled after sort
                    study_datetime=study_dt,
                    gls=gls,
                    view_dicoms=view_dicoms,
                    view_phase=view_phase,
                )
            )

        patient_records: List[PatientRecord] = []
        for pid, visits in patient_map.items():
            visits.sort(
                key=lambda v: (
                    0 if isinstance(v.study_datetime, pd.Timestamp) and not pd.isna(v.study_datetime) else 1,
                    v.study_datetime if isinstance(v.study_datetime, pd.Timestamp) and not pd.isna(v.study_datetime) else pd.Timestamp.max,
                    _numeric_hint(v.visit_id),
                    v.visit_id,
                )
            )
            months = _to_months([v.study_datetime for v in visits])
            for i, m in enumerate(months):
                visits[i].visit_time_months = m
            _validate_monotonic_visit_times(pid, [v.visit_time_months for v in visits])
            if len(visits) >= self.min_visits:
                patient_records.append(PatientRecord(patient_id=pid, visits=visits))

        patient_records.sort(key=lambda p: p.patient_id)
        logger.info("Prepared %d patient trajectories for training.", len(patient_records))
        return patient_records

    def __len__(self) -> int:
        return len(self.patient_records)

    def _load_clip(
        self,
        dicom_path: Path,
        end_diastole: Optional[int] = None,
        end_systole: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        clip = torch.zeros(self.t_frames, 3, self.image_size, self.image_size, dtype=torch.float32)
        clip_mask = torch.zeros(self.t_frames, dtype=torch.bool)

        frame_items = iter_cropped_frames(
            dicom_path=dicom_path,
            sampling_mode=self.sampling_mode,
            clip_length=self.t_frames,
            clip_stride=self.clip_stride,
            include_last=self.include_last_window,
            end_diastole=end_diastole,
            end_systole=end_systole,
            phase_aligned_include_midpoints=self.phase_aligned_include_midpoints,
            phase_aligned_strategy=self.phase_aligned_strategy,
        )
        if not frame_items:
            return clip, clip_mask

        for frame_idx, item in enumerate(frame_items[: self.t_frames]):
            tensor = item.get("tensor")
            if tensor is None:
                continue
            if tensor.ndim == 4 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            if tensor.ndim != 3:
                continue
            tensor = resize_tensor(tensor, size=self.image_size)
            if tensor.ndim == 4 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            if tensor.ndim != 3:
                continue
            clip[frame_idx] = tensor.float()
            clip_mask[frame_idx] = True
        return clip, clip_mask

    def __getitem__(self, idx: int) -> Dict[str, object]:
        patient = self.patient_records[idx]
        n_visits = min(len(patient.visits), self.max_visits)

        frames_by_view = {
            view: torch.zeros(
                self.max_visits, self.t_frames, 3, self.image_size, self.image_size, dtype=torch.float32
            )
            for view in VIEW_KEYS
        }
        frame_masks_by_view = {
            view: torch.zeros(self.max_visits, self.t_frames, dtype=torch.bool) for view in VIEW_KEYS
        }
        view_mask = torch.zeros(self.max_visits, len(VIEW_KEYS), dtype=torch.bool)
        visit_mask = torch.zeros(self.max_visits, dtype=torch.bool)
        visit_times = torch.zeros(self.max_visits, dtype=torch.float32)
        gls = torch.zeros(self.max_visits, dtype=torch.float32)
        gls_mask = torch.zeros(self.max_visits, dtype=torch.bool)

        for visit_idx in range(n_visits):
            visit = patient.visits[visit_idx]
            visit_mask[visit_idx] = True
            visit_times[visit_idx] = float(visit.visit_time_months)
            if visit.gls is not None and np.isfinite(visit.gls):
                gls[visit_idx] = float(visit.gls)
                gls_mask[visit_idx] = True

            for view_col_idx, view in enumerate(VIEW_KEYS):
                dicoms = visit.view_dicoms.get(view, [])
                if not dicoms:
                    continue
                if self.random_view_sampling and len(dicoms) > 1:
                    chosen = random.choice(dicoms)
                else:
                    chosen = dicoms[0]
                end_diastole, end_systole = visit.view_phase.get(view, (None, None))
                clip, clip_mask = self._load_clip(
                    chosen,
                    end_diastole=end_diastole,
                    end_systole=end_systole,
                )
                frames_by_view[view][visit_idx] = clip
                frame_masks_by_view[view][visit_idx] = clip_mask
                if clip_mask.any():
                    view_mask[visit_idx, view_col_idx] = True

        # Delta convention (single source of truth): visits are sorted earliest->latest.
        # latest_minus_earliest means delta = GLS_latest - GLS_earliest.
        delta_gls_target, delta_gls_mask, first_idx, last_idx = _compute_delta_target(
            gls=gls,
            gls_mask=gls_mask,
            n_visits=n_visits,
            convention=self.delta_gls_convention,
        )
        delta_first_visit_index = torch.tensor(first_idx if first_idx is not None else -1, dtype=torch.long)
        delta_last_visit_index = torch.tensor(last_idx if last_idx is not None else -1, dtype=torch.long)
        delta_first_visit_time = torch.tensor(
            float(visit_times[first_idx].item()) if first_idx is not None else 0.0,
            dtype=torch.float32,
        )
        delta_last_visit_time = torch.tensor(
            float(visit_times[last_idx].item()) if last_idx is not None else 0.0,
            dtype=torch.float32,
        )
        delta_first_gls = torch.tensor(
            float(gls[first_idx].item()) if first_idx is not None else 0.0,
            dtype=torch.float32,
        )
        delta_last_gls = torch.tensor(
            float(gls[last_idx].item()) if last_idx is not None else 0.0,
            dtype=torch.float32,
        )
        risk_label = torch.tensor(0.0, dtype=torch.float32)
        risk_mask = torch.tensor(False, dtype=torch.bool)
        if delta_gls_mask.item():
            risk_label = torch.tensor(
                float(delta_gls_target.item() >= self.risk_delta_threshold),
                dtype=torch.float32,
            )
            risk_mask = torch.tensor(True, dtype=torch.bool)

        return {
            "patient_id": patient.patient_id,
            "frames_by_view": frames_by_view,
            "frame_masks_by_view": frame_masks_by_view,
            "view_mask": view_mask,
            "visit_mask": visit_mask,
            "visit_times": visit_times,
            "gls": gls,
            "gls_mask": gls_mask,
            "delta_gls_target": delta_gls_target,
            "delta_gls_mask": delta_gls_mask,
            "delta_first_visit_index": delta_first_visit_index,
            "delta_last_visit_index": delta_last_visit_index,
            "delta_first_visit_time": delta_first_visit_time,
            "delta_last_visit_time": delta_last_visit_time,
            "delta_first_gls": delta_first_gls,
            "delta_last_gls": delta_last_gls,
            "risk_label": risk_label,
            "risk_mask": risk_mask,
        }


def visit_collate_fn(batch: List[Dict[str, object]]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    out["patient_id"] = [item["patient_id"] for item in batch]
    out["frames_by_view"] = {
        view: torch.stack([item["frames_by_view"][view] for item in batch], dim=0)
        for view in VIEW_KEYS
    }
    out["frame_masks_by_view"] = {
        view: torch.stack([item["frame_masks_by_view"][view] for item in batch], dim=0)
        for view in VIEW_KEYS
    }
    for key in (
        "view_mask",
        "visit_mask",
        "visit_times",
        "gls",
        "gls_mask",
        "delta_gls_target",
        "delta_gls_mask",
        "delta_first_visit_index",
        "delta_last_visit_index",
        "delta_first_visit_time",
        "delta_last_visit_time",
        "delta_first_gls",
        "delta_last_gls",
        "risk_label",
        "risk_mask",
    ):
        out[key] = torch.stack([item[key] for item in batch], dim=0)
    return out


class FrameEmbeddingVisitDataset(Dataset):
    """
    Visit dataset that consumes precomputed frame embeddings parquet/csv.
    """

    def __init__(
        self,
        input_embeddings: Path,
        views: str = "",
        t_frames: int = 16,
        max_visits: int = 5,
        min_visits: int = 2,
        risk_delta_threshold: float = 2.0,
        report_xlsx: Optional[Path] = None,
        patient_filter: Optional[Sequence[str]] = None,
        sampling_mode: str = "uniform",
        clip_stride: int = 1,
        include_last_window: bool = True,
        phase_aligned_include_midpoints: bool = True,
        phase_aligned_strategy: str = "cycle",
        delta_gls_convention: str = DELTA_GLS_CONVENTION,
    ) -> None:
        super().__init__()
        self.input_embeddings = Path(input_embeddings)
        self.selected_views = parse_views(views) or set(VIEW_KEYS)
        self.t_frames = int(t_frames)
        self.sampling_mode = str(sampling_mode).strip().lower()
        self.clip_stride = int(clip_stride)
        self.include_last_window = bool(include_last_window)
        self.phase_aligned_include_midpoints = bool(phase_aligned_include_midpoints)
        self.phase_aligned_strategy = str(phase_aligned_strategy).strip().lower()
        self.max_visits = int(max_visits)
        self.min_visits = int(min_visits)
        self.risk_delta_threshold = float(risk_delta_threshold)
        self.delta_gls_convention = str(delta_gls_convention).strip().lower()
        self.patient_filter = set(str(x) for x in patient_filter) if patient_filter else None

        if self.sampling_mode not in {"uniform", "sliding_window", "phase_aligned"}:
            raise ValueError("sampling_mode must be one of: uniform, sliding_window, phase_aligned.")
        if self.phase_aligned_strategy not in PHASE_ALIGNED_STRATEGY_CHOICES:
            raise ValueError(
                f"phase_aligned_strategy must be one of: {sorted(PHASE_ALIGNED_STRATEGY_CHOICES)}"
            )
        if self.delta_gls_convention not in DELTA_GLS_CONVENTION_CHOICES:
            raise ValueError(
                f"delta_gls_convention must be one of: {sorted(DELTA_GLS_CONVENTION_CHOICES)}"
            )

        if self.input_embeddings.suffix.lower() == ".parquet":
            df = pd.read_parquet(self.input_embeddings)
        else:
            df = pd.read_csv(self.input_embeddings)
        if report_xlsx is not None and report_xlsx.exists() and "gls" not in df.columns:
            try:
                df = add_gls_from_report(df, report_xlsx)
            except Exception as exc:
                logger.warning("Failed to attach GLS from report: %s", exc)
        self.patient_records = self._build_patient_records(df)
        if not self.patient_records:
            raise RuntimeError("No valid patient records found in frame embeddings.")

    @staticmethod
    def _visit_id_from_row(row: pd.Series) -> str:
        for col in ("visit_id", "visit", "study_id", "accession", "study_datetime"):
            if col in row and not pd.isna(row[col]):
                return str(row[col])
        return str(row.get("source_dicom", "unknown_visit"))

    def _build_patient_records(self, df: pd.DataFrame) -> List[PatientRecord]:
        if "embedding" not in df.columns:
            raise ValueError("Input embeddings dataframe missing 'embedding' column.")
        if "view" not in df.columns:
            raise ValueError("Input embeddings dataframe missing 'view' column.")

        patient_col = None
        for cand in ("patient_id", "patient_num", "patient"):
            if cand in df.columns:
                patient_col = cand
                break
        if patient_col is None:
            raise ValueError("No patient column found in embeddings (expected patient_id/patient_num).")

        work = df.copy()
        work = work[work["view"].isin(self.selected_views)]
        if self.patient_filter is not None:
            work = work[work[patient_col].astype(str).isin(self.patient_filter)]
        if "study_datetime" in work.columns:
            work["study_datetime"] = pd.to_datetime(work["study_datetime"], errors="coerce")
        else:
            work["study_datetime"] = pd.NaT
        if "frame_index" not in work.columns:
            work["frame_index"] = work.groupby(["source_dicom"]).cumcount()

        patient_map: Dict[str, List[VisitRecord]] = {}
        for pid, pgrp in work.groupby(patient_col):
            visit_rows: Dict[str, Dict[str, object]] = {}
            for _, row in pgrp.iterrows():
                visit_id = self._visit_id_from_row(row)
                row_dt = row.get("study_datetime")
                visit = visit_rows.setdefault(
                    visit_id,
                    {
                        "study_datetime": row_dt,
                        "gls_values": [],
                        "view_frames": {v: [] for v in VIEW_KEYS},
                        "view_phase": {},
                    },
                )
                if (
                    (visit.get("study_datetime") is None or pd.isna(visit.get("study_datetime")))
                    and row_dt is not None
                    and not pd.isna(row_dt)
                ):
                    visit["study_datetime"] = row_dt
                v = str(row.get("view"))
                emb = _parse_embedding_vector(row.get("embedding"))
                if emb is None:
                    continue
                frame_idx = int(row.get("frame_index", 0))
                visit["view_frames"][v].append((frame_idx, emb))
                ed = _coerce_optional_int(row.get("end_diastole"))
                es = _coerce_optional_int(row.get("end_systole"))
                if v in VIEW_KEYS and v not in visit["view_phase"]:
                    visit["view_phase"][v] = (ed, es)
                elif v in VIEW_KEYS:
                    prev_ed, prev_es = visit["view_phase"][v]
                    visit["view_phase"][v] = (
                        prev_ed if prev_ed is not None else ed,
                        prev_es if prev_es is not None else es,
                    )
                gls = row.get("gls")
                if gls is not None and not pd.isna(gls):
                    visit["gls_values"].append(float(gls))

            visits: List[VisitRecord] = []
            for visit_id, vdict in visit_rows.items():
                view_dicoms: Dict[str, List[Path]] = {}
                # Reuse VisitRecord container; embeddings are stored in synthetic path slots not used later.
                # TODO: replace this with a dedicated embedding visit dataclass if schema grows.
                keep_any = False
                for vw in VIEW_KEYS:
                    frames = sorted(vdict["view_frames"][vw], key=lambda x: x[0])
                    if not frames:
                        continue
                    keep_any = True
                    # Encode vector payload in an in-memory map keyed by synthetic path.
                    synth = Path(f"mem://{visit_id}/{vw}")
                    view_dicoms[vw] = [synth]
                    vdict.setdefault("embedding_map", {})[str(synth)] = frames
                if not keep_any:
                    continue
                gls_vals = vdict["gls_values"]
                visits.append(
                    VisitRecord(
                        visit_id=visit_id,
                        visit_time_months=0.0,
                        study_datetime=vdict["study_datetime"],
                        gls=float(np.mean(gls_vals)) if gls_vals else None,
                        view_dicoms=view_dicoms,
                        view_phase=dict(vdict.get("view_phase", {})),
                    )
                )
                visits[-1].__dict__["embedding_map"] = vdict.get("embedding_map", {})

            visits.sort(
                key=lambda v: (
                    0 if isinstance(v.study_datetime, pd.Timestamp) and not pd.isna(v.study_datetime) else 1,
                    v.study_datetime if isinstance(v.study_datetime, pd.Timestamp) and not pd.isna(v.study_datetime) else pd.Timestamp.max,
                    _numeric_hint(v.visit_id),
                    v.visit_id,
                )
            )
            months = _to_months([v.study_datetime for v in visits])
            for i, m in enumerate(months):
                visits[i].visit_time_months = m
            _validate_monotonic_visit_times(str(pid), [v.visit_time_months for v in visits])
            if len(visits) >= self.min_visits:
                patient_map[str(pid)] = visits

        records = [PatientRecord(patient_id=pid, visits=visits) for pid, visits in patient_map.items()]
        records.sort(key=lambda p: p.patient_id)
        return records

    def __len__(self) -> int:
        return len(self.patient_records)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        patient = self.patient_records[idx]
        n_visits = min(len(patient.visits), self.max_visits)

        # Infer embedding dim from first available frame embedding.
        emb_dim = 384
        for v in patient.visits:
            emap = v.__dict__.get("embedding_map", {})
            for seq in emap.values():
                if seq:
                    emb_dim = int(seq[0][1].shape[0])
                    break
            if emb_dim:
                break

        embeddings_by_view = {
            view: torch.zeros(self.max_visits, self.t_frames, emb_dim, dtype=torch.float32)
            for view in VIEW_KEYS
        }
        frame_masks_by_view = {
            view: torch.zeros(self.max_visits, self.t_frames, dtype=torch.bool) for view in VIEW_KEYS
        }
        visit_mask = torch.zeros(self.max_visits, dtype=torch.bool)
        visit_times = torch.zeros(self.max_visits, dtype=torch.float32)
        gls = torch.zeros(self.max_visits, dtype=torch.float32)
        gls_mask = torch.zeros(self.max_visits, dtype=torch.bool)

        for visit_idx in range(n_visits):
            visit = patient.visits[visit_idx]
            visit_mask[visit_idx] = True
            visit_times[visit_idx] = float(visit.visit_time_months)
            if visit.gls is not None and np.isfinite(visit.gls):
                gls[visit_idx] = float(visit.gls)
                gls_mask[visit_idx] = True

            emap = visit.__dict__.get("embedding_map", {})
            for view in VIEW_KEYS:
                synth_list = visit.view_dicoms.get(view, [])
                if not synth_list:
                    continue
                seq = emap.get(str(synth_list[0]), [])
                if not seq:
                    continue
                seq = sorted(seq, key=lambda x: int(x[0]))
                emb_arr = [x[1] for x in seq]
                end_diastole, end_systole = visit.view_phase.get(view, (None, None))
                sampled_pos = _sample_embedding_positions(
                    n_frames=len(emb_arr),
                    t_frames=self.t_frames,
                    sampling_mode=self.sampling_mode,
                    clip_stride=self.clip_stride,
                    include_last_window=self.include_last_window,
                    end_diastole=end_diastole,
                    end_systole=end_systole,
                    phase_aligned_include_midpoints=self.phase_aligned_include_midpoints,
                    phase_aligned_strategy=self.phase_aligned_strategy,
                )
                if sampled_pos.size == 0:
                    continue
                max_t = min(self.t_frames, int(sampled_pos.size))
                for t in range(max_t):
                    src_idx = int(np.clip(sampled_pos[t], 0, len(emb_arr) - 1))
                    emb = emb_arr[src_idx]
                    embeddings_by_view[view][visit_idx, t] = torch.tensor(
                        emb,
                        dtype=torch.float32,
                    )
                    frame_masks_by_view[view][visit_idx, t] = True

        delta_gls_target, delta_gls_mask, first_idx, last_idx = _compute_delta_target(
            gls=gls,
            gls_mask=gls_mask,
            n_visits=n_visits,
            convention=self.delta_gls_convention,
        )
        delta_first_visit_index = torch.tensor(first_idx if first_idx is not None else -1, dtype=torch.long)
        delta_last_visit_index = torch.tensor(last_idx if last_idx is not None else -1, dtype=torch.long)
        delta_first_visit_time = torch.tensor(
            float(visit_times[first_idx].item()) if first_idx is not None else 0.0,
            dtype=torch.float32,
        )
        delta_last_visit_time = torch.tensor(
            float(visit_times[last_idx].item()) if last_idx is not None else 0.0,
            dtype=torch.float32,
        )
        delta_first_gls = torch.tensor(
            float(gls[first_idx].item()) if first_idx is not None else 0.0,
            dtype=torch.float32,
        )
        delta_last_gls = torch.tensor(
            float(gls[last_idx].item()) if last_idx is not None else 0.0,
            dtype=torch.float32,
        )
        risk_label = torch.tensor(0.0, dtype=torch.float32)
        risk_mask = torch.tensor(False, dtype=torch.bool)
        if delta_gls_mask.item():
            risk_label = torch.tensor(
                float(delta_gls_target.item() >= self.risk_delta_threshold),
                dtype=torch.float32,
            )
            risk_mask = torch.tensor(True, dtype=torch.bool)

        return {
            "patient_id": patient.patient_id,
            "embeddings_by_view": embeddings_by_view,
            "frame_masks_by_view": frame_masks_by_view,
            "visit_mask": visit_mask,
            "visit_times": visit_times,
            "gls": gls,
            "gls_mask": gls_mask,
            "delta_gls_target": delta_gls_target,
            "delta_gls_mask": delta_gls_mask,
            "delta_first_visit_index": delta_first_visit_index,
            "delta_last_visit_index": delta_last_visit_index,
            "delta_first_visit_time": delta_first_visit_time,
            "delta_last_visit_time": delta_last_visit_time,
            "delta_first_gls": delta_first_gls,
            "delta_last_gls": delta_last_gls,
            "risk_label": risk_label,
            "risk_mask": risk_mask,
        }


def embedding_visit_collate_fn(batch: List[Dict[str, object]]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    out["patient_id"] = [item["patient_id"] for item in batch]
    out["embeddings_by_view"] = {
        view: torch.stack([item["embeddings_by_view"][view] for item in batch], dim=0)
        for view in VIEW_KEYS
    }
    out["frame_masks_by_view"] = {
        view: torch.stack([item["frame_masks_by_view"][view] for item in batch], dim=0)
        for view in VIEW_KEYS
    }
    for key in (
        "visit_mask",
        "visit_times",
        "gls",
        "gls_mask",
        "delta_gls_target",
        "delta_gls_mask",
        "delta_first_visit_index",
        "delta_last_visit_index",
        "delta_first_visit_time",
        "delta_last_visit_time",
        "delta_first_gls",
        "delta_last_gls",
        "risk_label",
        "risk_mask",
    ):
        out[key] = torch.stack([item[key] for item in batch], dim=0)
    return out


def split_patient_indices(
    n_patients: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    if n_patients <= 0:
        return [], [], []
    if val_ratio < 0 or test_ratio < 0 or (val_ratio + test_ratio) >= 1.0:
        raise ValueError("Expected 0 <= val_ratio,test_ratio and val_ratio+test_ratio < 1.")

    indices = list(range(n_patients))
    rng = random.Random(int(seed))
    rng.shuffle(indices)

    n_test = int(round(n_patients * float(test_ratio)))
    n_val = int(round(n_patients * float(val_ratio)))
    n_test = min(n_test, n_patients)
    n_val = min(n_val, n_patients - n_test)

    test_idx = indices[:n_test]
    val_idx = indices[n_test : n_test + n_val]
    train_idx = indices[n_test + n_val :]

    if not train_idx:
        train_idx = val_idx
        val_idx = []
    return train_idx, val_idx, test_idx

