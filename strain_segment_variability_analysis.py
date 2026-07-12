from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy import stats


NS = {"ss": "urn:schemas-microsoft-com:office:spreadsheet"}
SS_NS = NS["ss"]
SHEET_NAMES = ("Strain-Endo", "Strain-Myo")
TOP_LEVEL_EXCLUDES = {"Anonymous", "models", "processed"}
TARGET_VIEWS = ("2-chamber", "4-chamber")
DEFAULT_ROOTS = (
    Path(r"C:\Users\Oron\OneDrive - Technion\DS\Tags_SZMC\VVI"),
    Path(r"C:\Users\Oron\OneDrive - Technion\DS\Tags_Ichilov\VVI"),
)


@dataclass(frozen=True)
class XmlRecord:
    hospital: str
    root: Path
    xml_path: Path
    patient_id: str
    visit_date: str
    dicom_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze longitudinal strain segment variability from VVI SEG XML files."
    )
    parser.add_argument(
        "--root",
        type=Path,
        action="append",
        dest="roots",
        help="VVI tag root. May be passed more than once.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "strain_variability_analysis",
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--resample-len", type=int, default=96)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--hospital",
        action="append",
        default=None,
        help="Optional hospital filter, e.g. SZMC or Ichilov.",
    )
    parser.add_argument(
        "--include-view",
        action="append",
        default=None,
        choices=("2-chamber", "3-chamber", "4-chamber", "unknown"),
        help="Optional view filter. May be passed more than once.",
    )
    return parser.parse_args()


def infer_hospital(root: Path) -> str:
    parent = root.parent.name
    if parent.lower().startswith("tags_"):
        return parent[5:]
    return parent or root.name


def discover_xmls(roots: Iterable[Path], hospitals: Optional[set[str]]) -> list[XmlRecord]:
    records: list[XmlRecord] = []
    for root in roots:
        root = root.resolve()
        if not root.is_dir():
            continue
        hospital = infer_hospital(root)
        if hospitals and hospital.lower() not in hospitals:
            continue
        for xml_path in root.rglob("(SEG)*.xml"):
            try:
                rel = xml_path.relative_to(root).parts
            except ValueError:
                continue
            if len(rel) < 4:
                continue
            if rel[0] in TOP_LEVEL_EXCLUDES:
                continue
            records.append(
                XmlRecord(
                    hospital=hospital,
                    root=root,
                    xml_path=xml_path,
                    patient_id=rel[0],
                    visit_date=rel[1],
                    dicom_name=rel[2],
                )
            )
    records.sort(key=lambda r: (r.hospital, r.patient_id, r.visit_date, r.dicom_name, str(r.xml_path)))
    return records


def staged_xml_path(xml_path: Path, cache_dir: Path) -> Path:
    digest = hashlib.sha1(str(xml_path).encode("utf-8", errors="ignore")).hexdigest()[:16]
    return cache_dir / digest / xml_path.name


def stage_with_robocopy(xml_path: Path, cache_dir: Path) -> Path:
    target = staged_xml_path(xml_path, cache_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "robocopy",
            str(xml_path.parent),
            str(target.parent),
            xml_path.name,
            "/R:3",
            "/W:2",
            "/NP",
        ],
        capture_output=True,
    )
    if result.returncode >= 8:
        stdout = result.stdout.decode(errors="replace")
        stderr = result.stderr.decode(errors="replace")
        raise RuntimeError(
            f"robocopy staging failed for {xml_path} with code {result.returncode}\n"
            f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        )
    if not target.is_file():
        raise FileNotFoundError(f"robocopy did not create staged XML: {target}")
    return target


def parse_xml_tree(xml_path: Path, cache_dir: Path) -> tuple[ET.ElementTree, bool]:
    try:
        return ET.parse(xml_path), False
    except OSError:
        cached = stage_with_robocopy(xml_path, cache_dir)
        return ET.parse(cached), True


def worksheet_table(root: ET.Element, wanted_name: str) -> Optional[list[list[Optional[str]]]]:
    wanted_key = wanted_name.strip().lower()
    for ws in root.findall(".//ss:Worksheet", NS):
        name = ws.get(f"{{{SS_NS}}}Name") or ws.get("Name") or ""
        if name.strip().lower() != wanted_key:
            continue
        table_el = ws.find("ss:Table", NS)
        if table_el is None:
            return []

        table: list[list[Optional[str]]] = []
        row_cursor = 1
        for row_el in table_el.findall("ss:Row", NS):
            row_index = row_el.get(f"{{{SS_NS}}}Index")
            if row_index is not None:
                target = int(row_index)
                while row_cursor < target:
                    table.append([])
                    row_cursor += 1

            row: list[Optional[str]] = []
            col_cursor = 1
            for cell_el in row_el.findall("ss:Cell", NS):
                col_index = cell_el.get(f"{{{SS_NS}}}Index")
                if col_index is not None:
                    target = int(col_index)
                    while col_cursor < target:
                        row.append(None)
                        col_cursor += 1
                data_el = cell_el.find("ss:Data", NS)
                text = (data_el.text if data_el is not None else cell_el.text) or ""
                row.append(text.strip() or None)
                col_cursor += 1
            table.append(row)
            row_cursor += 1
        return table
    return None


def to_float(value: object) -> float:
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = str(value).strip().replace(",", ".")
    if not text:
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def parse_segment_label(label: object) -> tuple[Optional[int], str]:
    text = "" if label is None else str(label).strip()
    match = re.match(r"^\s*(\d+)\s*[-:_]?\s*(.*)$", text)
    if match:
        return int(match.group(1)), match.group(2).strip()
    return None, text


def infer_view_from_segments(segment_rows: list[dict]) -> str:
    labels = " ".join(str(row.get("segment_label") or "") for row in segment_rows).lower()
    names = [str(row.get("segment_name") or "").lower() for row in segment_rows]
    numbers = {row.get("segment_number") for row in segment_rows if row.get("segment_number") is not None}

    # Match the distinctive segment sets first. These are the six-segment VVI
    # exports for the apical views.
    if {"inferoseptal", "anterolateral"} & set(" ".join(names).split()):
        return "4-chamber"
    if "inferoseptal" in labels or "anterolateral" in labels:
        return "4-chamber"
    if "anteroseptal" in labels or "inferolateral" in labels or "posterior" in labels:
        return "3-chamber"

    # 2-chamber exports contain anterior/inferior walls without septal/lateral
    # qualifiers.
    if ("anterior" in labels or "inferior" in labels) and not (
        "anteroseptal" in labels
        or "inferoseptal" in labels
        or "anterolateral" in labels
        or "inferolateral" in labels
    ):
        return "2-chamber"

    if numbers == {1, 4, 7, 10, 13, 15}:
        return "2-chamber"
    if numbers == {3, 6, 9, 12, 14, 16}:
        return "4-chamber"

    return "unknown"


def layer_from_sheet(sheet_name: str) -> str:
    return "endo" if "endo" in sheet_name.lower() else "myo"


def extract_longitudinal_segments(table: list[list[Optional[str]]]) -> tuple[list[dict], Optional[np.ndarray]]:
    if not table:
        return [], None

    start_idx: Optional[int] = None
    for i, row in enumerate(table):
        first = "" if not row or row[0] is None else str(row[0])
        if "longitudinal strain" in first.lower():
            start_idx = i
            break
    if start_idx is None:
        return [], None

    segment_rows: list[dict] = []
    time_values: Optional[np.ndarray] = None
    for row in table[start_idx + 1 :]:
        if not row or not any(cell not in (None, "") for cell in row):
            break

        label = "" if row[0] is None else str(row[0]).strip()
        label_key = label.lower()
        values = np.asarray([to_float(v) for v in row[1:]], dtype=float)
        values = values[np.isfinite(values)]

        if label_key == "time":
            time_values = values
            continue
        if label_key.startswith("average") or label_key.startswith("standard"):
            continue
        if not values.size:
            continue

        segment_number, segment_name = parse_segment_label(label)
        segment_rows.append(
            {
                "segment_label": label,
                "segment_number": segment_number,
                "segment_name": segment_name,
                "curve": values,
            }
        )

    return segment_rows, time_values


def align_curve_and_time(curve: np.ndarray, time_ms: Optional[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    curve = np.asarray(curve, dtype=float)
    curve = curve[np.isfinite(curve)]
    if time_ms is None or len(time_ms) < 2:
        time_ms = np.arange(len(curve), dtype=float)
    else:
        time_ms = np.asarray(time_ms, dtype=float)
        time_ms = time_ms[np.isfinite(time_ms)]
    n = min(len(curve), len(time_ms))
    curve = curve[:n]
    time_ms = time_ms[:n]
    if n >= 2 and np.any(np.diff(time_ms) <= 0):
        time_ms = np.arange(n, dtype=float)
    return curve, time_ms


def normalized_time(time_ms: np.ndarray) -> np.ndarray:
    if len(time_ms) < 2:
        return np.zeros_like(time_ms, dtype=float)
    duration = float(time_ms[-1] - time_ms[0])
    if not math.isfinite(duration) or duration <= 0:
        return np.linspace(0.0, 1.0, len(time_ms))
    return (time_ms - time_ms[0]) / duration


def compute_segment_features(curve: np.ndarray, time_ms: np.ndarray) -> dict[str, float]:
    curve, time_ms = align_curve_and_time(curve, time_ms)
    if len(curve) < 3:
        return {}

    t_norm = normalized_time(time_ms)
    peak_idx = int(np.nanargmin(curve))
    peak_strain = float(curve[peak_idx])
    peak_abs = float(abs(peak_strain))
    time_to_peak_ms = float(time_ms[peak_idx] - time_ms[0])
    time_to_peak_norm = float(t_norm[peak_idx])

    negative_curve = np.maximum(-curve, 0.0)
    strain_burden = float(np.trapezoid(negative_curve, t_norm))
    rms_strain = float(np.sqrt(np.nanmean(np.square(curve))))

    dt_seconds = np.diff(time_ms) / 1000.0
    dy = np.diff(curve)
    valid_dt = np.where(np.abs(dt_seconds) > 1e-9, dt_seconds, np.nan)
    slopes = dy / valid_dt
    slopes = slopes[np.isfinite(slopes)]
    contraction_rate = float(np.nanmin(slopes)) if slopes.size else np.nan
    relaxation_rate = float(np.nanmax(slopes)) if slopes.size else np.nan

    recovery_fraction = float((curve[-1] - peak_strain) / (peak_abs + 1e-6))
    end_minus_start = float(curve[-1] - curve[0])
    roughness = float(np.nanstd(np.diff(curve, n=2))) if len(curve) >= 4 else np.nan

    return {
        "n_points": int(len(curve)),
        "duration_ms": float(time_ms[-1] - time_ms[0]),
        "peak_strain": peak_strain,
        "peak_abs": peak_abs,
        "peak_index": peak_idx,
        "time_to_peak_ms": time_to_peak_ms,
        "time_to_peak_norm": time_to_peak_norm,
        "strain_burden": strain_burden,
        "rms_strain": rms_strain,
        "contraction_rate_per_s": contraction_rate,
        "relaxation_rate_per_s": relaxation_rate,
        "recovery_fraction": recovery_fraction,
        "end_minus_start": end_minus_start,
        "curve_roughness": roughness,
        "start_strain": float(curve[0]),
        "end_strain": float(curve[-1]),
    }


def resample_curve(curve: np.ndarray, time_ms: np.ndarray, length: int) -> np.ndarray:
    curve, time_ms = align_curve_and_time(curve, time_ms)
    t = normalized_time(time_ms)
    grid = np.linspace(0.0, 1.0, length)
    return np.interp(grid, t, curve).astype(np.float32)


def parse_all(
    records: list[XmlRecord],
    cache_dir: Path,
    resample_len: int,
    include_views: Optional[set[str]] = None,
) -> tuple[pd.DataFrame, np.ndarray, list[dict]]:
    rows: list[dict] = []
    curves: list[np.ndarray] = []
    parse_logs: list[dict] = []
    raw_id = 0

    for rec in records:
        try:
            tree, staged = parse_xml_tree(rec.xml_path, cache_dir)
            root = tree.getroot()
        except Exception as exc:
            parse_logs.append(
                {
                    "xml_path": str(rec.xml_path),
                    "status": "parse_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        parsed_any = False
        excluded_views: set[str] = set()
        for sheet_name in SHEET_NAMES:
            table = worksheet_table(root, sheet_name)
            if table is None:
                parse_logs.append(
                    {
                        "xml_path": str(rec.xml_path),
                        "status": "missing_sheet",
                        "sheet": sheet_name,
                        "error": "",
                    }
                )
                continue
            segment_rows, time_ms = extract_longitudinal_segments(table)
            if not segment_rows:
                parse_logs.append(
                    {
                        "xml_path": str(rec.xml_path),
                        "status": "no_longitudinal_segments",
                        "sheet": sheet_name,
                        "error": "",
                    }
                )
                continue

            layer = layer_from_sheet(sheet_name)
            view = infer_view_from_segments(segment_rows)
            if include_views and view not in include_views:
                excluded_views.add(view)
                continue
            for segment in segment_rows:
                curve, aligned_time = align_curve_and_time(segment["curve"], time_ms)
                features = compute_segment_features(curve, aligned_time)
                if not features:
                    continue
                segment_number = segment["segment_number"]
                segment_key = (
                    f"{segment_number:02d}"
                    if segment_number is not None
                    else re.sub(r"[^a-z0-9]+", "_", segment["segment_label"].lower()).strip("_")
                )
                rows.append(
                    {
                        "raw_segment_id": raw_id,
                        "hospital": rec.hospital,
                        "patient_id": rec.patient_id,
                        "visit_date": rec.visit_date,
                        "dicom_name": rec.dicom_name,
                        "xml_path": str(rec.xml_path),
                        "sheet": sheet_name,
                        "view": view,
                        "layer": layer,
                        "segment_key": segment_key,
                        "segment_number": segment_number,
                        "segment_label": segment["segment_label"],
                        "segment_name": segment["segment_name"],
                        **features,
                    }
                )
                curves.append(resample_curve(curve, aligned_time, resample_len))
                raw_id += 1
                parsed_any = True

        parse_logs.append(
            {
                "xml_path": str(rec.xml_path),
                "status": "parsed",
                "staged_from_onedrive": staged,
                "excluded_views": ",".join(sorted(excluded_views)),
                "error": "",
            }
        )
        if not parsed_any:
            parse_logs[-1]["status"] = "excluded_by_view" if excluded_views else "parsed_no_segments"

    feature_df = pd.DataFrame(rows)
    curve_array = np.vstack(curves) if curves else np.empty((0, resample_len), dtype=np.float32)
    return feature_df, curve_array, parse_logs


def robust_std(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) < 2:
        return np.nan
    return float(np.std(values, ddof=1))


def iqr(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) < 2:
        return np.nan
    return float(np.nanpercentile(values, 75) - np.nanpercentile(values, 25))


def mad(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) < 2:
        return np.nan
    med = np.nanmedian(values)
    return float(np.nanmedian(np.abs(values - med)))


def visit_variability(feature_df: pd.DataFrame) -> pd.DataFrame:
    if feature_df.empty:
        return pd.DataFrame()

    group_cols = [
        "hospital",
        "patient_id",
        "visit_date",
        "view",
        "layer",
        "segment_key",
    ]
    numeric_cols = [
        "peak_abs",
        "peak_strain",
        "time_to_peak_ms",
        "time_to_peak_norm",
        "strain_burden",
        "rms_strain",
        "contraction_rate_per_s",
        "recovery_fraction",
        "curve_roughness",
    ]
    segment_level = (
        feature_df[group_cols + numeric_cols]
        .groupby(group_cols, dropna=False, as_index=False)
        .mean(numeric_only=True)
    )

    rows: list[dict] = []
    for keys, group in segment_level.groupby(["hospital", "patient_id", "visit_date", "view", "layer"], dropna=False):
        hospital, patient_id, visit_date, view, layer = keys
        row = {
            "hospital": hospital,
            "patient_id": patient_id,
            "visit_date": visit_date,
            "view": view,
            "layer": layer,
            "n_segments": int(group["segment_key"].nunique()),
        }
        for col in numeric_cols:
            row[f"{col}_mean"] = float(pd.to_numeric(group[col], errors="coerce").mean())
            row[f"{col}_std"] = robust_std(group[col])
            row[f"{col}_iqr"] = iqr(group[col])
            row[f"{col}_mad"] = mad(group[col])
        mean_peak = abs(row.get("peak_abs_mean", np.nan))
        row["peak_abs_cv"] = (
            row["peak_abs_std"] / mean_peak if mean_peak and math.isfinite(mean_peak) else np.nan
        )
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out["visit_dt"] = pd.to_datetime(out["visit_date"], errors="coerce", format="%Y_%m_%d")
        out = out.sort_values(["hospital", "patient_id", "view", "layer", "visit_dt", "visit_date"]).reset_index(drop=True)
        out["visit_order"] = out.groupby(["hospital", "patient_id", "view", "layer"]).cumcount() + 1
        baseline = out.groupby(["hospital", "patient_id", "view", "layer"])["visit_dt"].transform("min")
        out["days_since_baseline"] = (out["visit_dt"] - baseline).dt.days
    return out


def train_autoencoder(
    curves: np.ndarray,
    latent_dim: int,
    epochs: int,
    batch_size: int,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame, str]:
    if curves.size == 0:
        return np.empty((0, latent_dim)), pd.DataFrame(), "none"

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception:
        return pca_fallback(curves, latent_dim), pd.DataFrame(), "pca_fallback"

    torch.manual_seed(seed)
    np.random.seed(seed)

    mean = curves.mean(axis=0, keepdims=True)
    std = curves.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    x_np = ((curves - mean) / std).astype(np.float32)
    x = torch.from_numpy(x_np)
    loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=True)
    input_dim = x.shape[1]

    class CurveAutoencoder(nn.Module):
        def __init__(self, input_dim: int, latent_dim: int):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, input_dim),
            )

        def forward(self, value):
            z = self.encoder(value)
            return self.decoder(z)

    model = CurveAutoencoder(input_dim, latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    loss_rows: list[dict] = []

    model.train()
    for epoch in range(1, epochs + 1):
        losses = []
        for (batch,) in loader:
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        loss_rows.append({"epoch": epoch, "train_mse": float(np.mean(losses))})

    model.eval()
    with torch.no_grad():
        latent = model.encoder(x).cpu().numpy()
    return latent, pd.DataFrame(loss_rows), "torch_mlp_autoencoder"


def pca_fallback(curves: np.ndarray, latent_dim: int) -> np.ndarray:
    x = curves.astype(float)
    x = x - np.nanmean(x, axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    return (u[:, :latent_dim] * s[:latent_dim]).astype(float)


def mean_pairwise_distance(matrix: np.ndarray) -> float:
    matrix = np.asarray(matrix, dtype=float)
    if len(matrix) < 2:
        return np.nan
    diffs = matrix[:, None, :] - matrix[None, :, :]
    dists = np.sqrt(np.sum(diffs * diffs, axis=-1))
    upper = dists[np.triu_indices(len(matrix), k=1)]
    return float(np.nanmean(upper)) if upper.size else np.nan


def latent_outputs(feature_df: pd.DataFrame, latent: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    if feature_df.empty or latent.size == 0:
        return pd.DataFrame(), pd.DataFrame()

    latent_cols = [f"latent_{i:02d}" for i in range(latent.shape[1])]
    latent_df = pd.DataFrame(latent, columns=latent_cols)
    latent_df.insert(0, "raw_segment_id", feature_df["raw_segment_id"].to_numpy())
    meta_cols = [
        "hospital",
        "patient_id",
        "visit_date",
        "dicom_name",
        "view",
        "layer",
        "segment_key",
        "segment_label",
    ]
    latent_df = pd.concat([feature_df[meta_cols].reset_index(drop=True), latent_df], axis=1)

    segment_latent = (
        latent_df[["hospital", "patient_id", "visit_date", "view", "layer", "segment_key", *latent_cols]]
        .groupby(["hospital", "patient_id", "visit_date", "view", "layer", "segment_key"], dropna=False, as_index=False)
        .mean(numeric_only=True)
    )
    rows: list[dict] = []
    for keys, group in segment_latent.groupby(["hospital", "patient_id", "visit_date", "view", "layer"], dropna=False):
        hospital, patient_id, visit_date, view, layer = keys
        matrix = group[latent_cols].to_numpy(dtype=float)
        row = {
            "hospital": hospital,
            "patient_id": patient_id,
            "visit_date": visit_date,
            "view": view,
            "layer": layer,
            "n_segments": int(group["segment_key"].nunique()),
            "latent_pairwise_mean": mean_pairwise_distance(matrix),
            "latent_centroid_norm": float(np.linalg.norm(np.nanmean(matrix, axis=0))),
        }
        rows.append(row)

    variability = pd.DataFrame(rows)
    if not variability.empty:
        variability["visit_dt"] = pd.to_datetime(
            variability["visit_date"], errors="coerce", format="%Y_%m_%d"
        )
        variability = variability.sort_values(
            ["hospital", "patient_id", "view", "layer", "visit_dt", "visit_date"]
        ).reset_index(drop=True)
        variability["visit_order"] = variability.groupby(["hospital", "patient_id", "view", "layer"]).cumcount() + 1
        baseline = variability.groupby(["hospital", "patient_id", "view", "layer"])["visit_dt"].transform("min")
        variability["days_since_baseline"] = (variability["visit_dt"] - baseline).dt.days
    return latent_df, variability


def trend_tests(variability: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    if variability.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for keys, group in variability.groupby(["hospital", "patient_id", "view", "layer"], dropna=False):
        hospital, patient_id, view, layer = keys
        group = group.sort_values(["visit_order", "visit_date"])
        for metric in metrics:
            values = pd.to_numeric(group.get(metric), errors="coerce")
            valid = values.notna() & group["visit_order"].notna()
            g = group.loc[valid].copy()
            y = values.loc[valid].to_numpy(dtype=float)
            if len(g) < 2:
                continue
            x_order = g["visit_order"].to_numpy(dtype=float)
            slope_order = float(np.polyfit(x_order, y, 1)[0])
            days = pd.to_numeric(g.get("days_since_baseline"), errors="coerce").to_numpy(dtype=float)
            if np.isfinite(days).sum() >= 2 and np.nanmax(days) > np.nanmin(days):
                slope_per_year = float(np.polyfit(days / 365.25, y, 1)[0])
            else:
                slope_per_year = np.nan
            if len(g) >= 3:
                spearman = stats.spearmanr(x_order, y, nan_policy="omit")
                rho = float(spearman.statistic)
                p_value = float(spearman.pvalue)
            else:
                rho = np.nan
                p_value = np.nan
            rows.append(
                {
                    "hospital": hospital,
                    "patient_id": patient_id,
                    "view": view,
                    "layer": layer,
                    "metric": metric,
                    "n_visits": int(len(g)),
                    "first_value": float(y[0]),
                    "last_value": float(y[-1]),
                    "delta_last_first": float(y[-1] - y[0]),
                    "slope_per_visit": slope_order,
                    "slope_per_year": slope_per_year,
                    "spearman_r_visit_order": rho,
                    "spearman_p": p_value,
                    "supports_increase": bool(slope_order > 0),
                }
            )
    return pd.DataFrame(rows)


def trend_summary(trends: pd.DataFrame) -> pd.DataFrame:
    if trends.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for keys, group in trends.groupby(["hospital", "view", "layer", "metric"], dropna=False):
        hospital, view, layer, metric = keys
        pos = int((group["slope_per_visit"] > 0).sum())
        neg = int((group["slope_per_visit"] < 0).sum())
        zero = int((group["slope_per_visit"] == 0).sum())
        tested = pos + neg
        p_binom = stats.binomtest(pos, tested, 0.5, alternative="greater").pvalue if tested else np.nan
        rows.append(
            {
                "hospital": hospital,
                "view": view,
                "layer": layer,
                "metric": metric,
                "patients_tested": int(len(group)),
                "positive_slopes": pos,
                "negative_slopes": neg,
                "zero_slopes": zero,
                "fraction_positive": float(pos / tested) if tested else np.nan,
                "binomial_p_positive_gt_half": float(p_binom) if math.isfinite(p_binom) else np.nan,
                "median_slope_per_visit": float(group["slope_per_visit"].median()),
                "mean_slope_per_visit": float(group["slope_per_visit"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["hospital", "view", "layer", "metric"]).reset_index(drop=True)


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df.empty:
        return ""
    view = df.head(max_rows).copy()
    columns = [str(c) for c in view.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in view.iterrows():
        values = []
        for col in view.columns:
            value = row[col]
            if isinstance(value, float):
                if math.isnan(value):
                    text = ""
                else:
                    text = f"{value:.4g}"
            else:
                text = "" if pd.isna(value) else str(value)
            text = text.replace("|", "\\|")
            values.append(text)
        lines.append("| " + " | ".join(values) + " |")
    if len(df) > max_rows:
        lines.append(f"\nShowing first {max_rows} of {len(df)} rows.")
    return "\n".join(lines)


def write_report(
    output_dir: Path,
    records: list[XmlRecord],
    features: pd.DataFrame,
    visit_var: pd.DataFrame,
    latent_method: str,
    classic_summary: pd.DataFrame,
    latent_summary: pd.DataFrame,
    parse_logs: list[dict],
) -> None:
    parsed_logs = pd.DataFrame(parse_logs)
    staged = int(parsed_logs.get("staged_from_onedrive", pd.Series(dtype=object)).fillna(False).astype(bool).sum())
    parsed = int((parsed_logs["status"] == "parsed").sum()) if not parsed_logs.empty else 0
    excluded = int((parsed_logs["status"] == "excluded_by_view").sum()) if not parsed_logs.empty else 0
    failed = int((parsed_logs["status"] == "parse_failed").sum()) if not parsed_logs.empty else 0

    candidate_metrics = [
        "peak_abs_std",
        "time_to_peak_norm_std",
        "strain_burden_std",
        "contraction_rate_per_s_std",
    ]
    if "metric" in classic_summary.columns:
        top_classic = classic_summary[classic_summary["metric"].isin(candidate_metrics)].copy()
    else:
        top_classic = pd.DataFrame()
    top_latent = latent_summary.copy()

    report = [
        "# Strain Segment Variability Analysis",
        "",
        "## Scope",
        f"- XML files discovered: {len(records)}",
        f"- XML files parsed: {parsed}",
        f"- XML files parsed but excluded by view filter: {excluded}",
        f"- XML files staged locally because of OneDrive read errors: {staged}",
        f"- XML parse failures: {failed}",
        f"- Segment curves extracted: {len(features)}",
        f"- Visit/layer rows: {len(visit_var)}",
        f"- Latent method: {latent_method}",
        "",
        "Only `Strain-Endo` and `Strain-Myo` sheets were used. Within those sheets, only the `Longitudinal Strain` block was parsed; transverse strain blocks were ignored.",
        "",
        "Visit-level variability is computed within `patient + visit + view + layer`. This means `2-chamber_endo`, `2-chamber_myo`, `4-chamber_endo`, and `4-chamber_myo` are kept separate.",
        "",
        "## Segment Features",
        "- Classic feature: `time_to_peak_ms` and `time_to_peak_norm`, computed at the most negative longitudinal strain value.",
        "- Additional engineered features: `peak_abs`, normalized negative-strain area (`strain_burden`), maximum contraction/relaxation slopes, recovery fraction, RMS strain, and curve roughness.",
        "- Visit-level variability was computed after averaging duplicate segment labels within the same patient/visit/view/layer.",
        "",
        "## Classic Variability Summary",
        dataframe_to_markdown(top_classic) if not top_classic.empty else "No classic trends available.",
        "",
        "## Latent Variability Summary",
        dataframe_to_markdown(top_latent) if not top_latent.empty else "No latent trends available.",
        "",
        "## Interpretation Notes",
        "- A positive slope means segment variability increased over later visits for that patient, view, and layer.",
        "- `peak_abs_std` tests amplitude heterogeneity between segments.",
        "- `time_to_peak_norm_std` tests temporal dyssynchrony between segments.",
        "- `strain_burden_std` tests heterogeneity in the integrated negative strain load.",
        "- `latent_pairwise_mean` tests shape-level heterogeneity in autoencoder latent space.",
        "- Treat p-values as exploratory: the sample is small and there is no adjustment for repeated metrics/layers.",
    ]
    (output_dir / "research_report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or output_dir / "xml_cache"

    roots = args.roots or list(DEFAULT_ROOTS)
    hospitals = {h.lower() for h in args.hospital} if args.hospital else None
    include_views = set(args.include_view or TARGET_VIEWS)
    records = discover_xmls(roots, hospitals)
    if not records:
        raise RuntimeError("No SEG XML files found in the selected roots.")

    features, curves, parse_logs = parse_all(records, cache_dir, args.resample_len, include_views)
    features.to_csv(output_dir / "segment_features.csv", index=False)
    np.save(output_dir / "segment_curves_resampled.npy", curves)
    pd.DataFrame(parse_logs).to_csv(output_dir / "parse_log.csv", index=False)

    visit_var = visit_variability(features)
    visit_var.to_csv(output_dir / "visit_variability.csv", index=False)

    latent, loss_df, latent_method = train_autoencoder(
        curves,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    if not loss_df.empty:
        loss_df.to_csv(output_dir / "autoencoder_training_loss.csv", index=False)
    latent_segment, latent_var = latent_outputs(features, latent)
    latent_segment.to_csv(output_dir / "latent_segment_embeddings.csv", index=False)
    latent_var.to_csv(output_dir / "latent_visit_variability.csv", index=False)

    classic_metrics = [
        "peak_abs_std",
        "peak_abs_iqr",
        "time_to_peak_ms_std",
        "time_to_peak_norm_std",
        "strain_burden_std",
        "rms_strain_std",
        "contraction_rate_per_s_std",
        "recovery_fraction_std",
        "curve_roughness_std",
    ]
    classic_trends = trend_tests(visit_var, classic_metrics)
    classic_trends.to_csv(output_dir / "patient_trend_tests_classic.csv", index=False)
    classic_summary = trend_summary(classic_trends)
    classic_summary.to_csv(output_dir / "trend_summary_classic.csv", index=False)

    latent_trends = trend_tests(latent_var, ["latent_pairwise_mean", "latent_centroid_norm"])
    latent_trends.to_csv(output_dir / "patient_trend_tests_latent.csv", index=False)
    latent_summary = trend_summary(latent_trends)
    latent_summary.to_csv(output_dir / "trend_summary_latent.csv", index=False)

    metadata = {
        "roots": [str(r) for r in roots],
        "output_dir": str(output_dir),
        "resample_len": args.resample_len,
        "latent_dim": args.latent_dim,
        "epochs": args.epochs,
        "include_views": sorted(include_views),
        "latent_method": latent_method,
        "xml_records": len(records),
        "segment_features": len(features),
        "visit_variability_rows": len(visit_var),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    write_report(
        output_dir=output_dir,
        records=records,
        features=features,
        visit_var=visit_var,
        latent_method=latent_method,
        classic_summary=classic_summary,
        latent_summary=latent_summary,
        parse_logs=parse_logs,
    )

    print(f"Output: {output_dir}")
    print(f"XML files discovered: {len(records)}")
    print(f"Segment curves extracted: {len(features)}")
    print(f"Visit/layer rows: {len(visit_var)}")
    print(f"Latent method: {latent_method}")
    print(f"Report: {output_dir / 'research_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
