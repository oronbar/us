"""
Plot global longitudinal strain curves (mid/myo and endo) from a strain report.

Supports Excel 2003 XML (SpreadsheetML) exports like VVI reports. The script:
  - extracts the "Longitudinal Strain" block from Strain-Myo and Strain-Endo sheets
  - maps segment numbers to apical views (A2C/A3C/A4C)
  - averages segment curves per view to produce global curves
  - saves plots for mid and endo layers

Usage (PowerShell):
  .venv\\Scripts\\python plot_global_strain_curves.py ^
    --input "C:\\path\\to\\strain_report.xml"

Optional:
  --segment-order A4C,A2C,A3C  (override mapping by 6-segment blocks if needed)
"""
from __future__ import annotations

import argparse
import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("plot_global_strain_curves")


NS = {
    "ss": "urn:schemas-microsoft-com:office:spreadsheet",
}


# 18-segment model (6 basal + 6 mid + 6 apical). Each apical view has 6 segments.
SEGMENT_TO_VIEW_18: Dict[int, str] = {
    # A2C: anterior + inferior
    1: "A2C",
    4: "A2C",
    7: "A2C",
    10: "A2C",
    13: "A2C",
    16: "A2C",
    # A3C: anteroseptal + inferolateral (posterior)
    2: "A3C",
    5: "A3C",
    8: "A3C",
    11: "A3C",
    14: "A3C",
    17: "A3C",
    # A4C: inferoseptal + anterolateral
    3: "A4C",
    6: "A4C",
    9: "A4C",
    12: "A4C",
    15: "A4C",
    18: "A4C",
}


def _to_float(val: object) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, (int, float, np.number)):
        return float(val)
    s = str(val).strip()
    if not s:
        return None
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _worksheet_table(xml_path: Path, wanted_name: str) -> Optional[List[List[Optional[str]]]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    wanted_key = wanted_name.strip().lower()
    for ws in root.findall(".//ss:Worksheet", NS):
        name = ws.get("{%s}Name" % NS["ss"]) or ws.get("Name") or ""
        if name.strip().lower() != wanted_key:
            continue
        table_el = ws.find("ss:Table", NS)
        if table_el is None:
            return []
        table: List[List[Optional[str]]] = []
        row_cursor = 1
        for row_el in table_el.findall("ss:Row", NS):
            idx_attr_row = row_el.get("{%s}Index" % NS["ss"])
            if idx_attr_row is not None:
                target = int(idx_attr_row)
                while row_cursor < target:
                    table.append([])
                    row_cursor += 1
            row: List[Optional[str]] = []
            col_cursor = 1
            for cell_el in row_el.findall("ss:Cell", NS):
                idx_attr = cell_el.get("{%s}Index" % NS["ss"])
                if idx_attr is not None:
                    target = int(idx_attr)
                    while col_cursor < target:
                        row.append(None)
                        col_cursor += 1
                data_el = cell_el.find("ss:Data", NS)
                if data_el is not None:
                    txt = (data_el.text or "").strip()
                else:
                    txt = (cell_el.text or "").strip()
                row.append(txt if txt != "" else None)
                col_cursor += 1
            table.append(row)
            row_cursor += 1
        return table
    return None


def _table_to_df(table: List[List[Optional[str]]]) -> pd.DataFrame:
    if not table:
        return pd.DataFrame()
    max_cols = max((len(r) for r in table), default=0)
    rows = [r + [None] * (max_cols - len(r)) for r in table]
    df = pd.DataFrame(rows)
    # Column labels A, B, C, ...
    cols = []
    for i in range(max_cols):
        n = i + 1
        label = ""
        while n:
            n, rem = divmod(n - 1, 26)
            label = chr(65 + rem) + label
        cols.append(label)
    df.columns = cols
    return df


def _extract_longitudinal_block(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    col_a = df.get("A", pd.Series([], dtype=object)).astype(str)
    mask = col_a.str.contains("Longitudinal Strain", case=False, na=False)
    if not mask.any():
        return pd.DataFrame()
    start_idx = int(mask.idxmax())
    rows = []
    for _, row in df.iloc[start_idx + 1 :].iterrows():
        values = [row.get(c) for c in df.columns]
        if not any(v not in (None, "") for v in values):
            break
        rows.append(values)
    if not rows:
        return pd.DataFrame()
    headers = [r[0] for r in rows]
    data_rows = [r[1:] for r in rows]
    max_len = max(len(r) for r in data_rows)
    padded = [r + [None] * (max_len - len(r)) for r in data_rows]
    out = pd.DataFrame(padded).T
    out.columns = headers
    for col in out.columns:
        out[col] = pd.to_numeric(out[col].map(_to_float), errors="coerce")
    return out


def _parse_segment_label(label: object) -> Tuple[Optional[int], str]:
    if label is None:
        return None, ""
    s = str(label).strip()
    if not s:
        return None, ""
    m = re.match(r"^\s*(\d+)", s)
    num = int(m.group(1)) if m else None
    name = re.sub(r"^\s*\d+\s*[-:_]*\s*", "", s).strip().lower()
    return num, name


def _medical_name_from_text(name: str) -> Optional[str]:
    if not name:
        return None
    n = name.lower()
    if "anteroseptal" in n or "ant/sept" in n:
        return "ant/sept"
    if "inferoseptal" in n or "inf/sept" in n:
        return "inf/sept"
    if "anterolateral" in n or "ant/lat" in n:
        return "ant/lat"
    if "inferolateral" in n or "inf/lat" in n or "posterior" in n:
        return "inf/lat"
    if "anterior" in n or n == "ant":
        return "ant"
    if "inferior" in n or n == "inf":
        return "inf"
    if "septal" in n:
        return "sept"
    if "lateral" in n:
        return "lat"
    return None


def _view_from_name(name: str) -> Optional[str]:
    if not name:
        return None
    n = name.lower()
    if "anteroseptal" in n or "inferolateral" in n or "posterior" in n:
        return "A3C"
    if "inferoseptal" in n or "anterolateral" in n:
        return "A4C"
    if "septal" in n or "lateral" in n:
        return "A4C"
    if "anterior" in n or "inferior" in n:
        return "A2C"
    return None


def _build_block_map(order: Optional[str]) -> Dict[int, str]:
    if not order:
        return {}
    views = [v.strip().upper() for v in order.split(",") if v.strip()]
    if len(views) != 3 or any(v not in {"A2C", "A3C", "A4C"} for v in views):
        raise ValueError("segment-order must be 3 comma-separated views, e.g. A4C,A2C,A3C")
    mapping: Dict[int, str] = {}
    for i in range(1, 19):
        block = (i - 1) // 6
        mapping[i] = views[block]
    return mapping


def _assign_views(columns: List[object], block_map: Dict[int, str]) -> Dict[object, Optional[str]]:
    out: Dict[object, Optional[str]] = {}
    for col in columns:
        num, name = _parse_segment_label(col)
        view = _view_from_name(name)
        if view is None and num is not None:
            view = SEGMENT_TO_VIEW_18.get(num)
        if view is None and num is not None and block_map:
            view = block_map.get(num)
        out[col] = view
    return out


def _global_curve(df: pd.DataFrame, cols: List[object]) -> Optional[np.ndarray]:
    if not cols:
        return None
    arrays = []
    for col in cols:
        series = pd.to_numeric(df[col], errors="coerce")
        arr = series.to_numpy(dtype=float)
        if np.isfinite(arr).any():
            arrays.append(arr)
    if not arrays:
        return None
    min_len = min(len(a) for a in arrays)
    if min_len < 2:
        return None
    stack = np.vstack([a[:min_len] for a in arrays])
    with np.errstate(invalid="ignore"):
        return np.nanmean(stack, axis=0)


def _global_curve_from_arrays(arrays: List[np.ndarray]) -> Optional[np.ndarray]:
    if not arrays:
        return None
    min_len = min(len(a) for a in arrays)
    if min_len < 2:
        return None
    stack = np.vstack([a[:min_len] for a in arrays])
    with np.errstate(invalid="ignore"):
        return np.nanmean(stack, axis=0)

def _plot_curves(curves: Dict[str, np.ndarray], title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = {"A2C": "#1f77b4", "A3C": "#2ca02c", "A4C": "#d62728"}
    for view in ("A2C", "A3C", "A4C"):
        curve = curves.get(view)
        if curve is None:
            continue
        ax.plot(np.arange(len(curve)), curve, label=view, linewidth=1.8, color=colors.get(view))
    ax.set_title(title)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Strain")
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    logger.info("Saved plot: %s", out_path)


def _load_layer_df(xml_path: Path, sheet_name: str) -> pd.DataFrame:
    table = _worksheet_table(xml_path, sheet_name)
    if table is None:
        return pd.DataFrame()
    df = _table_to_df(table)
    return _extract_longitudinal_block(df)


def _plot_view_layers(view: str, curves: Dict[str, np.ndarray], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = {"endo": "#1f77b4", "mid": "#d62728"}
    for layer in ("endo", "mid"):
        curve = curves.get(layer)
        if curve is None:
            continue
        peak = float(np.nanmin(curve)) if np.isfinite(curve).any() else float("nan")
        label = f"{layer} (peak GLS {peak:.2f})" if np.isfinite(peak) else f"{layer} (peak GLS n/a)"
        ax.plot(np.arange(len(curve)), curve, label=label, linewidth=2.6, color=colors.get(layer))
    ax.set_title(f"{view} global strain")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Strain")
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    logger.info("Saved plot: %s", out_path)


def _plot_view_delta(view: str, curve: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(np.arange(len(curve)), curve, label="endo - mid", linewidth=2.6, color="#6a3d9a")
    ax.set_title(f"{view} transmural strain (endo - mid)")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Strain")
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    logger.info("Saved plot: %s", out_path)


def _is_valid_curve(curve: np.ndarray, eps: float = 1e-8) -> bool:
    if curve is None or len(curve) < 2:
        return False
    arr = np.asarray(curve, dtype=float)
    return np.isfinite(arr).any() and np.nanmax(np.abs(arr)) > eps


def _plot_segment_curves(
    layer: str, seg_curves: Dict[int, np.ndarray], seg_names: Dict[int, str], out_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for seg_id, curve in sorted(seg_curves.items()):
        name = seg_names.get(seg_id)
        label = f"Seg {seg_id} ({name})" if name else f"Seg {seg_id}"
        ax.plot(np.arange(len(curve)), curve, linewidth=2.2, alpha=0.8, label=label)
    ax.set_title(f"{layer} valid segment strains")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Strain")
    ax.grid(True, alpha=0.25, linewidth=0.8)
    legend_cols = 1 if len(seg_curves) <= 10 else 2
    ax.legend(loc="best", frameon=False, fontsize=7, ncol=legend_cols)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    logger.info("Saved plot: %s", out_path)


def _parse_curves_txt(lines: List[str]) -> Tuple[Dict[str, Dict[int, List[float]]], Dict[str, Dict[int, str]]]:
    sections: Dict[str, Dict[int, List[float]]] = {}
    names: Dict[str, Dict[int, str]] = {"endo": {}, "mid": {}}
    i = 0
    n = len(lines)
    in_curves = False
    while i < n:
        line = lines[i].strip()
        if line.startswith("Curves LvApical"):
            in_curves = True
            i += 1
            continue
        if line.startswith("Results LvApical"):
            in_curves = False
            i += 1
            continue
        if in_curves and line in ("Longitudinal Strain Endo", "Longitudinal Strain Mid"):
            layer = "endo" if "Endo" in line else "mid"
            i += 1
            # Skip unit line if present
            if i < n and lines[i].strip().startswith("Unit"):
                i += 1
            segment_map: Dict[int, List[float]] = {}
            while i < n:
                row = lines[i].strip()
                if not row:
                    i += 1
                    if segment_map:
                        break
                    continue
                if row.startswith("Curves LvApical") or row.startswith("Longitudinal Strain"):
                    break
                if row.startswith("-----") or row.startswith("====="):
                    i += 1
                    continue
                parts = [p.strip() for p in row.split(",") if p.strip() != ""]
                if not parts:
                    i += 1
                    continue
                label = parts[0].lower()
                if label.startswith("segment"):
                    m = re.match(r"segment\s+(\d+)", parts[0], re.IGNORECASE)
                    if m:
                        seg_idx = int(m.group(1))
                        name_match = re.search(r"\[(.+?)\]", parts[0])
                        if name_match:
                            seg_name = name_match.group(1).strip()
                            medical = _medical_name_from_text(seg_name)
                            if medical:
                                names[layer][seg_idx] = medical
                        values = [_to_float(v) for v in parts[1:]]
                        segment_map[seg_idx] = [v for v in values if v is not None]
                i += 1
            if segment_map:
                sections[layer] = segment_map
            continue
        i += 1
    return sections, names


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot global strain curves (mid/endo) from VVI XML report.")
    ap.add_argument("--input", type=Path, required=True, help="Path to SpreadsheetML XML report")
    ap.add_argument("--sheet-endo", type=str, default="Strain-Endo", help="Sheet name for endo curves")
    ap.add_argument("--sheet-mid", type=str, default="Strain-Myo", help="Sheet name for mid/myo curves")
    ap.add_argument(
        "--segment-order",
        type=str,
        default="",
        help="Optional 3-view order for segments 1-6/7-12/13-18 (e.g. A4C,A2C,A3C)",
    )
    ap.add_argument("--output-dir", type=Path, default=None, help="Output directory for plots")
    args = ap.parse_args()

    input_path = args.input
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    out_root = args.output_dir or (input_path.parent / "global_strain_plots")
    out_dir = out_root / input_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    block_map = _build_block_map(args.segment_order) if args.segment_order else {}

    try:
        endo_df = _load_layer_df(input_path, args.sheet_endo)
        mid_df = _load_layer_df(input_path, args.sheet_mid)
        curves_by_view: Dict[str, Dict[str, np.ndarray]] = {"A2C": {}, "A3C": {}, "A4C": {}}
        segment_curves: Dict[str, Dict[int, np.ndarray]] = {"endo": {}, "mid": {}}
        valid_by_view: Dict[str, Dict[str, List[np.ndarray]]] = {
            "endo": {"A2C": [], "A3C": [], "A4C": []},
            "mid": {"A2C": [], "A3C": [], "A4C": []},
        }
        seg_names: Dict[str, Dict[int, str]] = {"endo": {}, "mid": {}}
        for layer_name, df in [("endo", endo_df), ("mid", mid_df)]:
            if df.empty:
                logger.warning(
                    "No longitudinal strain data found in sheet '%s'.",
                    args.sheet_endo if layer_name == "endo" else args.sheet_mid,
                )
                continue
            view_map = _assign_views(list(df.columns), block_map)
            for col in df.columns:
                seg_id, seg_name = _parse_segment_label(col)
                if seg_id is None:
                    continue
                series = pd.to_numeric(df[col], errors="coerce")
                arr = series.to_numpy(dtype=float)
                if _is_valid_curve(arr):
                    segment_curves[layer_name][seg_id] = arr
                    medical = _medical_name_from_text(seg_name)
                    if medical:
                        seg_names[layer_name][seg_id] = medical
                    view = view_map.get(col)
                    if view in ("A2C", "A3C", "A4C"):
                        valid_by_view[layer_name][view].append(arr)
            for view in ("A2C", "A3C", "A4C"):
                curve = _global_curve_from_arrays(valid_by_view[layer_name][view])
                if curve is not None:
                    curves_by_view[view][layer_name] = curve
        for view, layer_curves in curves_by_view.items():
            if not layer_curves:
                continue
            out_path = out_dir / f"global_{view}_strain_curves.png"
            _plot_view_layers(view, layer_curves, out_path)
            if "endo" in layer_curves and "mid" in layer_curves:
                endo = layer_curves["endo"]
                mid = layer_curves["mid"]
                n = min(len(endo), len(mid))
                if n > 1:
                    delta = endo[:n] - mid[:n]
                    delta_path = out_dir / f"global_{view}_endo_minus_mid.png"
                    _plot_view_delta(view, delta, delta_path)
        for layer_name, segs in segment_curves.items():
            if not segs:
                continue
            seg_out = out_dir / f"segments_{layer_name}.png"
            _plot_segment_curves(layer_name, segs, seg_names.get(layer_name, {}), seg_out)
        return
    except ET.ParseError:
        logger.info("Input is not SpreadsheetML XML. Falling back to text/CSV export parser.")

    lines = input_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    sections, segment_names = _parse_curves_txt(lines)
    if not sections:
        raise RuntimeError("No longitudinal strain curve sections found in the text export.")

    view_to_segments: Dict[str, List[int]] = {"A2C": [], "A3C": [], "A4C": []}
    for seg_id, view in SEGMENT_TO_VIEW_18.items():
        view_to_segments[view].append(seg_id)

    curves_by_view: Dict[str, Dict[str, np.ndarray]] = {"A2C": {}, "A3C": {}, "A4C": {}}
    segment_curves: Dict[str, Dict[int, np.ndarray]] = {"endo": {}, "mid": {}}
    for layer_name in ("endo", "mid"):
        seg_map = sections.get(layer_name)
        if not seg_map:
            logger.warning("No %s curves found in text export.", layer_name)
            continue
        for seg_id, values in seg_map.items():
            arr = np.asarray(values, dtype=float)
            if _is_valid_curve(arr):
                segment_curves[layer_name][seg_id] = arr
        for view in ("A2C", "A3C", "A4C"):
            arrays = [segment_curves[layer_name][sid] for sid in view_to_segments[view] if sid in segment_curves[layer_name]]
            curve = _global_curve_from_arrays(arrays)
            if curve is not None:
                curves_by_view[view][layer_name] = curve
    for view, layer_curves in curves_by_view.items():
        if not layer_curves:
            continue
        out_path = out_dir / f"global_{view}_strain_curves.png"
        _plot_view_layers(view, layer_curves, out_path)
        if "endo" in layer_curves and "mid" in layer_curves:
            endo = layer_curves["endo"]
            mid = layer_curves["mid"]
            n = min(len(endo), len(mid))
            if n > 1:
                delta = endo[:n] - mid[:n]
                delta_path = out_dir / f"global_{view}_endo_minus_mid.png"
                _plot_view_delta(view, delta, delta_path)
    for layer_name, segs in segment_curves.items():
        if not segs:
            continue
        seg_out = out_dir / f"segments_{layer_name}.png"
        _plot_segment_curves(layer_name, segs, segment_names.get(layer_name, {}), seg_out)


if __name__ == "__main__":
    main()
