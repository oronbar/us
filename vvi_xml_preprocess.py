"""
VVI XML preprocessing pipeline
--------------------------------
Parses Siemens VVI SpreadsheetML XML files for strain analysis and builds a
patient-visit dataset with epicardium/endocardium/myocardium curves, their difference
(Endo - Epi), **GLS from myocardium**, and per-cycle segmentation. Cross-references an Excel registry
with patient metadata and DICOM filenames to tag each test with view (2C/4C).

Tested on Windows-style paths. Requires: pandas, numpy, lxml, openpyxl, pyarrow

Usage (PowerShell / cmd):

  python vvi_xml_preprocess.py \
    --vvi-dir "C:\\Users\\oronbarazani\\OneDrive - Technion\\DS\\Tags\\VVI\\Anonymous" \
    --registry-xlsx "C:\\Users\\oronbarazani\\OneDrive - Technion\\DS\\Cardio-Onco Echo SZMC\\SZMC_report.xlsx" \
    --echo-root "C:\\Users\\oronbarazani\\OneDrive - Technion\\DS\\Cardio-Onco Echo SZMC" \
    --out-parquet "C:\\Users\\oronbarazani\\OneDrive - Technion\\DS\\Tags\\VVI\\processed\\strain_dataset.parquet"

The resulting Parquet has one row per (patient, study, view, cycle, segment).
Columns include:
- short_id, name_initials, parent_folder, study_instance_uid, study_datetime, birth_datetime
- view ("2C"/"4C"), dicom_file, xml_path, cycle_index, segment (canonical key)
- time, epi, endo, myo, delta (lists of floats); gls (float), n_samples

Notes:
- Only Basal/Mid segments are kept; Apical rows are ignored.
- Cycle boundaries are inferred from **strain amplitude zeros**: a boundary is an index where **all kept segments** (endo & epi) are zero simultaneously. No extra knobs.
- Handles multiple cycles concatenated in each row (all segments/time synchronized).
- If a sheet or segment is missing, that XML is skipped with a warning.

"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from lxml import etree
import re

# ----------------------------- Logging ------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler(r"C:\work\us\vvi_xml_preprocess.log")  # File output
    ]
)
logger = logging.getLogger("vvi_xml_preprocess")

# ----------------------------- Constants ----------------------------------
# View-specific canonical segment maps (Basal/Mid only)
KEEP_SEGMENTS_4C = {
    "03-Basal inferoseptal": "basal_inferoseptal",
    "09-Mid inferoseptal": "mid_inferoseptal",
    "12-Mid anterolateral": "mid_anterolateral",
    "06-Basal anterolateral": "basal_anterolateral",
}
APICAL_SEGMENTS_4C = {"14-Apical septal", "16-Apical lateral"}

KEEP_SEGMENTS_2C = {
    "04-Basal inferior": "basal_inferior",
    "10-Mid inferior": "mid_inferior",
    "07-Mid anterior": "mid_anterior",
    "01-Basal anterior": "basal_anterior",
}
APICAL_SEGMENTS_2C = {"15-Apical inferior", "15-Apical inferior ", "13-Apical anterior"}

SHEETS_NEED = {"strain-epi", "strain-endo"}  # lowercase, stripped

# SpreadsheetML namespaces (Excel 2003 XML)
NS = {
    "ss": "urn:schemas-microsoft-com:office:spreadsheet",
    "o": "urn:schemas-microsoft-com:office:office",
    "x": "urn:schemas-microsoft-com:office:excel",
    "html": "http://www.w3.org/TR/REC-html40",
}

# Internal tolerance for zero detection (not exposed as CLI)
_ZERO_ATOL = 1e-12

def get_keep_map(view_tag: str) -> Dict[str, str]:
    view_tag = (view_tag or "").upper().strip()
    if view_tag == "2C":
        return KEEP_SEGMENTS_2C
    else:
        return KEEP_SEGMENTS_4C

# ----------------------------- Utilities ----------------------------------

def safe_float(x: str) -> Optional[float]:
    """Convert string to float, handling commas and stray spaces. Returns None on failure."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s == "":
        return None
    # Replace comma decimal separators if present
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def parse_datetime_underscore(s: str) -> Optional[pd.Timestamp]:
    """Parse YYYY_MM_DD_HH_MM_SS into pandas Timestamp. Returns None if empty/invalid."""
    if pd.isna(s):
        return None
    s = str(s).strip()
    if not s:
        return None
    try:
        parts = s.split("_")
        parts = [int(p) for p in parts]
        # Support YYYY_MM_DD or full YYYY_MM_DD_HH_MM_SS
        if len(parts) == 3:
            y, m, d = parts
            return pd.Timestamp(year=y, month=m, day=d)
        elif len(parts) >= 6:
            y, m, d, hh, mm, ss = parts[:6]
            return pd.Timestamp(year=y, month=m, day=d, hour=hh, minute=mm, second=ss)
        else:
            return None
    except Exception:
        return None


# ------------------------- SpreadsheetML Reader ---------------------------
@dataclass
class SheetTable:
    name: str
    table: List[List[Optional[str]]]

    def to_dataframe(self) -> pd.DataFrame:
        # Convert to DataFrame, filling missing columns to the max width
        max_cols = max((len(r) for r in self.table), default=0)
        rows = [r + [None] * (max_cols - len(r)) for r in self.table]
        df = pd.DataFrame(rows)
        # Name columns like Excel: A, B, C, ...
        cols = []
        for i in range(max_cols):
            n = i + 1
            label = ""
            while n:
                n, rem = divmod(n - 1, 26)
                label = chr(65 + rem) + label
            cols.append(label)
        df.columns = cols
        # Add 1-based row number to preserve Excel row addresses
        df["ROWNUM"] = np.arange(1, len(df) + 1)
        return df


class SpreadsheetML:
    def __init__(self, xml_path: Path):
        self.xml_path = Path(xml_path)
        self.tree = None

    def parse(self):
        try:
            parser = etree.XMLParser(recover=True, remove_comments=True)
            self.tree = etree.parse(str(self.xml_path), parser)
        except Exception as e:
            raise RuntimeError(f"Failed to parse XML {self.xml_path}: {e}")

    def list_sheets(self) -> List[str]:
        if self.tree is None:
            self.parse()
        sheets = self.tree.xpath("//ss:Worksheet", namespaces=NS)
        names = []
        for ws in sheets:
            name = ws.get("{%s}Name" % NS["ss"]) or ws.get("Name") or ""
            names.append(name)
        return names

    def get_sheet_table(self, wanted_name: str) -> Optional[SheetTable]:
        """
        Return a SheetTable for the worksheet whose name matches wanted_name
        ignoring case and surrounding spaces.
        Handles sparse rows via ss:Index by inserting blank rows.
        """
        if self.tree is None:
            self.parse()
        wanted_key = wanted_name.strip().lower()
        for ws in self.tree.xpath("//ss:Worksheet", namespaces=NS):
            name = ws.get("{%s}Name" % NS["ss"]) or ws.get("Name") or ""
            key = name.strip().lower()
            if key == wanted_key:
                table_el = ws.find("ss:Table", namespaces=NS)
                if table_el is None:
                    return SheetTable(name=name, table=[])
                table: List[List[Optional[str]]] = []
                row_cursor = 1  # 1-based Excel row numbering
                for row_el in table_el.findall("ss:Row", namespaces=NS):
                    # Honor ss:Index for row gaps
                    idx_attr_row = row_el.get("{%s}Index" % NS["ss"])  # 1-based
                    if idx_attr_row is not None:
                        target = int(idx_attr_row)
                        while row_cursor < target:
                            table.append([])  # blank row placeholder
                            row_cursor += 1
                    row: List[Optional[str]] = []
                    # Build dense cells honoring ss:Index on cells
                    col_cursor = 1
                    for cell_el in row_el.findall("ss:Cell", namespaces=NS):
                        idx_attr = cell_el.get("{%s}Index" % NS["ss"])  # 1-based
                        if idx_attr is not None:
                            target = int(idx_attr)
                            while col_cursor < target:
                                row.append(None)
                                col_cursor += 1
                        data_el = cell_el.find("ss:Data", namespaces=NS)
                        if data_el is not None:
                            txt = (data_el.text or "").strip()
                        else:
                            txt = (cell_el.text or "").strip()
                        row.append(txt if txt != "" else None)
                        col_cursor += 1
                    table.append(row)
                    row_cursor += 1
                return SheetTable(name=name, table=table)
        return None


# ---------------------------- VVI XML Parser ------------------------------
@dataclass
class VVIParseResult:
    xml_path: Path
    time: np.ndarray  # shape [N] (optional; empty if not found)
    epi: Dict[str, np.ndarray]  # canonical segment -> [N]
    endo: Dict[str, np.ndarray]
    myo: Dict[str, np.ndarray]
    delta: Dict[str, np.ndarray]


def extract_segments_from_sheet(df: pd.DataFrame, keep_map: Dict[str, str]) -> Dict[str, np.ndarray]:
    """
    Extract segments for **longitudinal strain only** from rows 13-18 inclusive
    (per your VVI layout). We ignore transverse blocks above. We also restrict
    to Column A labels matching `keep_map` to map to canonical names.
    """
    seg_series: Dict[str, np.ndarray] = {}
    # Restrict to longitudinal rows 13..18 (1-based)
    if "ROWNUM" in df.columns:
        df_long = df[(df["ROWNUM"] >= 13) & (df["ROWNUM"] <= 18)].copy()
    else:
        # Fallback: assume current order corresponds; slice by position
        df_long = df.iloc[12:18].copy()  # 0-based -> rows 13..18
    colA = df_long.get("A", pd.Series([], dtype=object)).astype(str).str.strip()
    for xml_label, canon in keep_map.items():
        matches = df_long[colA == xml_label]
        if matches.empty:
            # Allow minor trailing spaces in XML label
            matches = df_long[colA.str.replace("\s+$", "", regex=True) == xml_label]
        if matches.empty:
            logger.warning(f"Segment row not found in longitudinal rows 13-18: '{xml_label}'")
            continue
        row = matches.iloc[0]
        values: List[float] = []
        for col in df_long.columns:
            if col == "A" or col == "ROWNUM":
                continue
            v = row.get(col)
            f = safe_float(v)
            if f is None:
                continue
            values.append(f)
        if not values:
            logger.warning(f"No numeric values for segment (rows 13-18): '{xml_label}'")
            continue
        seg_series[canon] = np.asarray(values, dtype=float)
    return seg_series


def extract_time_row(df: pd.DataFrame) -> Optional[np.ndarray]:
    """Return the Time row from **row 21** primarily; fallback to label search."""
    # Prefer explicit row number 21 (longitudinal block time row)
    if "ROWNUM" in df.columns:
        hits = df[df["ROWNUM"] == 21]
        if not hits.empty:
            row = hits.iloc[0]
            values: List[float] = []
            for col in df.columns:
                if col in ("A", "ROWNUM"):
                    continue
                f = safe_float(row.get(col))
                if f is None:
                    continue
                values.append(f)
            if values:
                return np.asarray(values, dtype=float)
    # Fallback: name-based
    colA = df.get("A", pd.Series([], dtype=object)).astype(str).str.strip().str.lower()
    hits = df[colA == "time"]
    if hits.empty:
        hits = df[colA.str.contains("time", na=False)]
    if hits.empty:
        logger.warning("Time row not found in sheet; continuing without it.")
        return None
    row = hits.iloc[0]
    values: List[float] = []
    for col in df.columns:
        if col in ("A", "ROWNUM"):
            continue
        f = safe_float(row.get(col))
        if f is None:
            continue
        values.append(f)
    return np.asarray(values, dtype=float) if values else None


def split_cycles_by_all_zero(endo: Dict[str, np.ndarray], epi: Dict[str, np.ndarray]) -> List[Tuple[int, int]]:
    """Return (start_idx, end_idx) cycles using **only** indices where **all kept segments**
    (endo & epi) are exactly zero (within tiny internal tolerance). Consecutive zero indices are
    collapsed to a single boundary.
    """
    signals: List[np.ndarray] = []
    for d in (endo, epi):
        for seg, arr in d.items():
            if arr is not None:
                signals.append(np.asarray(arr, dtype=float))
    if not signals:
        return []
    N = min(len(a) for a in signals)
    if N == 0:
        return []

    # Boolean mask where **all** segments are (near) zero
    all_zero = np.ones(N, dtype=bool)
    for a in signals:
        all_zero &= np.isclose(a[:N], 0.0, atol=_ZERO_ATOL)

    idx = np.flatnonzero(all_zero)
    if idx.size < 2:
        return [(0, N - 1)]

    # Take the first index of each contiguous run as a boundary
    boundaries: List[int] = []
    prev = -10**9
    for i in idx:
        if i != prev + 1:
            boundaries.append(i)
        prev = i

    if len(boundaries) < 2:
        return [(0, N - 1)]

    cycles: List[Tuple[int, int]] = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        if b > a:
            cycles.append((a, b))
    return cycles if cycles else [(0, N - 1)]


def align_lengths(base: Optional[np.ndarray], *series_dicts: Dict[str, np.ndarray]) -> Tuple[Optional[np.ndarray], List[Dict[str, np.ndarray]]]:
    """Trim all arrays to the minimum length among base (if provided) and all provided series."""
    min_len = None if base is None else len(base)
    for d in series_dicts:
        for arr in d.values():
            L = len(arr)
            min_len = L if min_len is None else min(min_len, L)
    if min_len is None:
        return base, list(series_dicts)
    base_t = None if base is None else base[:min_len]
    out = []
    for d in series_dicts:
        out.append({k: v[:min_len] for k, v in d.items()})
    return base_t, out


def compute_gls_peak(myo: Dict[str, np.ndarray], a: int, b: int, keep_map: Dict[str, str]) -> Optional[float]:
    """
    GLS definition here: **peak (most negative) of the timewise average** across the 4 kept myocardium segments
    within the cycle slice [a:b]. Returns None if any required segment is missing or lengths mismatch.
    """
    seg_keys = list(keep_map.values())
    seg_arrays = []
    for k in seg_keys:
        arr = myo.get(k)
        if arr is None or len(arr) < (b + 1):
            logger.warning(f"Missing or short MYO segment '{k}' for GLS computation")
            return None
        seg_arrays.append(arr[a : b + 1])
    if not seg_arrays:
        return None
    try:
        stack = np.vstack(seg_arrays)  # [4, L]
        avg_curve = stack.mean(axis=0)  # [L]
        gls = float(avg_curve.min())    # most negative peak (typical GLS definition)
        return gls
    except Exception:
        return None


def parse_vvi_xml(xml_path: Path, keep_map: Dict[str, str]) -> Optional[VVIParseResult]:
    try:
        ssml = SpreadsheetML(xml_path)
        ssml.parse()
    except Exception as e:
        logger.error(str(e))
        return None

    # Fetch sheets (names may have leading spaces). Try exact then tolerant.
    epi_sheet = ssml.get_sheet_table("Strain-Epi") or ssml.get_sheet_table(" Strain-Epi")
    endo_sheet = ssml.get_sheet_table("Strain-Endo") or ssml.get_sheet_table(" Strain-Endo")
    myo_sheet = ssml.get_sheet_table("Strain-Myo") or ssml.get_sheet_table(" Strain-Myo")
    if epi_sheet is None or endo_sheet is None or myo_sheet is None:
        logger.warning(f"Missing required sheets in {xml_path.name} (found: {ssml.list_sheets()})")
        return None

    epi_df = epi_sheet.to_dataframe()
    endo_df = endo_sheet.to_dataframe()
    myo_df = myo_sheet.to_dataframe()

    # Extract time row (optional; we keep for reference/QC)
    time_arr = extract_time_row(endo_df)

    # Extract numeric series per kept segment
    epi = extract_segments_from_sheet(epi_df, keep_map)
    endo = extract_segments_from_sheet(endo_df, keep_map)
    myo = extract_segments_from_sheet(myo_df, keep_map)

    if not epi or not endo or not myo:
        logger.warning(f"No segment data in {xml_path.name}; skipping.")
        return None

    # Align lengths (trim to common length across all three layers)
    time_arr, (epi, endo, myo) = align_lengths(time_arr, epi, endo, myo)

    # Compute delta = Endo - Epi
    delta: Dict[str, np.ndarray] = {}
    for seg in keep_map.values():
        if seg in epi and seg in endo:
            delta[seg] = endo[seg] - epi[seg]
        else:
            logger.warning(f"Segment missing for delta: {seg}")

    return VVIParseResult(
        xml_path=xml_path,
        time=time_arr if time_arr is not None else np.array([]),
        epi=epi,
        endo=endo,
        myo=myo,
        delta=delta,
    )


# --------------------------- Registry join --------------------------------
@dataclass
class RegistryRow:
    short_id: str
    name_initials: str
    parent_folder: str
    study_instance_uid: str
    study_datetime: Optional[pd.Timestamp]
    birth_datetime: Optional[pd.Timestamp]
    dicoms_2c: List[str]
    dicoms_4c: List[str]


def _split_dicom_list(val: object) -> List[str]:
    if pd.isna(val):
        return []
    s = str(val).strip()
    if not s:
        return []
    # Allow separators ; , whitespace
    parts = [p.strip() for p in re_split(r"[;,\s]+", s) if p.strip()]
    # Ensure .dcm suffix consistency is handled later; store basename without extension
    parts = [p[:-4] if p.lower().endswith(".dcm") else p for p in parts]
    return parts



def re_split(pattern: str, s: str) -> List[str]:
    return re.split(pattern, s)


def load_registry(registry_xlsx: Path) -> pd.DataFrame:
    df = pd.read_excel(registry_xlsx, engine="openpyxl")
    # Standardize columns (strip)
    df.columns = [c.strip() for c in df.columns]
    required = [
        "Short ID",
        "Name intials",
        "Parent Folder",
        "Study Instance UID",
        "Study Date",
        "Birth Date",
        "2-Chambers",
        "4-Chambers",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in registry: {missing}")

    # Parse datetimes
    df["study_datetime"] = df["Study Date"].apply(parse_datetime_underscore)
    df["birth_datetime"] = df["Birth Date"].apply(parse_datetime_underscore)

    # DICOM file (basename without extension) lists
    df["dicoms_2c"] = df["2-Chambers"].apply(_split_dicom_list)
    df["dicoms_4c"] = df["4-Chambers"].apply(_split_dicom_list)

    # Keep trimmed essential fields
    keep_cols = [
        "Short ID",
        "Name intials",
        "Parent Folder",
        "Study Instance UID",
        "study_datetime",
        "birth_datetime",
        "dicoms_2c",
        "dicoms_4c",
    ]
    return df[keep_cols].copy()


# --------------------------- Dataset builder ------------------------------

def find_xml_for_dicom(vvi_dir: Path, dicom_basename: str) -> Optional[Path]:
    """Given a DICOM basename (without .dcm), locate its single XML under vvi_dir/dicom_basename.dcm/*.xml"""
    folder = vvi_dir / f"{dicom_basename}.dcm"
    if not folder.is_dir():
        return None
    xmls = list(folder.glob("*.xml"))
    if not xmls:
        return None
    if len(xmls) > 1:
        logger.warning(f"Multiple XMLs found in {folder}; using first: {xmls[0].name}")
    return xmls[0]


def dicom_full_path(echo_root: Path, parent_folder: str, study_uid: str, dicom_basename: str) -> Path:
    return echo_root / parent_folder / study_uid / f"{dicom_basename}.dcm"


def build_dataset(
    vvi_dir: Path,
    registry_xlsx: Path,
    echo_root: Path,
    out_parquet: Path,
) -> pd.DataFrame:
    reg = load_registry(registry_xlsx)

    records: List[dict] = []

    for idx, row in reg.iterrows():
        short_id = str(row["Short ID"]) if not pd.isna(row["Short ID"]) else None
        name_initials = str(row["Name intials"]).strip() if not pd.isna(row["Name intials"]) else None
        parent_folder = str(row["Parent Folder"]).strip() if not pd.isna(row["Parent Folder"]) else None
        study_uid = str(row["Study Instance UID"]).strip() if not pd.isna(row["Study Instance UID"]) else None
        study_dt = row.get("study_datetime")
        birth_dt = row.get("birth_datetime")

        for view_col, view_tag in [("dicoms_2c", "2C"), ("dicoms_4c", "4C")]:
            dicom_list: List[str] = row[view_col] or []
            for dicom_base in dicom_list:
                xml_path = find_xml_for_dicom(vvi_dir, dicom_base)
                if xml_path is None:
                    logger.warning(
                        f"XML not found for DICOM '{dicom_base}' under {vvi_dir} (row {idx})"
                    )
                    continue
                keep_map = get_keep_map(view_tag)
                parsed = parse_vvi_xml(xml_path, keep_map)
                if parsed is None:
                    continue

                # Split cycles at indices where **all segments (endo & epi) are zero**
                cycles = split_cycles_by_all_zero(parsed.endo, parsed.epi)
                if not cycles:
                    # treat as single series
                    total_len = len(next(iter(parsed.endo.values())))
                    cycles = [(0, total_len - 1)]

                for ci, (a, b) in enumerate(cycles):
                    t_slice = (
                        parsed.time[a : b + 1].tolist() if parsed.time is not None and parsed.time.size else list(range(b - a + 1))
                    )
                    n = len(t_slice)
                    # Compute cycle-level GLS once per cycle
                    gls_val = compute_gls_peak(parsed.myo, a, b, keep_map)
                    for seg in keep_map.values():
                        endo_arr = parsed.endo.get(seg)
                        epi_arr = parsed.epi.get(seg)
                        myo_arr = parsed.myo.get(seg)
                        delta_arr = parsed.delta.get(seg)
                        if endo_arr is None or epi_arr is None or delta_arr is None or myo_arr is None:
                            continue
                        rec = {
                            "short_id": short_id,
                            "name_initials": name_initials,
                            "parent_folder": parent_folder,
                            "study_instance_uid": study_uid,
                            "study_datetime": study_dt,
                            "birth_datetime": birth_dt,
                            "view": view_tag,
                            "dicom_file": f"{dicom_base}.dcm",
                            "xml_path": str(parsed.xml_path),
                            "cycle_index": ci,
                            "cycle_start_idx": int(a),
                            "cycle_end_idx": int(b),
                            "segment": seg,
                            "time": t_slice,
                            "endo": endo_arr[a : b + 1].tolist(),
                            "epi": epi_arr[a : b + 1].tolist(),
                            "myo": myo_arr[a : b + 1].tolist(),
                            "delta": delta_arr[a : b + 1].tolist(),
                            "gls": gls_val,
                            "n_samples": int(n),
                        }
                        # DICOM full path (useful for QC)
                        try:
                            rec["dicom_full_path"] = str(
                                dicom_full_path(echo_root, parent_folder, study_uid, dicom_base)
                            )
                        except Exception:
                            rec["dicom_full_path"] = None
                        records.append(rec)

    if not records:
        logger.error("No records parsed. Check paths and XML structure.")
        df_out = pd.DataFrame()
    else:
        df_out = pd.DataFrame.from_records(records)
        # Sort for readability
        df_out.sort_values(
            by=["short_id", "study_datetime", "view", "dicom_file", "cycle_index", "segment"],
            inplace=True,
        )

    # Ensure output directory exists
    out_parquet = Path(out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    if not df_out.empty:
        df_out.to_parquet(out_parquet, index=False)
        logger.info(
            f"Saved dataset: {out_parquet} with {len(df_out)} rows "
            f"({df_out['short_id'].nunique()} patients, {df_out['study_instance_uid'].nunique()} studies)."
        )
    else:
        logger.warning("Skipped saving empty dataset.")

    return df_out


# --------------------------- CLI Interface --------------------------------

def main():
    import os
    user_home = os.path.expanduser("~")  # Gets user home directory
    
    p = argparse.ArgumentParser(description="Parse VVI XML strain analyses into a dataset")
    p.add_argument(
        "--vvi-dir",
        type=Path,
        required=False,
        default=Path(user_home) / "OneDrive - Technion" / "DS" / "Tags" / "VVI" / "Anonymous",
        help="Path to VVI Anonymous directory containing <dicom>.dcm subfolders",
    )
    p.add_argument(
        "--registry-xlsx",
        type=Path,
        required=False,
        default=Path(user_home) / "OneDrive - Technion" / "DS" / "Cardio-Onco Echo SZMC" / "SZMC_report_oron.xlsx",
        help="Registry Excel path",
    )
    p.add_argument(
        "--echo-root",
        type=Path,
        required=False,
        default=Path(user_home) / "OneDrive - Technion" / "DS" / "Cardio-Onco Echo SZMC",
        help="Root directory where DICOMs reside (Parent_folder/Study_UID/<dicom>.dcm)",
    )
    p.add_argument(
        "--out-parquet",
        type=Path,
        required=False,
        default=Path(user_home) / "OneDrive - Technion" / "DS" / "Tags" / "VVI" / "processed" / "strain_dataset.parquet",
        help="Output Parquet path",
    )
    args = p.parse_args()

    _ = build_dataset(
        vvi_dir=args.vvi_dir,
        registry_xlsx=args.registry_xlsx,
        echo_root=args.echo_root,
        out_parquet=args.out_parquet,
    )
    
    # Load registry and compute statistics
    reg = pd.read_excel(args.registry_xlsx, engine="openpyxl")
    reg.columns = [c.strip() for c in reg.columns]
    
    # All patients with at least 3 studies
    all_study_counts = reg.groupby("Short ID")["Study Instance UID"].nunique()
    num_all_with_3plus = (all_study_counts >= 3).sum()
    
    # Processed patients (have both 2C and 4C DICOM filenames)
    reg_processed = reg[
        (reg["2-Chambers"].notna()) & (reg["2-Chambers"] != "") &
        (reg["4-Chambers"].notna()) & (reg["4-Chambers"] != "")
    ]
    proc_study_counts = reg_processed.groupby("Short ID")["Study Instance UID"].nunique()
    num_proc_with_3plus = (proc_study_counts >= 3).sum()
    
    logger.info(f"Total unique patients with ≥3 studies: {num_all_with_3plus}")
    logger.info(f"Processed patients with ≥3 studies: {num_proc_with_3plus}")

if __name__ == "__main__":
    main()