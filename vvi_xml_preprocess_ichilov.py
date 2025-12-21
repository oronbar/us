"""
Ichilov VVI XML preprocessing pipeline
---------------------------------------
Parses Siemens VVI SpreadsheetML XML files for strain analysis and builds a
patient-visit dataset with epicardium/endocardium/myocardium curves, their
difference (Endo - Epi), GLS from myocardium, and per-cycle segmentation.

Differences vs SZMC version:
- VVI exports are stored per patient folder under Tags_Ichilov\\VVI (no "Anonymous").
- Registry columns: PatientNum, ID, Study Date, 2-Chambers, 4-Chambers.
- DICOMs live under Ichilov\\<patient_num>\\<YYYY-MM-DD or YYYY_MM_DD>\\*.dcm
  where <patient_num> matches PatientNum (with or without zero padding).

Usage (PowerShell / cmd):

  python vvi_xml_preprocess_ichilov.py ^
    --vvi-dir "C:\\Users\\oronbarazani\\OneDrive - Technion\\DS\\Tags_Ichilov\\VVI" ^
    --registry-xlsx "C:\\Users\\oronbarazani\\OneDrive - Technion\\DS\\Report_Ichilov_oron.xlsx" ^
    --echo-root "C:\\Users\\oronbarazani\\OneDrive - Technion\\DS\\Ichilov" ^
    --out-parquet "C:\\Users\\oronbarazani\\OneDrive - Technion\\DS\\Tags_Ichilov\\VVI\\processed\\strain_dataset_ichilov.parquet"

The resulting Parquet has one row per (patient, study, view, cycle, segment).
Columns include:
- patient_num, patient_id, study_datetime
- view ("2C"/"4C"), dicom_file, dicom_full_path, xml_path, cycle_index, segment
- time, epi, endo, myo, delta (lists of floats); gls (float), n_samples

Notes:
- Only Basal/Mid segments are kept; Apical rows are ignored.
- Cycle boundaries are inferred from strain amplitude zeros: a boundary is an
  index where all kept segments (endo & epi) are zero simultaneously.
- Handles multiple cycles concatenated in each row (all segments/time synchronized).
- If a sheet or segment is missing, that XML is skipped with a warning.
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

import numpy as np
import pandas as pd
from lxml import etree

# ----------------------------- Logging ------------------------------------
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(stream=sys.stdout),
        logging.FileHandler(r"C:\work\us\vvi_xml_preprocess_ichilov.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("vvi_xml_preprocess_ichilov")

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

SHEETS_NEED = {"strain-epi", "strain-endo"}

NS = {
    "ss": "urn:schemas-microsoft-com:office:spreadsheet",
    "o": "urn:schemas-microsoft-com:office:office",
    "x": "urn:schemas-microsoft-com:office:excel",
    "html": "http://www.w3.org/TR/REC-html40",
}

_ZERO_ATOL = 1e-12


def get_keep_map(view_tag: str) -> Dict[str, str]:
    view_tag = (view_tag or "").upper().strip()
    if view_tag == "2C":
        return KEEP_SEGMENTS_2C
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
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def parse_datetime_generic(val: object) -> Optional[pd.Timestamp]:
    """Parse a wide range of date strings (supports underscores)."""
    if pd.isna(val):
        return None
    if isinstance(val, pd.Timestamp):
        return val
    if isinstance(val, (int, float)):
        # Excel serial or year digits
        try:
            return pd.to_datetime(val, errors="coerce")
        except Exception:
            return None
    s = str(val).strip()
    if not s:
        return None
    s = s.replace("_", "-").replace("\\", "-").replace("/", "-")
    try:
        dt = pd.to_datetime(s, errors="coerce")
        return None if pd.isna(dt) else dt
    except Exception:
        return None


def re_split(pattern: str, s: str) -> List[str]:
    return re.split(pattern, s)


def _split_dicom_list(val: object) -> List[str]:
    if pd.isna(val):
        return []
    s = str(val).strip()
    if not s:
        return []
    parts = [p.strip() for p in re_split(r"[;,\s]+", s) if p.strip()]
    parts = [p[:-4] if p.lower().endswith(".dcm") else p for p in parts]
    return parts


def _candidate_patient_keys(patient_num: str) -> List[str]:
    s = str(patient_num).strip()
    return list(dict.fromkeys([s, s.zfill(2), s.zfill(3)]))


def _match_patient_dir(name: str, patient_num: str) -> bool:
    keys = _candidate_patient_keys(patient_num)
    n = name.lower()
    for k in keys:
        k_low = k.lower()
        if n == k_low:
            return True
        if n.startswith(f"{k_low}_") or n.startswith(f"{k_low} "):
            return True
    return False


def _date_folder_candidates(dt: Optional[pd.Timestamp]) -> List[str]:
    if dt is None or pd.isna(dt):
        return []
    try:
        return [dt.strftime("%Y-%m-%d"), dt.strftime("%Y_%m_%d")]
    except Exception:
        return []


# ------------------------- SpreadsheetML Reader ---------------------------
@dataclass
class SheetTable:
    name: str
    table: List[List[Optional[str]]]

    def to_dataframe(self) -> pd.DataFrame:
        max_cols = max((len(r) for r in self.table), default=0)
        rows = [r + [None] * (max_cols - len(r)) for r in self.table]
        df = pd.DataFrame(rows)
        cols = []
        for i in range(max_cols):
            n = i + 1
            label = ""
            while n:
                n, rem = divmod(n - 1, 26)
                label = chr(65 + rem) + label
            cols.append(label)
        df.columns = cols
        df["ROWNUM"] = np.arange(1, len(df) + 1)
        return df


class SpreadsheetML:
    def __init__(self, xml_path: Path):
        self.xml_path = Path(xml_path)
        self.tree = None

    def parse(self):
        try:
            parser = etree.XMLParser(recover=True, remove_comments=True)
            if not self.xml_path.is_file():
                raise FileNotFoundError(f"XML file not found: {self.xml_path}")
            with self.xml_path.open("rb") as f:
                self.tree = etree.parse(f, parser)
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
                row_cursor = 1
                for row_el in table_el.findall("ss:Row", namespaces=NS):
                    idx_attr_row = row_el.get("{%s}Index" % NS["ss"])
                    if idx_attr_row is not None:
                        target = int(idx_attr_row)
                        while row_cursor < target:
                            table.append([])
                            row_cursor += 1
                    row: List[Optional[str]] = []
                    col_cursor = 1
                    for cell_el in row_el.findall("ss:Cell", namespaces=NS):
                        idx_attr = cell_el.get("{%s}Index" % NS["ss"])
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
    time: np.ndarray
    epi: Dict[str, np.ndarray]
    endo: Dict[str, np.ndarray]
    myo: Dict[str, np.ndarray]
    delta: Dict[str, np.ndarray]


def extract_segments_from_sheet(df: pd.DataFrame, keep_map: Dict[str, str]) -> Dict[str, np.ndarray]:
    seg_series: Dict[str, np.ndarray] = {}
    if "ROWNUM" in df.columns:
        df_long = df[(df["ROWNUM"] >= 13) & (df["ROWNUM"] <= 18)].copy()
    else:
        df_long = df.iloc[12:18].copy()
    colA = df_long.get("A", pd.Series([], dtype=object)).astype(str).str.strip()
    for xml_label, canon in keep_map.items():
        matches = df_long[colA == xml_label]
        if matches.empty:
            matches = df_long[colA.str.replace(r"\s+$", "", regex=True) == xml_label]
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
    signals: List[np.ndarray] = []
    for d in (endo, epi):
        for arr in d.values():
            if arr is not None:
                signals.append(np.asarray(arr, dtype=float))
    if not signals:
        return []
    N = min(len(a) for a in signals)
    if N == 0:
        return []
    all_zero = np.ones(N, dtype=bool)
    for a in signals:
        all_zero &= np.isclose(a[:N], 0.0, atol=_ZERO_ATOL)
    idx = np.flatnonzero(all_zero)
    if idx.size < 2:
        return [(0, N - 1)]
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
        stack = np.vstack(seg_arrays)
        avg_curve = stack.mean(axis=0)
        gls = float(avg_curve.min())
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

    epi_sheet = ssml.get_sheet_table("Strain-Epi") or ssml.get_sheet_table(" Strain-Epi")
    endo_sheet = ssml.get_sheet_table("Strain-Endo") or ssml.get_sheet_table(" Strain-Endo")
    myo_sheet = ssml.get_sheet_table("Strain-Myo") or ssml.get_sheet_table(" Strain-Myo")
    if epi_sheet is None or endo_sheet is None or myo_sheet is None:
        logger.warning(f"Missing required sheets in {xml_path.name} (found: {ssml.list_sheets()})")
        return None

    epi_df = epi_sheet.to_dataframe()
    endo_df = endo_sheet.to_dataframe()
    myo_df = myo_sheet.to_dataframe()

    time_arr = extract_time_row(endo_df)

    epi = extract_segments_from_sheet(epi_df, keep_map)
    endo = extract_segments_from_sheet(endo_df, keep_map)
    myo = extract_segments_from_sheet(myo_df, keep_map)

    if not epi or not endo or not myo:
        logger.warning(f"No segment data in {xml_path.name}; skipping.")
        return None

    time_arr, (epi, endo, myo) = align_lengths(time_arr, epi, endo, myo)

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
def load_registry(registry_xlsx: Path) -> pd.DataFrame:
    df = pd.read_excel(registry_xlsx, engine="openpyxl")
    df.columns = [c.strip() for c in df.columns]
    required = ["PatientNum", "ID", "Study Date", "2-Chambers", "4-Chambers"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in registry: {missing}")

    df["study_datetime"] = df["Study Date"].apply(parse_datetime_generic)
    df["dicoms_2c"] = df["2-Chambers"].apply(_split_dicom_list)
    df["dicoms_4c"] = df["4-Chambers"].apply(_split_dicom_list)

    keep_cols = [
        "PatientNum",
        "ID",
        "study_datetime",
        "dicoms_2c",
        "dicoms_4c",
    ]
    return df[keep_cols].copy()


# --------------------------- Path helpers ---------------------------------
def find_patient_vvi_dir(vvi_dir: Path, patient_num: str) -> Optional[Path]:
    if not vvi_dir.is_dir():
        return None
    for child in sorted(vvi_dir.iterdir()):
        if not child.is_dir():
            continue
        if _match_patient_dir(child.name, patient_num):
            return child
    return None


def _find_folder_candidates(root: Path, dicom_basename: str) -> List[Path]:
    names = [dicom_basename, f"{dicom_basename}.dcm"]
    paths: List[Path] = []
    for name in names:
        p = root / name
        if p.is_dir():
            paths.append(p)
    if not paths:
        targets = {n.lower() for n in names}
        for p in root.rglob("*"):
            if p.is_dir() and p.name.lower() in targets:
                paths.append(p)
    return paths


def find_xml_for_dicom(vvi_dir: Path, patient_num: str, dicom_basename: str) -> Optional[Path]:
    patient_root = find_patient_vvi_dir(vvi_dir, patient_num)
    search_roots = [patient_root] if patient_root else [vvi_dir]

    for root in search_roots:
        for folder in _find_folder_candidates(root, dicom_basename):
            # Prefer segmentation exports that start with "(SEG"
            xmls = [p for p in folder.glob("*.xml") if p.name.startswith("(SEG")]
            if not xmls:
                xmls = list(folder.glob("*.xml"))
            if not xmls:
                continue
            xmls = sorted(xmls)
            return xmls[0]
    logger.warning(
        f"XML not found for DICOM '{dicom_basename}' (patient {patient_num}); searched under {search_roots[0]}"
    )
    return None


def find_patient_echo_dir(echo_root: Path, patient_num: str) -> Optional[Path]:
    if not echo_root.is_dir():
        return None
    for cand in _candidate_patient_keys(str(patient_num)):
        p = echo_root / cand
        if p.is_dir():
            return p
    # fallback: first dir starting with patient num
    for child in sorted(echo_root.iterdir()):
        if child.is_dir() and child.name.startswith(str(patient_num)):
            return child
    return None


def find_dicom_full_path(
    echo_root: Path, patient_num: str, study_dt: Optional[pd.Timestamp], dicom_basename: str
) -> Optional[Path]:
    patient_dir = find_patient_echo_dir(echo_root, patient_num)
    if patient_dir is None:
        return None

    date_candidates = _date_folder_candidates(study_dt)
    # Prefer date-matching folders if available
    for dc in date_candidates:
        for sub in patient_dir.iterdir():
            if not sub.is_dir():
                continue
            if sub.name.startswith(dc):
                target = _locate_dicom_in_dir(sub, dicom_basename)
                if target:
                    return target

    # Fallback: search entire patient folder
    return _locate_dicom_in_dir(patient_dir, dicom_basename, recursive=True)


def _locate_dicom_in_dir(root: Path, dicom_basename: str, recursive: bool = False) -> Optional[Path]:
    targets = {f"{dicom_basename}.dcm".lower(), dicom_basename.lower()}
    if recursive:
        for p in root.rglob("*"):
            if p.is_file() and p.name.lower() in targets:
                return p
    else:
        for child in root.iterdir():
            if child.is_file() and child.name.lower() in targets:
                return child
    return None


# --------------------------- Dataset builder ------------------------------
def build_dataset(
    vvi_dir: Path,
    registry_xlsx: Path,
    echo_root: Path,
    out_parquet: Path,
) -> pd.DataFrame:
    reg = load_registry(registry_xlsx)

    records: List[dict] = []

    for idx, row in reg.iterrows():
        patient_num = str(row["PatientNum"]).strip() if not pd.isna(row["PatientNum"]) else None
        patient_id = str(row["ID"]).strip() if not pd.isna(row["ID"]) else None
        study_dt = row.get("study_datetime")

        for view_col, view_tag in [("dicoms_2c", "2C"), ("dicoms_4c", "4C")]:
            dicom_list: List[str] = row[view_col] or []
            for dicom_base in dicom_list:
                xml_path = find_xml_for_dicom(vvi_dir, patient_num, dicom_base)
                if xml_path is None:
                    continue
                keep_map = get_keep_map(view_tag)
                parsed = parse_vvi_xml(xml_path, keep_map)
                if parsed is None:
                    continue

                cycles = split_cycles_by_all_zero(parsed.endo, parsed.epi)
                if not cycles:
                    total_len = len(next(iter(parsed.endo.values())))
                    cycles = [(0, total_len - 1)]

                for ci, (a, b) in enumerate(cycles):
                    t_slice = (
                        parsed.time[a : b + 1].tolist()
                        if parsed.time is not None and parsed.time.size
                        else list(range(b - a + 1))
                    )
                    n = len(t_slice)
                    gls_val = compute_gls_peak(parsed.myo, a, b, keep_map)
                    for seg in keep_map.values():
                        endo_arr = parsed.endo.get(seg)
                        epi_arr = parsed.epi.get(seg)
                        myo_arr = parsed.myo.get(seg)
                        delta_arr = parsed.delta.get(seg)
                        if endo_arr is None or epi_arr is None or delta_arr is None or myo_arr is None:
                            continue
                        rec = {
                            "patient_num": patient_num,
                            "patient_id": patient_id,
                            "study_datetime": study_dt,
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
                        try:
                            dicom_path = find_dicom_full_path(echo_root, patient_num, study_dt, dicom_base)
                            rec["dicom_full_path"] = str(dicom_path) if dicom_path else None
                        except Exception:
                            rec["dicom_full_path"] = None
                        records.append(rec)

    if not records:
        logger.error("No records parsed. Check paths and XML structure.")
        df_out = pd.DataFrame()
    else:
        df_out = pd.DataFrame.from_records(records)
        df_out.sort_values(
            by=["patient_num", "study_datetime", "view", "dicom_file", "cycle_index", "segment"],
            inplace=True,
        )

    out_parquet = Path(out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    if not df_out.empty:
        df_out.to_parquet(out_parquet, index=False)
        logger.info(
            f"Saved dataset: {out_parquet} with {len(df_out)} rows "
            f"({df_out['patient_num'].nunique()} patients, {df_out['dicom_file'].nunique()} dicoms)."
        )
    else:
        logger.warning("Skipped saving empty dataset.")

    return df_out


# --------------------------- CLI Interface --------------------------------
def main():
    import os

    user_home = os.path.expanduser("~")

    p = argparse.ArgumentParser(description="Parse Ichilov VVI XML strain analyses into a dataset")
    p.add_argument(
        "--vvi-dir",
        type=Path,
        required=False,
        default=Path(user_home) / "OneDrive - Technion" / "DS" / "Tags_Ichilov" / "VVI",
        help="Path to Tags_Ichilov VVI directory containing patient folders (XX_Name)",
    )
    p.add_argument(
        "--registry-xlsx",
        type=Path,
        required=False,
        default=Path(user_home) / "OneDrive - Technion" / "DS" / "Report_Ichilov_oron.xlsx",
        help="Registry Excel path with PatientNum/ID/Study Date/2-Chambers/4-Chambers columns",
    )
    p.add_argument(
        "--echo-root",
        type=Path,
        required=False,
        default=Path(user_home) / "OneDrive - Technion" / "DS" / "Ichilov",
        help="Root directory where DICOMs reside (patient_num/date/<dicom>.dcm)",
    )
    p.add_argument(
        "--out-parquet",
        type=Path,
        required=False,
        default=Path(user_home) / "OneDrive - Technion" / "DS" / "Tags_Ichilov" / "VVI" / "processed" / "strain_dataset_ichilov.parquet",
        help="Output Parquet path",
    )
    args = p.parse_args()

    _ = build_dataset(
        vvi_dir=args.vvi_dir,
        registry_xlsx=args.registry_xlsx,
        echo_root=args.echo_root,
        out_parquet=args.out_parquet,
    )


if __name__ == "__main__":
    main()
