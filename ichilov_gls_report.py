"""
Build a GLS report from TomTec analysis DICOMs and map each GLS to its source cine.

Input:
  - Excel registry with PatientNum, ID, Study Date and optional 2/4-chamber DICOM names

Output:
  - Excel file with added GLS values and corresponding analysis/source DICOM paths

Usage (PowerShell):
  .venv\\Scripts\\python ichilov_gls_report.py ^
    --input-xlsx "C:\\Users\\oronbarazani\\OneDrive - Technion\\DS\\Report_Ichilov_expanded_oron.xlsx" ^
    --echo-root "C:\\Users\\oronbarazani\\OneDrive - Technion\\DS\\Ichilov" ^
    --output-xlsx "C:\\Users\\oronbarazani\\OneDrive - Technion\\DS\\Report_Ichilov_GLS_oron.xlsx"
"""
from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import xml.etree.ElementTree as ET

import pandas as pd
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("ichilov_gls_report")


VIEW_KEYS = ("A2C", "A3C", "A4C")


@dataclass
class AnalysisResult:
    analysis_path: Path
    view: Optional[str]
    gls: Optional[float]
    source_sop: Optional[str]
    label: Optional[str]


def _clean_xml_text(raw: bytes) -> str:
    text = raw.decode("utf-8", errors="ignore")
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", text)
    return text


def _extract_xtt_xml(data: bytes) -> Optional[str]:
    start = data.find(b"<?xml")
    if start < 0:
        return None
    end = data.find(b"</xtt>", start)
    if end < 0:
        return None
    chunk = data[start : end + len(b"</xtt>")]
    return _clean_xml_text(chunk)


def _extract_tt_xml(data: bytes) -> Optional[str]:
    start_tag = b"<TT_XML_DATACONTAINER_DOC_V2>"
    end_tag = b"</TT_XML_DATACONTAINER_DOC_V2>"
    start = data.find(start_tag)
    if start < 0:
        return None
    end = data.find(end_tag, start)
    if end < 0:
        return None
    chunk = data[start : end + len(end_tag)]
    return _clean_xml_text(chunk)


def _parse_tt_label(tt_xml: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not tt_xml:
        return None, None
    label = None
    file_name = None
    try:
        root = ET.fromstring(tt_xml)
        for di in root.iter("DI"):
            key = (di.attrib.get("K") or "").strip()
            val = (di.attrib.get("V") or "").strip()
            if key == "FileLabel" and val:
                label = val
            elif key == "FileName" and val:
                file_name = val
    except ET.ParseError:
        pass

    if label is None:
        m = re.search(r'K="FileLabel"[^>]*V="([^"]+)"', tt_xml)
        if m:
            label = m.group(1)
    if file_name is None:
        m = re.search(r'K="FileName"[^>]*V="([^"]+)"', tt_xml)
        if m:
            file_name = m.group(1)
    return label, file_name


def _parse_xtt_gls_and_uid(xtt_xml: Optional[str]) -> Tuple[Optional[float], Optional[str]]:
    if not xtt_xml:
        return None, None
    gls_val = None
    sop_uid = None
    try:
        root = ET.fromstring(xtt_xml)
        gls_node = root.find(".//Gls/Value")
        if gls_node is not None and gls_node.text:
            gls_val = _safe_float(gls_node.text)
        uid_node = root.find(".//SopInstanceUid")
        if uid_node is not None and uid_node.text:
            sop_uid = uid_node.text.strip()
    except ET.ParseError:
        pass

    if gls_val is None:
        m = re.search(r"<Gls[^>]*>.*?<Value[^>]*>([^<]+)</Value>", xtt_xml, re.DOTALL | re.IGNORECASE)
        if m:
            gls_val = _safe_float(m.group(1))
    if sop_uid is None:
        m = re.search(r"<SopInstanceUid[^>]*>([^<]+)</SopInstanceUid>", xtt_xml)
        if m:
            sop_uid = m.group(1).strip()
    return gls_val, sop_uid


def _safe_float(val: object) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if not s:
        return None
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _parse_view(label: Optional[str], xtt_xml: Optional[str]) -> Optional[str]:
    for source in (label, xtt_xml):
        if not source:
            continue
        m = re.search(r"A[234]C", source.upper())
        if m:
            return m.group(0)
    return None


def _get_tag_value(data: bytes, group: int, element: int) -> Optional[str]:
    tag = group.to_bytes(2, "little") + element.to_bytes(2, "little")
    long_vr = {b"OB", b"OW", b"OF", b"SQ", b"UT", b"UN"}
    i = 0
    n = len(data)
    while i <= n - 8:
        if data[i : i + 4] != tag:
            i += 1
            continue
        vr = data[i + 4 : i + 6]
        is_letter = all(65 <= b <= 90 for b in vr)
        if is_letter:
            if vr in long_vr:
                if i + 12 > n:
                    i += 1
                    continue
                length = int.from_bytes(data[i + 8 : i + 12], "little")
                val_start = i + 12
            else:
                if i + 8 > n:
                    i += 1
                    continue
                length = int.from_bytes(data[i + 6 : i + 8], "little")
                val_start = i + 8
        else:
            if i + 8 > n:
                i += 1
                continue
            length = int.from_bytes(data[i + 4 : i + 8], "little")
            val_start = i + 8
        if length <= 0 or val_start + length > n:
            i += 1
            continue
        raw_val = data[val_start : val_start + length]
        val = raw_val.decode("ascii", errors="ignore").strip("\x00").strip()
        if val:
            return val
        i += 1
    return None


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
    if dirs:
        return dirs
    return [patient_dir]


def _list_dicom_files(dirs: Iterable[Path]) -> List[Path]:
    files: List[Path] = []
    for d in dirs:
        files.extend(d.glob("*.dcm"))
        files.extend(d.glob("*.DCM"))
    if files:
        return sorted(set(files))
    for d in dirs:
        files.extend(d.rglob("*.dcm"))
        files.extend(d.rglob("*.DCM"))
    return sorted(set(files))


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


def _resolve_dicom_hint(hints: List[str], patient_dir: Path, study_dirs: List[Path]) -> Optional[Path]:
    for hint in hints:
        p = Path(hint)
        if p.is_file():
            return p
        base = p.stem if p.suffix else p.name
        for d in study_dirs:
            for cand in (d / f"{base}.dcm", d / base):
                if cand.is_file():
                    return cand
        for cand in patient_dir.rglob(f"{base}.dcm"):
            if cand.is_file():
                return cand
    return None


def _scan_study(dicom_paths: List[Path]) -> Tuple[Dict[str, Path], List[AnalysisResult]]:
    sop_map: Dict[str, Path] = {}
    analyses: List[AnalysisResult] = []
    for path in dicom_paths:
        try:
            data = path.read_bytes()
        except Exception as e:
            logger.warning(f"Failed reading DICOM: {path} ({e})")
            continue

        sop_uid = _get_tag_value(data, 0x0008, 0x0018)
        if sop_uid:
            sop_map[sop_uid] = path

        if b"<xtt" not in data:
            continue
        xtt_xml = _extract_xtt_xml(data)
        if not xtt_xml:
            continue
        tt_xml = _extract_tt_xml(data)
        label, _ = _parse_tt_label(tt_xml)
        gls, source_sop = _parse_xtt_gls_and_uid(xtt_xml)
        view = _parse_view(label, xtt_xml)
        analyses.append(
            AnalysisResult(
                analysis_path=path,
                view=view,
                gls=gls,
                source_sop=source_sop,
                label=label,
            )
        )
    return sop_map, analyses


def _pick_best(items: List[AnalysisResult]) -> Optional[AnalysisResult]:
    if not items:
        return None
    items_sorted = sorted(
        items,
        key=lambda x: (
            x.gls is None,
            x.source_sop is None,
            str(x.analysis_path),
        ),
    )
    return items_sorted[0]


def _get_sop_from_path(path: Path) -> Optional[str]:
    try:
        data = path.read_bytes()
    except Exception:
        return None
    return _get_tag_value(data, 0x0008, 0x0018)


def _parse_datetime(val: object) -> Optional[pd.Timestamp]:
    if pd.isna(val):
        return None
    if isinstance(val, pd.Timestamp):
        return val
    s = str(val).strip()
    if not s:
        return None
    s = s.replace("_", "-").replace("\\", "-").replace("/", "-")
    try:
        dt = pd.to_datetime(s, errors="coerce")
        return None if pd.isna(dt) else dt
    except Exception:
        return None


def _find_column(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
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


def build_gls_report(
    input_xlsx: Path,
    echo_root: Path,
    output_xlsx: Path,
) -> pd.DataFrame:
    df = pd.read_excel(input_xlsx, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]

    col_patient = _find_column(df, ["PatientNum", "Patient Num", "Patient Number"])
    col_id = _find_column(df, ["ID", "Patient ID", "Id"], required=False)
    col_date = _find_column(df, ["Study Date", "StudyDate", "Date"])
    col_2c = _find_column(df, ["2-Chambers", "2C", "2 Chambers"], required=False)
    col_4c = _find_column(df, ["4-Chambers", "4C", "4 Chambers"], required=False)
    col_3c = _find_column(df, ["3-Chambers", "3C", "3 Chambers"], required=False)

    out_cols = {}
    for view in VIEW_KEYS:
        out_cols[f"{view}_GLS"] = None
        out_cols[f"{view}_GLS_ANALYSIS_DICOM"] = None
        out_cols[f"{view}_GLS_SOURCE_DICOM"] = None

    for k, v in out_cols.items():
        if k not in df.columns:
            df[k] = v

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        if pd.isna(row[col_patient]):
            continue
        patient_num = str(row[col_patient]).strip()
        if not patient_num or patient_num.lower() == "nan":
            continue
        study_dt = _parse_datetime(row[col_date])

        if not patient_num:
            continue

        patient_dir = _find_patient_dir(echo_root, patient_num)
        if patient_dir is None:
            logger.warning(f"Patient dir not found for {patient_num}")
            continue

        study_dirs = _study_dirs(patient_dir, study_dt)
        if study_dirs == [patient_dir] and study_dt is not None:
            logger.info(f"No date folder match for {patient_num} {study_dt.date()}, scanning patient root")

        dicom_paths = _list_dicom_files(study_dirs)
        if not dicom_paths:
            logger.warning(f"No DICOMs found for {patient_num} ({study_dt})")
            continue

        sop_map, analyses = _scan_study(dicom_paths)
        by_view: Dict[str, List[AnalysisResult]] = {k: [] for k in VIEW_KEYS}
        for res in analyses:
            if res.view in by_view:
                by_view[res.view].append(res)

        # Manual hints (if provided)
        hint_2c = _split_dicom_list(row[col_2c]) if col_2c else []
        hint_4c = _split_dicom_list(row[col_4c]) if col_4c else []
        hint_3c = _split_dicom_list(row[col_3c]) if col_3c else []

        hint_map = {"A2C": hint_2c, "A3C": hint_3c, "A4C": hint_4c}
        hint_path_map: Dict[str, Optional[Path]] = {}
        hint_sop_map: Dict[str, Optional[str]] = {}
        for view, hints in hint_map.items():
            hint_path = _resolve_dicom_hint(hints, patient_dir, study_dirs) if hints else None
            hint_path_map[view] = hint_path
            hint_sop_map[view] = _get_sop_from_path(hint_path) if hint_path else None

        for view in VIEW_KEYS:
            candidates = by_view.get(view, [])
            hint_sop = hint_sop_map.get(view)
            if hint_sop:
                candidates = [c for c in candidates if c.source_sop == hint_sop] or candidates
            chosen = _pick_best(candidates)
            if chosen is None:
                continue
            source_path = sop_map.get(chosen.source_sop) if chosen.source_sop else None
            if source_path is None and hint_path_map.get(view):
                source_path = hint_path_map[view]

            df.at[idx, f"{view}_GLS"] = chosen.gls
            df.at[idx, f"{view}_GLS_ANALYSIS_DICOM"] = str(chosen.analysis_path)
            df.at[idx, f"{view}_GLS_SOURCE_DICOM"] = str(source_path) if source_path else None

    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_xlsx, index=False)
    logger.info(f"Saved: {output_xlsx}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract GLS per view and map to source DICOMs.")
    parser.add_argument(
        "--input-xlsx",
        type=Path,
        default=Path(r"C:\Users\oronbarazani\OneDrive - Technion\DS\Report_Ichilov_expanded_oron.xlsx"),
        help="Input registry Excel",
    )
    parser.add_argument(
        "--echo-root",
        type=Path,
        default=Path(r"D:\DS\Ichilov"),
        help="Root folder containing patient DICOMs",
    )
    parser.add_argument(
        "--output-xlsx",
        type=Path,
        default=Path(r"C:\Users\oronbarazani\OneDrive - Technion\DS\Report_Ichilov_GLS_oron.xlsx"),
        help="Output Excel path",
    )
    args = parser.parse_args()
    build_gls_report(
        input_xlsx=args.input_xlsx,
        echo_root=args.echo_root,
        output_xlsx=args.output_xlsx,
    )


if __name__ == "__main__":
    main()
