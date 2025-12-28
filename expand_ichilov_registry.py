"""
Expand Ichilov registry by scanning patient visit folders and adding missing visits.

Usage (PowerShell / cmd):
  python expand_ichilov_registry.py ^
    --registry-xlsx "C:\\Users\\oronbarazani\\OneDrive - Technion\\DS\\Report_Ichilov_oron.xlsx" ^
    --echo-root "C:\\Users\\oronbarazani\\OneDrive - Technion\\DS\\Ichilov" ^
    --out-xlsx "C:\\Users\\oronbarazani\\OneDrive - Technion\\DS\\Report_Ichilov_expanded_oron.xlsx"
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Set

import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("expand_ichilov_registry")

DATE_RE = re.compile(r"(\d{4})[-_](\d{2})[-_](\d{2})")


def parse_datetime_generic(val: object) -> Optional[pd.Timestamp]:
    if pd.isna(val):
        return None
    if isinstance(val, pd.Timestamp):
        return val
    if isinstance(val, (int, float)):
        try:
            dt = pd.to_datetime(val, errors="coerce")
            return None if pd.isna(dt) else dt
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


def _normalize_patient_num(val: object) -> Optional[str]:
    if pd.isna(val):
        return None
    if isinstance(val, (int,)):
        return str(val)
    if isinstance(val, float) and val.is_integer():
        return str(int(val))
    s = str(val).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s or None


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


def find_patient_echo_dir(echo_root: Path, patient_num: str) -> Optional[Path]:
    if not echo_root.is_dir():
        return None
    for cand in _candidate_patient_keys(str(patient_num)):
        p = echo_root / cand
        if p.is_dir():
            return p
    for child in sorted(echo_root.iterdir()):
        if child.is_dir() and _match_patient_dir(child.name, str(patient_num)):
            return child
    return None


def _extract_visit_date_from_name(name: str) -> Optional[pd.Timestamp]:
    match = DATE_RE.search(name)
    if not match:
        return None
    date_str = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    dt = pd.to_datetime(date_str, errors="coerce")
    if pd.isna(dt):
        return None
    return dt.normalize()


def collect_visit_dates(patient_dir: Path) -> List[pd.Timestamp]:
    dates: Set[pd.Timestamp] = set()
    for child in patient_dir.iterdir():
        if not child.is_dir():
            continue
        dt = _extract_visit_date_from_name(child.name)
        if dt is not None:
            dates.add(dt)
    return sorted(dates)


def expand_registry(registry_xlsx: Path, echo_root: Path) -> pd.DataFrame:
    df = pd.read_excel(registry_xlsx, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    required = ["PatientNum", "ID", "Study Date", "2-Chambers", "4-Chambers"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in registry: {missing}")

    df["_study_dt"] = df["Study Date"].apply(parse_datetime_generic)
    new_rows: List[Dict[str, object]] = []

    for patient_val, grp in df.groupby("PatientNum", dropna=True):
        patient_key = _normalize_patient_num(patient_val)
        if not patient_key:
            continue
        patient_dir = find_patient_echo_dir(echo_root, patient_key)
        if patient_dir is None:
            logger.warning(f"Patient folder not found for PatientNum={patient_val}")
            continue

        existing_dates: Set[pd.Timestamp] = set()
        for dt in grp["_study_dt"].dropna():
            ts = pd.Timestamp(dt)
            existing_dates.add(ts.normalize())
        visit_dates = collect_visit_dates(patient_dir)
        missing_dates = [dt for dt in visit_dates if dt not in existing_dates]
        if not missing_dates:
            continue

        first_id = grp["ID"].dropna().iloc[0] if not grp["ID"].dropna().empty else pd.NA
        for dt in missing_dates:
            row = {col: pd.NA for col in df.columns if col != "_study_dt"}
            row["PatientNum"] = patient_val
            row["ID"] = first_id
            row["Study Date"] = dt
            row["2-Chambers"] = pd.NA
            row["4-Chambers"] = pd.NA
            new_rows.append(row)

    if new_rows:
        df = pd.concat([df.drop(columns=["_study_dt"]), pd.DataFrame(new_rows)], ignore_index=True)
    else:
        df = df.drop(columns=["_study_dt"])

    df["_sort_dt"] = df["Study Date"].apply(parse_datetime_generic)
    df["_patient_sort"] = df["PatientNum"].apply(_normalize_patient_num)
    df.sort_values(by=["_patient_sort", "_sort_dt"], inplace=True, kind="stable")
    df.drop(columns=["_sort_dt", "_patient_sort"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def main() -> None:
    import os

    user_home = os.path.expanduser("~")

    p = argparse.ArgumentParser(description="Expand Ichilov registry with all visit folders")
    p.add_argument(
        "--registry-xlsx",
        type=Path,
        required=False,
        default=Path(user_home) / "OneDrive - Technion" / "DS" / "Report_Ichilov_oron.xlsx",
        help="Original registry Excel path",
    )
    p.add_argument(
        "--echo-root",
        type=Path,
        required=False,
        default=Path(user_home) / "OneDrive - Technion" / "DS" / "Ichilov",
        help="Root directory where DICOMs reside (patient_num/date/)",
    )
    p.add_argument(
        "--out-xlsx",
        type=Path,
        required=False,
        default=Path(user_home) / "OneDrive - Technion" / "DS" / "Report_Ichilov_expanded_oron.xlsx",
        help="Output Excel path",
    )
    args = p.parse_args()

    expanded = expand_registry(args.registry_xlsx, args.echo_root)
    args.out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    expanded.to_excel(args.out_xlsx, index=False, engine="openpyxl")
    logger.info(f"Saved expanded registry: {args.out_xlsx} ({len(expanded)} rows)")


if __name__ == "__main__":
    main()
