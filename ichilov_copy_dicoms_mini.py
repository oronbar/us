r"""
Copy a minimized Ichilov dataset using GLS source DICOM paths from Excel.

By default, this script reads the A2C/A3C/A4C_GLS_SOURCE_DICOM columns from
Report_Ichilov_GLS_and_Strain_oron.xlsx and copies only those DICOMs from
D:\ds\ichilov into D:\ds\ichilov_mini, preserving folder structure when
possible.

Usage (PowerShell):
  .venv\Scripts\python ichilov_copy_dicoms_mini.py ^
    --input-xlsx "$env:USERPROFILE\OneDrive - Technion\DS\Report_Ichilov_GLS_and_Strain_oron.xlsx" ^
    --source-root "D:\ds\ichilov" ^
    --output-root "D:\ds\ichilov_mini"
"""
from __future__ import annotations

import argparse
import logging
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("ichilov_copy_dicoms_mini")

USER_HOME = Path.home()
fallback_drive = Path("F:\\")
if not (USER_HOME / "OneDrive - Technion").exists() and (fallback_drive / "OneDrive - Technion").exists():
    USER_HOME = fallback_drive

VIEW_COLUMNS = (
    "A2C_GLS_SOURCE_DICOM",
    "A3C_GLS_SOURCE_DICOM",
    "A4C_GLS_SOURCE_DICOM",
)


def _find_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
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
    return None


def _split_paths(value: object) -> List[str]:
    if isinstance(value, (list, tuple, set)):
        out: List[str] = []
        for v in value:
            out.extend(_split_paths(v))
        return out
    if pd.isna(value):
        return []
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return []
    parts = [p.strip().strip('"').strip("'") for p in re.split(r"[;|,\r\n]+", s)]
    return [p for p in parts if p]


def _resolve_candidate(path_str: str, source_root: Path, name_index: Optional[Dict[str, Path]]) -> Optional[Path]:
    p = Path(path_str)
    if p.is_file():
        return p
    if not p.is_absolute():
        candidate = source_root / p
        if candidate.is_file():
            return candidate
    if p.suffix == "":
        candidate = p.with_suffix(".dcm")
        if candidate.is_file():
            return candidate
        if not candidate.is_absolute():
            candidate = source_root / candidate
            if candidate.is_file():
                return candidate
    # Try matching a suffix of the original path under the source root.
    parts = p.parts
    for idx in range(len(parts)):
        candidate = source_root.joinpath(*parts[idx:])
        if candidate.is_file():
            return candidate
    if name_index is not None:
        return name_index.get(p.name.lower())
    return None


def _build_name_index(root: Path) -> Dict[str, Path]:
    logger.info("Building filename index under %s (this can take a while)...", root)
    index: Dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        key = path.name.lower()
        if key not in index:
            index[key] = path
    return index


def _relative_to_root(path: Path, source_root: Path) -> Path:
    try:
        return path.resolve().relative_to(source_root.resolve())
    except Exception:
        source_tail = source_root.resolve().name.lower()
        parts = list(path.parts)
        for idx, part in enumerate(parts):
            if part.lower() == source_tail and idx + 1 < len(parts):
                return Path(*parts[idx + 1 :])
        return Path(path.name)


def _collect_raw_paths(df: pd.DataFrame, cols: Iterable[str]) -> List[str]:
    raw: List[str] = []
    for col in cols:
        for value in df[col].tolist():
            raw.extend(_split_paths(value))
    # Preserve order while de-duplicating.
    return list(dict.fromkeys(raw))


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy A2C/A3C/A4C GLS source DICOMs into a mini dataset.")
    parser.add_argument(
        "--input-xlsx",
        type=Path,
        default=USER_HOME / "OneDrive - Technion" / "DS" / "Report_Ichilov_GLS_and_Strain_oron.xlsx",
        help="Excel file containing GLS source DICOM paths.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(r"D:\ds\ichilov"),
        help="Root folder of the full Ichilov dataset.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"D:\ds\ichilov_mini"),
        help="Output folder for the minimized dataset.",
    )
    parser.add_argument(
        "--sheet",
        type=str,
        default=None,
        help="Optional Excel sheet name or index.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite files that already exist.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be copied without copying.")
    parser.add_argument(
        "--resolve-by-name",
        action="store_true",
        help="If a path does not resolve, search the source root by filename (slow).",
    )
    parser.add_argument(
        "--missing-report",
        type=Path,
        default=None,
        help="Optional path to write unresolved entries.",
    )
    parser.add_argument(
        "--max-missing-log",
        type=int,
        default=25,
        help="Maximum missing paths to print in the log.",
    )
    args = parser.parse_args()

    if not args.input_xlsx.exists():
        logger.error("Input Excel not found: %s", args.input_xlsx)
        return
    if not args.source_root.exists():
        logger.error("Source root not found: %s", args.source_root)
        return

    sheet = args.sheet
    if sheet is not None and sheet.isdigit():
        sheet = int(sheet)

    read_kwargs = {"engine": "openpyxl"}
    if sheet is not None:
        read_kwargs["sheet_name"] = sheet
    df = pd.read_excel(args.input_xlsx, **read_kwargs)
    df.columns = [str(c).strip() for c in df.columns]

    cols: List[str] = []
    for cand in VIEW_COLUMNS:
        col = _find_column(df, [cand])
        if col:
            cols.append(col)

    if not cols:
        logger.error("No GLS source DICOM columns found. Expected: %s", ", ".join(VIEW_COLUMNS))
        return

    raw_paths = _collect_raw_paths(df, cols)
    if not raw_paths:
        logger.warning("No DICOM paths found in columns: %s", ", ".join(cols))
        return

    logger.info("Found %d unique entries across %s", len(raw_paths), ", ".join(cols))

    resolved: Dict[str, Path] = {}
    missing: List[str] = []

    for path_str in raw_paths:
        resolved_path = _resolve_candidate(path_str, args.source_root, None)
        if resolved_path:
            resolved[str(resolved_path)] = resolved_path
        else:
            missing.append(path_str)

    if missing and args.resolve_by_name:
        name_index = _build_name_index(args.source_root)
        remaining: List[str] = []
        for path_str in missing:
            resolved_path = _resolve_candidate(path_str, args.source_root, name_index)
            if resolved_path:
                resolved[str(resolved_path)] = resolved_path
            else:
                remaining.append(path_str)
        missing = remaining

    dicom_paths = list(resolved.values())
    if not dicom_paths:
        logger.error("No DICOMs could be resolved. Check the Excel columns and paths.")
        return

    if missing:
        logger.warning("Missing %d entries.", len(missing))
        for item in missing[: max(0, args.max_missing_log)]:
            logger.warning("Missing: %s", item)
        if len(missing) > args.max_missing_log:
            logger.warning("... %d more missing entries", len(missing) - args.max_missing_log)
        if args.missing_report:
            args.missing_report.parent.mkdir(parents=True, exist_ok=True)
            args.missing_report.write_text("\n".join(missing), encoding="utf-8")
            logger.info("Wrote missing report to %s", args.missing_report)

    copied = 0
    skipped = 0

    for src in dicom_paths:
        rel = _relative_to_root(src, args.source_root)
        dest = args.output_root / rel
        if dest.exists() and not args.overwrite:
            skipped += 1
            continue
        if not args.dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
        copied += 1

    logger.info(
        "Done. Resolved=%d Copied=%d SkippedExisting=%d Missing=%d DryRun=%s",
        len(dicom_paths),
        copied,
        skipped,
        len(missing),
        args.dry_run,
    )


if __name__ == "__main__":
    main()
