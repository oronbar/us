"""
Prepare a DICOM/view-level strain curve dataset for Ichilov pipeline5.

Example:
  python ichilov_prepare_strain_curve_dataset.py ^
    --strain-xlsx "C:\\path\\strain.xlsx" ^
    --ed-es-xlsx "C:\\path\\ed_es.xlsx" ^
    --echo-root "D:\\DS\\Ichilov" ^
    --output-parquet "C:\\path\\dataset.parquet"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ichilov_strain_curve_utils import (
    add_view_disagreement,
    attach_ed_es,
    clean_string,
    curve_peak_and_time,
    curve_quality_heuristics,
    dataframe_preview_for_csv,
    ed_es_report_to_rows,
    infer_report_columns,
    is_missing,
    make_sample_id,
    parse_global_curve,
    parse_views_list,
    read_dicom_basic_metadata,
    read_excel_first_sheet,
    resample_curve,
    strain_report_to_view_rows,
    summarize_samples,
    to_float,
    to_int,
    write_json,
)


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("ichilov_prepare_strain_curve_dataset")


def _load_column_map(text: str) -> Dict[str, Any]:
    return json.loads(text) if text else {}


def _stable_split_id(patient: object) -> Optional[int]:
    s = clean_string(patient)
    if not s:
        return None
    digest = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100


def _resolve_cropped(dicom_path: object, echo_root: Path, cropped_root: Optional[Path]) -> Optional[str]:
    if cropped_root is None or is_missing(dicom_path):
        return None
    try:
        p = Path(str(dicom_path))
        if not p.is_file():
            return None
        rel = p.resolve().relative_to(echo_root.resolve())
        direct = cropped_root / rel
        if direct.is_file():
            return str(direct)
        # Avoid a broad basename search here: a base cropped root can contain
        # multiple crop runs with identical patient/visit/file suffixes.
        return None
    except Exception:
        return None


def _summary_md(summary: Dict[str, Any], skipped: Dict[str, int]) -> str:
    lines = [
        "# Pipeline5 Prepared Strain Curve Dataset",
        "",
        "## Summary",
        f"- Samples: {summary.get('n_samples', 0)}",
        f"- Patients: {summary.get('n_patients', 0)}",
        f"- Views: {summary.get('views', {})}",
        "",
        "## Skipped Rows",
    ]
    for key, value in skipped.items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Missing Values",
            f"- Missing ED: {summary.get('missing_ed_index', 0)}",
            f"- Missing ES: {summary.get('missing_es_index', 0)}",
            f"- Missing report GLS: {summary.get('missing_peak_gls_from_report', 0)}",
            f"- Missing curve peak: {summary.get('missing_peak_gls_from_curve', 0)}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare DICOM/view-level strain curve dataset.")
    parser.add_argument("--strain-xlsx", type=Path, required=True)
    parser.add_argument("--ed-es-xlsx", type=Path, required=True)
    parser.add_argument("--echo-root", type=Path, required=True)
    parser.add_argument("--cropped-root", type=Path, default=None)
    parser.add_argument("--output-parquet", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--column-map-json", type=str, default="")
    parser.add_argument("--curve-length", type=int, default=64)
    parser.add_argument("--views", type=str, default="A2C,A3C,A4C")
    args = parser.parse_args()

    output_dir = args.output_dir or args.output_parquet.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_csv or output_dir / "prepared_curve_dataset_preview.csv"

    logger.info("Loading strain report: %s", args.strain_xlsx)
    logger.info("Loading ED/ES report: %s", args.ed_es_xlsx)
    strain_df = read_excel_first_sheet(args.strain_xlsx)
    ed_df = read_excel_first_sheet(args.ed_es_xlsx)
    views = parse_views_list(args.views)
    column_map = _load_column_map(args.column_map_json)
    resolved = infer_report_columns(ed_df, strain_df, column_map=column_map, views=views)
    write_json(output_dir / "resolved_column_map.json", resolved)

    strain_rows = strain_report_to_view_rows(strain_df, resolved, echo_root=args.echo_root, views=views)
    ed_rows = ed_es_report_to_rows(ed_df, resolved, echo_root=args.echo_root, views=views)
    merged = attach_ed_es(strain_rows, ed_rows)
    logger.info("Normalized strain samples: %d", len(strain_rows))
    logger.info("Normalized ED/ES rows: %d", len(ed_rows))

    skipped = {
        "no_dicom_path": 0,
        "unparseable_curve": 0,
        "wrong_view": 0,
    }
    rows: List[Dict[str, Any]] = []
    for idx, row in merged.iterrows():
        view = clean_string(row.get("view"))
        if view not in views:
            skipped["wrong_view"] += 1
            continue
        dicom_path = clean_string(row.get("dicom_path") or row.get("dicom_path_ed"))
        if not dicom_path:
            skipped["no_dicom_path"] += 1
            continue

        global_curve, raw_curves = parse_global_curve(row.get("strain_curve_raw"))
        if global_curve is None or len(global_curve) < 2:
            skipped["unparseable_curve"] += 1
            continue

        resampled = resample_curve(global_curve, args.curve_length)
        peak_curve, ttp_fraction, ttp_index = curve_peak_and_time(resampled)
        quality = curve_quality_heuristics(resampled)

        metadata = read_dicom_basic_metadata(dicom_path)
        n_frames = to_int(row.get("n_frames_report")) or metadata.get("n_frames")
        frame_time_ms = to_float(row.get("frame_time_ms_report")) or metadata.get("frame_time_ms")
        heart_rate = metadata.get("heart_rate")
        ed_index = to_int(row.get("ed_index"))
        es_index = to_int(row.get("es_index"))
        if ed_index is None:
            ed_index = to_int(row.get("strain_ed_index"))
        if es_index is None:
            es_index = to_int(row.get("strain_es_index"))

        out = {
            "sample_id": make_sample_id(row, int(idx)),
            "source_row_index": to_int(row.get("source_row_index")),
            "ed_es_row_index": to_int(row.get("ed_es_row_index")),
            "ed_es_match_method": clean_string(row.get("ed_es_match_method")),
            "patient_id": clean_string(row.get("patient_id")),
            "patient_num": clean_string(row.get("patient_num")),
            "patient_key": clean_string(row.get("patient_key")),
            "split_id": _stable_split_id(row.get("patient_key")),
            "visit_id": clean_string(row.get("visit_date")),
            "visit_date": clean_string(row.get("visit_date")),
            "dicom_path": dicom_path,
            "cropped_path": _resolve_cropped(dicom_path, args.echo_root, args.cropped_root),
            "view": view,
            "ed_index": ed_index,
            "es_index": es_index,
            "n_frames": n_frames,
            "frame_time_ms": frame_time_ms,
            "heart_rate": heart_rate,
            "raw_strain_curves": raw_curves,
            "strain_curve": [float(x) if np.isfinite(x) else None for x in global_curve],
            "strain_curve_length": int(len(global_curve)),
            "resampled_strain_curve": [float(x) if np.isfinite(x) else None for x in resampled],
            "curve_length": int(args.curve_length),
            "peak_gls_from_report": to_float(row.get("peak_gls_from_report")),
            "peak_gls_from_curve": peak_curve,
            "time_to_peak_from_curve": ttp_fraction,
            "time_to_peak_index_from_curve": ttp_index,
        }
        out.update(quality)
        rows.append(out)

    dataset = pd.DataFrame(rows)
    dataset = add_view_disagreement(dataset)
    if "patient_key" in dataset:
        dataset = dataset.sort_values(["patient_key", "visit_date", "view", "sample_id"], na_position="last")

    dataset.to_parquet(args.output_parquet, index=False)
    dataframe_preview_for_csv(dataset).to_csv(output_csv, index=False)

    summary = summarize_samples(dataset)
    summary["skipped"] = skipped
    summary["output_parquet"] = str(args.output_parquet)
    summary["output_csv"] = str(output_csv)
    write_json(output_dir / "dataset_summary.json", summary)
    (output_dir / "dataset_summary.md").write_text(_summary_md(summary, skipped), encoding="utf-8")

    logger.info("Saved prepared dataset: %s (%d samples)", args.output_parquet, len(dataset))
    logger.info("Saved CSV preview: %s", output_csv)
    logger.info("Skipped rows: %s", skipped)


if __name__ == "__main__":
    main()
