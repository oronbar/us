"""
Validate ED/ES and strain curve report schemas for Ichilov pipeline5.

Example:
  python ichilov_validate_strain_reports.py ^
    --ed-es-xlsx "C:\\path\\ed_es.xlsx" ^
    --strain-xlsx "C:\\path\\strain.xlsx" ^
    --output-dir "C:\\path\\validation"
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from ichilov_strain_curve_utils import (
    attach_ed_es,
    duplicate_key_summary,
    ed_es_report_to_rows,
    infer_report_columns,
    is_missing,
    parse_global_curve,
    parse_views_list,
    read_excel_first_sheet,
    strain_report_to_view_rows,
    summarize_samples,
    write_json,
)


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("ichilov_validate_strain_reports")


def _load_column_map(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    return json.loads(text)


def _columns_df(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "column": [str(c) for c in df.columns],
            "dtype": [str(df[c].dtype) for c in df.columns],
            "non_null": [int(df[c].notna().sum()) for c in df.columns],
        }
    )


def _parsed_curve_ok(value: object) -> bool:
    curve, _ = parse_global_curve(value)
    return curve is not None and len(curve) >= 2


def _write_md(path: Path, report: Dict[str, Any]) -> None:
    lines = [
        "# Pipeline5 Report Validation",
        "",
        "## Inputs",
        f"- ED/ES rows: {report['ed_es']['rows']}",
        f"- Strain rows: {report['strain']['rows']}",
        "",
        "## Detected Columns",
        "```json",
        json.dumps(report["resolved_column_map"], indent=2),
        "```",
        "",
        "## Row Summaries",
        f"- ED/ES normalized samples: {report['ed_es_normalized'].get('n_samples', 0)}",
        f"- ED/ES patients: {report['ed_es_normalized'].get('n_patients', 0)}",
        f"- Strain normalized samples: {report['strain_normalized'].get('n_samples', 0)}",
        f"- Strain patients: {report['strain_normalized'].get('n_patients', 0)}",
        "",
        "## Matching",
        f"- Matched strain rows: {report['matching']['matched_strain_rows']}",
        f"- Unmatched strain rows: {report['matching']['unmatched_strain_rows']}",
        f"- Unmatched ED/ES rows: {report['matching']['unmatched_ed_es_rows']}",
        "",
        "## Missing Labels",
        f"- Missing ED: {report['missing']['ed_index']}",
        f"- Missing ES: {report['missing']['es_index']}",
        f"- Missing strain curve: {report['missing']['strain_curve']}",
        f"- Unparseable strain curve: {report['missing']['unparseable_strain_curve']}",
        f"- Missing GLS: {report['missing']['gls_peak']}",
        "",
        "## Duplicates",
        "```json",
        json.dumps(report["duplicates"], indent=2),
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Ichilov strain report inputs.")
    parser.add_argument("--ed-es-xlsx", type=Path, required=True)
    parser.add_argument("--strain-xlsx", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--column-map-json", type=str, default="")
    parser.add_argument("--views", type=str, default="A2C,A3C,A4C")
    args = parser.parse_args()

    output_dir = args.output_dir or Path("pipeline5_validate_reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading ED/ES report: %s", args.ed_es_xlsx)
    logger.info("Loading strain report: %s", args.strain_xlsx)
    ed_df = read_excel_first_sheet(args.ed_es_xlsx)
    strain_df = read_excel_first_sheet(args.strain_xlsx)
    views = parse_views_list(args.views)
    column_map = _load_column_map(args.column_map_json)
    resolved = infer_report_columns(ed_df, strain_df, column_map=column_map, views=views)

    _columns_df(ed_df).to_csv(output_dir / "ed_es_columns.csv", index=False)
    _columns_df(strain_df).to_csv(output_dir / "strain_columns.csv", index=False)
    write_json(output_dir / "resolved_column_map.json", resolved)

    ed_rows = ed_es_report_to_rows(ed_df, resolved, echo_root=None, views=views)
    strain_rows = strain_report_to_view_rows(strain_df, resolved, echo_root=None, views=views)
    merged = attach_ed_es(strain_rows, ed_rows) if not strain_rows.empty and not ed_rows.empty else strain_rows

    matched_ed = set(
        int(v)
        for v in merged.get("ed_es_row_index", pd.Series(dtype=float)).dropna().tolist()
    )
    unmatched_strain = merged[merged.get("ed_es_match_method", "none").eq("none")] if not merged.empty else merged
    unmatched_ed = ed_rows[~ed_rows["ed_es_row_index"].isin(matched_ed)] if not ed_rows.empty else ed_rows

    unmatched_strain.to_csv(output_dir / "unmatched_strain_rows.csv", index=False)
    unmatched_ed.to_csv(output_dir / "unmatched_ed_es_rows.csv", index=False)

    duplicate_report = {
        "ed_es": duplicate_key_summary(ed_rows.rename(columns={"dicom_path_ed": "dicom_path", "view_ed": "view", "patient_key_ed": "patient_key", "visit_date_ed": "visit_date"})),
        "strain": duplicate_key_summary(strain_rows),
    }
    write_json(output_dir / "duplicate_keys.json", duplicate_report)

    missing_ed = int(merged["ed_index"].map(is_missing).sum()) if "ed_index" in merged else len(merged)
    missing_es = int(merged["es_index"].map(is_missing).sum()) if "es_index" in merged else len(merged)
    missing_curve = int(strain_rows["strain_curve_raw"].map(is_missing).sum()) if "strain_curve_raw" in strain_rows else len(strain_rows)
    parsed_ok = (
        strain_rows["strain_curve_raw"].map(_parsed_curve_ok)
        if "strain_curve_raw" in strain_rows
        else pd.Series([], dtype=bool)
    )
    missing_gls = int(strain_rows["peak_gls_from_report"].map(is_missing).sum()) if "peak_gls_from_report" in strain_rows else len(strain_rows)

    report: Dict[str, Any] = {
        "ed_es": {
            "path": str(args.ed_es_xlsx),
            "rows": int(len(ed_df)),
            "columns": [str(c) for c in ed_df.columns],
        },
        "strain": {
            "path": str(args.strain_xlsx),
            "rows": int(len(strain_df)),
            "columns": [str(c) for c in strain_df.columns],
        },
        "resolved_column_map": resolved,
        "ed_es_normalized": summarize_samples(
            ed_rows.rename(columns={"patient_key_ed": "patient_key", "view_ed": "view"})
        ),
        "strain_normalized": summarize_samples(strain_rows),
        "available_views": {
            "ed_es": ed_rows.get("view_ed", pd.Series(dtype=object)).dropna().value_counts().to_dict(),
            "strain": strain_rows.get("view", pd.Series(dtype=object)).dropna().value_counts().to_dict(),
        },
        "matching": {
            "matched_strain_rows": int((merged.get("ed_es_match_method", pd.Series(dtype=object)) != "none").sum()) if not merged.empty else 0,
            "unmatched_strain_rows": int(len(unmatched_strain)),
            "unmatched_ed_es_rows": int(len(unmatched_ed)),
        },
        "missing": {
            "ed_index": missing_ed,
            "es_index": missing_es,
            "strain_curve": missing_curve,
            "unparseable_strain_curve": int((~parsed_ok).sum()) if len(parsed_ok) else int(len(strain_rows)),
            "gls_peak": missing_gls,
        },
        "duplicates": duplicate_report,
    }

    write_json(output_dir / "validation_report.json", report)
    pd.DataFrame([report["missing"]]).to_csv(output_dir / "missing_summary.csv", index=False)
    _write_md(output_dir / "validation_report.md", report)

    logger.info("ED/ES rows: %d", len(ed_df))
    logger.info("Strain rows: %d", len(strain_df))
    logger.info("Normalized strain samples: %d", len(strain_rows))
    logger.info("Matched strain samples: %d", report["matching"]["matched_strain_rows"])
    logger.info("Saved validation report: %s", output_dir / "validation_report.md")


if __name__ == "__main__":
    main()
