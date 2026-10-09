"""Deduplicate read-only GE cine inventory and attach existing view labels.

The resulting manifest is private data under D:\DS. Unknown views are left for
the released EchoPrime view classifier; this script never infers them from a
filename or from the downstream label.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(r"D:\DS\ichilov_ge_inventory_20260930")
E2_CACHE = Path(r"D:\us\output\dicom_prediction")
E3_SELECTION = Path(r"D:\DS\ichilov3_stage2_padded_20260926\full_manifest.json")
VIEWS = {"A2C", "A3C", "A4C"}
PREFERENCE = {"Ichilov3": 0, "Ichilov2": 1, "Ichilov1": 2}


def normal(path: str) -> str:
    return os.path.normcase(os.path.normpath(path))


def existing_labels() -> dict[str, dict]:
    spatial = pd.read_parquet(E2_CACHE / "spatial_manifest.parquet")
    views = pd.read_parquet(E2_CACHE / "view_manifest.parquet")
    table = spatial.merge(views, on=["file_id", "study_uid"], validate="one_to_one")
    result = {}
    for row in table.itertuples():
        if row.view in VIEWS and bool(row.is_bmode_candidate) and bool(row.eligible_motion):
            old_path = Path(str(row.path))
            current_path = old_path
            # The earlier EchoPrime inventory predates moving these exports
            # under E:\Ichilov2. Keep the model label only after a one-to-one
            # relative-path remap to an existing DICOM file.
            if not current_path.exists() and old_path.drive.upper() == "E:":
                relocated = Path(r"E:\Ichilov2") / old_path.relative_to(Path("E:/"))
                if relocated.is_file():
                    current_path = relocated
            if not current_path.is_file() or current_path.stat().st_size != int(row.source_size):
                continue
            result[normal(str(current_path))] = {"view": row.view,
                                              "view_source": "prior EchoPrime classifier",
                                              "view_confidence": float(row.confidence)}
    for record in json.loads(E3_SELECTION.read_text(encoding="utf-8")):
        result[normal(record["source_path"])] = {"view": record["view"],
                  "view_source": record["selection_source"], "view_confidence": 1.0,
                  "selected_ichilov3_file_id": record["file_id"]}
    return result


def main() -> None:
    records = []
    for source in ("Ichilov1", "Ichilov2", "Ichilov3"):
        path = ROOT / f"{source.lower()}_all_ge_cines.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Full inventory unavailable: {path}")
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row["eligible_min16"] and Path(row["source_path"]).is_file():
                    records.append(row)
    labels = existing_labels()
    labels_by_sop = {}
    for row in records:
        label = labels.get(normal(row["source_path"]))
        if label and row["sop_uid"]:
            labels_by_sop[row["sop_uid"]] = label
    # A nonempty SOPInstanceUID is the first line of duplicate detection. The
    # source path remains the fallback key if an export omitted that UID.
    best = {}
    for row in records:
        uid = row["sop_uid"]
        key = ("sop", uid) if uid else ("path", normal(row["source_path"]))
        old = best.get(key)
        if old is not None and old["patient_id"] != row["patient_id"]:
            raise ValueError(f"One SOP UID maps to multiple patients: {uid}")
        if old is None or PREFERENCE[row["dataset"]] < PREFERENCE[old["dataset"]]:
            best[key] = row
    selected = sorted(best.values(), key=lambda r: (r["patient_id"], r["study_date"],
                                                   r["study_uid"], r["sop_uid"], r["source_path"]))
    for row in selected:
        row.update(labels.get(normal(row["source_path"])) or labels_by_sop.get(row["sop_uid"], {
            "view": "", "view_source": "pending EchoPrime classification", "view_confidence": 0.0}))
    tmp = ROOT / "unique_ge_cines.partial.jsonl"
    final = ROOT / "unique_ge_cines.jsonl"
    with tmp.open("w", encoding="utf-8") as stream:
        for row in selected:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    tmp.replace(final)
    summary = {"input_eligible_cines": len(records), "unique_sop_or_path": len(selected),
               "duplicate_removed": len(records) - len(selected),
               "patients": len({r["patient_id"] for r in selected}),
               "by_dataset": dict(Counter(r["dataset"] for r in selected)),
               "by_view": dict(Counter(r["view"] or "pending" for r in selected)),
               "view_source": dict(Counter(r["view_source"] for r in selected)),
               "manifest": str(final)}
    temp = ROOT / "unique_ge_cines_summary.tmp"
    temp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    temp.replace(ROOT / "unique_ge_cines_summary.json")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
