"""Read-only full DICOM-header inventory of GE ultrasound cines across datasets."""
from __future__ import annotations

import argparse
import json
import os
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pydicom

from ichilov_vendor_overlap_inventory import (classify_vendor, norm_date,
                                             patient_hash_from_path, patient_map)

SOURCES = {
    "Ichilov1": Path(r"F:\DS\Ichilov"),
    "Ichilov2": Path(r"E:\Ichilov2"),
    "Ichilov3": Path(r"E:\Ichilov3"),
}
OUTPUT = Path(r"D:\DS\ichilov_ge_inventory_20260930")
TAGS = ["PatientID", "StudyDate", "StudyInstanceUID", "SOPInstanceUID", "Manufacturer",
        "ManufacturerModelName", "Modality", "NumberOfFrames", "Rows", "Columns",
        "ImageType", "FrameTime", "CineRate", "HeartRate", "PhotometricInterpretation"]


def scan_one(dataset: str):
    root = SOURCES[dataset]
    if not root.exists():
        raise FileNotFoundError(root)
    mapping = patient_map() if dataset == "Ichilov2" else {}
    outfile = OUTPUT / f"{dataset.lower()}_all_ge_cines.jsonl"
    partial = outfile.with_suffix(".partial.jsonl")
    stats = Counter()
    vendors = Counter()
    patients = set()
    with partial.open("w", encoding="utf-8") as stream:
        for directory, _, names in os.walk(root):
            folder = Path(directory)
            candidates = [name for name in names if not Path(name).suffix or
                          Path(name).suffix.lower() == ".dcm"]
            for name in sorted(candidates):
                path = folder / name
                stats["candidate_files"] += 1
                try:
                    d = pydicom.dcmread(path, stop_before_pixels=True, force=True,
                                       specific_tags=TAGS)
                    val = lambda key: str(getattr(d, key, "") or "").strip()
                    relative = path.relative_to(root)
                    patient = val("PatientID")
                    if dataset == "Ichilov2":
                        patient = mapping.get(patient or patient_hash_from_path(path), "")
                    elif dataset == "Ichilov3":
                        # The study folder contains the stable research patient ID.
                        patient = relative.parts[0]
                    date = norm_date(val("StudyDate"))
                    if not date and dataset != "Ichilov2" and len(relative.parts) > 1:
                        date = norm_date(relative.parts[1])
                    frame_text = val("NumberOfFrames")
                    frames = int(frame_text) if frame_text.isdecimal() else 0
                    vendor, _ = classify_vendor(val("Manufacturer"), val("ManufacturerModelName"))
                    vendors[vendor] += 1
                    stats["readable_headers"] += 1
                    if not vendor.startswith("GE ") or val("Modality") != "US" or frames < 2:
                        continue
                    rows = int(val("Rows")) if val("Rows").isdecimal() else 0
                    cols = int(val("Columns")) if val("Columns").isdecimal() else 0
                    if not rows or not cols:
                        continue
                    record = {
                        "dataset": dataset, "source_path": str(path), "patient_id": patient,
                        "patient_id_available": bool(patient), "study_date": date,
                        "study_uid": val("StudyInstanceUID"), "sop_uid": val("SOPInstanceUID"),
                        "vendor": vendor, "manufacturer": val("Manufacturer"),
                        "scanner_model": val("ManufacturerModelName"), "frames": frames,
                        "rows": rows, "columns": cols, "image_type": val("ImageType"),
                        "photometric": val("PhotometricInterpretation"),
                        "frame_time": val("FrameTime"), "cine_rate": val("CineRate"),
                        "heart_rate": val("HeartRate"), "file_size": path.stat().st_size,
                        "eligible_min16": frames >= 16 and bool(patient) and bool(date)
                    }
                    stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                    stats["ge_cines"] += 1
                    if record["eligible_min16"]:
                        stats["eligible_min16"] += 1
                    patients.add(patient)
                except Exception:
                    stats["header_errors"] += 1
                if stats["candidate_files"] % 5000 == 0:
                    print(f"{dataset}: {stats['candidate_files']} candidate files, "
                          f"{stats['ge_cines']} GE cines, {stats['header_errors']} errors", flush=True)
    partial.replace(outfile)
    summary = {"dataset": dataset, "source": str(root),
               "created_utc": datetime.now(timezone.utc).isoformat(),
               "counts": dict(stats), "vendors_by_file": dict(vendors),
               "patients_with_ge_cines": len(patients), "manifest": str(outfile)}
    temp = (OUTPUT / f"{dataset.lower()}_summary.tmp")
    temp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    temp.replace(OUTPUT / f"{dataset.lower()}_summary.json")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--dataset", choices=SOURCES, required=True)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    try:
        scan_one(args.dataset)
    except Exception:
        (OUTPUT / f"{args.dataset.lower()}_error.txt").write_text(traceback.format_exc(),
                                                                  encoding="utf-8")
        raise
