"""Read-only DICOM header inventory for Ichilov2/3 overlap and legacy vendor mix."""
from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pydicom


ROOT2 = Path(r"E:\Ichilov2")
ROOT3 = Path(r"E:\Ichilov3")
ROOT1 = Path(r"F:\DS\Ichilov")
OUTPUT = Path(r"D:\DS\ichilov_vendor_overlap_20260930")
TAGS = ["PatientID", "StudyDate", "StudyInstanceUID", "Manufacturer",
        "ManufacturerModelName", "Modality", "NumberOfFrames", "Rows", "Columns"]
MODEL_VENDOR = {
    "Vivid S70": "GE Healthcare Ultrasound",
    "Vivid E95": "GE Vingmed Ultrasound",
    "iE33": "Philips Medical Systems",
    "EPIQ 7C": "Philips Medical Systems",
    "EPIQ CVx": "Philips Medical Systems",
    "EPIQ CVxi": "Philips Medical Systems",
}


def norm_date(value: object) -> str:
    s = str(value or "").strip().replace("_", "").replace("-", "")
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}" if re.fullmatch(r"\d{8}", s) else ""


def classify_vendor(manufacturer: str, model: str) -> tuple[str, str]:
    manufacturer, model = manufacturer.strip(), model.strip()
    if manufacturer in ("GE Healthcare Ultrasound", "GE Vingmed Ultrasound", "Philips Medical Systems"):
        return manufacturer, "DICOM Manufacturer"
    if model in MODEL_VENDOR:
        return MODEL_VENDOR[model], "DICOM model inferred from Ichilov3"
    if manufacturer.lower().startswith("tomtec"):
        return "TomTec post-processing", "DICOM Manufacturer"
    return manufacturer or "Unknown", "DICOM Manufacturer" if manufacturer else "Unknown"


def header(path: Path) -> dict:
    ds = pydicom.dcmread(path, stop_before_pixels=True, force=True, specific_tags=TAGS)
    val = lambda name: str(getattr(ds, name, "") or "").strip()
    frames = val("NumberOfFrames")
    return {
        "patient_dicom": val("PatientID"), "date": norm_date(val("StudyDate")),
        "study_uid": val("StudyInstanceUID"), "manufacturer": val("Manufacturer"),
        "model": val("ManufacturerModelName"), "modality": val("Modality"),
        "frames": int(frames) if frames.isdecimal() else 0,
        "rows": int(val("Rows")) if val("Rows").isdecimal() else 0,
        "columns": int(val("Columns")) if val("Columns").isdecimal() else 0,
    }


def patient_hash_from_path(path: Path) -> str:
    for part in path.relative_to(ROOT2).parts:
        if len(part) >= 30 and part.startswith("P") and re.fullmatch(r"[A-Za-z0-9_-]+", part):
            return part
    return ""


def patient_map() -> dict[str, str]:
    mapping = pd.read_excel(ROOT2 / "anonymization_mapping_hashed.xlsx")
    mapping["original_id"] = mapping.source_file.astype(str).str.extract(r"AutoStrainCap_(\d+-\d+)", expand=False)
    pairs = mapping[["PatID", "original_id"]].dropna().drop_duplicates()
    if pairs.groupby("PatID").original_id.nunique().gt(1).any():
        raise ValueError("A hashed patient maps to several original IDs")
    return dict(zip(pairs.PatID, pairs.original_id))


def annotate_report_availability(rows: list[dict]) -> None:
    mapping = pd.read_excel(ROOT2 / "anonymization_mapping_hashed.xlsx")
    report_dates = pd.to_datetime(mapping.study_date_and_time.astype(str).str.slice(0, 10), errors="coerce")
    keys = set(zip(mapping.PatID.astype(str), report_dates.dt.strftime("%Y-%m-%d")))
    for row in rows:
        row["strain_report_available"] = (row["patient_key"], row["visit_date"]) in keys


def scan(root: Path, dataset: str) -> dict:
    studies: dict[tuple, dict] = {}
    files = scanned = errors = folders = 0
    for directory, _, names in os.walk(root):
        folder = Path(directory)
        candidates = [name for name in names if not Path(name).suffix or Path(name).suffix.lower() == ".dcm"]
        if not candidates:
            continue
        folders += 1
        candidates.sort()
        files += len(candidates)
        n = len(candidates)
        positions = sorted({round(i * (n - 1) / (min(n, 12) - 1)) for i in range(min(n, 12))}) if n > 1 else [0]
        for position in positions:
            name = candidates[position]
            path = folder / name
            scanned += 1
            try:
                d = header(path)
                if not (d["date"] or d["study_uid"] or d["patient_dicom"]):
                    raise ValueError("No identifying DICOM tags")
                relative = path.relative_to(root)
                if dataset == "Ichilov2":
                    patient = d["patient_dicom"] or patient_hash_from_path(path)
                elif dataset == "Ichilov_original":
                    patient = d["patient_dicom"] or relative.parts[0]
                else:
                    patient = relative.parts[0]
                folder_date = norm_date(relative.parts[1]) if dataset != "Ichilov2" and len(relative.parts) > 1 else ""
                date = d["date"] or folder_date
                uid = d["study_uid"] or str(folder)
                key = (patient, date, uid)
                if key not in studies:
                    studies[key] = {
                        "patient_key": patient, "date": date, "study_uid": d["study_uid"],
                        "file_count": 0, "cine_count": 0, "vendor_cines": Counter(),
                        "vendor_us": Counter(), "models": Counter(), "manufacturers": Counter(),
                        "folders": set(), "categories": set(), "dicom_patient_ids": set(),
                    }
                s = studies[key]
                s["file_count"] += 1
                s["folders"].add(str(folder))
                s["dicom_patient_ids"].add(d["patient_dicom"])
                if dataset == "Ichilov2":
                    s["categories"].add(relative.parts[0])
                if d["model"]:
                    s["models"][d["model"]] += 1
                if d["manufacturer"]:
                    s["manufacturers"][d["manufacturer"]] += 1
                vendor, _ = classify_vendor(d["manufacturer"], d["model"])
                if d["modality"] == "US" and vendor != "TomTec post-processing":
                    s["vendor_us"][vendor] += 1
                    if d["frames"] > 1 and d["rows"] > 0 and d["columns"] > 0:
                        s["cine_count"] += 1
                        s["vendor_cines"][vendor] += 1
            except Exception:
                errors += 1
        if folders % 100 == 0:
            print(f"{dataset}: {folders} folders, {scanned} headers, {len(studies)} study keys, {errors} read errors", flush=True)
    rows = []
    for s in studies.values():
        use = s["vendor_cines"] or s["vendor_us"]
        vendors = {k: v for k, v in use.items() if k != "Unknown"}
        vendor = max(vendors, key=vendors.get) if vendors else "Unknown"
        rows.append({
            "patient_key": s["patient_key"], "date": s["date"], "study_uid": s["study_uid"],
            "vendor": vendor, "vendor_basis": "cine" if s["vendor_cines"] else "US header",
            "vendor_counts": dict(use), "mixed_vendor": len(vendors) > 1,
            "models": dict(s["models"]), "manufacturers": dict(s["manufacturers"]),
            "file_count": s["file_count"], "cine_count": s["cine_count"],
            "folders": sorted(s["folders"]), "categories": sorted(s["categories"]),
            "dicom_patient_ids": sorted(s["dicom_patient_ids"]),
        })
    print(f"{dataset}: complete, {files} candidate files, {scanned} sampled headers, {len(rows)} study keys, {errors} read errors", flush=True)
    return {"dataset": dataset, "files": files, "headers_scanned": scanned,
            "read_errors": errors, "studies": rows}


def visit_rows(studies: list[dict], dataset: str, mapping: dict[str, str] | None = None) -> list[dict]:
    by_visit = defaultdict(list)
    for study in studies:
        original = mapping.get(study["patient_key"], "") if mapping is not None else study["patient_key"]
        by_visit[(study["patient_key"], study["date"], original)].append(study)
    result = []
    for (key, date, original), group in by_visit.items():
        cine_vendors = Counter()
        for s in group:
            cine_vendors.update(s["vendor_counts"])
        valid_vendors = {v: n for v, n in cine_vendors.items() if v != "Unknown"}
        vendor = max(valid_vendors, key=valid_vendors.get) if valid_vendors else "Unknown"
        result.append({
            "dataset": dataset, "patient_key": key, "original_patient_id": original,
            "visit_date": date, "vendor": vendor, "mixed_vendor": len(valid_vendors) > 1,
            "study_count": len(group), "study_uids": sorted({s["study_uid"] for s in group if s["study_uid"]}),
            "file_count": sum(s["file_count"] for s in group),
            "cine_count": sum(s["cine_count"] for s in group),
            "scanner_models": sorted({m for s in group for m in s["models"]}),
            "categories": sorted({c for s in group for c in s["categories"]}),
            "folders": sorted({f for s in group for f in s["folders"]}),
        })
    return sorted(result, key=lambda r: (r["original_patient_id"], r["visit_date"], r["patient_key"]))


def finalize(raw: dict) -> None:
    mapping = patient_map()
    visits = {
        "Ichilov2": visit_rows(raw["Ichilov2"]["studies"], "Ichilov2", mapping),
        "Ichilov3": visit_rows(raw["Ichilov3"]["studies"], "Ichilov3"),
        "Ichilov_original": visit_rows(raw["Ichilov_original"]["studies"], "Ichilov_original"),
    }
    e3 = {(r["original_patient_id"], r["visit_date"]): r for r in visits["Ichilov3"]}
    e3_uids = {uid: r for r in visits["Ichilov3"] for uid in r["study_uids"]}
    original = {(r["original_patient_id"], r["visit_date"]): r for r in visits["Ichilov_original"]}
    original_uids = {uid: r for r in visits["Ichilov_original"] for uid in r["study_uids"]}
    for r in visits["Ichilov2"]:
        key = (r["original_patient_id"], r["visit_date"])
        e3_match = e3.get(key) if r["original_patient_id"] else None
        shared_uid = [uid for uid in r["study_uids"] if uid in e3_uids]
        if e3_match:
            r["overlap_status"] = "patient_date_and_uid" if set(r["study_uids"]) & set(e3_match["study_uids"]) else "patient_date_only_uid_differs"
        elif shared_uid:
            r["overlap_status"] = "study_uid_only_check_identity"
        else:
            r["overlap_status"] = "Ichilov2_only"
        r["ichilov3_vendor"] = e3_match["vendor"] if e3_match else ""
        r["ichilov3_study_uids"] = e3_match["study_uids"] if e3_match else []
        old_match = original.get(key) if r["original_patient_id"] else None
        old_shared_uid = [uid for uid in r["study_uids"] if uid in original_uids]
        if old_match:
            r["original_ichilov_overlap"] = "patient_date_and_uid" if set(r["study_uids"]) & set(old_match["study_uids"]) else "patient_date_only_uid_differs"
        elif old_shared_uid:
            r["original_ichilov_overlap"] = "study_uid_only_check_identity"
        else:
            r["original_ichilov_overlap"] = "not_found"
    annotate_report_availability(visits["Ichilov2"])
    for name in raw:
        (OUTPUT / f"{name.lower()}_study_inventory.json").write_text(json.dumps(raw[name],indent=2),encoding="utf-8")
        (OUTPUT / f"{name.lower()}_visits.json").write_text(json.dumps(visits[name],indent=2),encoding="utf-8")
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "datasets": {name: {
            "source_files": raw[name]["files"], "sampled_headers": raw[name]["headers_scanned"],
            "header_read_errors": raw[name]["read_errors"],
            "study_keys": len(raw[name]["studies"]), "visits": len(visits[name]),
            "patients": len({r["patient_key"] for r in visits[name]}),
            "vendors_by_visit": dict(Counter(r["vendor"] for r in visits[name])),
            "mixed_vendor_visits": sum(r["mixed_vendor"] for r in visits[name]),
        } for name in raw},
        "ichilov2_overlap": dict(Counter(r["overlap_status"] for r in visits["Ichilov2"])),
        "ichilov2_unmapped_patients": len({r["patient_key"] for r in visits["Ichilov2"] if not r["original_patient_id"]}),
        "ichilov2_unique_vendor": dict(Counter(r["vendor"] for r in visits["Ichilov2"] if r["overlap_status"] == "Ichilov2_only")),
        "ichilov2_only_overlap_with_original": dict(Counter(r["original_ichilov_overlap"] for r in visits["Ichilov2"] if r["overlap_status"] == "Ichilov2_only")),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            "Ichilov2": pool.submit(scan, ROOT2, "Ichilov2"),
            "Ichilov3": pool.submit(scan, ROOT3, "Ichilov3"),
            "Ichilov_original": pool.submit(scan, ROOT1, "Ichilov_original"),
        }
        raw = {name: future.result() for name, future in futures.items()}
    finalize(raw)


def rebuild_saved() -> None:
    raw = {name: json.loads((OUTPUT / f"{name.lower()}_study_inventory.json").read_text(encoding="utf-8"))
           for name in ("Ichilov2", "Ichilov3", "Ichilov_original")}
    for study in raw["Ichilov2"]["studies"]:
        ids = {value for value in study["dicom_patient_ids"] if value}
        if len(ids) != 1:
            raise ValueError("Ichilov2 study lacks one unambiguous DICOM PatientID")
        study["patient_key"] = ids.pop()
    finalize(raw)


if __name__ == "__main__":
    import sys
    rebuild_saved() if "--rebuild-saved" in sys.argv else main()
