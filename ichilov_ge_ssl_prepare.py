"""Cache two native-color temporal windows per GE apical cine for SSL.

Original DICOMs and prior crops are read-only. The cache is resumable and is
shared by fold-specific training runs; every cache item records its source
fingerprint and exact frame indices.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pydicom

from dicom_prediction_video import decode_frames, spatial_geometry

ROOT = Path(r"D:\DS\ichilov_ge_inventory_20260930")
SOURCE = ROOT / "classified_ge_cines.jsonl"
CACHE = Path(r"D:\DS\ichilov_ge_ssl_clips_20260930")
VIEWS = {"A2C", "A3C", "A4C"}
VERSION = "native-color-16x224-two-period-windows-v1"


def key(row: dict) -> str:
    return hashlib.sha256((row["dataset"] + "|" + row["source_path"] + "|" +
                           row["sop_uid"]).encode()).hexdigest()[:24]


def frame_indices(row: dict) -> np.ndarray:
    n = int(row["frames"])
    fps = None
    try:
        if float(row.get("frame_time") or 0) > 0:
            fps = 1000 / float(row["frame_time"])
        elif float(row.get("cine_rate") or 0) > 0:
            fps = float(row["cine_rate"])
    except (TypeError, ValueError):
        pass
    try:
        hr = float(row["heart_rate"])
    except (TypeError, ValueError):
        hr = 0
    if fps and 30 <= hr <= 220:
        period = round(60 * fps / hr)
        if 16 <= period <= n:
            starts = [0, n - period]
            return np.stack([np.rint(np.linspace(s, s + period - 1, 16)).astype(np.int32)
                             for s in starts])
    # When timing is unavailable, use two overlapping observed cine windows.
    span = max(16, min(n, round(n * .7)))
    return np.stack([np.rint(np.linspace(s, s + span - 1, 16)).astype(np.int32)
                     for s in (0, n - span)])


def create(row: dict, target: Path) -> dict:
    source = Path(row["source_path"])
    stat = source.stat()
    indices = frame_indices(row)
    d = pydicom.dcmread(source, stop_before_pixels=True, force=True)
    geometry = spatial_geometry(d)
    if not geometry["eligible_region"]:
        raise ValueError("No declared tissue-region geometry")
    pixels = decode_frames(source, indices.flatten(), geometry["crop"])
    pixels = pixels.reshape(2, 16, 224, 224, 3)
    if pixels.dtype != np.uint8:
        raise ValueError("Unexpected cine pixel type")
    temp = target.with_suffix(".partial.npz")
    np.savez_compressed(temp, windows=pixels, indices=indices)
    temp.replace(target)
    return {"source_size": stat.st_size, "source_mtime_ns": stat.st_mtime_ns,
            "crop": geometry["crop"], "indices": indices.tolist()}


def process_row(row: dict) -> tuple[dict | None, dict | None]:
    cv2.setNumThreads(1)
    file_id = key(row)
    target = CACHE / f"{file_id}.npz"
    metadata = CACHE / f"{file_id}.json"
    if metadata.exists() and target.exists():
        saved = json.loads(metadata.read_text(encoding="utf-8"))
        stat = Path(row["source_path"]).stat()
        if (saved["source_size"], saved["source_mtime_ns"]) != (stat.st_size, stat.st_mtime_ns):
            raise ValueError(f"Source changed since cache creation: {row['source_path']}")
        return row | {"cache_id": file_id}, None
    try:
        meta = create(row, target)
        saved = {"cache_id": file_id, "version": VERSION, "source_path": row["source_path"],
                 **meta}
        tmp = metadata.with_suffix(".tmp")
        tmp.write_text(json.dumps(saved), encoding="utf-8")
        tmp.replace(metadata)
        return row | {"cache_id": file_id}, None
    except Exception as exc:
        return None, {"source_path": row["source_path"],
                      "reason": f"{type(exc).__name__}: {str(exc)[:200]}"}


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    CACHE.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    with SOURCE.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    rows = [r for r in rows if r["view"] in VIEWS]
    started = time.time()
    valid = []
    failures = []
    # Decoding is CPU-heavy and independent per cine. Three bounded workers
    # shorten preparation while leaving CPU and I/O headroom for the GPU job.
    with ProcessPoolExecutor(max_workers=3) as pool:
        for index, (accepted, failure) in enumerate(pool.map(process_row, rows,
                                                               chunksize=1), 1):
            if accepted is not None:
                valid.append(accepted)
            if failure is not None:
                failures.append(failure)
            if index % 100 == 0 or index == len(rows):
                print(f"GE SSL cache {index}/{len(rows)}, valid {len(valid)}, "
                      f"failed {len(failures)} ({time.time()-started:.0f}s)", flush=True)
    manifest = CACHE / "training_manifest.jsonl"
    temp = manifest.with_suffix(".partial.jsonl")
    with temp.open("w", encoding="utf-8") as stream:
        for row in valid:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    temp.replace(manifest)
    errors = CACHE / "cache_failures.json"
    errors.write_text(json.dumps(failures, indent=2), encoding="utf-8")
    (CACHE / "run_complete.json").write_text(json.dumps({
        "input_apical_cines": len(rows), "cached": len(valid), "failed": len(failures),
        "manifest": str(manifest), "version": VERSION}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        CACHE.mkdir(parents=True, exist_ok=True)
        (CACHE / "fatal_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
