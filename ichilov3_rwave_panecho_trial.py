"""Frozen PanEcho trial with 16 native frames from first to last R-wave marker.

Only selected cines with DICOM RWaveTimeVector are re-encoded. Other views
retain their exact previously saved estimated-heartbeat embedding. Source
DICOMs, crops, and prior model artifacts are read-only.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch

from dicom_prediction_encode import load_encoder
from dicom_prediction_video import normalize
from ichilov3_encode_selected import timestamps, verify_artifacts
from ichilov3_train_fusion import atomic, hashfile


ROOT = Path(r"D:\DS\ichilov3_rwave_panecho_20261005")
CROPS = Path(r"D:\DS\ichilov3_stage2_padded_20260926")
BASE = Path(r"D:\DS\ichilov3_temporal_trial_20260927")
ALIGN = Path(r"D:\DS\ichilov3_aligned_20260927")
ART = Path(r"D:\us\output\dicom_prediction")
POLICY = "16 nearest native frames from first to last RWaveTimeVector marker, inclusive; one window"


def rwave_window(record):
    ds = pydicom.dcmread(record["source_path"], stop_before_pixels=True)
    if str(ds.SOPInstanceUID) != str(record["expected_sop"]):
        raise ValueError("Source SOP differs from selected manifest")
    r = np.asarray(getattr(ds, "RWaveTimeVector", []), dtype=float)
    if len(r) < 2:
        return None, {"reason": "no_rwave_pair", "rwave_markers": int(len(r))}
    if not np.isfinite(r).all() or np.any(np.diff(r) <= 0):
        raise ValueError("Invalid R-wave time vector")
    n = int(record["frames"])
    timeline, source = timestamps(record, np.arange(n))
    if not np.isfinite(timeline).all() or np.any(np.diff(timeline) <= 0):
        raise ValueError("Invalid frame timeline")
    start, end = float(r[0]) / 1000, float(r[-1]) / 1000
    frame_margin = float(np.median(np.diff(timeline))) / 2
    if start < timeline[0] - frame_margin or end > timeline[-1] + frame_margin:
        raise ValueError("R-wave marker outside cine frame timeline")
    targets = np.linspace(start, end, 16)
    right = np.clip(np.searchsorted(timeline, targets), 0, n - 1)
    left = np.maximum(0, right - 1)
    indices = np.where(np.abs(timeline[left] - targets) <=
                       np.abs(timeline[right] - targets), left, right).astype(int)
    if np.any(np.diff(indices) <= 0):
        raise ValueError("Fewer than 16 unique native frames between R markers")
    return indices, {
        "reason": "rwave_first_to_last",
        "rwave_markers": int(len(r)),
        "cardiac_cycles_spanned": int(len(r) - 1),
        "rwave_first_ms": float(r[0]),
        "rwave_last_ms": float(r[-1]),
        "rwave_span_ms": float(r[-1] - r[0]),
        "first_frame": int(indices[0]),
        "last_frame": int(indices[-1]),
        "native_frame_intervals_spanned": int(indices[-1] - indices[0]),
        "maximum_sample_time_error_ms": float(np.max(np.abs(timeline[indices] - targets)) * 1000),
        "timing_source": source,
    }


def export_visits(manifest):
    with np.load(BASE / "panecho_visits.npz") as f:
        visits = {k: f[k].copy() for k in f.files}
    lookup = {(str(p), str(d)): i for i, (p, d) in enumerate(zip(visits["patient_ids"], visits["visit_dates"]))}
    views = {"A2C": 0, "A3C": 1, "A4C": 2}
    for record in manifest:
        with np.load(record["embedding_path"]) as f:
            visits["embeddings"][lookup[(record["patient"], record["visit_date"])], views[record["view"]]] = f["view_embedding"]
    if not np.isfinite(visits["embeddings"]).all():
        raise ValueError("Nonfinite visit embedding")
    np.savez_compressed(ROOT / "panecho_visits.npz", **visits)

    shutil.copy2(BASE / "echoprime_visits.npz", ROOT / "echoprime_visits.npz")
    paired = ROOT / "aligned"
    paired.mkdir(exist_ok=True)
    for name in ("cohort.parquet", "patient_folds.parquet", "retained_baseline_oof.parquet"):
        shutil.copy2(ALIGN / name, paired / name)
    with np.load(BASE / "aligned" / "features.npz") as f:
        features = {k: f[k].copy() for k in f.files}
    cohort = pd.read_parquet(ALIGN / "cohort.parquet")
    aligned_visits = pd.read_parquet(ALIGN / "visit_alignment.parquet").set_index("visit_id")
    visit_embeddings = {(str(p), str(d)): v for p, d, v in zip(
        visits["patient_ids"], visits["visit_dates"], visits["embeddings"])}
    for i, row in cohort.iterrows():
        visit = aligned_visits.loc[row.current_visit_id]
        features["panecho_current"][i] = visit_embeddings[(str(visit.patient_id), str(visit.visit_date))]
    np.savez_compressed(paired / "features.npz", **features)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    records = json.loads((CROPS / "full_manifest.json").read_text(encoding="utf-8"))
    baseline = {r["file_id"]: r for r in json.loads((BASE / "panecho_manifest.json").read_text(encoding="utf-8"))}
    plans = {r["file_id"]: rwave_window(r) for r in records}
    counts = Counter(info["reason"] for _, info in plans.values())
    if counts != {"rwave_first_to_last": 919, "no_rwave_pair": 314}:
        raise ValueError(f"R-wave availability changed: {counts}")
    protocol = {
        "policy": POLICY,
        "selected_cines": len(records),
        "sampling_counts": dict(counts),
        "rwave_endpoint_policy": "first and last listed R-wave marker, including 14 multicycle cines",
        "fallback": "exact previous PanEcho estimated-heartbeat embedding for no-R-wave cines",
        "preprocessing": "original padded crop, native RGB, PanEcho normalization, 224px",
        "model": "frozen PanEcho pretrained checkpoint",
        "evaluation": "same matched visits and 3x5 patient-held-out folds as previous physiology and future tests",
        "limitation": "one R-anchored window versus up to four averaged estimated-heartbeat windows; timing and aggregation both change",
        "code_sha256": hashfile(Path(__file__)),
        "crop_manifest_sha256": hashfile(CROPS / "full_manifest.json"),
        "baseline_manifest_sha256": hashfile(BASE / "panecho_manifest.json"),
        "checkpoint_sha256": hashfile(ART / "weights" / "panecho.pt"),
    }
    path = ROOT / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Existing protocol differs; use a fresh output directory")
    atomic(path, protocol)
    audit = [dict(file_id=r["file_id"], patient=r["patient"], visit_date=r["visit_date"],
                  view=r["view"], manufacturer=r["manufacturer"], **plans[r["file_id"]][1])
             for r in records]
    pd.DataFrame(audit).to_parquet(ROOT / "sampling_audit.parquet", index=False)
    atomic(ROOT / "status.json", {"stage": "loading_encoder", "pid": os.getpid(), "counts": dict(counts)})
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    verify_artifacts(ART)
    model = load_encoder("panecho", ART, "cuda")
    output_records = []

    for n, record in enumerate(records, 1):
        indices, info = plans[record["file_id"]]
        dest = ROOT / "panecho" / (record["file_id"] + ".npz")
        meta = dest.with_suffix(".json")
        if meta.exists():
            saved = json.loads(meta.read_text(encoding="utf-8"))
            if saved["sampling_policy"] != POLICY or not Path(saved["embedding_path"]).exists():
                raise ValueError("Invalid resume artifact")
        elif indices is None:
            saved = dict(baseline[record["file_id"]], sampling_policy=POLICY,
                         sampling_info=info, rwave_resampled=False)
            dest.parent.mkdir(parents=True, exist_ok=True)
            atomic(meta, saved)
        else:
            old = baseline[record["file_id"]]
            if hashfile(Path(record["crop_output"]).with_suffix(".json")) != old["crop_metadata_sha256"]:
                raise ValueError("Crop metadata changed")
            with np.load(record["crop_output"]) as f:
                frames = f["frames"]
                if not np.array_equal(f["source_frame_indices"], np.arange(len(frames))):
                    raise ValueError("Crop changed frame order")
            if frames.dtype != np.uint8:
                raise ValueError("Unexpected crop dtype")
            window = frames[indices]
            if window.ndim == 3:
                window = np.repeat(window[..., None], 3, axis=-1)
            resized = np.stack([cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
                                for frame in window])
            tensor = torch.from_numpy(resized.copy()).permute(3, 0, 1, 2).unsqueeze(0).float().cuda()
            with torch.inference_mode():
                embedding = model(normalize(tensor, "panecho")).cpu().numpy()[0].astype(np.float32)
            if embedding.shape != (768,) or not np.isfinite(embedding).all():
                raise ValueError("Invalid PanEcho embedding")
            times, source = timestamps(record, indices)
            dest.parent.mkdir(parents=True, exist_ok=True)
            temp = dest.with_suffix(".partial.npz")
            np.savez_compressed(temp, window_embeddings=embedding[None, :],
                                view_embedding=embedding, source_frame_indices=indices[None, :],
                                sampled_times_seconds=times[None, :])
            temp.replace(dest)
            saved = dict(old, embedding_path=str(dest), sampling_policy=POLICY,
                         sampling_info=info, rwave_resampled=True, windows=1,
                         temporal_stride=None, timing_source=source,
                         version="rwave-first-last-v1",
                         baseline_embedding_path=old["embedding_path"])
            atomic(meta, saved)
        output_records.append(saved)
        if n % 25 == 0 or n == len(records):
            atomic(ROOT / "panecho_manifest.json", output_records)
            atomic(ROOT / "status.json", {"stage": "encoding_panecho", "completed": n,
                                           "total": len(records), "pid": os.getpid(),
                                           "elapsed_seconds": time.time() - started})
            print(f"PanEcho {n}/{len(records)}, elapsed {time.time() - started:.0f}s", flush=True)

    del model
    torch.cuda.empty_cache()
    export_visits(output_records)
    atomic(ROOT / "status.json", {"stage": "same_visit_gls", "pid": os.getpid()})
    subprocess.run([sys.executable, r"D:\us\ichilov3_physiology_probe.py",
                    "--model-inputs", str(ROOT), "--output", str(ROOT / "physiology")], check=True)
    atomic(ROOT / "status.json", {"stage": "next_visit_deterioration", "pid": os.getpid()})
    subprocess.run([sys.executable, r"D:\us\ichilov3_train_late_fusion.py",
                    "--input", str(ROOT / "aligned"), "--output", str(ROOT / "late_fusion")], check=True)
    atomic(ROOT / "run_complete.json", {"clips": len(records), "rwave_reencoded": 919,
                                        "elapsed_seconds": time.time() - started})
    atomic(ROOT / "status.json", {"stage": "complete", "pid": os.getpid(),
                                   "elapsed_seconds": time.time() - started})


if __name__ == "__main__":
    if "--preflight" in sys.argv:
        records = json.loads((CROPS / "full_manifest.json").read_text(encoding="utf-8"))
        counts = Counter()
        multicycle = Counter()
        for record in records:
            _, info = rwave_window(record)
            counts[info["reason"]] += 1
            if info["reason"] == "rwave_first_to_last":
                multicycle[info["cardiac_cycles_spanned"]] += 1
        print(json.dumps({"sampling_counts": dict(counts),
                          "cycles_spanned": dict(multicycle)}, indent=2))
    else:
        try:
            main()
        except Exception:
            ROOT.mkdir(parents=True, exist_ok=True)
            atomic(ROOT / "status.json", {"stage": "error", "error": traceback.format_exc()})
            raise
