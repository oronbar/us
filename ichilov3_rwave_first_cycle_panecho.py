"""Correct PanEcho R-wave sampling to first-to-second R marker (one cycle)."""

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import pydicom
import torch

import ichilov3_rwave_panecho_trial as prior
from dicom_prediction_encode import load_encoder
from dicom_prediction_video import normalize
from ichilov3_encode_selected import timestamps, verify_artifacts
from ichilov3_train_fusion import atomic, hashfile


ROOT = Path(r"D:\DS\ichilov3_rwave_first_cycle_panecho_20261005")
SOURCE = Path(r"D:\DS\ichilov3_rwave_panecho_20261005")
CROPS = Path(r"D:\DS\ichilov3_stage2_padded_20260926")
POLICY = "16 nearest native frames between first and second R-wave markers, inclusive; one cardiac cycle"


def first_cycle_indices(record, markers):
    n = int(record["frames"])
    timeline, source = timestamps(record, np.arange(n))
    targets = np.linspace(float(markers[0]) / 1000, float(markers[1]) / 1000, 16)
    right = np.clip(np.searchsorted(timeline, targets), 0, n - 1)
    left = np.maximum(0, right - 1)
    indices = np.where(np.abs(timeline[left] - targets) <=
                       np.abs(timeline[right] - targets), left, right).astype(int)
    if np.any(np.diff(indices) <= 0):
        raise ValueError("First R-R cycle has fewer than 16 unique native frames")
    return indices, source


def main():
    ROOT.mkdir(exist_ok=True, parents=True)
    records = json.loads((CROPS / "full_manifest.json").read_text(encoding="utf-8"))
    old = {r["file_id"]: r for r in json.loads((SOURCE / "panecho_manifest.json").read_text(encoding="utf-8"))}
    multi = []
    for record in records:
        ds = pydicom.dcmread(record["source_path"], stop_before_pixels=True)
        markers = getattr(ds, "RWaveTimeVector", [])
        if len(markers) > 2:
            multi.append((record, [float(x) for x in markers]))
    if len(multi) != 14:
        raise ValueError(f"Expected 14 multicycle cines; found {len(multi)}")
    protocol = {
        "policy": POLICY,
        "reencode": len(multi),
        "reuse": len(records) - len(multi),
        "source_trial": str(SOURCE),
        "source_trial_protocol_sha256": hashfile(SOURCE / "protocol.json"),
        "crop_manifest_sha256": hashfile(CROPS / "full_manifest.json"),
        "code_sha256": hashfile(Path(__file__)),
        "evaluation": "same GLS and future held-out patient folds as first-last trial",
    }
    pp = ROOT / "protocol.json"
    if pp.exists() and json.loads(pp.read_text()) != protocol:
        raise ValueError("Protocol differs from resumed run")
    atomic(pp, protocol)
    atomic(ROOT / "status.json", {"stage": "loading_panecho", "pid": os.getpid()})
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    verify_artifacts(prior.ART)
    model = load_encoder("panecho", prior.ART, "cuda")
    changed = {}
    for n, (record, markers) in enumerate(multi, 1):
        dest = ROOT / "panecho" / (record["file_id"] + ".npz")
        meta = dest.with_suffix(".json")
        if meta.exists():
            saved = json.loads(meta.read_text(encoding="utf-8"))
            if saved.get("sampling_policy") != POLICY or not dest.exists():
                raise ValueError("Invalid resume artifact")
        else:
            indices, source = first_cycle_indices(record, markers)
            with np.load(record["crop_output"]) as f:
                frames = f["frames"]
                if not np.array_equal(f["source_frame_indices"], np.arange(len(frames))):
                    raise ValueError("Crop frame order changed")
            window = frames[indices]
            if window.ndim == 3:
                window = np.repeat(window[..., None], 3, axis=-1)
            resized = np.stack([cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
                                for frame in window])
            tensor = torch.from_numpy(resized.copy()).permute(3, 0, 1, 2).unsqueeze(0).float().cuda()
            with torch.inference_mode():
                vector = model(normalize(tensor, "panecho")).cpu().numpy()[0].astype(np.float32)
            if vector.shape != (768,) or not np.isfinite(vector).all():
                raise ValueError("Invalid embedding")
            times, _ = timestamps(record, indices)
            dest.parent.mkdir(exist_ok=True)
            temp = dest.with_suffix(".partial.npz")
            np.savez_compressed(temp, window_embeddings=vector[None, :],
                                view_embedding=vector, source_frame_indices=indices[None, :],
                                sampled_times_seconds=times[None, :])
            temp.replace(dest)
            saved = dict(old[record["file_id"]], embedding_path=str(dest),
                         sampling_policy=POLICY, first_cycle_only=True, windows=1,
                         timing_source=source,
                         first_rwave_ms=markers[0], second_rwave_ms=markers[1],
                         source_frame_indices=indices.tolist(),
                         previous_embedding_path=old[record["file_id"]]["embedding_path"])
            atomic(meta, saved)
        changed[record["file_id"]] = saved
        atomic(ROOT / "status.json", {"stage": "encoding_14_corrected_cines", "completed": n,
                                       "total": len(multi), "pid": os.getpid()})
        print(f"Corrected {n}/{len(multi)}", flush=True)
    del model
    torch.cuda.empty_cache()
    manifest = [changed.get(record["file_id"], old[record["file_id"]]) for record in records]
    atomic(ROOT / "panecho_manifest.json", manifest)
    prior.ROOT = ROOT
    prior.export_visits(manifest)
    atomic(ROOT / "status.json", {"stage": "same_visit_gls", "pid": os.getpid()})
    subprocess.run([sys.executable, r"D:\us\ichilov3_physiology_probe.py",
                    "--model-inputs", str(ROOT), "--output", str(ROOT / "physiology")], check=True)
    atomic(ROOT / "status.json", {"stage": "next_visit_deterioration", "pid": os.getpid()})
    subprocess.run([sys.executable, r"D:\us\ichilov3_train_late_fusion.py",
                    "--input", str(ROOT / "aligned"), "--output", str(ROOT / "late_fusion")], check=True)
    atomic(ROOT / "run_complete.json", {"corrected_cines": len(multi),
                                        "reused_cines": len(records) - len(multi)})
    atomic(ROOT / "status.json", {"stage": "complete", "pid": os.getpid()})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        ROOT.mkdir(exist_ok=True, parents=True)
        atomic(ROOT / "status.json", {"stage": "error", "error": traceback.format_exc()})
        raise
