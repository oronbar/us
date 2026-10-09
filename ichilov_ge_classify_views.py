"""Resumable EchoPrime view screening for GE cines without existing labels.

Reads original DICOMs; writes only private classification metadata. Run as the
sole GPU process, after the EchoFM job releases the GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import traceback
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pydicom
import torch

from dicom_prediction_video import (VIEWS, decode_frames, doppler_pixels,
                                    load_view_model, normalize,
                                    normalize_appearance, spatial_geometry)

ROOT = Path(r"D:\DS\ichilov_ge_inventory_20260930")
SOURCE = ROOT / "unique_ge_cines.jsonl"
CACHE = ROOT / "view_predictions"
OUTPUT = ROOT / "classified_ge_cines.jsonl"
WEIGHTS = Path(r"D:\us\output\dicom_prediction\weights")
TARGETS = {"A2C", "A3C", "A4C"}
VERSION = "EchoPrime released 11-view classifier, 3-frame B-mode screen v1"


def cache_key(row: dict) -> str:
    stat = Path(row["source_path"]).stat()
    datum = f"{VERSION}|{row['source_path']}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha256(datum.encode()).hexdigest()[:32]


def classify(row: dict, model, device: str) -> dict:
    path = row["source_path"]
    d = pydicom.dcmread(path, stop_before_pixels=True, force=True)
    geo = spatial_geometry(d)
    if not geo["eligible_region"] or geo.get("multiple_tissue_regions", False):
        return {"view": "", "view_source": VERSION,
                "view_confidence": 0., "screen": "ambiguous_tissue_region"}
    if 2 in geo["region_types"]:
        return {"view": "", "view_source": VERSION,
                "view_confidence": 0., "screen": "doppler_region"}
    n = int(row["frames"])
    indices = np.unique(np.rint(np.linspace(n * .2, n * .8, 3)).astype(int)).tolist()
    pixels = decode_frames(path, indices, geo["crop"])
    if doppler_pixels(pixels):
        return {"view": "", "view_source": VERSION,
                "view_confidence": 0., "screen": "color_doppler"}
    display, appearance = normalize_appearance(pixels)
    tensor = torch.from_numpy(display.copy()).permute(0, 3, 1, 2).float().to(device)
    with torch.inference_mode():
        probs = model(normalize(tensor, "echoprime")).softmax(-1).cpu().numpy()
    mean = probs.mean(0)
    choice = VIEWS[int(mean.argmax())]
    agreement = float(np.mean(np.argmax(probs, axis=1) == int(mean.argmax())))
    confidence = float(mean.max())
    accepted = choice in TARGETS and confidence >= .50 and agreement >= 2 / 3
    return {"view": choice if accepted else "", "predicted_view": choice,
            "view_source": VERSION, "view_confidence": confidence,
            "frame_agreement": agreement, "screen": "accepted" if accepted else "uncertain_view",
            "appearance": appearance, "sampled_indices": indices}


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    with SOURCE.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    pending = [r for r in rows if not r["view"]]
    if not pending:
        print("All GE cines already have a view label", flush=True)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda" and not args.smoke:
        raise RuntimeError("GPU unavailable for the GE view-classification job")
    model = load_view_model(WEIGHTS, device) if pending else None
    started = time.time()
    choices = pending[:1] if args.smoke else pending
    for index, row in enumerate(choices, 1):
        key = cache_key(row)
        dest = CACHE / f"{key}.json"
        if dest.exists():
            saved = json.loads(dest.read_text(encoding="utf-8"))
            if saved["cache_key"] != key or saved["source_path"] != row["source_path"]:
                raise ValueError(f"View cache collision: {dest}")
            continue
        try:
            prediction = classify(row, model, device)
        except Exception as exc:
            prediction = {"view": "", "view_source": VERSION, "view_confidence": 0.,
                          "screen": "error", "error": f"{type(exc).__name__}: {str(exc)[:250]}"}
        saved = {"cache_key": key, "source_path": row["source_path"], **prediction}
        temp = dest.with_suffix(".tmp")
        temp.write_text(json.dumps(saved), encoding="utf-8")
        temp.replace(dest)
        if index % 100 == 0 or args.smoke:
            print(f"GE view screening {index}/{len(choices)} ({time.time()-started:.0f}s)",
                  flush=True)
    if args.smoke:
        return
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    counts = Counter()
    tmp = OUTPUT.with_suffix(".partial.jsonl")
    with tmp.open("w", encoding="utf-8") as stream:
        for row in rows:
            if not row["view"]:
                saved = json.loads((CACHE / f"{cache_key(row)}.json").read_text(encoding="utf-8"))
                row.update({k: v for k, v in saved.items() if k not in ("cache_key", "source_path")})
            counts[row["view"] or "unclassified"] += 1
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    tmp.replace(OUTPUT)
    (ROOT / "view_classification_complete.json").write_text(json.dumps({
        "total": len(rows), "new_screened": len(pending), "views": dict(counts),
        "manifest": str(OUTPUT), "version": VERSION}, indent=2), encoding="utf-8")
    print(json.dumps(dict(counts), indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        ROOT.mkdir(parents=True, exist_ok=True)
        (ROOT / "view_classification_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
