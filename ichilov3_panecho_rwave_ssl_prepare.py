"""Cache the exact first R-R cycle pixels for fold-isolated PanEcho SSL."""

import json
import os
from pathlib import Path

import cv2
import numpy as np

from ichilov3_train_fusion import atomic, hashfile


ROOT = Path(r"D:\DS\ichilov3_panecho_rwave_ssl_20261005")
SOURCE = Path(r"D:\DS\ichilov3_rwave_first_cycle_panecho_20261005")
MANIFEST = SOURCE / "panecho_manifest.json"


def main():
    cache = ROOT / "clips"
    cache.mkdir(parents=True, exist_ok=True)
    records = json.loads(MANIFEST.read_text(encoding="utf-8"))
    eligible = [r for r in records if r.get("rwave_resampled") or r.get("first_cycle_only")]
    if len(eligible) != 919:
        raise ValueError(f"Expected 919 R-wave cines, found {len(eligible)}")
    protocol = {"policy": "first R marker to second R marker, 16 nearest native frames",
                "source_manifest_sha256": hashfile(MANIFEST), "eligible_cines": len(eligible),
                "input": "original padded crop; native RGB; no change to source data",
                "cache": "one uint8 16-frame 224px cine; augmentations only during SSL"}
    target = ROOT / "cache_protocol.json"
    if target.exists() and json.loads(target.read_text()) != protocol:
        raise ValueError("Existing cache protocol changed")
    atomic(target, protocol)
    cv2.setNumThreads(1)
    rows = []
    for count, record in enumerate(eligible, 1):
        ident = record["file_id"]
        path = cache / f"{ident}.npz"
        with np.load(record["embedding_path"]) as f:
            indices = f["source_frame_indices"][0].astype(int)
        if len(indices) != 16 or np.any(np.diff(indices) <= 0):
            raise ValueError(f"Invalid 16-frame R-R sample: {ident}")
        if not path.exists():
            with np.load(record["crop_output"]) as f:
                frames = f["frames"]
                if not np.array_equal(f["source_frame_indices"], np.arange(len(frames))):
                    raise ValueError(f"Crop order changed: {ident}")
            window = frames[indices]
            if window.ndim == 3:
                window = np.repeat(window[..., None], 3, axis=-1)
            window = np.stack([cv2.resize(x, (224, 224), interpolation=cv2.INTER_AREA)
                               for x in window])
            if window.shape != (16, 224, 224, 3) or window.dtype != np.uint8:
                raise ValueError(f"Invalid prepared pixels: {ident}")
            tmp = path.with_suffix(".partial.npz")
            np.savez_compressed(tmp, frames=window, source_frame_indices=indices)
            os.replace(tmp, path)
        rows.append({"patient_id": record["patient"], "visit_date": record["visit_date"],
                     "view": record["view"], "cache_id": ident,
                     "source_embedding_path": record["embedding_path"]})
        if count % 25 == 0 or count == len(eligible):
            atomic(ROOT / "status.json", {"stage": "preparing_cines", "completed": count,
                                         "total": len(eligible), "pid": os.getpid()})
            print(f"Prepared {count}/{len(eligible)}", flush=True)
    lines = "\n".join(json.dumps(row) for row in rows) + "\n"
    (ROOT / "training_manifest.jsonl").write_text(lines, encoding="utf-8")
    atomic(ROOT / "cache_complete.json", {"cines": len(rows), "patients": len(set(
        row["patient_id"] for row in rows)), "manifest_sha256": hashfile(MANIFEST)})


if __name__ == "__main__":
    main()
