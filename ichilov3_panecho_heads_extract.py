"""Apply released PanEcho regression heads to cached frozen Ichilov3 video embeddings."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ichilov3_train_fusion import atomic


ROOT = Path(r"D:\DS\ichilov3_temporal_trial_20260927")
MANIFEST = ROOT / "panecho_manifest.json"
CROPS = Path(r"D:\DS\ichilov3_stage2_padded_20260926\full_manifest.json")
CHECKPOINT = Path(r"D:\us\output\dicom_prediction\weights\panecho.pt")
OUTPUT = Path(r"D:\DS\ichilov3_panecho_heads_20261004")
TASKS = ("EF", "GLS", "LVEDV", "LVESV", "LAVol", "LAIDs2D", "TAPSE", "RVSVel")
VIEWS = ("A2C", "A3C", "A4C")


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manufacturers = {r["file_id"]: r["manufacturer"] for r in json.loads(CROPS.read_text(encoding="utf-8"))}
    if len(records) != 1233 or {r["checkpoint_sha256"] for r in records} != {digest(CHECKPOINT)}:
        raise ValueError("PanEcho checkpoint identity or clip count mismatch")
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)["weights"]
    weights, biases = [], []
    for task in TASKS:
        w = state[f"{task}_head.1.weight"].numpy()
        b = state[f"{task}_head.1.bias"].numpy()
        if w.shape != (1, 768) or b.shape != (1,):
            raise ValueError(f"Unexpected regression head: {task}")
        weights.append(w[0]); biases.append(b[0])
    W = np.stack(weights); b = np.asarray(biases)
    protocol = {
        "manifest": str(MANIFEST), "manifest_sha256": digest(MANIFEST),
        "crop_manifest": str(CROPS), "crop_manifest_sha256": digest(CROPS),
        "checkpoint": str(CHECKPOINT), "checkpoint_sha256": digest(CHECKPOINT),
        "tasks": TASKS, "views": VIEWS,
        "method": "official regression head linear layer applied to saved per-window PanEcho encoder outputs; mean across windows",
        "source_video_protocol": "estimated-heartbeat 16-frame windows where eligible; original stride-1 windows for short cines",
        "no_outcome_based_filtering": True,
    }
    old = OUTPUT / "protocol.json"
    if old.exists() and json.loads(old.read_text(encoding="utf-8")) != json.loads(json.dumps(protocol)):
        raise ValueError("Existing output has a different protocol")
    atomic(old, protocol)
    rows = []
    for number, r in enumerate(records, 1):
        with np.load(r["embedding_path"]) as f:
            x = f["window_embeddings"].astype(np.float32)
            if x.ndim != 2 or x.shape[1] != 768 or not np.isfinite(x).all():
                raise ValueError(f"Invalid PanEcho window embeddings: {r['file_id']}")
        window = x @ W.T + b
        if not np.isfinite(window).all():
            raise ValueError(f"Nonfinite head output: {r['file_id']}")
        row = {"file_id": r["file_id"], "patient_id": r["patient"],
               "visit_date": r["visit_date"], "view": r["view"],
               "manufacturer": manufacturers[r["file_id"]], "n_windows": len(x),
               "resampled": bool(r.get("resampled", False)),
               "embedding_path": r["embedding_path"]}
        for index, task in enumerate(TASKS):
            row[task] = float(window[:, index].mean())
            row[task + "_window_sd"] = float(window[:, index].std())
        rows.append(row)
        if number % 100 == 0 or number == len(records):
            atomic(OUTPUT / "status.json", {"processed": number, "total": len(records)})
            print(f"PanEcho heads {number}/{len(records)}", flush=True)
    clips = pd.DataFrame(rows)
    if clips.duplicated(["patient_id", "visit_date", "view"]).any():
        raise ValueError("Duplicate visit/view")
    clips.to_parquet(OUTPUT / "clip_measurements.parquet", index=False)
    visits = []
    for (patient, date), group in clips.groupby(["patient_id", "visit_date"]):
        if set(group.view) != set(VIEWS):
            raise ValueError(f"Missing visit view: {patient}, {date}")
        by = group.set_index("view")
        row = {"patient_id": patient, "visit_date": date, "manufacturer": group.manufacturer.iloc[0]}
        for task in TASKS:
            for view in VIEWS:
                row[f"{task}_{view}"] = float(by.loc[view, task])
            selected = ("A2C", "A3C", "A4C") if task in {"EF", "GLS", "LVEDV", "LVESV"} else (("A2C", "A4C") if task in {"LAVol", "LAIDs2D"} else ("A4C",))
            row[task] = float(np.mean([by.loc[v, task] for v in selected]))
        visits.append(row)
    visit_table = pd.DataFrame(visits).sort_values(["patient_id", "visit_date"])
    if len(visit_table) != 411 or not np.isfinite(visit_table[list(TASKS)].to_numpy()).all():
        raise ValueError("Visit head output incomplete")
    visit_table.to_parquet(OUTPUT / "visit_measurements.parquet", index=False)
    atomic(OUTPUT / "run_complete.json", {"clips": len(clips), "visits": len(visit_table),
                                          "patients": int(visit_table.patient_id.nunique()),
                                          "clip_table": str(OUTPUT / "clip_measurements.parquet"),
                                          "visit_table": str(OUTPUT / "visit_measurements.parquet")})


if __name__ == "__main__":
    main()
