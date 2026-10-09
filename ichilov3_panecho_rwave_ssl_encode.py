"""Encode selected visits with one fold's adapted PanEcho checkpoint."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from dicom_prediction_encode import load_encoder
from dicom_prediction_video import normalize
from ichilov3_train_fusion import atomic, hashfile


ROOT = Path(r"D:\DS\ichilov3_panecho_rwave_ssl_20261005")
SOURCE = Path(r"D:\DS\ichilov3_rwave_first_cycle_panecho_20261005")
ART = Path(r"D:\us\output\dicom_prediction")
VIEWS = ("A2C", "A3C", "A4C")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()
    name = f"r{args.repeat}_f{args.fold}"
    adapter = ROOT / "training" / name / "encoder_adapter.pt"
    if not adapter.exists():
        raise FileNotFoundError(adapter)
    out = ROOT / "evaluation_embeddings" / name
    out.mkdir(parents=True, exist_ok=True)
    finished = out / "run_complete.json"
    if finished.exists():
        print("Already complete", flush=True)
        return
    manifest = json.loads((SOURCE / "panecho_manifest.json").read_text(encoding="utf-8"))
    selected = [r for r in manifest if r.get("rwave_resampled") or r.get("first_cycle_only")]
    if len(selected) != 919:
        raise ValueError("Changed eligible cine count")
    model = load_encoder("panecho", ART, "cuda")
    model.eval()
    saved = torch.load(adapter, map_location="cpu", weights_only=False)
    if saved["protocol"]["repeat"] != args.repeat or saved["protocol"]["fold"] != args.fold:
        raise ValueError("Wrong fold adapter")
    model.encoder.model.features[-1].load_state_dict(saved["last_convnext"])
    model.transformer.load_state_dict(saved["transformer"])
    del saved
    torch.set_num_threads(4)
    started = time.time()
    clips = out / "clips"
    clips.mkdir(exist_ok=True)
    for i, record in enumerate(selected, 1):
        ident = record["file_id"]
        dest = clips / f"{ident}.npz"
        if not dest.exists():
            with np.load(ROOT / "clips" / f"{ident}.npz") as data:
                frames = data["frames"]
            x = torch.from_numpy(frames.copy()).permute(3, 0, 1, 2).unsqueeze(0).float().cuda()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                vector = model(normalize(x, "panecho")).float().cpu().numpy()[0]
            if vector.shape != (768,) or not np.isfinite(vector).all():
                raise ValueError(f"Invalid adapted vector: {ident}")
            tmp = dest.with_suffix(".partial.npz")
            np.savez_compressed(tmp, view_embedding=vector.astype(np.float32))
            os.replace(tmp, dest)
        if i % 100 == 0 or i == len(selected):
            atomic(out / "status.json", {"stage": "encoding", "clips": i,
                                        "total": len(selected), "elapsed_seconds": time.time()-started})
            print(f"Adapted PanEcho {name}: {i}/{len(selected)}", flush=True)
    with np.load(SOURCE / "panecho_visits.npz") as data:
        visits = {k: data[k].copy() for k in data.files}
    lookup = {(str(p), str(d)): i for i, (p, d) in enumerate(zip(
        visits["patient_ids"], visits["visit_dates"]))}
    view_index = {v: i for i, v in enumerate(VIEWS)}
    for record in selected:
        with np.load(clips / f"{record['file_id']}.npz") as data:
            visits["embeddings"][lookup[(record["patient"], record["visit_date"])],
                                  view_index[record["view"]]] = data["view_embedding"]
    if len(visits["patient_ids"]) != 411 or not np.isfinite(visits["embeddings"]).all():
        raise ValueError("Invalid visit embedding array")
    target = out / "visit_embeddings.npz"
    temp = out / "visit_embeddings.partial.npz"
    np.savez_compressed(temp, **visits)
    os.replace(temp, target)
    atomic(finished, {"visits": 411, "adapted_rwave_cines": 919,
                      "frozen_fallback_cines": 314, "adapter_sha256": hashfile(adapter),
                      "elapsed_seconds": time.time()-started})


if __name__ == "__main__":
    main()
