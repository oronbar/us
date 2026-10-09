"""Apply a fold's adapted EchoPrime final block to cached cine tokens."""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from dicom_prediction_encode import load_encoder

ROOT = Path(r"D:\DS\ichilov_ge_ssl_20260930")
PREFIX = Path(r"D:\DS\ichilov_ge_ssl_prefix_20260930")
CROPS = Path(r"D:\DS\ichilov3_stage2_padded_20260926\full_manifest.json")
ART = Path(r"D:\us\output\dicom_prediction")
VIEWS = ("A2C", "A3C", "A4C")


def load_adapter(model, path: Path):
    item = torch.load(path, map_location="cpu", weights_only=False)
    model.blocks[-1].load_state_dict(item["last_block"])
    model.norm.load_state_dict(item["norm"])
    model.head.load_state_dict(item["head"])
    return item["protocol"]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", choices=("all", "specific"), required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    out = ROOT / "evaluation_embeddings" / args.config / f"r{args.repeat}_f{args.fold}"
    out.mkdir(parents=True, exist_ok=True)
    records = json.loads(CROPS.read_text(encoding="utf-8"))
    adapters = {v: ROOT / ("all" if args.config == "all" else v) /
                f"r{args.repeat}_f{args.fold}" / "encoder_adapter.pt" for v in VIEWS}
    if not all(p.exists() for p in adapters.values()):
        raise FileNotFoundError("Required fold adapter is incomplete")
    torch.set_num_threads(4)
    model = load_encoder("echoprime", ART, "cuda")
    model.eval()
    started = time.time()
    selected = records[:1] if args.smoke else records
    by_view = {v: [r for r in selected if r["view"] == v] for v in VIEWS}
    for view in VIEWS:
        if not by_view[view]:
            continue
        protocol = load_adapter(model, adapters[view])
        if protocol["repeat"] != args.repeat or protocol["fold"] != args.fold:
            raise ValueError("Adapter does not match outer fold")
        for index, row in enumerate(by_view[view], 1):
            dest = out / f"{row['file_id']}.npz"
            if dest.exists() and not args.smoke:
                continue
            with np.load(PREFIX / f"{row['file_id']}.npz") as source:
                tokens = source["tokens"].astype(np.float32)
                thw = tuple(source["thw"].astype(int))
            embeddings = []
            for window in tokens:
                x = torch.from_numpy(window).unsqueeze(0).cuda()
                with torch.inference_mode():
                    latent, _ = model.blocks[-1](x, thw)
                    vector = model.head(model.norm(latent)[:, 0]).float().cpu().numpy()[0]
                embeddings.append(vector)
            vector = np.asarray(embeddings, np.float32).mean(axis=0)
            if vector.shape != (512,) or not np.isfinite(vector).all():
                raise ValueError(f"Invalid adapted embedding: {row['file_id']}")
            if not args.smoke:
                temp = out / f"{row['file_id']}.partial.npz"
                np.savez_compressed(temp, view_embedding=vector)
                temp.replace(dest)
            if index % 100 == 0 or args.smoke:
                print(f"Adapter {args.config} r{args.repeat} f{args.fold} {view}: "
                      f"{index}/{len(by_view[view])} ({time.time()-started:.0f}s)", flush=True)
    if args.smoke:
        return
    groups = {}
    for row in records:
        groups.setdefault((row["patient"], row["visit_date"]), {})[row["view"]] = row["file_id"]
    keys = sorted(groups)
    values = np.empty((len(keys), 3, 512), np.float32)
    for i, key in enumerate(keys):
        if set(groups[key]) != set(VIEWS):
            raise ValueError(f"Incomplete selected view triplet: {key}")
        for j, view in enumerate(VIEWS):
            with np.load(out / f"{groups[key][view]}.npz") as data:
                values[i, j] = data["view_embedding"]
    result = out / "visit_embeddings.npz"
    temp = out / "visit_embeddings.partial.npz"
    np.savez_compressed(temp, patient_ids=np.array([x[0] for x in keys]),
                        visit_dates=np.array([x[1] for x in keys]),
                        views=np.array(VIEWS), embeddings=values)
    temp.replace(result)
    (out / "run_complete.json").write_text(json.dumps({
        "visits": len(keys), "clips": len(records), "embedding_path": str(result),
        "elapsed_seconds": time.time()-started}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        ROOT.mkdir(parents=True, exist_ok=True)
        (ROOT / "encode_eval_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
