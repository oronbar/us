"""Cache frozen EchoPrime prefix tokens for identical Ichilov3 cine windows.

This makes 60 fold-specific adapter evaluations cheap and verifies that the
cached-prefix frozen output reproduces the earlier full encoder embeddings.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch

from dicom_prediction_encode import load_encoder
from dicom_prediction_video import normalize

ROOT = Path(r"D:\DS\ichilov_ge_ssl_prefix_20260930")
CROPS = Path(r"D:\DS\ichilov3_stage2_padded_20260926\full_manifest.json")
TEMPORAL = Path(r"D:\DS\ichilov3_temporal_trial_20260927\echoprime_manifest.json")
ART = Path(r"D:\us\output\dicom_prediction")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    records = json.loads(CROPS.read_text(encoding="utf-8"))
    if args.smoke:
        records = records[:1]
    temporal = {r["file_id"]: r for r in json.loads(TEMPORAL.read_text(encoding="utf-8"))}
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    model = load_encoder("echoprime", ART, "cuda")
    model.eval()
    started = time.time()
    for n, row in enumerate(records, 1):
        file_id = row["file_id"]
        target = ROOT / f"{file_id}.npz"
        if target.exists():
            continue
        prior = temporal[file_id]
        with np.load(prior["embedding_path"]) as source:
            indices = source["source_frame_indices"].astype(np.int32)
            reference = source["window_embeddings"].astype(np.float32)
        with np.load(row["crop_output"]) as source:
            frames = source["frames"]
        tokens = []
        outputs = []
        thw_out = None
        for window in indices:
            pixels = np.stack([cv2.resize(frames[int(i)], (224, 224),
                                          interpolation=cv2.INTER_AREA) for i in window])
            if pixels.ndim == 3:
                pixels = np.repeat(pixels[..., None], 3, axis=-1)
            tensor = torch.from_numpy(pixels.copy()).permute(3, 0, 1, 2).unsqueeze(0)
            tensor = normalize(tensor.float().cuda(), "echoprime")
            with torch.inference_mode():
                x = model.conv_proj(tensor).flatten(2).transpose(1, 2)
                x = model.pos_encoding(x)
                thw = (model.pos_encoding.temporal_size,) + model.pos_encoding.spatial_size
                for block in model.blocks[:-1]:
                    x, thw = block(x, thw)
                latent, _ = model.blocks[-1](x, thw)
                output = model.head(model.norm(latent)[:, 0]).float().cpu().numpy()[0]
            tokens.append(x.float().cpu().numpy()[0])
            outputs.append(output)
            thw_out = thw
        tokens = np.stack(tokens).astype(np.float32)
        outputs = np.stack(outputs).astype(np.float32)
        error = float(np.max(np.abs(outputs - reference)))
        if error > 2e-3:
            raise ValueError(f"Frozen EchoPrime does not reproduce prior embedding: {file_id}, {error}")
        temp = ROOT / f"{file_id}.partial.npz"
        np.savez_compressed(temp, tokens=tokens, thw=np.array(thw_out, np.int32),
                            frame_indices=indices, frozen_embeddings=outputs)
        temp.replace(target)
        if n % 25 == 0 or n == len(records):
            print(f"EchoPrime prefix {n}/{len(records)} ({time.time()-started:.0f}s)", flush=True)
    if not args.smoke:
        (ROOT / "run_complete.json").write_text(json.dumps({
            "clips": len(records), "source": str(CROPS), "sampling": str(TEMPORAL),
            "max_reproduction_error_allowed": 2e-3, "elapsed_seconds": time.time()-started},
            indent=2), encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        ROOT.mkdir(parents=True, exist_ok=True)
        (ROOT / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
