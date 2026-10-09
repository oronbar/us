"""Resumable frozen EchoJEPA encoding of the selected Ichilov3 crops."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch

from ichilov3_encode_selected import timestamps

SOURCE = Path(r"D:\DS\ichilov3_stage2_padded_20260926\full_manifest.json")
REPO = Path(r"D:\DS\ichilov_models\EchoJEPA")
CHECKPOINT = Path(r"D:\DS\ichilov_models\EchoJEPA_weights\vitl-vmix22m-pt220-c55.pt")
OUTPUT = Path(r"D:\DS\ichilov3_echojepa_20261004")
VIEWS = ("A2C", "A3C", "A4C")
FRAMES = 16
SIDE = 224
DIM = 1024


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def sample_indices(record: dict) -> tuple[np.ndarray, dict]:
    n = int(record["frames"])
    times, timing_source = timestamps(record, np.arange(n))
    hr = record.get("heart_rate")
    usable = n >= 2 and np.isfinite(times).all() and np.all(np.diff(times) > 0)
    if usable and hr is not None and np.isfinite(hr) and 30 <= hr <= 220:
        period = 60.0 / float(hr)
        duration = float(times[-1] - times[0])
        if duration >= period:
            starts = np.unique(np.linspace(float(times[0]), float(times[-1] - period), 4))
            target = starts[:, None] + np.linspace(0, period, FRAMES)[None, :]
            right = np.clip(np.searchsorted(times, target), 0, n - 1)
            left = np.maximum(right - 1, 0)
            idx = np.where(np.abs(times[left] - target) <= np.abs(times[right] - target), left, right)
            return idx.astype(np.int32), {"policy": "estimated heartbeat, 16 nearest native frames",
                                         "timing_source": timing_source, "windows": len(idx),
                                         "estimated_period_seconds": period}
    idx = np.rint(np.linspace(0, n - 1, FRAMES)).astype(np.int32)[None, :]
    return idx, {"policy": "16 frames across observed cine", "timing_source": timing_source,
                 "windows": 1}


def load_model(device: str):
    sys.path.insert(0, str(REPO))
    from src.models.vision_transformer import vit_large

    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict):
        raise ValueError("Unexpected EchoJEPA checkpoint structure")
    state = ckpt.get("target_encoder", ckpt.get("encoder"))
    if state is None:
        raise ValueError(f"No encoder weights in checkpoint; keys={list(ckpt)[:15]}")
    state = {k.removeprefix("module.").removeprefix("backbone."): v for k, v in state.items()}
    model = vit_large(img_size=SIDE, num_frames=FRAMES, tubelet_size=2,
                      use_rope=True, uniform_power=True, use_sdpa=True)
    model_keys = set(model.state_dict())
    ckpt_keys = set(state)
    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys
    if missing or unexpected:
        raise ValueError(f"Checkpoint architecture mismatch: missing={list(sorted(missing))[:8]}, "
                         f"unexpected={list(sorted(unexpected))[:8]}, counts=({len(missing)},{len(unexpected)})")
    model.load_state_dict(state, strict=True)
    del ckpt, state
    return model.eval().requires_grad_(False).to(device)


def encode_one(record: dict, model, device: str):
    indices, info = sample_indices(record)
    with np.load(record["crop_output"]) as cached:
        frames = cached["frames"]
        if frames.dtype != np.uint8 or len(frames) != record["frames"]:
            raise ValueError(f"Invalid crop cache: {record['file_id']}")
    mean = torch.tensor((.485, .456, .406), device=device).view(1, 3, 1, 1, 1)
    std = torch.tensor((.229, .224, .225), device=device).view(1, 3, 1, 1, 1)
    vectors = []
    for sequence in indices:
        pixels = np.stack([cv2.resize(frames[int(i)], (SIDE, SIDE), interpolation=cv2.INTER_AREA)
                           for i in sequence])
        if pixels.ndim == 3:
            pixels = np.repeat(pixels[..., None], 3, axis=-1)
        x = torch.from_numpy(pixels.copy()).permute(3, 0, 1, 2).unsqueeze(0)
        x = x.to(device=device, dtype=torch.float32).div_(255.0)
        x = (x - mean) / std
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16,
                                                    enabled=device == "cuda"):
            tokens = model(x)
            vector = tokens.mean(dim=1).float().cpu().numpy()[0]
        if vector.shape != (DIM,) or not np.isfinite(vector).all():
            raise ValueError(f"Invalid EchoJEPA embedding: {record['file_id']}")
        vectors.append(vector)
    return np.asarray(vectors, np.float32), indices, info


def export_visits(records: list[dict]) -> Path:
    grouped: dict[tuple[str, str], dict[str, dict]] = {}
    for r in records:
        grouped.setdefault((r["patient"], r["visit_date"]), {})[r["view"]] = r
    keys = sorted(grouped)
    embeddings = np.empty((len(keys), 3, DIM), np.float32)
    for row, key in enumerate(keys):
        if set(grouped[key]) != set(VIEWS):
            raise ValueError(f"Incomplete visit: {key}")
        for col, view in enumerate(VIEWS):
            file_id = grouped[key][view]["file_id"]
            with np.load(OUTPUT / "clips" / f"{file_id}.npz") as f:
                embeddings[row, col] = f["view_embedding"]
    if not np.isfinite(embeddings).all():
        raise ValueError("Nonfinite visit embeddings")
    target = OUTPUT / "echojepa_visits.npz"
    temp = OUTPUT / "echojepa_visits.partial.npz"
    np.savez_compressed(temp, patient_ids=np.array([p for p, _ in keys]),
                        visit_dates=np.array([d for _, d in keys]),
                        views=np.array(VIEWS), embeddings=embeddings)
    temp.replace(target)
    return target


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "clips").mkdir(exist_ok=True)
    records = json.loads(SOURCE.read_text(encoding="utf-8"))
    if len(records) != 1233 or any(not Path(r["crop_output"]).is_file() for r in records):
        raise ValueError("Selected-cine manifest incomplete")
    protocol = {"source_manifest": str(SOURCE), "source_sha256": sha256(SOURCE),
                "checkpoint": str(CHECKPOINT), "checkpoint_sha256": sha256(CHECKPOINT),
                "model_repo": str(REPO), "model": "EchoJEPA V-JEPA2 ViT-L VideoMix22M MIMIC pt220 c55",
                "frames": FRAMES, "resolution": SIDE, "color": "native RGB",
                "normalization": "ImageNet mean/std per official evaluation transform",
                "sampling": "16 frames over HR-estimated heartbeat, up to four starts; short-cine fallback",
                "pooling": "mean of final tokens per window, then mean across windows",
                "views": list(VIEWS)}
    protocol_path = OUTPUT / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
        raise ValueError("Existing output has a different encoding protocol")
    atomic_json(protocol_path, protocol)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)
    selected = records[:1] if args.smoke else records
    start = time.monotonic()
    for number, record in enumerate(selected, 1):
        file_id = record["file_id"]
        target = OUTPUT / "clips" / f"{file_id}.npz"
        meta_path = OUTPUT / "clips" / f"{file_id}.json"
        if target.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta["source_sha256"] != record["source_sha256"] or meta["checkpoint_sha256"] != protocol["checkpoint_sha256"]:
                raise ValueError(f"Resume identity mismatch for {file_id}")
            continue
        vectors, indices, info = encode_one(record, model, device)
        temp = OUTPUT / "clips" / f"{file_id}.partial.npz"
        np.savez_compressed(temp, window_embeddings=vectors,
                            view_embedding=vectors.mean(axis=0), source_frame_indices=indices)
        temp.replace(target)
        atomic_json(meta_path, {"file_id": file_id, "patient": record["patient"],
                                "visit_date": record["visit_date"], "view": record["view"],
                                "manufacturer": record["manufacturer"],
                                "source_sha256": record["source_sha256"],
                                "checkpoint_sha256": protocol["checkpoint_sha256"], **info})
        if number % 25 == 0 or number == len(selected) or args.smoke:
            elapsed = time.monotonic() - start
            atomic_json(OUTPUT / "status.json", {"stage": "encoding", "completed": number,
                                                 "total": len(selected), "elapsed_seconds": elapsed,
                                                 "pid": os.getpid()})
            print(f"EchoJEPA {number}/{len(selected)} ({elapsed:.0f}s)", flush=True)
    if not args.smoke:
        visit_path = export_visits(records)
        atomic_json(OUTPUT / "run_complete.json", {"clips": len(records), "visits": len(records)//3,
                                                   "visit_embeddings": str(visit_path),
                                                   "elapsed_seconds": time.monotonic()-start})
        atomic_json(OUTPUT / "status.json", {"stage": "complete", "completed": len(records),
                                             "total": len(records), "elapsed_seconds": time.monotonic()-start})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        atomic_json(OUTPUT / "status.json", {"stage": "error", "pid": os.getpid(),
                                             "error": traceback.format_exc()})
        raise
