"""Resumable read-only EchoFM encoding of the selected Ichilov3 cine crops."""
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
MODEL_REPO = Path(r"D:\DS\ichilov_models\EchoFM")
CHECKPOINT = Path(r"D:\DS\ichilov_models\EchoFM_weights\echofm_vitl.pth")
OUTPUT = Path(r"D:\DS\ichilov3_echofm_20260930")
VIEWS = ("A2C", "A3C", "A4C")


def atomic_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temp.replace(path)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sample_indices(record: dict) -> tuple[np.ndarray, dict]:
    n = int(record["frames"])
    times, time_source = timestamps(record, np.arange(n))
    if n < 2:
        raise ValueError(f"Insufficient cine frames: {record['file_id']}")
    valid_timing = np.isfinite(times).all() and np.all(np.diff(times) > 0)
    hr = record.get("heart_rate")
    if valid_timing and hr is not None and np.isfinite(hr) and 30 <= hr <= 220:
        duration = float(times[-1] - times[0])
        period = 60.0 / float(hr)
        if duration >= period:
            starts = np.unique(np.linspace(float(times[0]), float(times[-1] - period), 4))
            targets = starts[:, None] + np.linspace(0, period, 32)[None, :]
            right = np.clip(np.searchsorted(times, targets), 0, n - 1)
            left = np.maximum(right - 1, 0)
            indices = np.where(np.abs(times[left] - targets) <= np.abs(times[right] - targets), left, right)
            return indices.astype(np.int32), {
                "policy": "32 native frames nearest four estimated heartbeat windows",
                "timing_source": time_source, "windows": len(indices),
                "estimated_period_seconds": period, "source_duration_seconds": duration,
            }
    # The checkpoint needs 32 frames. Nearest-frame resampling covers only the
    # observed cine; it never wraps or invents an extra cardiac cycle.
    indices = np.rint(np.linspace(0, n - 1, 32)).astype(np.int32)[None, :]
    return indices, {
        "policy": "32 nearest frames across observed short/unreliable-timing cine",
        "timing_source": time_source, "windows": 1,
    }


def load_model(device: str):
    sys.path.insert(0, str(MODEL_REPO))
    from EchoFM import models_mae

    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    args = {key: ckpt["model_args"][key] for key in
            ("num_frames", "t_patch_size", "pred_t_dim", "sep_pos_embed", "cls_embed", "norm_pix_loss")}
    if int(args["num_frames"]) != 32:
        raise ValueError(f"Unexpected EchoFM frame count: {args['num_frames']}")
    model = models_mae.mae_vit_large_patch16(**args)
    result = model.load_state_dict(ckpt["model"], strict=False)
    if result.unexpected_keys:
        raise ValueError(f"Unexpected checkpoint keys: {result.unexpected_keys[:8]}")
    model.eval().to(device)
    return model, args, len(result.missing_keys)


def encode_one(record: dict, model, device: str) -> dict:
    indices, sampling = sample_indices(record)
    with np.load(record["crop_output"]) as f:
        frames = f["frames"]
        if frames.dtype != np.uint8 or len(frames) != record["frames"]:
            raise ValueError(f"Changed crop: {record['file_id']}")
    embeddings = []
    for sequence in indices:
        pixels = np.stack([cv2.resize(frames[int(i)], (224, 224), interpolation=cv2.INTER_AREA)
                           for i in sequence])
        if pixels.ndim == 3:
            pixels = np.repeat(pixels[..., None], 3, axis=-1)
        tensor = torch.from_numpy(pixels.copy()).permute(3, 0, 1, 2).unsqueeze(0)
        tensor = tensor.to(device=device, dtype=torch.float32).div_(255.0)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16,
                                                    enabled=device == "cuda"):
            latent, _, _ = model.forward_encoder(tensor, mask_ratio=0.0)
            vector = latent.mean(dim=1).float().cpu().numpy()[0]
        if vector.shape != (1024,) or not np.isfinite(vector).all():
            raise ValueError(f"Bad embedding: {record['file_id']}")
        embeddings.append(vector)
    return {"windows": np.asarray(embeddings, np.float32), "indices": indices,
            "sampling": sampling}


def export_visits(records: list[dict]) -> Path:
    groups: dict[tuple[str, str], dict[str, dict]] = {}
    for record in records:
        groups.setdefault((record["patient"], record["visit_date"]), {})[record["view"]] = record
    keys = sorted(groups)
    embeddings = np.empty((len(keys), 3, 1024), dtype=np.float32)
    for row, key in enumerate(keys):
        if set(groups[key]) != set(VIEWS):
            raise ValueError(f"Incomplete view triplet: {key}")
        for col, view in enumerate(VIEWS):
            path = OUTPUT / "clips" / f"{groups[key][view]['file_id']}.npz"
            with np.load(path) as f:
                embeddings[row, col] = f["view_embedding"]
    if not np.isfinite(embeddings).all():
        raise ValueError("Nonfinite visit embeddings")
    result = OUTPUT / "echofm_visits.npz"
    temporary = OUTPUT / "echofm_visits.partial.npz"
    np.savez_compressed(temporary, patient_ids=np.array([p for p, _ in keys]),
                        visit_dates=np.array([d for _, d in keys]), views=np.array(VIEWS),
                        embeddings=embeddings)
    temporary.replace(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "clips").mkdir(exist_ok=True)
    records = json.loads(SOURCE.read_text(encoding="utf-8"))
    if len(records) != 1233:
        raise ValueError(f"Unexpected selected cine count: {len(records)}")
    if any(not Path(r["crop_output"]).is_file() for r in records):
        raise FileNotFoundError("A selected cine crop is unavailable")
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    protocol = {"source_manifest": str(SOURCE), "source_sha256": digest(SOURCE),
                "checkpoint": str(CHECKPOINT), "checkpoint_sha256": digest(CHECKPOINT),
                "model_repo": str(MODEL_REPO), "model": "EchoFM ViT-L", "frames": 32,
                "resolution": 224, "native_color": True, "views": list(VIEWS),
                "sampling": "HR-estimated 32-frame heartbeat, up to four windows; observed cine fallback",
                "embedding": "mean of encoder tokens per window, then mean of windows"}
    protocol_path = OUTPUT / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Existing EchoFM protocol differs; use a new output directory")
    atomic_json(protocol_path, protocol)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, model_args, missing = load_model(device)
    if missing:
        print(f"Checkpoint missing keys: {missing}", flush=True)
    selected = records[:1] if args.smoke else records
    started = time.time()
    completed = 0
    for record in selected:
        file_id = record["file_id"]
        output = OUTPUT / "clips" / f"{file_id}.npz"
        metadata = OUTPUT / "clips" / f"{file_id}.json"
        if metadata.exists() and output.exists():
            saved = json.loads(metadata.read_text())
            if saved.get("source_sha256") != record["source_sha256"]:
                raise ValueError(f"Source mismatch on resume: {file_id}")
            completed += 1
            continue
        result = encode_one(record, model, device)
        temp = OUTPUT / "clips" / f"{file_id}.partial.npz"
        np.savez_compressed(temp, window_embeddings=result["windows"],
                            view_embedding=result["windows"].mean(axis=0),
                            source_frame_indices=result["indices"])
        temp.replace(output)
        atomic_json(metadata, {"file_id": file_id, "patient": record["patient"],
                               "visit_date": record["visit_date"], "view": record["view"],
                               "manufacturer": record["manufacturer"],
                               "source_sha256": record["source_sha256"],
                               "embedding_path": str(output), **result["sampling"]})
        completed += 1
        if completed % 25 == 0 or args.smoke or completed == len(selected):
            atomic_json(OUTPUT / "status.json", {"stage": "encoding", "completed": completed,
                                                  "total": len(selected), "pid": os.getpid(),
                                                  "elapsed_seconds": time.time() - started})
            print(f"EchoFM {completed}/{len(selected)} ({time.time()-started:.0f}s)", flush=True)
    if not args.smoke:
        result_path = export_visits(records)
        atomic_json(OUTPUT / "run_complete.json", {"clips": completed,
                    "visits": len(records) // 3, "visit_embeddings": str(result_path),
                    "elapsed_seconds": time.time() - started})
        atomic_json(OUTPUT / "status.json", {"stage": "complete", "completed": completed,
                    "total": len(records), "elapsed_seconds": time.time() - started})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        atomic_json(OUTPUT / "status.json", {"stage": "error", "pid": os.getpid(),
                                           "error": traceback.format_exc()})
        raise
