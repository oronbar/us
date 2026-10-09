"""Fold-isolated, label-free PanEcho adaptation on first-cycle R-wave cines."""

import argparse
import json
import os
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

import ichilov_ge_ssl_train as ssl
from dicom_prediction_encode import load_encoder
from dicom_prediction_video import normalize
from ichilov3_train_fusion import hashfile


ROOT = Path(r"D:\DS\ichilov3_panecho_rwave_ssl_20261005")
FOLDS = Path(r"D:\DS\ichilov3_nonphilips_future_20260930\aligned\patient_folds.parquet")
ART = Path(r"D:\us\output\dicom_prediction")
VIEWS = ("A2C", "A3C", "A4C")


def fold_inventory(repeat, fold):
    assignments = pd.read_parquet(FOLDS)
    held = set(assignments.loc[assignments.repeat.eq(repeat) &
                               assignments.fold.eq(fold), "patient_id"].astype(str))
    if not held:
        raise ValueError("Unknown fold")
    by_view = {v: defaultdict(list) for v in VIEWS}
    with (ROOT / "training_manifest.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if str(row["patient_id"]) not in held:
                by_view[row["view"]][str(row["patient_id"])].append(row["cache_id"])
    n = sum(len(clips) for view in by_view.values() for clips in view.values())
    return by_view, held, n


def load_batch(schedule_row):
    patients, windows = [], []
    for patient, cache_id in schedule_row:
        patients.append(patient)
        with np.load(ROOT / "clips" / f"{cache_id}.npz") as data:
            window = data["frames"]
        if window.shape != (16, 224, 224, 3):
            raise ValueError(f"Bad cache shape: {cache_id}")
        windows.append(window)
    if len(set(patients)) != len(patients):
        raise ValueError("Repeated patient in contrastive batch")
    return patients, np.stack(windows)


def encode(model, x):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model(normalize(x, "panecho")).float()


def save(path, model, projector, order_head, optimizer, step, rng, queue, queue_patients, protocol):
    ssl.atomic_torch(path, {
        "last_convnext": model.encoder.model.features[-1].state_dict(),
        "transformer": model.transformer.state_dict(),
        "projector": projector.state_dict(), "order_head": order_head.state_dict(),
        "optimizer": optimizer.state_dict(), "step": step,
        "numpy_rng_state": rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(),
        "queue": queue.detach().cpu(), "queue_patients": queue_patients,
        "protocol": protocol})


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if not (ROOT / "cache_complete.json").exists():
        raise RuntimeError("First-cycle cache is incomplete")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA GPU required")
    torch.set_num_threads(4)
    seed = 20261005 + args.repeat * 10 + args.fold
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    by_view, held, train_cines = fold_inventory(args.repeat, args.fold)
    out = ROOT / "training" / f"r{args.repeat}_f{args.fold}"
    out.mkdir(parents=True, exist_ok=True)
    schedule_path = out / "schedule.json"
    if schedule_path.exists():
        schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    else:
        schedule = []
        for epoch in range(args.epochs):
            schedule.extend(ssl.full_coverage_schedule(by_view, "all", rng, args.batch_size))
        ssl.atomic_json(schedule_path, schedule)
    protocol = {"encoder": "PanEcho ConvNeXt-Tiny + 4-layer temporal Transformer",
                "checkpoint_sha256": hashfile(ART / "weights" / "panecho.pt"),
                "source_manifest_sha256": hashfile(ROOT / "training_manifest.jsonl"),
                "sampling": "first-to-second R-wave, 16 frames, one cycle",
                "trainable": "last ConvNeXt feature stage and temporal Transformer",
                "objective": "patient-aware cine contrast + 0.2 temporal chunk order",
                "labels_used": False, "epochs": args.epochs, "batch_size": args.batch_size,
                "repeat": args.repeat, "fold": args.fold,
                "train_cines": train_cines, "heldout_patients": len(held),
                "total_steps": len(schedule)}
    old_protocol = out / "protocol.json"
    if old_protocol.exists() and json.loads(old_protocol.read_text()) != protocol:
        raise ValueError("Protocol mismatch on resume")
    ssl.atomic_json(old_protocol, protocol)
    if (out / "run_complete.json").exists():
        print("Already complete", flush=True)
        return
    model = load_encoder("panecho", ART, "cuda")
    model.eval().requires_grad_(False)
    model.encoder.model.features[-1].requires_grad_(True)
    model.transformer.requires_grad_(True)
    projector = nn.Sequential(nn.Linear(768, 256), nn.GELU(), nn.Linear(256, 128)).cuda()
    order_head = nn.Linear(768, 2).cuda()
    trainable = ([p for p in model.parameters() if p.requires_grad] +
                 list(projector.parameters()) + list(order_head.parameters()))
    optimizer = torch.optim.AdamW(trainable, lr=1e-5, weight_decay=.01)
    checkpoint = out / "latest.pt"
    queue = torch.empty((0, 128), device="cuda")
    queue_patients = []
    start = 0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["protocol"] != protocol:
            raise ValueError("Checkpoint protocol mismatch")
        model.encoder.model.features[-1].load_state_dict(saved["last_convnext"])
        model.transformer.load_state_dict(saved["transformer"])
        projector.load_state_dict(saved["projector"])
        order_head.load_state_dict(saved["order_head"])
        optimizer.load_state_dict(saved["optimizer"])
        rng.bit_generator.state = saved["numpy_rng_state"]
        torch.set_rng_state(saved["torch_rng_state"])
        torch.cuda.set_rng_state(saved["cuda_rng_state"])
        queue = saved["queue"].cuda()
        queue_patients = saved["queue_patients"]
        start = saved["step"]
    started = time.time()
    for step in range(start + 1, len(schedule) + 1):
        patients, windows = load_batch(schedule[step - 1])
        x = torch.from_numpy(windows).permute(0, 4, 1, 2, 3).float().cuda()
        first, second = ssl.perturb(x), ssl.perturb(x)
        wrong = ssl.perturb(ssl.shuffled_chunks(x, rng))
        optimizer.zero_grad(set_to_none=True)
        a, b, bad = encode(model, torch.cat((first, second, wrong))).chunk(3)
        contrast, detached = ssl.contrastive(projector(a), projector(b), patients,
                                              queue, queue_patients)
        logits = torch.cat((order_head(a), order_head(bad)))
        targets = torch.cat((torch.zeros(len(a), dtype=torch.long, device="cuda"),
                             torch.ones(len(bad), dtype=torch.long, device="cuda")))
        order = F.cross_entropy(logits, targets)
        loss = contrast + .2 * order
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite SSL loss at step {step}")
        loss.backward()
        nn.utils.clip_grad_norm_(trainable, 1.)
        optimizer.step()
        queue = torch.cat((queue, detached), dim=0)[-512:]
        queue_patients = (queue_patients + patients + patients)[-512:]
        if step % 25 == 0 or step == len(schedule):
            save(checkpoint, model, projector, order_head, optimizer, step, rng,
                 queue, queue_patients, protocol)
            ssl.atomic_json(out / "status.json", {"stage": "training", "step": step,
                             "total": len(schedule), "loss": float(loss.item()),
                             "contrast": float(contrast.item()), "order": float(order.item()),
                             "elapsed_seconds": time.time() - started, "pid": os.getpid()})
            print(f"PanEcho SSL r{args.repeat} f{args.fold} "
                  f"{step}/{len(schedule)} loss={loss.item():.3f} "
                  f"elapsed={time.time()-started:.0f}s", flush=True)
    adapter = out / "encoder_adapter.pt"
    ssl.atomic_torch(adapter, {"last_convnext": model.encoder.model.features[-1].state_dict(),
                               "transformer": model.transformer.state_dict(),
                               "protocol": protocol})
    ssl.atomic_json(out / "run_complete.json", {"adapter": str(adapter),
                    "steps": len(schedule), "elapsed_seconds": time.time() - started})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        ROOT.mkdir(parents=True, exist_ok=True)
        (ROOT / "training_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
