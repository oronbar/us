"""Fold-isolated self-supervised adaptation of EchoPrime on GE apical cines.

One process trains one fold and one configuration. The pretrained encoder's
final MViT block, norm, and 512-dimensional head are updated with cine-pair
contrast and temporal-order objectives. No GLS or deterioration labels enter
training. Earlier MViT blocks stay frozen to fit the 8 GB GPU.
"""
from __future__ import annotations

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

from dicom_prediction_encode import load_encoder
from dicom_prediction_video import normalize

ROOT = Path(r"D:\DS\ichilov_ge_ssl_20260930")
CACHE = Path(r"D:\DS\ichilov_ge_ssl_clips_20260930")
ART = Path(r"D:\us\output\dicom_prediction")
FOLDS = Path(r"D:\DS\ichilov3_nonphilips_future_20260930\aligned\patient_folds.parquet")
VIEWS = ("A2C", "A3C", "A4C")
CHECKPOINT_INTERVAL = 25


def atomic_torch(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".partial.pt")
    torch.save(payload, temporary)
    replace_with_retry(temporary, path)


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    replace_with_retry(temporary, path)


def replace_with_retry(temporary: Path, target: Path) -> None:
    """Allow brief Windows reader/antivirus locks without losing a checkpoint."""
    for attempt in range(20):
        try:
            temporary.replace(target)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(min(0.05 * (attempt + 1), 0.5))


def load_manifest(repeat: int, fold: int, view: str) -> tuple[dict, dict]:
    assignments = pd.read_parquet(FOLDS)
    held = set(assignments.loc[assignments.repeat.eq(repeat) &
                               assignments.fold.eq(fold), "patient_id"].astype(str))
    if not held:
        raise ValueError(f"Unknown outer fold: repeat={repeat} fold={fold}")
    by_view = {v: defaultdict(list) for v in VIEWS}
    seen = set()
    with (CACHE / "training_manifest.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            patient = row["patient_id"]
            if patient in held or row["view"] not in VIEWS:
                continue
            cache_id = row["cache_id"]
            if cache_id in seen:
                raise ValueError(f"Duplicate training cine: {cache_id}")
            seen.add(cache_id)
            by_view[row["view"]][patient].append(cache_id)
    needed = VIEWS if view == "all" else (view,)
    if any(len(by_view[v]) < 4 for v in needed):
        raise ValueError("Too few distinct GE patients for SSL batch")
    return by_view, {"heldout_patients": len(held),
                     "train_patients": len({p for v in needed for p in by_view[v]}),
                     "train_cines": len(seen), "view": view}


def choose_batch(by_view, view: str, rng: np.random.Generator, batch_size: int):
    records = []
    used = set()
    view_offset = int(rng.integers(0, 3)) if view == "all" else 0
    for i in range(batch_size):
        candidates = list(VIEWS) if view == "all" else [view]
        if view == "all":
            rotation = (view_offset + i) % 3
            candidates = candidates[rotation:] + candidates[:rotation]
        chosen_view = None
        chosen_patient = None
        for candidate in candidates:
            available = [p for p in by_view[candidate] if p not in used]
            if available:
                chosen_view = candidate
                chosen_patient = str(rng.choice(available))
                break
        if chosen_patient is None:
            raise ValueError("Insufficient distinct patients for an SSL batch")
        used.add(chosen_patient)
        cache_id = str(rng.choice(by_view[chosen_view][chosen_patient]))
        with np.load(CACHE / f"{cache_id}.npz") as data:
            windows = data["windows"].copy()
        if windows.shape != (2, 16, 224, 224, 3):
            raise ValueError(f"Bad cine cache shape: {cache_id}")
        records.append((chosen_patient, windows))
    return [r[0] for r in records], np.stack([r[1] for r in records])


def full_coverage_schedule(by_view, view: str, rng: np.random.Generator,
                           batch_size: int) -> list[list[tuple[str, str]]]:
    """One shuffled pass over every eligible cine, with distinct patients/batch."""
    needed = VIEWS if view == "all" else (view,)
    pool: dict[str, list[str]] = defaultdict(list)
    for selected_view in needed:
        for patient, clips in by_view[selected_view].items():
            pool[patient].extend(clips)
    for clips in pool.values():
        rng.shuffle(clips)
    originals = {file_id for clips in pool.values() for file_id in clips}
    schedule = []
    all_patients = list(pool)
    while any(pool.values()):
        available = [p for p, clips in pool.items() if clips]
        chosen = [str(p) for p in rng.choice(available,
                                           size=min(batch_size, len(available)), replace=False)]
        batch = [(patient, pool[patient].pop()) for patient in chosen]
        if len(batch) < batch_size:
            extras = [p for p in all_patients if p not in chosen]
            filler = rng.choice(extras, size=batch_size-len(batch), replace=False)
            for patient in filler:
                patient = str(patient)
                # Fill a late batch from the already eligible pool; this does
                # not substitute for visiting all original cines above.
                candidates = [clip for selected_view in needed
                              for clip in by_view[selected_view].get(patient, [])]
                batch.append((patient, str(rng.choice(candidates))))
        schedule.append(batch)
    if {file_id for batch in schedule for _, file_id in batch} != originals:
        raise ValueError("SSL schedule does not cover every eligible cine")
    return schedule


def scheduled_batch(batch: list[list[str]]) -> tuple[list[str], np.ndarray]:
    patients, windows = [], []
    for patient, cache_id in batch:
        patients.append(patient)
        with np.load(CACHE / f"{cache_id}.npz") as data:
            pixels = data["windows"].copy()
        if pixels.shape != (2, 16, 224, 224, 3):
            raise ValueError(f"Bad cine cache shape: {cache_id}")
        windows.append(pixels)
    if len(set(patients)) != len(patients):
        raise ValueError("SSL batch contains repeated patient")
    return patients, np.stack(windows)


def perturb(x: torch.Tensor) -> torch.Tensor:
    factor = torch.empty((len(x), 1, 1, 1, 1), device=x.device).uniform_(.88, 1.12)
    offset = torch.empty((len(x), 1, 1, 1, 1), device=x.device).uniform_(-5., 5.)
    return (x * factor + offset).clamp_(0, 255)


def shuffled_chunks(x: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    # Four intact four-frame segments are permuted. Static appearance is kept;
    # the auxiliary task requires the encoder to detect temporal continuity.
    pieces = []
    for item in x:
        perm = rng.permutation(4)
        if np.array_equal(perm, np.arange(4)):
            perm = np.array([2, 0, 3, 1])
        pieces.append(torch.cat([item[:, 4*j:4*(j+1)] for j in perm], dim=1))
    return torch.stack(pieces)


def encode(model, pixels: torch.Tensor) -> torch.Tensor:
    x = normalize(pixels, "echoprime")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        x = model.conv_proj(x).flatten(2).transpose(1, 2)
        x = model.pos_encoding(x)
        thw = (model.pos_encoding.temporal_size,) + model.pos_encoding.spatial_size
        for block in model.blocks[:-1]:
            x, thw = block(x, thw)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        x, thw = model.blocks[-1](x, thw)
        x = model.head(model.norm(x)[:, 0])
    return x.float()


def contrastive(z1, z2, patients, queue, queue_patients, temperature=.12):
    z = F.normalize(torch.cat((z1, z2)), dim=1)
    batch = len(patients)
    labels = torch.cat((torch.arange(batch, 2*batch, device=z.device),
                        torch.arange(batch, device=z.device)))
    similarity = z @ z.T / temperature
    similarity.fill_diagonal_(-1e4)
    if len(queue):
        other = z @ queue.T / temperature
        mask = torch.tensor([[q == p for q in queue_patients]
                             for p in patients + patients], device=z.device)
        other = other.masked_fill(mask, -1e4)
        similarity = torch.cat((similarity, other), dim=1)
    return F.cross_entropy(similarity, labels), z.detach()


def pack(model, projector, order_head, optimizer, step, rng, queue, queue_patients, protocol):
    return {"last_block": model.blocks[-1].state_dict(), "norm": model.norm.state_dict(),
            "head": model.head.state_dict(), "projector": projector.state_dict(),
            "order_head": order_head.state_dict(), "optimizer": optimizer.state_dict(),
            "step": step, "numpy_rng_state": rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(),
            "queue": queue.detach().cpu(), "queue_patients": queue_patients,
            "protocol": protocol}


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--view", choices=("all", *VIEWS), required=True)
    parser.add_argument("--steps", type=int, default=0,
                        help="0 means one complete pass over every eligible cine")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This protocol needs a BF16-capable CUDA GPU")
    out = ROOT / args.view / f"r{args.repeat}_f{args.fold}"
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(20260930 + args.repeat * 10 + args.fold)
    rng = np.random.default_rng(20260930 + args.repeat * 10 + args.fold)
    by_view, cohort = load_manifest(args.repeat, args.fold, args.view)
    schedule_path = out / "full_coverage_schedule.json"
    if args.steps == 0:
        if schedule_path.exists():
            schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
        else:
            schedule = full_coverage_schedule(by_view, args.view, rng, args.batch_size)
            atomic_json(schedule_path, schedule)
        total_steps = len(schedule)
    else:
        schedule = None
        total_steps = args.steps
    protocol = {"repeat": args.repeat, "fold": args.fold, "view": args.view,
                "steps": total_steps, "batch_size": args.batch_size,
                "coverage": "one complete pass over all eligible fold-training GE cines"
                if schedule is not None else "limited-step diagnostic",
                "loss": "patient-aware cine-pair InfoNCE + 0.2 temporal-chunk-order CE",
                "backbone": "EchoPrime MViT-v2-S pretrained; final block, norm, head trainable",
                "pretraining_labels": "none", **cohort}
    path = out / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Existing SSL run has a different protocol")
    atomic_json(path, protocol)
    if (out / "run_complete.json").exists():
        print("Already complete", flush=True)
        return
    model = load_encoder("echoprime", ART, "cuda")
    model.eval().requires_grad_(False)
    for module in (model.blocks[-1], model.norm, model.head):
        module.requires_grad_(True)
    projector = nn.Sequential(nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 128)).cuda()
    order_head = nn.Linear(512, 2).cuda()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad] +
                                  list(projector.parameters()) + list(order_head.parameters()),
                                  lr=1e-4, weight_decay=.01)
    checkpoint = out / "latest.pt"
    queue = torch.empty((0, 128), device="cuda")
    queue_patients = []
    start = 0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["protocol"] != protocol:
            raise ValueError("SSL checkpoint protocol differs")
        model.blocks[-1].load_state_dict(saved["last_block"])
        model.norm.load_state_dict(saved["norm"])
        model.head.load_state_dict(saved["head"])
        projector.load_state_dict(saved["projector"])
        order_head.load_state_dict(saved["order_head"])
        optimizer.load_state_dict(saved["optimizer"])
        rng.bit_generator.state = saved["numpy_rng_state"]
        torch.set_rng_state(saved["torch_rng_state"])
        torch.cuda.set_rng_state(saved["cuda_rng_state"])
        queue = saved["queue"].cuda()
        queue_patients = saved["queue_patients"]
        start = int(saved["step"])
    started = time.time()
    for step in range(start + 1, total_steps + 1):
        patients, batch = (scheduled_batch(schedule[step-1]) if schedule is not None else
                           choose_batch(by_view, args.view, rng, args.batch_size))
        x = torch.from_numpy(batch).permute(0, 1, 5, 2, 3, 4).float().cuda()
        first, second = perturb(x[:, 0]), perturb(x[:, 1])
        wrong = perturb(shuffled_chunks(x[:, 0], rng))
        optimizer.zero_grad(set_to_none=True)
        embeddings = encode(model, torch.cat((first, second, wrong), dim=0))
        a, b, bad = embeddings.chunk(3, dim=0)
        contrast_loss, detached = contrastive(projector(a), projector(b), patients,
                                              queue, queue_patients)
        order_logits = torch.cat((order_head(a), order_head(bad)))
        targets = torch.cat((torch.zeros(len(a), dtype=torch.long, device="cuda"),
                             torch.ones(len(bad), dtype=torch.long, device="cuda")))
        order_loss = F.cross_entropy(order_logits, targets)
        loss = contrast_loss + .2 * order_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite SSL loss at step {step}")
        loss.backward()
        nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad] +
                                 list(projector.parameters()) + list(order_head.parameters()), 1.)
        optimizer.step()
        queue = torch.cat((queue, detached), dim=0)[-512:]
        queue_patients = (queue_patients + patients + patients)[-512:]
        if step % CHECKPOINT_INTERVAL == 0 or step == total_steps:
            atomic_torch(checkpoint, pack(model, projector, order_head, optimizer,
                                          step, rng, queue, queue_patients, protocol))
            atomic_json(out / "status.json", {"status": "training", "step": step,
                         "total": total_steps, "loss": float(loss.item()),
                         "contrast_loss": float(contrast_loss.item()),
                         "order_loss": float(order_loss.item()),
                         "elapsed_seconds": time.time() - started, "pid": os.getpid()})
            print(f"SSL {args.view} r{args.repeat} f{args.fold}: {step}/{total_steps} "
                  f"loss={loss.item():.3f} ({time.time()-started:.0f}s)", flush=True)
    final = out / "encoder_adapter.pt"
    atomic_torch(final, {"last_block": model.blocks[-1].state_dict(),
                         "norm": model.norm.state_dict(), "head": model.head.state_dict(),
                         "protocol": protocol})
    atomic_json(out / "run_complete.json", {"adapter": str(final), "steps": total_steps,
                                              "elapsed_seconds": time.time() - started})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        ROOT.mkdir(parents=True, exist_ok=True)
        (ROOT / "last_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
