"""Durable sequential GE self-supervised training and held-out evaluation."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
import psutil

ROOT = Path(r"D:\DS\ichilov_ge_ssl_20260930")
PREP = Path(r"D:\DS\ichilov_ge_inventory_20260930")
CACHE = Path(r"D:\DS\ichilov_ge_ssl_clips_20260930")
PREFIX = Path(r"D:\DS\ichilov_ge_ssl_prefix_20260930")
EXPERIMENT = Path(r"D:\us\output\ichilov_video_experiments_20260930")
FOLDS = Path(r"D:\DS\ichilov3_nonphilips_future_20260930\aligned\patient_folds.parquet")
SCRIPT = Path(__file__).parent
GPU_LOCK = EXPERIMENT / "gpu_job.lock"


def status(**fields):
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / "runner_status.json"
    value = json.loads(path.read_text()) if path.exists() else {}
    value.update(fields, pid=os.getpid())
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.replace(path)


def wait_for(path: Path, reason: str):
    status(status="waiting", waiting_for=reason)
    while not path.exists():
        time.sleep(45)


def run(script: str, *args: str, label: str):
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    logfile = logs / f"{label}.log"
    errfile = logs / f"{label}_errors.log"
    status(status="running", job=label, log=str(logfile))
    with logfile.open("a", encoding="utf-8") as stdout, errfile.open("a", encoding="utf-8") as stderr:
        subprocess.run([sys.executable, "-u", str(SCRIPT / script), *args],
                       stdout=stdout, stderr=stderr, check=True)


def gpu_busy():
    needles = ("ichilov3_encode_echofm.py", "ichilov_ge_classify_views.py",
               "ichilov_ge_ssl_prefix_cache.py", "ichilov_ge_ssl_train.py",
               "ichilov_ge_ssl_encode_eval.py")
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = " ".join(p.info["cmdline"] or []).lower()
            if any(name in cmd for name in needles):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return False


def acquire_gpu_lock():
    status(status="waiting", waiting_for="GPU and prior prep lock")
    while gpu_busy() or GPU_LOCK.exists():
        time.sleep(30)
    fd = os.open(GPU_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps({"pid": os.getpid(), "job": "GE SSL sequential training"}))


def fold_list():
    table = pd.read_parquet(FOLDS)
    return sorted((int(r), int(f)) for r, f in
                  table[["repeat", "fold"]].drop_duplicates().itertuples(index=False, name=None))


def train(view: str, repeat: int, fold: int):
    key = f"{view}_r{repeat}_f{fold}"
    complete = ROOT / view / f"r{repeat}_f{fold}" / "run_complete.json"
    if complete.exists():
        return
    run("ichilov_ge_ssl_train.py", "--view", view, "--repeat", str(repeat),
        "--fold", str(fold), "--steps", "0", "--batch-size", "4",
        label=f"train_{key}")
    if not complete.exists():
        raise RuntimeError(f"Training returned without completion manifest: {key}")


def encode(config: str, repeat: int, fold: int):
    key = f"{config}_r{repeat}_f{fold}"
    complete = ROOT / "evaluation_embeddings" / config / f"r{repeat}_f{fold}" / "run_complete.json"
    if complete.exists():
        return
    run("ichilov_ge_ssl_encode_eval.py", "--config", config, "--repeat", str(repeat),
        "--fold", str(fold), label=f"encode_{key}")
    if not complete.exists():
        raise RuntimeError(f"Adapter encoding incomplete: {key}")


def checkpoint(config: str):
    import pandas as pd
    folder = ROOT / "evaluation" / config
    future = pd.read_csv(folder / "metrics.csv").to_dict("records")
    gls = pd.read_csv(folder / "gls" / "metrics.csv").to_dict("records")
    output = {"config": config, "future": future, "gls": gls,
              "baseline_stage2": str(EXPERIMENT / "stage2" / "future" / "metrics.csv")}
    temp = folder / "checkpoint_comparison.tmp"
    temp.write_text(json.dumps(output, indent=2), encoding="utf-8")
    temp.replace(folder / "checkpoint_comparison.json")


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    wait_for(PREP / "view_classification_complete.json", "GE view inventory and classification")
    if not (CACHE / "run_complete.json").exists():
        run("ichilov_ge_ssl_prepare.py", label="prepare_ssl_cine_cache")
    if not (CACHE / "run_complete.json").exists():
        raise RuntimeError("Incomplete GE SSL cine cache")
    acquire_gpu_lock()
    try:
        if not (PREFIX / "run_complete.json").exists():
            run("ichilov_ge_ssl_prefix_cache.py", label="echoprime_prefix_cache")
        if not (PREFIX / "run_complete.json").exists():
            raise RuntimeError("Incomplete EchoPrime prefix cache")
        folds = fold_list()
        if len(folds) != 15:
            raise ValueError(f"Unexpected number of outer folds: {len(folds)}")
        for repeat, fold in folds:
            train("all", repeat, fold)
            encode("all", repeat, fold)
        if not (ROOT / "evaluation" / "all" / "run_complete.json").exists():
            run("ichilov_ge_ssl_evaluate.py", "--config", "all", label="evaluate_all")
        checkpoint("all")
        status(status="shared_complete", next_action="Train three independent view-specific adapters")
        for repeat, fold in folds:
            for view in ("A2C", "A3C", "A4C"):
                train(view, repeat, fold)
            encode("specific", repeat, fold)
        if not (ROOT / "evaluation" / "specific" / "run_complete.json").exists():
            run("ichilov_ge_ssl_evaluate.py", "--config", "specific", label="evaluate_specific")
        checkpoint("specific")
        status(status="complete", next_action="Compare all stages and stop 30-minute follow-up")
        (ROOT / "run_complete.json").write_text(json.dumps({
            "shared": str(ROOT / "evaluation" / "all" / "checkpoint_comparison.json"),
            "specific": str(ROOT / "evaluation" / "specific" / "checkpoint_comparison.json")},
            indent=2), encoding="utf-8")
    finally:
        if GPU_LOCK.exists():
            GPU_LOCK.unlink()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        ROOT.mkdir(parents=True, exist_ok=True)
        (ROOT / "runner_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        status(status="error", error=traceback.format_exc())
        raise
