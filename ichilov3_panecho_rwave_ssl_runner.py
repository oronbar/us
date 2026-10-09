"""Sequential, resumable PanEcho first-cycle SSL training and evaluation."""

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

from ichilov3_train_fusion import atomic


ROOT = Path(r"D:\DS\ichilov3_panecho_rwave_ssl_20261005")
SCRIPT = Path(r"D:\us")
FOLDS = Path(r"D:\DS\ichilov3_nonphilips_future_20260930\aligned\patient_folds.parquet")
GPU_LOCK = Path(r"D:\us\output\ichilov_video_experiments_20260930\gpu_job.lock")


def run(script, label, *args):
    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    atomic(ROOT / "runner_status.json", {"stage": label, "pid": os.getpid(),
                                        "started_at": time.time()})
    with (logs / f"{label}.log").open("a", encoding="utf-8") as stdout, \
         (logs / f"{label}_errors.log").open("a", encoding="utf-8") as stderr:
        subprocess.run([sys.executable, "-u", str(SCRIPT / script), *args],
                       stdout=stdout, stderr=stderr, check=True)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = ROOT / "runner.lock"
    if lock.exists():
        raise RuntimeError(f"Runner lock already exists: {lock}")
    with os.fdopen(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY), "w") as f:
        f.write(json.dumps({"pid": os.getpid(), "started_at": time.time()}))
    try:
        while not (ROOT / "cache_complete.json").exists():
            atomic(ROOT / "runner_status.json", {"stage": "waiting_for_cine_cache",
                                                "pid": os.getpid()})
            time.sleep(10)
        while GPU_LOCK.exists():
            atomic(ROOT / "runner_status.json", {"stage": "waiting_for_shared_gpu",
                                                "pid": os.getpid()})
            time.sleep(20)
        with os.fdopen(os.open(GPU_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY), "w") as f:
            f.write(json.dumps({"pid": os.getpid(), "job": "PanEcho first-cycle SSL"}))
        try:
            folds = sorted((int(r), int(f)) for r, f in pd.read_parquet(FOLDS)[
                ["repeat", "fold"]].drop_duplicates().itertuples(index=False, name=None))
            if len(folds) != 15:
                raise ValueError("Expected 3x5 patient folds")
            for repeat, fold in folds:
                name = f"r{repeat}_f{fold}"
                if not (ROOT / "training" / name / "run_complete.json").exists():
                    run("ichilov3_panecho_rwave_ssl_train.py", "train_" + name,
                        "--repeat", str(repeat), "--fold", str(fold), "--epochs", "3")
                if not (ROOT / "evaluation_embeddings" / name / "run_complete.json").exists():
                    run("ichilov3_panecho_rwave_ssl_encode.py", "encode_" + name,
                        "--repeat", str(repeat), "--fold", str(fold))
        finally:
            if GPU_LOCK.exists():
                data = json.loads(GPU_LOCK.read_text(encoding="utf-8"))
                if data.get("pid") == os.getpid():
                    GPU_LOCK.unlink()
        if not (ROOT / "evaluation" / "run_complete.json").exists():
            run("ichilov3_panecho_rwave_ssl_evaluate.py", "evaluate")
        atomic(ROOT / "runner_status.json", {"stage": "complete", "pid": os.getpid()})
        atomic(ROOT / "run_complete.json", {"evaluation": str(ROOT / "evaluation" /
                                                               "results.json"), "folds": 15})
    finally:
        if lock.exists():
            lock.unlink()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        ROOT.mkdir(parents=True, exist_ok=True)
        error = traceback.format_exc()
        (ROOT / "runner_error.txt").write_text(error, encoding="utf-8")
        atomic(ROOT / "runner_status.json", {"stage": "error", "pid": os.getpid(),
                                            "error": error})
        raise
