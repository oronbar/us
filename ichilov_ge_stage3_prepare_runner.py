"""Resumable GE inventory, deduplication and view screening after EchoFM."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import psutil

ROOT = Path(r"D:\DS\ichilov_ge_inventory_20260930")
EXPERIMENT = Path(r"D:\us\output\ichilov_video_experiments_20260930")
ECHOFM = Path(r"D:\DS\ichilov3_echofm_20260930\run_complete.json")
GPU_LOCK = EXPERIMENT / "gpu_job.lock"
SCRIPT = Path(__file__).parent


def state(**fields):
    target = ROOT / "prepare_status.json"
    data = json.loads(target.read_text()) if target.exists() else {}
    data.update(fields, pid=os.getpid())
    temp = target.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temp.replace(target)


def wait_for(path: Path, reason: str):
    state(status="waiting", waiting_for=reason)
    while not path.exists():
        time.sleep(45)


def run(script: str, *arguments: str, log_name: str):
    log = ROOT / log_name
    error = ROOT / (Path(log_name).stem + "_errors.log")
    state(status="running", script=script, log=str(log))
    with log.open("a", encoding="utf-8") as stdout, error.open("a", encoding="utf-8") as stderr:
        subprocess.run([sys.executable, "-u", str(SCRIPT / script), *arguments],
                       stdout=stdout, stderr=stderr, check=True)


def gpu_processes():
    needles = ("ichilov3_encode_echofm.py", "ichilov_ge_classify_views.py",
               "ichilov_ge_ssl_train.py", "ichilov_ge_ssl_encode.py")
    found = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            command = " ".join(p.info["cmdline"] or []).lower()
            if any(name in command for name in needles):
                found.append(p.info["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return found


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    wait_for(ECHOFM, "EchoFM completion and GPU release")
    for dataset in ("Ichilov2", "Ichilov3"):
        summary = ROOT / f"{dataset.lower()}_summary.json"
        if not summary.is_file():
            run("ichilov_ge_cine_inventory.py", "--dataset", dataset,
                log_name=f"{dataset.lower()}.log")
    wait_for(ROOT / "ichilov1_summary.json", "Ichilov1 full inventory")
    if not (ROOT / "unique_ge_cines_summary.json").exists():
        run("ichilov_ge_prepare_manifest.py", log_name="deduplicate.log")
    if not (ROOT / "view_classification_complete.json").exists():
        state(status="waiting", waiting_for="exclusive GPU")
        while gpu_processes():
            time.sleep(30)
        try:
            lock_fd = os.open(GPU_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise RuntimeError(f"Another GPU job owns {GPU_LOCK}")
        with os.fdopen(lock_fd, "w") as lock:
            lock.write(json.dumps({"pid": os.getpid(), "job": "GE view classification"}))
        try:
            run("ichilov_ge_classify_views.py", log_name="classify_views.log")
        finally:
            if GPU_LOCK.exists():
                GPU_LOCK.unlink()
    state(status="complete", next_action="Train shared EchoPrime SSL fold models")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        ROOT.mkdir(parents=True, exist_ok=True)
        (ROOT / "prepare_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        state(status="error", error=traceback.format_exc())
        raise
