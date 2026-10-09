"""Fill the read-only EchoPrime prefix cache while GE inventories run."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

LOCK = Path(r"D:\us\output\ichilov_video_experiments_20260930\gpu_job.lock")
ROOT = Path(r"D:\DS\ichilov_ge_ssl_prefix_20260930")


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / "run_complete.json").exists():
        return
    ready = Path(r"D:\DS\ichilov_ge_inventory_20260930\view_classification_complete.json")
    while not ready.exists() or LOCK.exists():
        time.sleep(30)
    fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps({"pid": os.getpid(), "job": "EchoPrime prefix cache"}))
    try:
        with (ROOT / "run.log").open("a", encoding="utf-8") as stdout, \
             (ROOT / "run_errors.log").open("a", encoding="utf-8") as stderr:
            subprocess.run([sys.executable, "-u", str(Path(__file__).parent /
                                                 "ichilov_ge_ssl_prefix_cache.py")],
                           stdout=stdout, stderr=stderr, check=True)
    finally:
        if LOCK.exists():
            LOCK.unlink()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        ROOT.mkdir(parents=True, exist_ok=True)
        (ROOT / "runner_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
