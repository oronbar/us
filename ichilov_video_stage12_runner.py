"""Resume-safe local continuation from EchoFM encoding through Stage 2 evaluation."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

STATE = Path(r"D:\us\output\ichilov_video_experiments_20260930\state.json")
ENCODING = Path(r"D:\DS\ichilov3_echofm_20260930")
ROOT = STATE.parent


def set_state(**kwargs):
    state = json.loads(STATE.read_text(encoding="utf-8"))
    state.update(kwargs)
    temp = STATE.with_suffix(".tmp")
    temp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temp.replace(STATE)


def run_stage(stage):
    destination = ROOT / stage / "run_complete.json"
    if destination.exists():
        return
    state = json.loads(STATE.read_text(encoding="utf-8"))
    state[stage]["status"] = "evaluating"
    state["stage"] = 1 if stage == "stage1" else 2
    state["status"] = "running"
    state["active_gpu_job"] = None
    state["next_action"] = f"Complete {stage} held-out evaluation."
    temp = STATE.with_suffix(".tmp")
    temp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temp.replace(STATE)
    subprocess.run([sys.executable, "-u", r"D:\us\ichilov_video_experiment_eval.py",
                    "--stage", stage], check=True)
    if not destination.exists():
        raise RuntimeError(f"{stage} evaluation exited without completion artifact")
    metrics = pd.read_csv(ROOT / stage / "future" / "metrics.csv").set_index("model")
    baseline = metrics.loc["strain_clinical_reference"]
    if stage == "stage1":
        gls = pd.read_csv(ROOT / stage / "gls" / "metrics.csv").set_index("model")
        check = {
            "stage": stage,
            "reference": {"auc": float(baseline.auc), "ap": float(baseline.ap),
                          "ppv_top20": float(baseline.ppv_top20)},
            "echofm": metrics.loc["echofm_current_late25"]["auc":"ppv_top20"].to_dict(),
            "echoprime": metrics.loc["echoprime_current_late25"]["auc":"ppv_top20"].to_dict(),
            "panecho": metrics.loc["panecho_current_late25"]["auc":"ppv_top20"].to_dict(),
            "gls_pearson_r": {name: float(row.pearson_r) for name, row in gls.iterrows()},
        }
    else:
        check = {"stage": stage,
                 "reference": {"auc": float(baseline.auc), "ap": float(baseline.ap),
                               "ppv_top20": float(baseline.ppv_top20)},
                 "history_vs_current": {}}
        for encoder in ("echoprime", "panecho", "echofm"):
            old = metrics.loc[f"{encoder}_current_late25"]
            new = metrics.loc[f"{encoder}_history_late25"]
            check["history_vs_current"][encoder] = {
                "auc_delta": float(new.auc - old.auc), "ap_delta": float(new.ap - old.ap),
                "ppv_top20_delta": float(new.ppv_top20 - old.ppv_top20),
                "history_auc": float(new.auc), "history_ap": float(new.ap),
            }
    checkpoint = ROOT / stage / "checkpoint_comparison.json"
    temporary = checkpoint.with_suffix(".tmp")
    temporary.write_text(json.dumps(check, indent=2), encoding="utf-8")
    temporary.replace(checkpoint)
    state = json.loads(STATE.read_text(encoding="utf-8"))
    state[stage]["status"] = "complete"
    state["next_action"] = "Review metrics and continue to the next stage."
    temp = STATE.with_suffix(".tmp")
    temp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temp.replace(STATE)


def main():
    set_state(runner={"pid": os.getpid(), "status": "waiting_for_echofm"})
    while not (ENCODING / "run_complete.json").exists():
        info = ENCODING / "status.json"
        if info.exists():
            status = json.loads(info.read_text(encoding="utf-8"))
            if status.get("stage") == "error":
                raise RuntimeError("EchoFM encoding failed: " + status.get("error", "unknown"))
        time.sleep(30)
    done = json.loads((ENCODING / "run_complete.json").read_text(encoding="utf-8"))
    if done.get("clips") != 1233 or done.get("visits") != 411:
        raise ValueError(f"Incomplete EchoFM encoding: {done}")
    with (ENCODING / "echofm_visits.npz").open("rb") as stream:
        if not stream.read(4) == b"PK\x03\x04":
            raise ValueError("Invalid EchoFM visit archive")
    run_stage("stage1")
    run_stage("stage2")
    state = json.loads(STATE.read_text(encoding="utf-8"))
    state["stage"] = 3
    state["stage3"]["status"] = "preparing"
    state["runner"] = {"pid": os.getpid(), "status": "complete"}
    state["next_action"] = "Review Stage 1 and 2 metrics; build cross-dataset GE cine inventory and begin shared self-supervised EchoPrime adaptation."
    temp = STATE.with_suffix(".tmp")
    temp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temp.replace(STATE)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        ROOT.mkdir(parents=True, exist_ok=True)
        (ROOT / "stage12_runner_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        set_state(status="error", next_action="Repair stage12_runner_error.txt; resume the runner after checking existing processes and fold files.")
        raise
