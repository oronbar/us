"""Precompute local cine suggestions for the no-bookmark review queue.

Uses the running review server so model jobs are serialized with interactive
requests. Never saves review decisions or changes source DICOMs.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen


BASE_URL = "http://127.0.0.1:8765"
OUTPUT = Path(r"D:\us\output\dicom_prediction\ichilov3_manual_selection")
STATUS = OUTPUT / "batch_analysis_status.json"


def request(path: str, **params) -> dict | list:
    url = BASE_URL + path + ("?" + urlencode(params) if params else "")
    last_error = None
    for attempt in range(5):
        try:
            with urlopen(url, timeout=30) as response:
                return json.load(response)
        except (URLError, TimeoutError, OSError) as exc:
            last_error = exc
            time.sleep(min(2**attempt, 15))
    raise RuntimeError(f"Review server unavailable: {last_error}")


def write_status(state: dict) -> None:
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    temporary = STATUS.with_name(f"{STATUS.stem}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    # Windows readers may briefly hold the destination open. Retry the
    # atomic replacement rather than losing the entire background run.
    for attempt in range(20):
        try:
            os.replace(temporary, STATUS)
            return
        except PermissionError:
            time.sleep(min(0.05 * (attempt + 1), 0.5))
    os.replace(temporary, STATUS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-patient", default="", help="Optional starting patient ID")
    args = parser.parse_args()

    visits = request("/api/visits")
    # Match the app's "No bookmark" queue, including visits with no DICOMs.
    queue = [visit for visit in visits if visit["patient"] >= args.start_patient
             and visit["tomtec_bookmark"] != "Yes"]
    queue.sort(key=lambda visit: (visit["patient"], visit["date"]))
    state = {"status": "running", "start_patient": args.start_patient,
             "total": len(queue), "completed": 0, "ready": 0, "no_cines": 0,
             "errors": 0, "current": "", "current_clip": 0, "current_total_clips": 0,
             "results": []}
    write_status(state)
    print(f"Batch started: {len(queue)} visits from {args.start_patient}", flush=True)

    for visit in queue:
        key = visit["key"]
        state.update(current=key, current_clip=0, current_total_clips=visit["video_count"])
        write_status(state)
        try:
            result = request("/api/suggestions", key=key, start=1)
            while result["status"] in ("queued", "running"):
                state["current_clip"] = result.get("done", 0)
                write_status(state)
                time.sleep(2)
                result = request("/api/suggestions", key=key)
            outcome = {"visit_key": key, "status": result["status"]}
            if result["status"] == "ready":
                outcome["selected"] = result["selected"]
                outcome["analyzed"] = result["analyzed"]
                outcome["clip_errors"] = result["errors"]
                state["ready"] += 1
            elif result["status"] == "no_cines":
                state["no_cines"] += 1
            else:
                outcome["error"] = result.get("error", "Unexpected status")
                state["errors"] += 1
        except Exception as exc:
            outcome = {"visit_key": key, "status": "error",
                       "error": f"{type(exc).__name__}: {exc}"}
            state["errors"] += 1
        state["results"].append(outcome)
        state["completed"] += 1
        write_status(state)
        print(f"{state['completed']}/{state['total']} {key}: {outcome['status']}", flush=True)

    state.update(status="complete", current="", current_clip=0, current_total_clips=0)
    write_status(state)
    print(f"Complete: {state['ready']} ready, {state['no_cines']} without cines, "
          f"{state['errors']} errors", flush=True)


if __name__ == "__main__":
    main()
