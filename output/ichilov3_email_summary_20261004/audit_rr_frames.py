"""Measure selected cine R-R intervals in native frames, using DICOM timing tags."""

import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pydicom


OUT = Path(__file__).resolve().parent
MANIFEST = Path(r"D:\DS\ichilov3_stage2_padded_20260926\full_manifest.json")


def describe(a):
    arr = np.asarray(a, dtype=float)
    return {
        "n": len(arr),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "population_sd": float(arr.std(ddof=0)),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
    }


def main():
    entries = json.loads(MANIFEST.read_text(encoding="utf-8"))
    warnings.filterwarnings("ignore", category=UserWarning, module="pydicom")
    continuous_frames = []
    nearest_frame_intervals = []
    rr_ms = []
    cine_first_cycle_frames = []
    cine_count = 0
    timing = Counter()
    issues = Counter()

    for entry in entries:
        ds = pydicom.dcmread(entry["source_path"], stop_before_pixels=True)
        r = np.asarray(getattr(ds, "RWaveTimeVector", []), dtype=float)
        if len(r) < 2:
            continue
        cine_count += 1
        n = int(ds.NumberOfFrames)
        if getattr(ds, "FrameTimeVector", None) is not None:
            # FrameTimeVector gives a time increment for each frame. Its first
            # item is typically zero and subsequent items advance the clock.
            v = np.asarray(ds.FrameTimeVector, dtype=float)
            if len(v) != n:
                issues["bad_frame_time_vector_length"] += 1
                continue
            t = np.cumsum(v)
            t -= t[0]
            timing["FrameTimeVector"] += 1
            mean_dt = float(np.mean(np.diff(t)))
        elif getattr(ds, "FrameTime", None) is not None:
            dt = float(ds.FrameTime)
            t = np.arange(n) * dt
            timing["FrameTime"] += 1
            mean_dt = dt
        else:
            issues["no_frame_timing"] += 1
            continue
        if mean_dt <= 0 or not np.isfinite(t).all():
            issues["invalid_frame_timing"] += 1
            continue

        first = None
        for start, end in zip(r[:-1], r[1:]):
            if not (np.isfinite(start) and np.isfinite(end) and end > start):
                issues["invalid_rwave_pair"] += 1
                continue
            if start < t[0] - mean_dt / 2 or end > t[-1] + mean_dt / 2:
                issues["rwave_pair_outside_cine"] += 1
                continue
            # Intervals between the nearest native frames, not endpoint-inclusive.
            i = int(np.argmin(np.abs(t - start)))
            j = int(np.argmin(np.abs(t - end)))
            frame_delta = j - i
            if frame_delta <= 0:
                issues["same_or_reverse_frame"] += 1
                continue
            nearest_frame_intervals.append(frame_delta)
            continuous_frames.append((end - start) / mean_dt)
            rr_ms.append(end - start)
            if first is None:
                first = frame_delta
        if first is not None:
            cine_first_cycle_frames.append(first)

    result = {
        "definition": "Native-frame index difference between adjacent R-wave timestamps, using each cine's DICOM frame time; endpoint-exclusive cycle length.",
        "selected_cines_with_2plus_rwave_markers": cine_count,
        "valid_rr_intervals": describe(nearest_frame_intervals),
        "continuous_rr_in_frame_units": describe(continuous_frames),
        "rr_duration_ms": describe(rr_ms),
        "first_valid_rr_per_cine_frame_intervals": describe(cine_first_cycle_frames),
        "timing_source_cines": dict(timing),
        "issues": dict(issues),
    }
    (OUT / "selected_cine_rr_frame_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
