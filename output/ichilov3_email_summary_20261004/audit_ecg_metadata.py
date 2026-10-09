"""Count ECG timing metadata in selected Ichilov3 source cines, without changing them."""

import json
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import pydicom


ROOT = Path(__file__).resolve().parent
MANIFEST = Path(r"D:\DS\ichilov3_stage2_padded_20260926\full_manifest.json")
OUT = ROOT / "selected_cine_ecg_metadata_summary.json"


def summarize(values):
    return {key: int(value) for key, value in sorted(values.items())}


def main():
    entries = json.loads(MANIFEST.read_text(encoding="utf-8"))
    overall = Counter()
    by_vendor = defaultdict(Counter)
    by_view = defaultdict(Counter)
    rwave_visits = set()
    rwave_views_per_visit = Counter()
    ecg_visits = set()
    all_visits = set()
    selected_sops = set()
    errors = []
    warnings.filterwarnings("ignore", category=UserWarning, module="pydicom")

    for entry in entries:
        vendor = "Philips" if "PHILIPS" in entry.get("manufacturer", "").upper() else (
            "GE" if "GE" in entry.get("manufacturer", "").upper() else "Other")
        view = entry["view"]
        visit = (entry["patient"], entry["visit_date"])
        all_visits.add(visit)
        counters = [overall, by_vendor[vendor], by_view[view]]
        for counter in counters:
            counter["selected_cines"] += 1
        path = Path(entry["source_path"])
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            sop = str(getattr(ds, "SOPInstanceUID", ""))
            selected_sops.add(sop)
            rwave = getattr(ds, "RWaveTimeVector", None)
            n_rwave = len(rwave) if rwave is not None else 0
            waveform = getattr(ds, "WaveformSequence", None)
            n_waveform = len(waveform) if waveform is not None else 0
            if n_rwave:
                rwave_visits.add(visit)
                rwave_views_per_visit[visit] += 1
            if n_waveform:
                ecg_visits.add(visit)
            for counter in counters:
                counter["read_ok"] += 1
                counter["rwavetimevector_present"] += int(n_rwave > 0)
                counter[f"rwave_peak_count_{n_rwave if n_rwave < 5 else '5plus'}"] += 1
                counter["rwavetimevector_two_or_more"] += int(n_rwave >= 2)
                counter["rwavetimevector_one"] += int(n_rwave == 1)
                counter["rwavetimevector_zero"] += int(n_rwave == 0)
                counter["waveform_sequence_present"] += int(n_waveform > 0)
                counter["trigger_time_present"] += int(getattr(ds, "TriggerTime", None) is not None)
                counter["heart_rate_present"] += int(getattr(ds, "HeartRate", None) is not None)
        except Exception as exc:
            for counter in counters:
                counter["read_error"] += 1
            errors.append({"index": len(selected_sops), "type": type(exc).__name__})

    result = {
        "scope": "selected A2C/A3C/A4C source DICOMs in full_manifest; metadata only",
        "total_selected": len(entries),
        "unique_sop_count": len(selected_sops),
        "unique_visits": len(all_visits),
        "visits_with_rwave_vector": len(rwave_visits),
        "visits_by_number_of_views_with_rwave_vector": {
            str(n): sum(rwave_views_per_visit[visit] == n for visit in all_visits)
            for n in range(4)
        },
        "visits_with_waveform_sequence": len(ecg_visits),
        "overall": summarize(overall),
        "by_vendor": {k: summarize(v) for k, v in sorted(by_vendor.items())},
        "by_view": {k: summarize(v) for k, v in sorted(by_view.items())},
        "errors": errors,
        "note": "RWaveTimeVector contains timestamp markers, not a continuous ECG waveform."
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
