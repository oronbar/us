"""GE-only patient-held-out evaluation of old versus R-wave PanEcho vectors."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ichilov_video_experiment_eval import future_evaluation, gls_evaluation
from ichilov3_train_fusion import atomic, hashfile


ROOT = Path(r"D:\DS\ichilov3_rwave_panecho_20261005")
OLD = Path(r"D:\DS\ichilov3_temporal_trial_20260927\panecho_visits.npz")
NEW = ROOT / "panecho_visits.npz"
ALIGN = Path(r"D:\DS\ichilov3_aligned_20260927\visit_alignment.parquet")
LABELS = Path(r"D:\us\amber_full_105_preprocessed\Ichilov_july_visits.parquet")
COHORT = Path(r"D:\DS\ichilov3_nonphilips_future_20260930\aligned\cohort.parquet")


def load(path):
    with np.load(path) as f:
        return {(str(p), str(d)): v.copy() for p, d, v in zip(
            f["patient_ids"], f["visit_dates"], f["embeddings"])}


def main():
    out = ROOT / "ge_only"
    out.mkdir(exist_ok=True)
    protocol = {
        "models": ["panecho_hr_estimated", "panecho_rwave"],
        "scope": "same 182 GE next-visit transitions and 298 matched GE GLS visits",
        "method": "same 3x5 patient-held-out GE-only evaluation from ichilov_video_experiment_eval, stage1 current-video features",
        "input_hashes": {str(p): hashfile(p) for p in (OLD, NEW, ALIGN, LABELS, COHORT)},
        "evaluation_code_sha256": hashfile(Path(r"D:\us\ichilov_video_experiment_eval.py")),
    }
    path = out / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("GE-only evaluation protocol changed")
    atomic(path, protocol)
    models = {"panecho_hr_estimated": load(OLD), "panecho_rwave": load(NEW)}
    visits = pd.read_parquet(ALIGN)
    labels = pd.read_parquet(LABELS)
    cohort = pd.read_parquet(COHORT)
    future = future_evaluation(out / "future", "stage1", models, visits, labels, cohort)
    gls = gls_evaluation(out, models, visits, labels)
    lines = ["# GE-only R-wave PanEcho evaluation", "",
             "Same 182 GE transitions and 298 matched GE GLS visits, identical 3x5 held-out patient folds. Separate GE-only training for each PanEcho sampling arm.",
             "", "## Next-visit deterioration", "", future.to_string(index=False),
             "", "## Same-visit Mid-GLS", "", gls.to_string(index=False),
             "", "The R-wave arm uses one window versus up to four averaged HR-estimated windows. Fourteen source cines span multiple cycles between first and last R marker."]
    (out / "results.md").write_text("\n".join(lines), encoding="utf-8")
    atomic(out / "run_complete.json", {"future": future.model.tolist(),
                                       "gls": gls.model.tolist()})


if __name__ == "__main__":
    main()
