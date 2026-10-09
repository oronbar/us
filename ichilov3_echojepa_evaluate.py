"""Evaluate frozen EchoJEPA on the retained Ichilov3 patient-held-out folds."""
from __future__ import annotations

import json
from pathlib import Path

import ichilov_video_experiment_eval as ev
from ichilov3_train_fusion import atomic, hashfile


EMBEDDINGS = Path(r"D:\DS\ichilov3_echojepa_20261004\echojepa_visits.npz")
OUTPUT = Path(r"D:\us\output\ichilov_echojepa_20261004")


def main() -> None:
    if not EMBEDDINGS.is_file():
        raise FileNotFoundError(EMBEDDINGS)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    ev.PATHS = {"echojepa": EMBEDDINGS}
    models, visits, labels, cohort = ev.load_visits()
    protocol = {
        "model": "frozen EchoJEPA V-JEPA2 ViT-L VideoMix22M MIMIC pt220 c55",
        "embeddings": str(EMBEDDINGS),
        "embeddings_sha256": hashfile(EMBEDDINGS),
        "prediction_goal": "next-visit deterioration; same-visit Mid-GLS probe",
        "cohort": len(cohort),
        "patients": int(cohort.patient_id.nunique()),
        "events": int(cohort.label.sum()),
        "outer_folds": "retained 3x5 patient-held-out",
        "inner_folds": 3,
        "video_features": "current and current plus previous-visit delta",
        "fusion": "prespecified 25% video, 75% strain-clinical held-out scores",
    }
    path = OUTPUT / "protocol.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != protocol:
        raise ValueError("Evaluation protocol changed; use a fresh output directory")
    atomic(path, protocol)
    future = ev.future_evaluation(OUTPUT / "future", "stage2", models, visits, labels, cohort)
    gls = ev.gls_evaluation(OUTPUT, models, visits, labels)
    text = "# EchoJEPA frozen encoder evaluation\n\nNext visit:\n\n"
    text += future.to_string(index=False) + "\n\nCurrent Mid-GLS:\n\n"
    text += gls.to_string(index=False) + "\n"
    (OUTPUT / "results.md").write_text(text, encoding="utf-8")
    atomic(OUTPUT / "run_complete.json", {"future_models": future.model.tolist(),
                                          "gls_models": gls.model.tolist(),
                                          "output": str(OUTPUT)})
    print(text, flush=True)


if __name__ == "__main__":
    main()
