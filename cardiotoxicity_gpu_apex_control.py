from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import cardiotoxicity_next_visit_gpu as core
import cardiotoxicity_nonapical_qc as qc


OUTPUT = Path(r"D:\us\cardiotoxicity_nonapical_qc_results")
BASE_RESULTS = Path(r"D:\us\cardiotoxicity_next_visit_gpu_results")
CURVES = Path(r"D:\us\amber_full_105_preprocessed\Ichilov_july_dataset.parquet")
VISITS = Path(r"D:\us\amber_full_105_preprocessed\Ichilov_july_visits.parquet")
SEED = 20260722


def transition_inputs(
    transitions: pd.DataFrame,
    visits: pd.DataFrame,
    tensors: dict[str, np.ndarray],
) -> np.ndarray:
    visit_order = {
        (str(row.patient_id), int(row.visit_order)): str(row.visit_id)
        for row in visits[["patient_id", "visit_order", "visit_id"]].itertuples(index=False)
    }
    result = []
    for row in transitions.itertuples(index=False):
        current_id = str(row.current_visit_id)
        order = int(row.current_visit_order)
        previous_id = current_id if order == 1 else visit_order[(str(row.patient_id), order - 1)]
        current = tensors[current_id]
        previous = tensors[previous_id]
        endo = current[:, 0]
        mid = current[:, 1]
        delta_endo = endo - previous[:, 0] if order > 1 else np.zeros_like(endo)
        delta_mid = mid - previous[:, 1] if order > 1 else np.zeros_like(mid)
        value = np.stack(
            [endo, mid, endo - mid, delta_endo, delta_mid, delta_endo - delta_mid], axis=1
        ).astype(np.float32)
        result.append(np.clip(value / 30.0, -2.0, 2.0))
    return np.stack(result)


def simple_policy_tensors(curves: pd.DataFrame, policy: str) -> dict[str, np.ndarray]:
    accept = f"accept__{policy}"
    segment_count = int(curves["segment_number"].max())
    tensors = {}
    for visit_id, group in curves.groupby("visit_id", sort=False):
        tensor = np.empty((segment_count, 2, 96), dtype=np.float32)
        for layer_index, layer in enumerate(("endo", "mid")):
            layer_group = group[group["layer"] == layer]
            accepted = layer_group[layer_group[accept]]
            fallback_source = accepted if len(accepted) else layer_group
            fallback = np.median(np.stack(fallback_source["curve_values"].to_numpy()), axis=0)
            for segment in range(1, segment_count + 1):
                values = accepted[accepted["segment_number"] == segment]
                tensor[segment - 1, layer_index] = (
                    np.mean(np.stack(values["curve_values"].to_numpy()), axis=0)
                    if len(values)
                    else fallback
                )
        tensors[str(visit_id)] = tensor
    return tensors


def main() -> int:
    transitions = pd.read_parquet(BASE_RESULTS / "next_visit_transitions.parquet")
    visits = pd.read_parquet(VISITS)
    manifest = pd.read_csv(BASE_RESULTS / "feature_manifest.csv")
    clinical = manifest[manifest["feature_set"] == "clinical"]["feature"].tolist()
    feature_sets = {"gpu_scalars": clinical}
    tasks = core.task_specs()
    audit = core.label_audit(transitions, tasks)
    active_names = set(
        audit[
            (audit["eligible_transitions"] >= 30)
            & (audit["events"] >= 8)
            & ((audit["eligible_transitions"] - audit["events"]) >= 8)
        ]["task"]
    )
    active_all = [task for task in tasks if task.name in active_names]
    selected_names = {
        task.name
        for task in active_all
        if task.name.startswith("mid_") and "_roll3_" not in task.name
    }
    splits, _ = core.patient_splits(transitions, 3, SEED)

    prior_predictions_path = OUTPUT / "controlled_gpu_oof_predictions.parquet"
    prior_logs_path = OUTPUT / "controlled_gpu_training_log.csv"
    if prior_predictions_path.exists():
        parts = [pd.read_parquet(prior_predictions_path)]
        logs = [pd.read_csv(prior_logs_path)]
        prepared18 = qc.prepare_curves(CURVES, max_segment=18)
        tensor_sets = {
            "controlled18_fixed_qc": simple_policy_tensors(prepared18, "noapex_fixed_qc"),
            "controlled18_shape_qc": simple_policy_tensors(prepared18, "noapex_shape_qc"),
        }
    else:
        parts = []
        logs = []
        prepared = qc.prepare_curves(CURVES)
        prepared18 = qc.prepare_curves(CURVES, max_segment=18)
        tensor_sets = {
            "controlled18": core.build_visit_curve_tensors(CURVES),
            "controlled_noapex": simple_policy_tensors(prepared, "noapex"),
            "controlled_fixed_qc": simple_policy_tensors(prepared, "noapex_fixed_qc"),
            "controlled_shape_qc": simple_policy_tensors(prepared, "noapex_shape_qc"),
            "controlled18_fixed_qc": simple_policy_tensors(prepared18, "noapex_fixed_qc"),
            "controlled18_shape_qc": simple_policy_tensors(prepared18, "noapex_shape_qc"),
        }
    for name, tensors in tensor_sets.items():
        curve_inputs = transition_inputs(transitions, visits, tensors)
        predictions, log, metadata = core.gpu_oof_predictions(
            transitions,
            curve_inputs,
            feature_sets,
            active_all,
            splits,
            epochs=250,
            patience=35,
            seed=SEED,
        )
        predictions = predictions[predictions["task"].isin(selected_names)].copy()
        predictions["model"] = name
        parts.append(predictions)
        log.insert(0, "model", name)
        logs.append(log)
    predictions = pd.concat(parts, ignore_index=True)
    metrics, _ = core.evaluate_predictions(predictions, 1000, SEED + 200000)
    comparisons = [
        ("remove_apex_controlled", "controlled18", "controlled_noapex"),
        ("fixed_qc_controlled", "controlled_noapex", "controlled_fixed_qc"),
        ("shape_qc_controlled", "controlled_noapex", "controlled_shape_qc"),
        ("fixed_qc_all18_controlled", "controlled18", "controlled18_fixed_qc"),
        ("shape_qc_all18_controlled", "controlled18", "controlled18_shape_qc"),
    ]
    deltas = qc.paired_variant_deltas(predictions, comparisons, 1000, SEED + 300000)
    predictions.to_parquet(OUTPUT / "controlled_gpu_oof_predictions.parquet", index=False)
    metrics.to_csv(OUTPUT / "controlled_gpu_metrics.csv", index=False)
    deltas.to_csv(OUTPUT / "controlled_gpu_deltas.csv", index=False)
    pd.concat(logs, ignore_index=True).to_csv(OUTPUT / "controlled_gpu_training_log.csv", index=False)
    metadata_out = {
        "device": metadata["device"],
        "scalar_features": "same 27 clinical trajectory features in all four models",
        "controlled18_segments": 18,
        "controlled_nonapical_segments": 12,
        "all18_qc_models": ["controlled18_fixed_qc", "controlled18_shape_qc"],
        "cv": "3 repeated 5-fold patient-held-out",
        "bootstraps": 1000,
    }
    (OUTPUT / "controlled_gpu_metadata.json").write_text(
        json.dumps(metadata_out, indent=2), encoding="utf-8"
    )
    print(metrics[metrics["task"] == "mid_first_rel15"].to_string(index=False))
    print(deltas[deltas["task"] == "mid_first_rel15"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
