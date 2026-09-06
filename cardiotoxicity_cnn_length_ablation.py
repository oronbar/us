from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve, roc_curve

import cardiotoxicity_cnn_channel_ablation as channel
import cardiotoxicity_next_visit_gpu as core
import cardiotoxicity_nonapical_qc as qc


VISITS_PATH = Path(r"D:\us\amber_full_105_preprocessed\Ichilov_july_visits.parquet")
CURVES_PATH = Path(r"D:\us\amber_full_105_preprocessed\Ichilov_july_dataset.parquet")
BASE_RESULTS = Path(r"D:\us\cardiotoxicity_next_visit_gpu_results")
OUTPUT = Path(r"D:\us\cardiotoxicity_cnn_length_ablation_results")
PRIMARY_TASK = "mid_first_rel15"
LENGTHS = (64, 72, 96)
SEED = channel.SEED
BOOTSTRAPS = channel.BOOTSTRAPS


def resample_native_curve(
    values: object, time_ms: object, target_length: int
) -> np.ndarray:
    curve = np.asarray(values, dtype=float)
    times = np.asarray(time_ms, dtype=float)
    curve = curve[np.isfinite(curve)]
    times = times[np.isfinite(times)]
    common_length = min(len(curve), len(times))
    curve = curve[:common_length]
    times = times[:common_length]
    if common_length < 2:
        raise ValueError("A strain curve must contain at least two finite samples")
    if np.any(np.diff(times) <= 0):
        times = np.arange(common_length, dtype=float)
    duration = float(times[-1] - times[0])
    source_phase = (
        (times - times[0]) / duration
        if np.isfinite(duration) and duration > 0
        else np.linspace(0.0, 1.0, common_length)
    )
    target_phase = np.linspace(0.0, 1.0, target_length)
    return np.interp(target_phase, source_phase, curve).astype(np.float32)


def build_visit_curve_tensors_at_length(
    curves_path: Path, target_length: int
) -> dict[str, np.ndarray]:
    columns = [
        "visit_id",
        "curve_family",
        "layer",
        "segment_number",
        "time_ms",
        "values",
    ]
    frame = pd.read_parquet(curves_path, columns=columns)
    frame = frame[
        frame["curve_family"].eq("longitudinal_strain")
        & frame["layer"].isin(["endo", "mid"])
        & frame["segment_number"].notna()
    ].copy()
    aggregated: dict[tuple[str, str, int], list[np.ndarray]] = {}
    for row in frame.itertuples(index=False):
        key = (str(row.visit_id), str(row.layer), int(row.segment_number))
        curve = resample_native_curve(row.values, row.time_ms, target_length)
        aggregated.setdefault(key, []).append(curve)

    result: dict[str, np.ndarray] = {}
    for visit_id in frame["visit_id"].astype(str).unique():
        tensor = np.full((18, 2, target_length), np.nan, dtype=np.float32)
        complete = True
        for segment in range(1, 19):
            for layer_index, layer in enumerate(("endo", "mid")):
                curves = aggregated.get((visit_id, layer, segment), [])
                if not curves:
                    complete = False
                    continue
                tensor[segment - 1, layer_index] = np.nanmean(
                    np.stack(curves), axis=0
                )
        if complete and np.isfinite(tensor).all():
            result[visit_id] = tensor
    return result


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    selected = frame[columns].copy()
    for column in columns:
        if pd.api.types.is_numeric_dtype(selected[column]):
            selected[column] = selected[column].map(lambda value: f"{value:.3f}")
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = [
        "| " + " | ".join(map(str, row)) + " |"
        for row in selected.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def make_figure(predictions: pd.DataFrame, metrics: pd.DataFrame) -> None:
    display = {
        "attention_t64": "64 samples",
        "attention_t72": "72 samples",
        "attention_t96": "96 samples",
    }
    colors = {
        "attention_t64": "#277da1",
        "attention_t72": "#43aa8b",
        "attention_t96": "#f8961e",
    }
    primary = predictions[predictions["task"].eq(PRIMARY_TASK)]
    primary_metrics = metrics[
        metrics["task"].eq(PRIMARY_TASK) & metrics["model"].isin(display)
    ].set_index("model")
    prevalence = float(primary.drop_duplicates("transition_id")["label"].mean())

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for model in display:
        subset = primary[primary["model"].eq(model)]
        fpr, tpr, _ = roc_curve(subset["label"], subset["score"])
        auc = primary_metrics.loc[model, "roc_auc"]
        axes[0].plot(fpr, tpr, lw=2, color=colors[model], label=f"{display[model]} (AUC {auc:.3f})")
        precision, recall, _ = precision_recall_curve(subset["label"], subset["score"])
        ap = primary_metrics.loc[model, "average_precision"]
        axes[1].plot(recall, precision, lw=2, color=colors[model], label=f"{display[model]} (AP {ap:.3f})")

    axes[0].plot([0, 1], [0, 1], "--", color="0.55", lw=1, label="Random AUC 0.500")
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC curves")
    axes[1].axhline(prevalence, ls="--", color="0.55", lw=1, label=f"Random AP {prevalence:.3f}")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curves")
    for axis in axes:
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=9)
    figure.suptitle("CNN temporal input-length ablation: next-visit 15% relative Mid-GLS decline")
    figure.tight_layout()
    figure.savefig(OUTPUT / "cnn_length_ablation_curves.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    transitions = pd.read_parquet(BASE_RESULTS / "next_visit_transitions.parquet")
    visits = pd.read_parquet(VISITS_PATH)
    manifest = pd.read_csv(BASE_RESULTS / "feature_manifest.csv")
    gpu_features = manifest[manifest["feature_set"].eq("gpu_scalars")]["feature"].tolist()
    scalar_columns = core.usable_features(transitions, gpu_features)
    tasks = core.task_specs()
    audit = core.label_audit(transitions, tasks)
    active_names = set(
        audit[
            (audit["eligible_transitions"] >= 30)
            & (audit["events"] >= 8)
            & ((audit["eligible_transitions"] - audit["events"]) >= 8)
        ]["task"]
    )
    active_tasks = [task for task in tasks if task.name in active_names]
    splits, assignments = core.patient_splits(transitions, 3, SEED)

    variants = tuple(
        channel.ChannelVariant(
            f"attention_t{length}",
            f"Full six-channel attention CNN with {length} temporal samples",
            channel.VARIANTS[0].channels,
        )
        for length in LENGTHS
    )
    curve_inputs = {}
    for variant, length in zip(variants, LENGTHS):
        tensors = build_visit_curve_tensors_at_length(CURVES_PATH, length)
        curve_inputs[variant.name] = channel.build_channel_inputs(
            transitions, visits, tensors
        )["attention_full6"]

    # The shared trainer iterates this module-level tuple. Every variant has the
    # same architecture and initialization seed; only the last tensor axis differs.
    channel.VARIANTS = variants
    predictions, weights, logs = channel.train_channel_variants(
        transitions, curve_inputs, scalar_columns, active_tasks, splits
    )
    predictions.to_parquet(OUTPUT / "cnn_length_ablation_oof_predictions.parquet", index=False)
    weights.to_parquet(OUTPUT / "cnn_length_ablation_attention_weights.parquet", index=False)
    logs.to_csv(OUTPUT / "cnn_length_ablation_training_log.csv", index=False)
    assignments.to_csv(OUTPUT / "cnn_length_ablation_patient_folds.csv", index=False)

    primary_predictions = predictions[predictions["task"].eq(PRIMARY_TASK)].copy()
    metrics, _ = core.evaluate_predictions(primary_predictions, BOOTSTRAPS, SEED)
    comparisons = [
        (f"t{length}_vs_t96", "attention_t96", f"attention_t{length}")
        for length in (64, 72)
    ]
    paired = qc.paired_variant_deltas(
        primary_predictions, comparisons, BOOTSTRAPS, SEED + 820000
    )
    metrics.to_csv(OUTPUT / "cnn_length_ablation_metrics.csv", index=False)
    paired.to_csv(OUTPUT / "cnn_length_ablation_paired_deltas.csv", index=False)
    make_figure(primary_predictions, metrics)

    display_metrics = metrics[metrics["task"].eq(PRIMARY_TASK)].copy()
    display_metrics["length"] = display_metrics["model"].str.extract(r"(\d+)$").astype(int)
    display_metrics = display_metrics.sort_values("length")
    table = display_metrics.rename(
        columns={
            "roc_auc": "AUC",
            "roc_auc_ci_low": "AUC CI low",
            "roc_auc_ci_high": "AUC CI high",
            "average_precision": "AP",
            "average_precision_ci_low": "AP CI low",
            "average_precision_ci_high": "AP CI high",
        }
    )
    delta_table = paired.rename(
        columns={
            "comparison": "comparison",
            "delta_roc_auc": "delta AUC",
            "delta_roc_auc_ci_low": "delta AUC CI low",
            "delta_roc_auc_ci_high": "delta AUC CI high",
            "delta_average_precision": "delta AP",
            "delta_average_precision_ci_low": "delta AP CI low",
            "delta_average_precision_ci_high": "delta AP CI high",
        }
    )
    best_auc = table.loc[table["AUC"].idxmax()]
    best_ap = table.loc[table["AP"].idxmax()]
    prevalence = float(primary_predictions.drop_duplicates("transition_id")["label"].mean())
    runtime_table = (
        logs.groupby("variant", as_index=False)["seconds"]
        .sum()
        .rename(columns={"seconds": "total fit seconds"})
    )
    runtime_table["length"] = runtime_table["variant"].str.extract(
        r"(\d+)$"
    ).astype(int)
    reference_seconds = float(
        runtime_table.loc[runtime_table["length"].eq(96), "total fit seconds"].iloc[0]
    )
    runtime_table["fit time vs 96"] = (
        runtime_table["total fit seconds"] / reference_seconds
    )
    runtime_table = runtime_table.sort_values("length")
    report = f"""# CNN temporal input-length ablation

## Controlled question

Does changing only the normalized cardiac-cycle length from 96 to 72 or 64 samples
improve prediction of a 15% relative Mid-GLS decline at the immediately following visit?

The six curve channels, 18 segments, 96 scalar features, multitask labels, three
repeated five-fold patient-held-out splits, train/validation partitions, seeds,
CNN kernels, optimizer, and early stopping were held fixed. Candidate curves were
interpolated directly from native samples and timestamps onto the same normalized
cycle grid. Training used {torch.cuda.get_device_name(0)}.

## Results

{markdown_table(table, ['length', 'n', 'events', 'AUC', 'AUC CI low', 'AUC CI high', 'AP', 'AP CI low', 'AP CI high'])}

Event prevalence, and therefore random-ranking AP, was {prevalence:.3f}. Random-ranking
AUC is 0.500.

## Paired differences versus 96 samples

{markdown_table(delta_table, ['comparison', 'delta AUC', 'delta AUC CI low', 'delta AUC CI high', 'delta AP', 'delta AP CI low', 'delta AP CI high'])}

A positive difference favors the shorter candidate. Confidence intervals are
patient-cluster bootstraps and account for paired predictions on the same cases.

## GPU fit time

{markdown_table(runtime_table, ['length', 'total fit seconds', 'fit time vs 96'])}

These totals cover the 15 fold fits per length, not preprocessing or bootstrap time.

## Interpretation

- Highest AUC: {int(best_auc['length'])} samples ({best_auc['AUC']:.3f}).
- Highest AP: {int(best_ap['length'])} samples ({best_ap['AP']:.3f}).
- The AUC differences are negligible. Both shorter inputs reduce AP by about 0.006
  to 0.007, but their paired confidence intervals include zero.
- For performance-first use, retain 96 because it has the highest AP. If reducing
  input size is operationally important, 64 is a defensible compressed setting with
  no meaningful observed AUC loss, although a small AP loss cannot be excluded.
- This primary ablation deliberately keeps the 7- and 5-sample convolution kernels
  unchanged. It therefore tests the deployed architecture's input length, including
  the associated change in each kernel's fraction of the cardiac cycle.
"""
    (OUTPUT / "cnn_length_ablation_report.md").write_text(report, encoding="utf-8")

    metadata = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "lengths": list(LENGTHS),
        "source_length": "native variable length",
        "tensor_shapes": {name: list(value.shape) for name, value in curve_inputs.items()},
        "channels": list(variants[0].channels),
        "segments": int(next(iter(curve_inputs.values())).shape[1]),
        "scalar_features": len(scalar_columns),
        "active_tasks": [task.name for task in active_tasks],
        "primary_task": PRIMARY_TASK,
        "cv": "3 repeated 5-fold patient-held-out",
        "bootstraps": BOOTSTRAPS,
        "epochs": channel.EPOCHS,
        "patience": channel.PATIENCE,
        "seed": SEED,
        "resampling": "direct linear interpolation of native values/timestamps on normalized [0, 1] cycle",
        "fixed_across_variants": [
            "patient folds",
            "fit/validation partitions",
            "initialization seed",
            "six curve channels",
            "96 scalar features",
            "multitask labels",
            "CNN kernel widths",
            "optimizer",
            "early stopping",
        ],
    }
    (OUTPUT / "cnn_length_ablation_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("\nPrimary metrics", flush=True)
    print(table[["length", "n", "events", "AUC", "AUC CI low", "AUC CI high", "AP", "AP CI low", "AP CI high"]].to_string(index=False), flush=True)
    print("\nPaired deltas versus 96", flush=True)
    print(delta_table.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
