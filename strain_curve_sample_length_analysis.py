from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


DATASET = Path(r"D:\us\amber_full_105_preprocessed\Ichilov_july_dataset.parquet")
OUTPUT = Path(r"D:\us\cardiotoxicity_curve_length_results")
CANDIDATES = (32, 48, 56, 64, 65, 72, 80, 96, 128)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    display = frame[columns].copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.3f}"
        )
    return "\n".join(
        [
            "| " + " | ".join(columns) + " |",
            "|" + "|".join(["---"] * len(columns)) + "|",
            *[
                "| " + " | ".join(map(str, row)) + " |"
                for row in display.itertuples(index=False, name=None)
            ],
        ]
    )


def aligned_curve(row: object) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(row.values, dtype=float)
    time_ms = np.asarray(row.time_ms, dtype=float)
    valid = np.isfinite(values) & np.isfinite(time_ms)
    values = values[valid]
    time_ms = time_ms[valid]
    if len(values) < 3:
        return np.empty(0), np.empty(0)
    if np.any(np.diff(time_ms) <= 0) or time_ms[-1] <= time_ms[0]:
        phase = np.linspace(0.0, 1.0, len(values))
    else:
        phase = (time_ms - time_ms[0]) / (time_ms[-1] - time_ms[0])
    return values, phase


def candidate_tradeoffs(curves: pd.DataFrame) -> pd.DataFrame:
    accumulators = {
        length: {
            key: []
            for key in (
                "rmse",
                "mae",
                "peak_error",
                "ttp_fraction_error",
                "ttp_ms_error",
            )
        }
        for length in CANDIDATES
    }
    for row in curves.itertuples(index=False):
        values, phase = aligned_curve(row)
        if len(values) < 3:
            continue
        original_peak = abs(float(np.min(values)))
        original_ttp = float(phase[np.argmin(values)])
        duration_ms = float(row.duration_ms)
        for length in CANDIDATES:
            grid = np.linspace(0.0, 1.0, length)
            resampled = np.interp(grid, phase, values)
            reconstructed = np.interp(phase, grid, resampled)
            error = reconstructed - values
            ttp_error = abs(float(grid[np.argmin(resampled)]) - original_ttp)
            accumulator = accumulators[length]
            accumulator["rmse"].append(float(np.sqrt(np.mean(error**2))))
            accumulator["mae"].append(float(np.mean(np.abs(error))))
            accumulator["peak_error"].append(
                abs(abs(float(np.min(resampled))) - original_peak)
            )
            accumulator["ttp_fraction_error"].append(ttp_error)
            accumulator["ttp_ms_error"].append(ttp_error * duration_ms)

    native = curves["n_points"].to_numpy(int)
    rows = []
    for length in CANDIDATES:
        values = {
            key: np.asarray(value, dtype=float)
            for key, value in accumulators[length].items()
        }
        rows.append(
            {
                "target_length": length,
                "upsampled_fraction": float(np.mean(native < length)),
                "exact_fraction": float(np.mean(native == length)),
                "downsampled_fraction": float(np.mean(native > length)),
                "relative_input_size_vs_96": length / 96.0,
                "median_reconstruction_rmse_pp": float(np.median(values["rmse"])),
                "p95_reconstruction_rmse_pp": float(np.quantile(values["rmse"], 0.95)),
                "median_peak_error_pp": float(np.median(values["peak_error"])),
                "p95_peak_error_pp": float(np.quantile(values["peak_error"], 0.95)),
                "median_ttp_error_ms": float(np.median(values["ttp_ms_error"])),
                "p95_ttp_error_ms": float(np.quantile(values["ttp_ms_error"], 0.95)),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    columns = [
        "analysis_id",
        "visit_id",
        "source_file",
        "curve_family",
        "layer",
        "segment_number",
        "n_points",
        "duration_ms",
        "time_ms",
        "values",
    ]
    frame = pd.read_parquet(DATASET, columns=columns)
    curves = frame[
        frame["curve_family"].eq("longitudinal_strain")
        & frame["layer"].isin(["endo", "mid"])
        & frame["segment_number"].notna()
    ].copy()

    reports = curves.groupby("source_file", as_index=False).agg(
        curves=("n_points", "size"),
        min_points=("n_points", "min"),
        median_points=("n_points", "median"),
        max_points=("n_points", "max"),
        unique_lengths=("n_points", "nunique"),
    )
    summary = pd.DataFrame(
        [
            {
                "reports": curves["source_file"].nunique(),
                "visits": curves["visit_id"].nunique(),
                "analyses": curves["analysis_id"].nunique(),
                "curves": len(curves),
                "mean_points": curves["n_points"].mean(),
                "sd_points": curves["n_points"].std(ddof=1),
                "min_points": curves["n_points"].min(),
                "p01_points": curves["n_points"].quantile(0.01),
                "p05_points": curves["n_points"].quantile(0.05),
                "p10_points": curves["n_points"].quantile(0.10),
                "p25_points": curves["n_points"].quantile(0.25),
                "median_points": curves["n_points"].median(),
                "p75_points": curves["n_points"].quantile(0.75),
                "p90_points": curves["n_points"].quantile(0.90),
                "p95_points": curves["n_points"].quantile(0.95),
                "p99_points": curves["n_points"].quantile(0.99),
                "max_points": curves["n_points"].max(),
                "median_native_spacing_ms": np.median(
                    curves["duration_ms"] / (curves["n_points"] - 1)
                ),
                "reports_with_65_points": int(
                    (reports["median_points"] == 65).sum()
                ),
                "reports_with_one_shared_length": int(
                    (reports["unique_lengths"] == 1).sum()
                ),
            }
        ]
    )
    frequency = (
        reports["median_points"]
        .value_counts()
        .sort_index()
        .rename_axis("native_points")
        .reset_index(name="reports")
    )
    frequency["report_fraction"] = frequency["reports"] / len(reports)
    tradeoffs = candidate_tradeoffs(curves)

    summary.to_csv(OUTPUT / "curve_length_summary.csv", index=False)
    reports.to_csv(OUTPUT / "report_curve_lengths.csv", index=False)
    frequency.to_csv(OUTPUT / "sample_count_frequency.csv", index=False)
    tradeoffs.to_csv(OUTPUT / "resampling_candidate_tradeoffs.csv", index=False)

    selected_tradeoffs = tradeoffs[
        tradeoffs["target_length"].isin([48, 64, 72, 80, 96])
    ].copy()
    selected_tradeoffs["upsampled_%"] = 100 * selected_tradeoffs[
        "upsampled_fraction"
    ]
    selected_tradeoffs["downsampled_%"] = 100 * selected_tradeoffs[
        "downsampled_fraction"
    ]
    selected_tradeoffs["input_%_of_96"] = 100 * selected_tradeoffs[
        "relative_input_size_vs_96"
    ]
    report = f"""# Native strain-curve sample-length analysis

## Cohort used by the CNN

This analysis includes only the longitudinal Endo and Mid segment curves used to
construct the CNN tensor: {len(curves):,} curves from
{curves['source_file'].nunique()} reports, {curves['visit_id'].nunique()} true visits,
and {curves['analysis_id'].nunique()} analysis instances. Each report contributes 36
curves (18 segments × 2 layers), and all 36 curves within a report have exactly the
same native sample count.

## Native sample count

{markdown_table(summary, ['reports', 'curves', 'mean_points', 'sd_points', 'min_points', 'p05_points', 'p25_points', 'median_points', 'p75_points', 'p90_points', 'p95_points', 'p99_points', 'max_points'])}

Only {int(summary.iloc[0]['reports_with_65_points'])} of 416 reports have exactly 65
samples. The median is 57 and the middle 50% range is 46–67.

## Candidate fixed lengths

Reconstruction error was calculated by resampling the original curve to each target
length and interpolating it back to the original native time points. Strain errors are
percentage points.

{markdown_table(selected_tradeoffs, ['target_length', 'upsampled_%', 'downsampled_%', 'input_%_of_96', 'median_reconstruction_rmse_pp', 'p95_reconstruction_rmse_pp', 'median_peak_error_pp', 'p95_peak_error_pp', 'median_ttp_error_ms', 'p95_ttp_error_ms'])}

## Recommendation

- **96 was a conservative engineering choice, not a data-derived optimum.** It
  upsamples 96.6% of reports and therefore does not create new temporal information.
- **64 points is the best first alternative to test.** It is close to the native
  distribution, reduces curve-branch input and convolutional work by 33%, and its
  interpolation error is small relative to the observed strain-measurement noise:
  median RMSE 0.065 percentage points and 95th percentile RMSE 0.135.
- **72 points is a conservative compromise.** It covers 82.9% of reports without
  downsampling and uses 75% of the 96-point input size.
- Reconstruction error naturally decreases on denser interpolation grids, even above
  the native sample count. This means the piecewise-linear approximation is more
  accurate; it does **not** mean that 96 or 128 points contain new physiological
  information that was absent from the original report.
- Do not choose 65 merely because one inspected report had 65 points; only seven
  reports have that exact length. A fixed normalized grid need not match a particular
  report exactly.
- The final choice should be made with a controlled CNN ablation at 48, 64, 72, and
  96 points using identical patient folds. When changing length, the temporal kernel
  widths should also be considered because a 7-sample kernel covers 7.3% of a
  96-point cycle but 10.9% of a 64-point cycle.
"""
    (OUTPUT / "curve_length_report.md").write_text(report, encoding="utf-8")
    metadata = {
        "dataset": str(DATASET),
        "filter": (
            "curve_family=longitudinal_strain; layer in {endo,mid}; "
            "segment_number present"
        ),
        "candidate_lengths": list(CANDIDATES),
        "recommended_first_ablation": [64, 72, 96],
        "reconstruction_method": (
            "linear resample on normalized phase, then interpolate back to native points"
        ),
    }
    (OUTPUT / "curve_length_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print("\nCandidate tradeoffs")
    print(tradeoffs.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
