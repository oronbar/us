"""
Repeated patient-level split evaluation for Ichilov pipeline3.

This script runs short training jobs across K group-aware splits and aggregates
mean/std metrics per (variant, sampling_mode).
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

if __package__ is None or __package__ == "":  # pragma: no cover - script execution path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Ichilov_pipeline3.datasets.visit_dataset import FrameEmbeddingVisitDataset, VisitDataset

try:
    from sklearn.model_selection import GroupShuffleSplit
except Exception:  # pragma: no cover - optional dependency
    GroupShuffleSplit = None  # type: ignore


def _load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def _extract_step_args(cfg: Dict) -> Dict:
    steps = cfg.get("steps", {}) if isinstance(cfg.get("steps"), dict) else {}
    node = steps.get("longitudinal_train", {}) if isinstance(steps.get("longitudinal_train"), dict) else {}
    args = node.get("args", {}) if isinstance(node.get("args"), dict) else {}
    return dict(args)


def _extract_cfg_value(cfg: Dict, key: str, default: str = "") -> str:
    args = _extract_step_args(cfg)
    if key in args and args.get(key) is not None:
        return str(args.get(key))
    paths = cfg.get("paths", {}) if isinstance(cfg.get("paths"), dict) else {}
    if key in paths and paths.get(key) is not None:
        return str(paths.get(key))
    return default


def _variant_registry() -> Dict[str, Dict[str, str]]:
    return {
        "baseline": {
            "temporal_mode": "attn_pool",
            "view_fusion_mode": "attn",
            "longitudinal_ablation_mode": "inherit",
            "time_encoding_mode": "inherit",
        },
        "no_time_encoding": {"time_encoding_mode": "off"},
        "long_last_visit_linear": {"longitudinal_ablation_mode": "last_visit_linear"},
        "temporal_mean_pool": {"temporal_mode": "mean_pool"},
        "view_mean": {"view_fusion_mode": "mean"},
    }


def _parse_csv_list(raw: str) -> List[str]:
    out: List[str] = []
    for token in str(raw).split(","):
        t = token.strip()
        if t:
            out.append(t)
    return out


def _build_dataset_for_splits(
    cfg: Dict,
    input_embeddings: str,
    report_xlsx: str,
    sampling_mode: str,
    phase_aligned_include_midpoints: bool,
    phase_aligned_strategy: str,
    max_visits: int,
    min_visits: int,
    t_frames: int,
    clip_stride: int,
    include_last_window: bool,
    risk_threshold: float,
    delta_gls_convention: str,
) -> object:
    views = _extract_cfg_value(cfg, "views", "")
    if input_embeddings.strip():
        return FrameEmbeddingVisitDataset(
            input_embeddings=Path(input_embeddings),
            report_xlsx=Path(report_xlsx) if report_xlsx.strip() else None,
            views=views,
            t_frames=t_frames,
            sampling_mode=sampling_mode,
            clip_stride=clip_stride,
            include_last_window=include_last_window,
            phase_aligned_include_midpoints=phase_aligned_include_midpoints,
            phase_aligned_strategy=phase_aligned_strategy,
            max_visits=max_visits,
            min_visits=min_visits,
            risk_delta_threshold=risk_threshold,
            delta_gls_convention=delta_gls_convention,
        )
    input_xlsx = _extract_cfg_value(cfg, "input_xlsx", "")
    echo_root = _extract_cfg_value(cfg, "echo_root", "")
    cropped_root = _extract_cfg_value(cfg, "cropped_root", "")
    if not input_xlsx or not echo_root:
        raise ValueError("Raw-frame dataset requires input_xlsx and echo_root in config.")
    return VisitDataset(
        input_xlsx=Path(input_xlsx),
        echo_root=Path(echo_root),
        cropped_root=Path(cropped_root) if cropped_root else None,
        views=views,
        t_frames=t_frames,
        sampling_mode=sampling_mode,
        clip_stride=clip_stride,
        include_last_window=include_last_window,
        phase_aligned_include_midpoints=phase_aligned_include_midpoints,
        phase_aligned_strategy=phase_aligned_strategy,
        max_visits=max_visits,
        min_visits=min_visits,
        risk_delta_threshold=risk_threshold,
        delta_gls_convention=delta_gls_convention,
        random_view_sampling=False,
    )


def _split_patient_ids(
    patient_ids: Sequence[str],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    n = len(patient_ids)
    if n == 0:
        return [], [], []
    idx = np.arange(n, dtype=np.int64)
    groups = np.asarray([str(x) for x in patient_ids])
    holdout_ratio = float(val_ratio + test_ratio)
    if holdout_ratio <= 0.0:
        return list(groups.tolist()), [], []

    if GroupShuffleSplit is None:
        rng = np.random.default_rng(int(seed))
        shuffled = idx.copy()
        rng.shuffle(shuffled)
        n_hold = int(round(n * holdout_ratio))
        hold_idx = shuffled[:n_hold]
        train_idx = shuffled[n_hold:]
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=holdout_ratio, random_state=int(seed))
        train_idx, hold_idx = next(gss.split(idx, groups=groups))
    train_ids = groups[train_idx].tolist()
    hold_ids = groups[hold_idx].tolist()
    if not hold_ids:
        return train_ids, [], []

    if test_ratio <= 0:
        return train_ids, hold_ids, []
    test_frac_in_hold = float(test_ratio / max(holdout_ratio, 1e-12))
    hold_idx_arr = np.arange(len(hold_ids), dtype=np.int64)
    hold_groups = np.asarray(hold_ids)
    if GroupShuffleSplit is None:
        rng = np.random.default_rng(int(seed) + 1)
        shuffled = hold_idx_arr.copy()
        rng.shuffle(shuffled)
        n_test = int(round(len(hold_ids) * test_frac_in_hold))
        test_sub = shuffled[:n_test]
        val_sub = shuffled[n_test:]
    else:
        gss2 = GroupShuffleSplit(n_splits=1, test_size=test_frac_in_hold, random_state=int(seed) + 1)
        val_sub, test_sub = next(gss2.split(hold_idx_arr, groups=hold_groups))
    val_ids = hold_groups[val_sub].tolist()
    test_ids = hold_groups[test_sub].tolist()
    return train_ids, val_ids, test_ids


def _build_train_cmd(
    python_exe: str,
    train_script: Path,
    config_path: Path,
    run_name: str,
    run_dir: Path,
    split_json: Path,
    epochs: int,
    seed: int,
    risk_threshold: float,
    input_embeddings: str,
    report_xlsx: str,
    device: str,
    num_workers: int,
    batch_size: int,
    sampling_mode: str,
    phase_aligned_include_midpoints: bool,
    phase_aligned_strategy: str,
    delta_gls_convention: str,
    variant_overrides: Dict[str, str],
    enable_diagnostics: bool,
    debug_delta: bool,
) -> List[str]:
    out_dir = run_dir / "outputs"
    log_dir = run_dir / "tensorboard"
    pred_path = run_dir / f"{run_name}_predictions.parquet"
    summary_path = run_dir / f"{run_name}_summary.json"
    cmd = [
        python_exe,
        str(train_script),
        "--config",
        str(config_path),
        "--run-name",
        run_name,
        "--output-dir",
        str(out_dir),
        "--log-dir",
        str(log_dir),
        "--output-parquet",
        str(pred_path),
        "--summary-json",
        str(summary_path),
        "--epochs",
        str(int(epochs)),
        "--seed",
        str(int(seed)),
        "--risk-delta-threshold",
        str(float(risk_threshold)),
        "--device",
        str(device),
        "--num-workers",
        str(int(num_workers)),
        "--batch-size",
        str(int(batch_size)),
        "--sampling-mode",
        str(sampling_mode),
        "--phase-aligned-strategy",
        str(phase_aligned_strategy),
        "--delta-gls-convention",
        str(delta_gls_convention),
        "--split-json",
        str(split_json),
    ]
    if phase_aligned_include_midpoints:
        cmd.append("--phase-aligned-include-midpoints")
    else:
        cmd.append("--no-phase-aligned-include-midpoints")
    cmd.append("--enable-diagnostics" if enable_diagnostics else "--no-enable-diagnostics")
    cmd.append("--debug-delta" if debug_delta else "--no-debug-delta")
    if input_embeddings.strip():
        cmd.extend(["--input-embeddings", input_embeddings])
    if report_xlsx.strip():
        cmd.extend(["--report-xlsx", report_xlsx])
    for key, val in variant_overrides.items():
        cmd.extend(["--" + key.replace("_", "-"), str(val)])
    return cmd


def _flatten_summary(summary: Dict, variant: str, sampling_mode: str, split_id: int, split_seed: int) -> Dict[str, object]:
    split_val = summary.get("split_metrics", {}).get("val", {})
    return {
        "variant": variant,
        "sampling_mode": sampling_mode,
        "split_id": int(split_id),
        "split_seed": int(split_seed),
        "status": "ok",
        "best_epoch": summary.get("best_epoch"),
        "best_val_loss": summary.get("best_val_loss"),
        "best_val_delta_mae": summary.get("best_val_delta_mae"),
        "val_delta_mae": split_val.get("delta_mae"),
        "val_delta_pearson": split_val.get("delta_pearson"),
        "val_delta_spearman": split_val.get("delta_spearman"),
        "val_corr_signflip_advantage": split_val.get("corr_signflip_advantage"),
        "val_risk_auc": split_val.get("risk_auc"),
    }


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "best_val_loss",
        "best_val_delta_mae",
        "val_delta_mae",
        "val_delta_pearson",
        "val_delta_spearman",
        "val_corr_signflip_advantage",
        "val_risk_auc",
    ]
    rows: List[Dict[str, object]] = []
    if df.empty:
        return pd.DataFrame(rows)
    for (variant, sampling_mode), grp in df.groupby(["variant", "sampling_mode"], dropna=False):
        row: Dict[str, object] = {
            "variant": variant,
            "sampling_mode": sampling_mode,
            "n_runs": int(len(grp)),
        }
        for m in metrics:
            vals = pd.to_numeric(grp[m], errors="coerce").to_numpy(dtype=float)
            row[f"{m}_mean"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
            row[f"{m}_std"] = float(np.nanstd(vals, ddof=0)) if np.isfinite(vals).any() else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty and "best_val_delta_mae_mean" in out.columns:
        out = out.sort_values(
            ["best_val_delta_mae_mean", "best_val_loss_mean"],
            ascending=[True, True],
        )
    return out


def _write_markdown(df: pd.DataFrame, out_md: Path) -> None:
    cols = [
        "variant",
        "sampling_mode",
        "n_runs",
        "best_val_delta_mae_mean",
        "best_val_delta_mae_std",
        "val_delta_pearson_mean",
        "val_delta_pearson_std",
        "val_delta_spearman_mean",
        "val_delta_spearman_std",
        "val_corr_signflip_advantage_mean",
        "val_corr_signflip_advantage_std",
    ]
    use = [c for c in cols if c in df.columns]
    lines = ["# Pipeline3 CV Summary", "", "| " + " | ".join(use) + " |", "| " + " | ".join(["---"] * len(use)) + " |"]
    for _, row in df[use].iterrows():
        vals = []
        for c in use:
            v = row[c]
            if isinstance(v, float):
                vals.append("" if pd.isna(v) else f"{v:.6f}")
            else:
                vals.append("" if pd.isna(v) else str(v))
        lines.append("| " + " | ".join(vals) + " |")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repeated patient-level CV for pipeline3.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().with_name("config_pipeline3.yaml"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"C:\Users\Oron\OneDrive - Technion\Experiments\DinoPipeline_21\cv_suite_pipeline3"),
    )
    parser.add_argument("--k-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--python-exe", type=str, default="")
    parser.add_argument(
        "--variants",
        type=str,
        default="baseline,no_time_encoding,long_last_visit_linear,temporal_mean_pool,view_mean",
    )
    parser.add_argument("--sampling-modes", type=str, default="uniform,phase_aligned")
    parser.add_argument("--input-embeddings", type=str, default="")
    parser.add_argument("--report-xlsx", type=str, default="")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--risk-threshold", type=float, default=2.0)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--test-ratio", type=float, default=None)
    parser.add_argument("--max-runs", type=int, default=0, help="If >0, stop after N runs.")
    parser.add_argument("--phase-aligned-include-midpoints", dest="phase_aligned_include_midpoints", action="store_true")
    parser.add_argument("--no-phase-aligned-include-midpoints", dest="phase_aligned_include_midpoints", action="store_false")
    parser.set_defaults(phase_aligned_include_midpoints=True)
    parser.add_argument("--phase-aligned-strategy", type=str, choices=["cycle", "segment"], default="cycle")
    parser.add_argument(
        "--delta-gls-convention",
        type=str,
        choices=["latest_minus_earliest", "earliest_minus_latest"],
        default="latest_minus_earliest",
    )
    parser.add_argument("--enable-diagnostics", dest="enable_diagnostics", action="store_true")
    parser.add_argument("--no-enable-diagnostics", dest="enable_diagnostics", action="store_false")
    parser.set_defaults(enable_diagnostics=False)
    parser.add_argument("--debug-delta", dest="debug_delta", action="store_true")
    parser.add_argument("--no-debug-delta", dest="debug_delta", action="store_false")
    parser.set_defaults(debug_delta=True)
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    steps_args = _extract_step_args(cfg)
    run_prefix = str((cfg.get("pipeline", {}) or {}).get("run_name_prefix") or "pipeline3")
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_dir = args.output_root / f"{run_prefix}_cv_{run_stamp}"
    suite_dir.mkdir(parents=True, exist_ok=True)

    input_embeddings = args.input_embeddings.strip() or _extract_cfg_value(cfg, "input_embeddings", "")
    report_xlsx = args.report_xlsx.strip() or _extract_cfg_value(cfg, "report_xlsx", "")
    max_visits = int(steps_args.get("max_visits", 5))
    min_visits = int(steps_args.get("min_visits", 2))
    t_frames = int(steps_args.get("t_frames", 16))
    clip_stride = int(steps_args.get("clip_stride", 1))
    include_last_window = bool(steps_args.get("include_last_window", True))
    val_ratio = float(args.val_ratio if args.val_ratio is not None else steps_args.get("val_ratio", 0.2))
    test_ratio = float(args.test_ratio if args.test_ratio is not None else steps_args.get("test_ratio", 0.1))

    split_dataset = _build_dataset_for_splits(
        cfg=cfg,
        input_embeddings=input_embeddings,
        report_xlsx=report_xlsx,
        sampling_mode="uniform",
        phase_aligned_include_midpoints=args.phase_aligned_include_midpoints,
        phase_aligned_strategy=args.phase_aligned_strategy,
        max_visits=max_visits,
        min_visits=min_visits,
        t_frames=t_frames,
        clip_stride=clip_stride,
        include_last_window=include_last_window,
        risk_threshold=args.risk_threshold,
        delta_gls_convention=args.delta_gls_convention,
    )
    patient_ids = [str(r.patient_id) for r in getattr(split_dataset, "patient_records")]
    if len(patient_ids) == 0:
        raise RuntimeError("No patients available for CV splits.")

    registry = _variant_registry()
    variant_names = _parse_csv_list(args.variants)
    if not variant_names:
        raise ValueError("No variants requested.")
    for v in variant_names:
        if v not in registry:
            raise ValueError(f"Unknown variant '{v}'. Available: {sorted(registry.keys())}")
    sampling_modes = _parse_csv_list(args.sampling_modes)
    if not sampling_modes:
        raise ValueError("No sampling modes requested.")
    for sm in sampling_modes:
        if sm not in {"uniform", "sliding_window", "phase_aligned"}:
            raise ValueError(f"Unsupported sampling mode '{sm}'")

    train_script = Path(__file__).resolve().with_name("train_pipeline3.py")
    python_exe = args.python_exe.strip() or "python"

    run_records: List[Dict[str, object]] = []
    ok_rows: List[Dict[str, object]] = []
    run_counter = 0
    total_planned = int(args.k_splits) * len(variant_names) * len(sampling_modes)
    for split_id in range(1, int(args.k_splits) + 1):
        split_seed = int(args.seed) + split_id - 1
        train_ids, val_ids, test_ids = _split_patient_ids(
            patient_ids=patient_ids,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=split_seed,
        )
        split_obj = {
            "train_patient_ids": train_ids,
            "val_patient_ids": val_ids,
            "test_patient_ids": test_ids,
        }
        for sampling_mode in sampling_modes:
            for variant_name in variant_names:
                run_counter += 1
                if args.max_runs > 0 and run_counter > int(args.max_runs):
                    break
                run_dir = suite_dir / f"split_{split_id:02d}" / f"{sampling_mode}" / variant_name
                run_dir.mkdir(parents=True, exist_ok=True)
                split_json = run_dir / "split.json"
                split_json.write_text(json.dumps(split_obj, indent=2), encoding="utf-8")
                run_name = f"{run_prefix}_{variant_name}_{sampling_mode}_split{split_id:02d}_{run_stamp}"
                cmd = _build_train_cmd(
                    python_exe=python_exe,
                    train_script=train_script,
                    config_path=args.config,
                    run_name=run_name,
                    run_dir=run_dir,
                    split_json=split_json,
                    epochs=args.epochs,
                    seed=split_seed,
                    risk_threshold=args.risk_threshold,
                    input_embeddings=input_embeddings,
                    report_xlsx=report_xlsx,
                    device=args.device,
                    num_workers=args.num_workers,
                    batch_size=args.batch_size,
                    sampling_mode=sampling_mode,
                    phase_aligned_include_midpoints=args.phase_aligned_include_midpoints,
                    phase_aligned_strategy=args.phase_aligned_strategy,
                    delta_gls_convention=args.delta_gls_convention,
                    variant_overrides=registry[variant_name],
                    enable_diagnostics=args.enable_diagnostics,
                    debug_delta=args.debug_delta,
                )
                print(f"[{run_counter}/{total_planned}] split={split_id} sampling={sampling_mode} variant={variant_name}")
                proc = subprocess.run(cmd, text=True, capture_output=True)
                log_path = run_dir / "stdout_stderr.log"
                log_path.write_text(
                    f"COMMAND:\n{' '.join(cmd)}\n\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}\n",
                    encoding="utf-8",
                )
                summary_path = run_dir / f"{run_name}_summary.json"
                status = "ok" if proc.returncode == 0 and summary_path.exists() else "failed"
                record: Dict[str, object] = {
                    "variant": variant_name,
                    "sampling_mode": sampling_mode,
                    "split_id": split_id,
                    "split_seed": split_seed,
                    "status": status,
                    "return_code": int(proc.returncode),
                    "log_path": str(log_path),
                    "summary_path": str(summary_path),
                }
                run_records.append(record)
                if status != "ok":
                    continue
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                row = _flatten_summary(
                    summary=summary,
                    variant=variant_name,
                    sampling_mode=sampling_mode,
                    split_id=split_id,
                    split_seed=split_seed,
                )
                ok_rows.append(row)
            if args.max_runs > 0 and run_counter >= int(args.max_runs):
                break
        if args.max_runs > 0 and run_counter >= int(args.max_runs):
            break

    runs_df = pd.DataFrame(run_records)
    runs_csv = suite_dir / "cv_runs.csv"
    runs_df.to_csv(runs_csv, index=False)

    per_split_df = pd.DataFrame(ok_rows)
    per_split_csv = suite_dir / "cv_per_split.csv"
    per_split_df.to_csv(per_split_csv, index=False)

    agg_df = _aggregate(per_split_df)
    agg_csv = suite_dir / "cv_summary.csv"
    agg_md = suite_dir / "cv_summary.md"
    agg_df.to_csv(agg_csv, index=False)
    _write_markdown(agg_df, agg_md)

    print(f"Saved run log table: {runs_csv}")
    print(f"Saved per-split metrics: {per_split_csv}")
    print(f"Saved summary: {agg_csv}")
    print(f"Saved markdown: {agg_md}")


if __name__ == "__main__":
    main()
