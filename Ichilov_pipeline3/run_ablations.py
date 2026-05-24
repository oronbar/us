"""
Run a compact ablation grid for Ichilov pipeline3 and aggregate metrics.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml


def _load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping YAML in {path}")
    return data


def _get_run_prefix(cfg: Dict) -> str:
    pipeline = cfg.get("pipeline", {}) if isinstance(cfg.get("pipeline"), dict) else {}
    prefix = str(pipeline.get("run_name_prefix") or "pipeline3")
    return prefix


def _variant_list() -> List[Dict[str, str]]:
    return [
        {"name": "baseline", "temporal_mode": "attn_pool", "view_fusion_mode": "attn", "longitudinal_ablation_mode": "inherit", "time_encoding_mode": "inherit"},
        {"name": "temporal_mean_pool", "temporal_mode": "mean_pool"},
        {"name": "temporal_last_frame", "temporal_mode": "last_frame"},
        {"name": "view_mean", "view_fusion_mode": "mean"},
        {"name": "view_single_best", "view_fusion_mode": "single_best_view", "single_best_view": "A4C"},
        {"name": "view_concat_then_linear", "view_fusion_mode": "concat_then_linear"},
        {"name": "long_last_visit_linear", "longitudinal_ablation_mode": "last_visit_linear"},
        {"name": "long_mean_visit_linear", "longitudinal_ablation_mode": "mean_visit_linear"},
        {"name": "long_delta_only_linear", "longitudinal_ablation_mode": "delta_only_linear"},
        {"name": "no_time_encoding", "time_encoding_mode": "off"},
    ]


def _extract_default_arg(cfg: Dict, key: str) -> str:
    steps = cfg.get("steps", {}) if isinstance(cfg.get("steps"), dict) else {}
    long_cfg = steps.get("longitudinal_train", {}) if isinstance(steps.get("longitudinal_train"), dict) else {}
    args = long_cfg.get("args", {}) if isinstance(long_cfg.get("args"), dict) else {}
    val = args.get(key)
    if val is None:
        paths = cfg.get("paths", {}) if isinstance(cfg.get("paths"), dict) else {}
        val = paths.get(key)
    return str(val).strip() if val is not None else ""


def _build_train_cmd(
    python_exe: str,
    train_script: Path,
    config_path: Path,
    variant: Dict[str, str],
    run_name: str,
    run_dir: Path,
    epochs: int,
    seed: int,
    risk_threshold: float,
    input_embeddings: str,
    report_xlsx: str,
    device: str,
    num_workers: int,
    batch_size: int,
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
        "--enable-diagnostics",
        "--diagnostics-batches",
        "2",
        "--diagnostics-topk",
        "5",
        "--diagnostics-every",
        "1",
    ]
    if input_embeddings.strip():
        cmd.extend(["--input-embeddings", input_embeddings])
    if report_xlsx.strip():
        cmd.extend(["--report-xlsx", report_xlsx])
    for key in (
        "temporal_mode",
        "view_fusion_mode",
        "single_best_view",
        "longitudinal_ablation_mode",
        "time_encoding_mode",
    ):
        val = variant.get(key)
        if val is None:
            continue
        flag = "--" + key.replace("_", "-")
        cmd.extend([flag, str(val)])
    return cmd


def _flatten_summary(variant_name: str, summary: Dict) -> Dict[str, object]:
    split_val = summary.get("split_metrics", {}).get("val", {})
    diag = summary.get("diagnostics", {})
    frame_entropy = (
        diag.get("frame_attention", {}).get("entropy_mean")
        if isinstance(diag, dict)
        else None
    )
    view_entropy = (
        diag.get("view_attention", {}).get("entropy_mean")
        if isinstance(diag, dict)
        else None
    )
    probes = summary.get("probes", {})
    pid_probe = probes.get("patient_id_probe", {}) if isinstance(probes, dict) else {}
    delta_probe = probes.get("delta_gls_probe", {}) if isinstance(probes, dict) else {}
    return {
        "variant": variant_name,
        "best_epoch": summary.get("best_epoch"),
        "best_val_loss": summary.get("best_val_loss"),
        "best_val_delta_mae": summary.get("best_val_delta_mae"),
        "val_delta_mae_from_preds": split_val.get("delta_mae"),
        "val_delta_pearson": split_val.get("delta_pearson"),
        "val_delta_spearman": split_val.get("delta_spearman"),
        "val_corr_signflip_advantage": split_val.get("corr_signflip_advantage"),
        "val_risk_auc": split_val.get("risk_auc"),
        "frame_attention_entropy": frame_entropy,
        "view_attention_entropy": view_entropy,
        "pid_probe_train_acc": pid_probe.get("train_acc"),
        "pid_probe_val_acc": pid_probe.get("val_acc"),
        "delta_probe_baseline_pearson": delta_probe.get("baseline_pearson"),
        "delta_probe_model_pearson": delta_probe.get("model_pearson"),
    }


def _write_markdown(df: pd.DataFrame, out_md: Path) -> None:
    cols = [
        "variant",
        "best_val_loss",
        "best_val_delta_mae",
        "val_delta_pearson",
        "val_delta_spearman",
        "val_corr_signflip_advantage",
        "val_risk_auc",
        "frame_attention_entropy",
        "view_attention_entropy",
        "pid_probe_train_acc",
        "pid_probe_val_acc",
    ]
    use = [c for c in cols if c in df.columns]
    md = "# Pipeline3 Ablation Summary\n\n"
    header = "| " + " | ".join(use) + " |\n"
    sep = "| " + " | ".join(["---"] * len(use)) + " |\n"
    rows = []
    for _, row in df[use].iterrows():
        vals = []
        for c in use:
            v = row[c]
            if isinstance(v, float):
                vals.append("" if pd.isna(v) else f"{v:.6f}")
            else:
                vals.append("" if pd.isna(v) else str(v))
        rows.append("| " + " | ".join(vals) + " |")
    md += header + sep + "\n".join(rows) + "\n"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pipeline3 ablations and aggregate metrics.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().with_name("config_pipeline3.yaml"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"C:\Users\Oron\OneDrive - Technion\Experiments\DinoPipeline_21\ablation_suite_pipeline3"),
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--risk-threshold", type=float, default=2.0)
    parser.add_argument("--python-exe", type=str, default="")
    parser.add_argument("--max-runs", type=int, default=0, help="If > 0, run only first N variants.")
    parser.add_argument("--input-embeddings", type=str, default="")
    parser.add_argument("--report-xlsx", type=str, default="")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    prefix = _get_run_prefix(cfg)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_dir = args.output_root / f"{prefix}_ablations_{run_stamp}"
    suite_dir.mkdir(parents=True, exist_ok=True)

    train_script = Path(__file__).resolve().with_name("train_pipeline3.py")
    python_exe = args.python_exe.strip() or "python"
    input_embeddings = args.input_embeddings.strip() or _extract_default_arg(cfg, "input_embeddings")
    report_xlsx = args.report_xlsx.strip() or _extract_default_arg(cfg, "report_xlsx")

    variants = _variant_list()
    if args.max_runs > 0:
        variants = variants[: int(args.max_runs)]

    summary_rows: List[Dict[str, object]] = []
    run_records: List[Dict[str, object]] = []

    for i, variant in enumerate(variants, start=1):
        name = variant["name"]
        run_name = f"{prefix}_{name}_{run_stamp}"
        run_dir = suite_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = _build_train_cmd(
            python_exe=python_exe,
            train_script=train_script,
            config_path=args.config,
            variant=variant,
            run_name=run_name,
            run_dir=run_dir,
            epochs=args.epochs,
            seed=args.seed,
            risk_threshold=args.risk_threshold,
            input_embeddings=input_embeddings,
            report_xlsx=report_xlsx,
            device=args.device,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
        )
        print(f"[{i}/{len(variants)}] Running {name}")
        proc = subprocess.run(cmd, text=True, capture_output=True)
        log_path = run_dir / "stdout_stderr.log"
        log_path.write_text(
            f"COMMAND:\n{' '.join(cmd)}\n\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}\n",
            encoding="utf-8",
        )
        summary_path = run_dir / f"{run_name}_summary.json"
        status = "ok" if proc.returncode == 0 and summary_path.exists() else "failed"
        record = {
            "variant": name,
            "status": status,
            "return_code": int(proc.returncode),
            "log_path": str(log_path),
            "summary_path": str(summary_path),
        }
        run_records.append(record)
        if status != "ok":
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        row = _flatten_summary(name, summary)
        summary_rows.append(row)

    runs_df = pd.DataFrame(run_records)
    runs_csv = suite_dir / "ablation_runs.csv"
    runs_df.to_csv(runs_csv, index=False)

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        if "best_val_delta_mae" in df.columns:
            df = df.sort_values(["best_val_delta_mae", "best_val_loss"], ascending=[True, True])
        summary_csv = suite_dir / "ablation_summary.csv"
        summary_md = suite_dir / "ablation_summary.md"
        df.to_csv(summary_csv, index=False)
        _write_markdown(df, summary_md)
        print(f"Saved summary: {summary_csv}")
        print(f"Saved markdown: {summary_md}")
    print(f"Saved run log table: {runs_csv}")


if __name__ == "__main__":
    main()
