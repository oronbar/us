"""
Run Ichilov pipeline5 from a YAML configuration file.

Config path:
  - default: ichilov_pipeline5.yaml
  - override with env var: ICHILOV_PIPELINE5_CONFIG

Pipeline stages:
  1) validate_reports
  2) prepare_curve_dataset
  3) encode_phase_frames
  4) train_strain_curve_model
  5) evaluate_strain_curve_model
  6) optional_quality_report
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required. Install with: python -m pip install pyyaml") from exc


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("ichilov_pipeline5")

CONFIG_ENV = "ICHILOV_PIPELINE5_CONFIG"
DEFAULT_CONFIG_NAME = "ichilov_pipeline5.yaml"
DEFAULT_EXPERIMENTS_ROOT = Path(r"C:\Users\Oron\OneDrive - Technion\Experiments")


def _unique_dir(path: Path) -> Path:
    if not path.exists():
        return path
    for idx in range(1, 1000):
        candidate = path.parent / f"{path.name}_{idx}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find unique directory for {path}")


def _latest_prefixed_dir(root: Path, prefix: str) -> Optional[Path]:
    if not root.exists():
        return None
    candidates = [p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _latest_file(root: Path, pattern: str) -> Optional[Path]:
    if not root.exists():
        return None
    candidates = [p for p in root.rglob(pattern) if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _expand_path(value: Optional[Any]) -> Optional[Path]:
    if value is None:
        return None
    if isinstance(value, Path):
        s = str(value)
    else:
        s = str(value).strip()
    if not s or s.lower() in {"none", "null", "auto"}:
        return None
    return Path(os.path.expandvars(s)).expanduser()


def _stringify_paths(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _stringify_paths(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_stringify_paths(v) for v in obj]
    return obj


def _resolve_config_path() -> Path:
    env_value = os.environ.get(CONFIG_ENV)
    if env_value:
        return Path(os.path.expandvars(env_value)).expanduser()
    return Path(__file__).resolve().with_name(DEFAULT_CONFIG_NAME)


def _load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Pipeline config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML config at {path} (expected mapping).")
    return data


def _step_enabled(cfg: Dict[str, Any]) -> bool:
    if "run" in cfg:
        return bool(cfg["run"])
    if "skip" in cfg:
        return not bool(cfg["skip"])
    return True


def _parse_bool_flags(step_cfg: Dict[str, Any]) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
    flags: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
    raw = step_cfg.get("bool_flags", {})
    if not isinstance(raw, dict):
        return flags
    for key, value in raw.items():
        if isinstance(value, (list, tuple)) and len(value) == 2:
            flags[key] = (value[0] or None, value[1] or None)
        elif isinstance(value, str):
            flags[key] = (value, None)
    return flags


def _merge_bool_flags(
    defaults: Dict[str, Tuple[Optional[str], Optional[str]]],
    overrides: Dict[str, Tuple[Optional[str], Optional[str]]],
) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
    out = dict(defaults)
    out.update(overrides)
    return out


def _normalize_args(raw: Dict[str, Any], path_keys: Iterable[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, str) and not value.strip():
            value = None
        if key in path_keys:
            value = _expand_path(value)
        out[key] = value
    return out


def _merged_args(defaults: Dict[str, Any], step_cfg: Dict[str, Any], path_keys: Iterable[str]) -> Dict[str, Any]:
    merged = dict(defaults)
    merged.update(step_cfg.get("args", {}) or {})
    return _normalize_args(merged, path_keys)


def _build_cmd(
    script_path: Path,
    args: Dict[str, Any],
    bool_flags: Dict[str, Tuple[Optional[str], Optional[str]]],
    python_exe: Path,
) -> List[str]:
    cmd = [str(python_exe), str(script_path)]
    for key, value in args.items():
        if value is None:
            continue
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            true_flag, false_flag = bool_flags.get(key, (flag, None))
            if value:
                if true_flag:
                    cmd.append(true_flag)
            else:
                if false_flag:
                    cmd.append(false_flag)
            continue
        cmd.extend([flag, str(value)])
    return cmd


def _step_script(step_cfg: Dict[str, Any], default_script: str, project_root: Path) -> Path:
    raw = step_cfg.get("script")
    script_path = _expand_path(raw) if raw is not None else None
    if script_path is None:
        script_path = project_root / default_script
    elif not script_path.is_absolute():
        script_path = project_root / script_path
    return script_path


def _ensure_script_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} script not found: {path}")


def _run_step(label: str, cmd: List[str]) -> None:
    logger.info("Running %s: %s", label, " ".join(cmd))
    subprocess.run(cmd, check=True)


def _require_artifact(path: Optional[Path], label: str) -> Path:
    if path is None:
        raise FileNotFoundError(f"Missing required artifact for {label}. Configure use_if_skipped or run upstream step.")
    return path


def main() -> None:
    if len(sys.argv) > 1:
        logger.warning("Command-line arguments are ignored. Configure via YAML or %s.", CONFIG_ENV)

    config_path = _resolve_config_path()
    config = _load_config(config_path)
    project_root = Path(__file__).resolve().parent

    pipeline_cfg = config.get("pipeline", {})
    paths_cfg = config.get("paths", {})
    reports_cfg = config.get("reports", {})
    data_cfg = config.get("data", {})
    encoder_cfg = config.get("encoder", {})
    train_cfg = config.get("train", {})
    cv_cfg = config.get("cv", {})
    steps_cfg = config.get("steps", {})
    if not isinstance(steps_cfg, dict) or not steps_cfg:
        raise ValueError("Missing required 'steps' section.")

    prefix = str(pipeline_cfg.get("run_name_prefix") or "ichilov_pipeline5").strip()
    run_name_cfg = str(pipeline_cfg.get("run_name") or "").strip()
    run_name = run_name_cfg or f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    continue_last_run = bool(pipeline_cfg.get("continue_last_run", False))
    experiments_root = _expand_path(paths_cfg.get("experiments_root")) or DEFAULT_EXPERIMENTS_ROOT
    experiments_root.mkdir(parents=True, exist_ok=True)

    if continue_last_run:
        if run_name_cfg:
            run_dir = experiments_root / run_name_cfg
            if not run_dir.exists():
                raise FileNotFoundError(f"continue_last_run=true but run folder does not exist: {run_dir}")
        else:
            latest = _latest_prefixed_dir(experiments_root, prefix)
            if latest is None:
                raise FileNotFoundError(f"No previous run found under {experiments_root} with prefix {prefix}")
            run_dir = latest
        logger.info("Continuing existing pipeline run folder: %s", run_dir)
    else:
        run_dir = _unique_dir(experiments_root / run_name)
        run_dir.mkdir(parents=True, exist_ok=True)
    run_name = run_dir.name

    python_exe = _expand_path(pipeline_cfg.get("python_exe")) or Path(sys.executable)
    if pipeline_cfg.get("save_config_copy", True):
        config_copy = run_dir / "config.yaml"
        if continue_last_run and config_copy.exists():
            config_copy = run_dir / f"config_resume_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        shutil.copy2(config_path, config_copy)

    column_map_json = json.dumps(reports_cfg.get("column_map", {}) or {})
    ed_es_xlsx = _expand_path(paths_cfg.get("ed_es_xlsx"))
    strain_xlsx = _expand_path(paths_cfg.get("strain_xlsx"))
    echo_root = _expand_path(paths_cfg.get("echo_root"))
    cropped_root = _expand_path(paths_cfg.get("cropped_root"))
    views = str(data_cfg.get("views") or "A2C,A3C,A4C")

    artifacts: Dict[str, Optional[Path]] = {
        "prepared_dataset_path": None,
        "phase_embeddings_path": None,
        "best_checkpoint_path": None,
        "predictions_path": None,
        "metrics_path": None,
        "quality_report_path": None,
    }

    resolved = deepcopy(config)
    resolved.setdefault("pipeline", {})
    resolved["pipeline"].update(
        {
            "run_name": run_name,
            "run_dir": str(run_dir),
            "config_path": str(config_path),
            "python_exe": str(python_exe),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "continue_last_run": continue_last_run,
        }
    )

    # 1) validate_reports
    step_name = "validate_reports"
    step_cfg = steps_cfg.get(step_name, {})
    step_run = _step_enabled(step_cfg)
    defaults = {
        "ed_es_xlsx": ed_es_xlsx,
        "strain_xlsx": strain_xlsx,
        "output_dir": run_dir / step_name,
        "column_map_json": column_map_json,
        "views": views,
    }
    args = _merged_args(defaults, step_cfg, path_keys=("ed_es_xlsx", "strain_xlsx", "output_dir"))
    if step_run:
        script = _step_script(step_cfg, "ichilov_validate_strain_reports.py", project_root)
        _ensure_script_exists(script, step_name)
        _run_step(step_name, _build_cmd(script, args, _parse_bool_flags(step_cfg), python_exe))
    resolved.setdefault("steps", {}).setdefault(step_name, {})
    resolved["steps"][step_name]["run"] = step_run
    resolved["steps"][step_name]["args"] = _stringify_paths(args)

    # 2) prepare_curve_dataset
    step_name = "prepare_curve_dataset"
    step_cfg = steps_cfg.get(step_name, {})
    step_run = _step_enabled(step_cfg)
    prepare_dir = run_dir / "curve_dataset"
    defaults = {
        "strain_xlsx": strain_xlsx,
        "ed_es_xlsx": ed_es_xlsx,
        "echo_root": echo_root,
        "cropped_root": cropped_root,
        "output_parquet": prepare_dir / "prepared_strain_curve_dataset.parquet",
        "output_csv": prepare_dir / "prepared_strain_curve_dataset_preview.csv",
        "output_dir": prepare_dir,
        "column_map_json": column_map_json,
        "curve_length": int(data_cfg.get("curve_length", 64)),
        "views": views,
    }
    args = _merged_args(defaults, step_cfg, path_keys=("strain_xlsx", "ed_es_xlsx", "echo_root", "cropped_root", "output_parquet", "output_csv", "output_dir"))
    if step_run:
        script = _step_script(step_cfg, "ichilov_prepare_strain_curve_dataset.py", project_root)
        _ensure_script_exists(script, step_name)
        _run_step(step_name, _build_cmd(script, args, _parse_bool_flags(step_cfg), python_exe))
        artifacts["prepared_dataset_path"] = args["output_parquet"]
    else:
        artifacts["prepared_dataset_path"] = _expand_path(step_cfg.get("use_if_skipped"))
        logger.info("Skipping %s. Using: %s", step_name, artifacts["prepared_dataset_path"])
    resolved["steps"].setdefault(step_name, {})
    resolved["steps"][step_name]["run"] = step_run
    resolved["steps"][step_name]["args"] = _stringify_paths(args)

    # 3) encode_phase_frames
    step_name = "encode_phase_frames"
    step_cfg = steps_cfg.get(step_name, {})
    step_run = _step_enabled(step_cfg)
    encode_dir = run_dir / "phase_embeddings"
    defaults = {
        "input_parquet": artifacts["prepared_dataset_path"],
        "echo_root": echo_root,
        "cropped_root": cropped_root,
        "weights": _expand_path(encoder_cfg.get("weights")),
        "output_parquet": encode_dir / "phase_frame_embeddings.parquet",
        "output_dir": encode_dir,
        "backbone_name": encoder_cfg.get("backbone_name", "vit_small_patch14_dinov2.lvd142m"),
        "image_size": int(data_cfg.get("image_size", encoder_cfg.get("image_size", 518))),
        "t_frames": int(data_cfg.get("t_frames", 16)),
        "phase_sampling_mode": data_cfg.get("phase_sampling_mode", "ed_es_cycle"),
        "batch_size": int(encoder_cfg.get("batch_size", 16)),
        "num_workers": int(encoder_cfg.get("num_workers", 0)),
        "use_amp": bool(encoder_cfg.get("use_amp", True)),
        "freeze_backbone": bool(encoder_cfg.get("freeze_backbone", True)),
    }
    args = _merged_args(defaults, step_cfg, path_keys=("input_parquet", "echo_root", "cropped_root", "weights", "output_parquet", "output_dir"))
    if step_run:
        args["input_parquet"] = _require_artifact(args.get("input_parquet"), step_name)
        script = _step_script(step_cfg, "ichilov_encode_phase_frames_dinov2.py", project_root)
        _ensure_script_exists(script, step_name)
        bool_flags = _merge_bool_flags(
            {
                "use_amp": ("--use-amp", "--no-use-amp"),
                "freeze_backbone": ("--freeze-backbone", "--no-freeze-backbone"),
            },
            _parse_bool_flags(step_cfg),
        )
        _run_step(step_name, _build_cmd(script, args, bool_flags, python_exe))
        artifacts["phase_embeddings_path"] = args["output_parquet"]
    else:
        artifacts["phase_embeddings_path"] = _expand_path(step_cfg.get("use_if_skipped"))
        logger.info("Skipping %s. Using: %s", step_name, artifacts["phase_embeddings_path"])
    resolved["steps"].setdefault(step_name, {})
    resolved["steps"][step_name]["run"] = step_run
    resolved["steps"][step_name]["args"] = _stringify_paths(args)

    # 4) train_strain_curve_model
    step_name = "train_strain_curve_model"
    step_cfg = steps_cfg.get(step_name, {})
    step_run = _step_enabled(step_cfg)
    train_dir = run_dir / "strain_curve_model"
    cv_seeds = cv_cfg.get("seeds", [])
    if isinstance(cv_seeds, list):
        cv_seeds_arg = ",".join(str(s) for s in cv_seeds)
    else:
        cv_seeds_arg = str(cv_seeds)
    defaults = {
        "input_parquet": artifacts["phase_embeddings_path"],
        "output_dir": train_dir,
        "model_type": train_cfg.get("model_type", "temporal_transformer"),
        "batch_size": int(train_cfg.get("batch_size", 16)),
        "epochs": int(train_cfg.get("epochs", 50)),
        "lr": train_cfg.get("lr", 1e-4),
        "weight_decay": train_cfg.get("weight_decay", 0.01),
        "val_ratio": train_cfg.get("val_ratio", 0.2),
        "seed": int(train_cfg.get("seed", 42)),
        "device": train_cfg.get("device", "auto"),
        "early_stopping_patience": int(train_cfg.get("early_stopping_patience", 10)),
        "curve_loss": train_cfg.get("curve_loss", "huber"),
        "curve_loss_weight": train_cfg.get("curve_loss_weight", 1.0),
        "peak_loss_weight": train_cfg.get("peak_loss_weight", 0.5),
        "ttp_loss_weight": train_cfg.get("ttp_loss_weight", 0.1),
        "derivative_loss_weight": train_cfg.get("derivative_loss_weight", 0.2),
        "smoothness_loss_weight": train_cfg.get("smoothness_loss_weight", 0.01),
        "use_amp": bool(train_cfg.get("use_amp", True)),
        "cv_run": bool(cv_cfg.get("run", False)),
        "cv_seeds": cv_seeds_arg,
    }
    for optional_key in ("hidden_dim", "num_layers", "num_heads", "dropout", "num_workers", "diagnostic_samples", "derive_predicted_peak_from_curve"):
        if optional_key in train_cfg:
            defaults[optional_key] = train_cfg[optional_key]
    args = _merged_args(defaults, step_cfg, path_keys=("input_parquet", "output_dir"))
    if step_run:
        args["input_parquet"] = _require_artifact(args.get("input_parquet"), step_name)
        script = _step_script(step_cfg, "train_strain_curve_model.py", project_root)
        _ensure_script_exists(script, step_name)
        bool_flags = _merge_bool_flags(
            {
                "use_amp": ("--use-amp", "--no-use-amp"),
                "cv_run": ("--cv-run", None),
                "derive_predicted_peak_from_curve": ("--derive-predicted-peak-from-curve", None),
            },
            _parse_bool_flags(step_cfg),
        )
        _run_step(step_name, _build_cmd(script, args, bool_flags, python_exe))
        best = Path(args["output_dir"]) / "strain_curve_model_best.pt"
        if not best.exists():
            best = _latest_file(Path(args["output_dir"]), "*best*.pt") or best
        artifacts["best_checkpoint_path"] = best
    else:
        artifacts["best_checkpoint_path"] = _expand_path(step_cfg.get("use_if_skipped"))
        logger.info("Skipping %s. Using: %s", step_name, artifacts["best_checkpoint_path"])
    resolved["steps"].setdefault(step_name, {})
    resolved["steps"][step_name]["run"] = step_run
    resolved["steps"][step_name]["args"] = _stringify_paths(args)

    # 5) evaluate_strain_curve_model
    step_name = "evaluate_strain_curve_model"
    step_cfg = steps_cfg.get(step_name, {})
    step_run = _step_enabled(step_cfg)
    eval_dir = run_dir / "evaluation"
    defaults = {
        "input_parquet": artifacts["phase_embeddings_path"],
        "checkpoint": artifacts["best_checkpoint_path"],
        "output_dir": eval_dir,
        "batch_size": int(train_cfg.get("batch_size", 16)),
        "device": train_cfg.get("device", "auto"),
    }
    args = _merged_args(defaults, step_cfg, path_keys=("input_parquet", "checkpoint", "output_dir"))
    if step_run:
        args["input_parquet"] = _require_artifact(args.get("input_parquet"), step_name)
        args["checkpoint"] = _require_artifact(args.get("checkpoint"), step_name)
        script = _step_script(step_cfg, "evaluate_strain_curve_model.py", project_root)
        _ensure_script_exists(script, step_name)
        _run_step(step_name, _build_cmd(script, args, _parse_bool_flags(step_cfg), python_exe))
        artifacts["predictions_path"] = Path(args["output_dir"]) / "predictions.parquet"
        artifacts["metrics_path"] = Path(args["output_dir"]) / "metrics.json"
    else:
        logger.info("Skipping %s.", step_name)
    resolved["steps"].setdefault(step_name, {})
    resolved["steps"][step_name]["run"] = step_run
    resolved["steps"][step_name]["args"] = _stringify_paths(args)

    # 6) optional_quality_report
    step_name = "optional_quality_report"
    step_cfg = steps_cfg.get(step_name, {})
    step_run = _step_enabled(step_cfg)
    quality_dir = run_dir / "quality_report"
    defaults = {
        "prepared_parquet": artifacts["prepared_dataset_path"],
        "predictions_parquet": artifacts["predictions_path"],
        "output_dir": quality_dir,
    }
    args = _merged_args(defaults, step_cfg, path_keys=("prepared_parquet", "predictions_parquet", "output_dir"))
    if step_run:
        args["prepared_parquet"] = _require_artifact(args.get("prepared_parquet"), step_name)
        script = _step_script(step_cfg, "ichilov_strain_quality_report.py", project_root)
        _ensure_script_exists(script, step_name)
        _run_step(step_name, _build_cmd(script, args, _parse_bool_flags(step_cfg), python_exe))
        artifacts["quality_report_path"] = Path(args["output_dir"]) / "quality_report.md"
    else:
        logger.info("Skipping %s.", step_name)
    resolved["steps"].setdefault(step_name, {})
    resolved["steps"][step_name]["run"] = step_run
    resolved["steps"][step_name]["args"] = _stringify_paths(args)

    resolved["artifacts"] = _stringify_paths(artifacts)
    resolved_path = run_dir / "resolved_config.yaml"
    with resolved_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(_stringify_paths(resolved), handle, sort_keys=False)
    if continue_last_run:
        resume_path = run_dir / f"resolved_config_resume_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        with resume_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(_stringify_paths(resolved), handle, sort_keys=False)

    logger.info("Pipeline5 complete. Run name: %s", run_name)
    logger.info("Run folder: %s", run_dir)
    logger.info("Prepared dataset: %s", artifacts["prepared_dataset_path"])
    logger.info("Phase embeddings: %s", artifacts["phase_embeddings_path"])
    logger.info("Best checkpoint: %s", artifacts["best_checkpoint_path"])
    logger.info("Predictions: %s", artifacts["predictions_path"])
    logger.info("Metrics: %s", artifacts["metrics_path"])
    logger.info("Quality report: %s", artifacts["quality_report_path"])


if __name__ == "__main__":
    main()
