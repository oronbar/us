"""
Run the Ichilov GLS pipeline (v2) from a YAML configuration file.

Config path (no CLI args):
  - default: ichilov_pipeline2.yaml (next to this file)
  - override with env var: ICHILOV_PIPELINE2_CONFIG

Outputs:
  - Each run writes outputs (except cropped DICOMs) into a timestamped
    folder under paths.experiments_root.

Pipeline stages (default order):
  1) crop           -> create cropped DICOMs (optional)
  2) frame_pretrain -> 2D Frame-MAE pretraining (self-supervised)
  3) frame_encode   -> extract per-frame embeddings
  4) temporal_train -> train temporal model (cine-level)
  5) cine_encode    -> extract per-cine embeddings
  6) visit_fusion   -> aggregate per-cine embeddings into visit embeddings
  7) longitudinal_train -> model patient trajectories over time
"""
from __future__ import annotations

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
except Exception as exc:  # pragma: no cover - runtime check
    raise RuntimeError(
        "PyYAML is required. Install with: .venv\\Scripts\\python -m pip install pyyaml"
    ) from exc


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("ichilov_pipeline2")

CONFIG_ENV = "ICHILOV_PIPELINE2_CONFIG"
DEFAULT_CONFIG_NAME = "ichilov_pipeline2.yaml"

DEFAULT_EXPERIMENTS_ROOT = Path(r"C:\Users\oron\OneDrive - Technion\Experiments")
DEFAULT_CROPPED_ROOT_BASE = Path(r"D:\DS\Ichilov_cropped")


def _unique_dir(path: Path) -> Path:
    if not path.exists():
        return path
    for idx in range(1, 1000):
        candidate = path.parent / f"{path.name}_{idx}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find unique directory for {path}")


def _unique_file(path: Path) -> Path:
    if not path.exists():
        return path
    for idx in range(1, 1000):
        candidate = path.with_name(f"{path.stem}_{idx}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find unique file for {path}")


def _latest_dir(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    candidates = [p for p in root.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _latest_file(root: Path, pattern: str, dir_prefix: Optional[str] = None) -> Optional[Path]:
    if not root.exists():
        return None
    candidate_dirs: List[Path] = []
    if dir_prefix:
        candidate_dirs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith(dir_prefix)]
    if not candidate_dirs:
        candidate_dirs = [root]
    candidates: List[Path] = []
    for base in candidate_dirs:
        candidates.extend([p for p in base.rglob(pattern) if p.is_file()])
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _run_step(label: str, args: List[str]) -> None:
    logger.info("Running %s: %s", label, " ".join(str(a) for a in args))
    subprocess.run(args, check=True)


def _expand_path(value: Optional[Any]) -> Optional[Path]:
    if value is None:
        return None
    if isinstance(value, Path):
        s = str(value)
    else:
        s = str(value).strip()
    if not s or s.lower() in {"none", "null", "auto"}:
        return None
    s = os.path.expandvars(s)
    return Path(s).expanduser()


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
        raise ValueError(f"Invalid YAML config at {path} (expected a mapping).")
    return data


def _step_enabled(cfg: Dict[str, Any]) -> bool:
    if "run" in cfg:
        return bool(cfg["run"])
    if "skip" in cfg:
        return not bool(cfg["skip"])
    return True


def _normalize_args(raw: Dict[str, Any], path_keys: Iterable[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, str) and not value.strip():
            value = None
        if key in path_keys:
            value = _expand_path(value)
        out[key] = value
    return out


def _parse_bool_flags(step_cfg: Dict[str, Any]) -> Dict[str, Tuple[str, Optional[str]]]:
    flags: Dict[str, Tuple[str, Optional[str]]] = {}
    raw = step_cfg.get("bool_flags", {})
    if not isinstance(raw, dict):
        return flags
    for key, value in raw.items():
        if isinstance(value, (list, tuple)) and len(value) == 2:
            true_flag = value[0] if value[0] else None
            false_flag = value[1] if value[1] else None
            flags[key] = (str(true_flag) if true_flag else None, str(false_flag) if false_flag else None)
        elif isinstance(value, str):
            flags[key] = (value, None)
    return flags


def _merge_bool_flags(
    default_flags: Dict[str, Tuple[str, Optional[str]]],
    override_flags: Dict[str, Tuple[str, Optional[str]]],
) -> Dict[str, Tuple[str, Optional[str]]]:
    merged = dict(default_flags)
    merged.update(override_flags)
    return merged

def _build_cmd(
    script_path: Path,
    args: Dict[str, Any],
    bool_flags: Dict[str, Tuple[str, Optional[str]]],
    python_exe: Path,
) -> List[str]:
    cmd = [str(python_exe), str(script_path)]
    for key, value in args.items():
        if value is None:
            continue
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if key in bool_flags:
                true_flag, false_flag = bool_flags[key]
                if value:
                    if true_flag:
                        cmd.append(true_flag)
                else:
                    if false_flag:
                        cmd.append(false_flag)
                    else:
                        logger.warning(
                            "%s: '%s' is false but no '--no-...' flag exists; "
                            "leaving script default unchanged.",
                            script_path.name,
                            key,
                        )
            else:
                if value:
                    cmd.append(flag)
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


def _get_git_commit(root: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _git_has_changes(root: Path) -> bool:
    result = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def _commit_and_push(root: Path, message: str) -> str:
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-m", message], check=True)
    subprocess.run(["git", "-C", str(root), "push", "origin", "master"], check=True)
    commit_hash = _get_git_commit(root)
    if not commit_hash:
        raise RuntimeError("Git commit hash unavailable after commit.")
    return commit_hash


def _ensure_git_commit(root: Path, run_name: str) -> Optional[str]:
    if not (root / ".git").exists():
        logger.warning("No .git directory found at %s; skipping git commit capture.", root)
        return None
    try:
        if _git_has_changes(root):
            message = f"Ichilov pipeline2 run {run_name}"
            return _commit_and_push(root, message)
        return _get_git_commit(root)
    except Exception as exc:
        raise RuntimeError(f"Git commit/push failed: {exc}") from exc


def _default_best_name(step_cfg: Dict[str, Any], fallback: str) -> str:
    best_name = str(step_cfg.get("best_weights_name") or "").strip()
    return best_name or fallback

def main() -> None:
    if len(sys.argv) > 1:
        logger.warning("Command-line arguments are ignored. Configure via YAML instead.")

    config_path = _resolve_config_path()
    config = _load_config(config_path)

    pipeline_cfg = config.get("pipeline", {})
    paths_cfg = config.get("paths", {})
    steps_cfg = config.get("steps", {})
    if not isinstance(steps_cfg, dict) or not steps_cfg:
        raise ValueError("Missing required 'steps' section in YAML config.")

    prefix = str(pipeline_cfg.get("run_name_prefix") or "ichilov_pipeline2").strip()
    pipeline_views = str(pipeline_cfg.get("views") or "").strip()
    run_name_cfg = str(pipeline_cfg.get("run_name") or "").strip()
    run_name = run_name_cfg or f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    experiments_root = _expand_path(paths_cfg.get("experiments_root")) or DEFAULT_EXPERIMENTS_ROOT
    experiments_root.mkdir(parents=True, exist_ok=True)
    run_dir = _unique_dir(experiments_root / run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_name = run_dir.name

    python_exe = _expand_path(pipeline_cfg.get("python_exe")) or Path(sys.executable)
    project_root = Path(__file__).resolve().parent
    git_commit = _ensure_git_commit(project_root, run_name)
    if git_commit:
        (run_dir / "git_commit.txt").write_text(f"{git_commit}\n", encoding="utf-8")

    default_weights = (
        Path.home()
        / "OneDrive - Technion"
        / "models"
        / "Encoder_weights"
        / "pytorch_model.bin"
    )
    repo_weights = project_root / "Echo-Vison-FM" / "weights" / "pytorch_model.bin"
    if repo_weights.exists():
        default_weights = repo_weights

    if pipeline_cfg.get("save_config_copy", True):
        config_copy = run_dir / "config.yaml"
        shutil.copy2(config_path, config_copy)

    resolved = deepcopy(config)
    resolved.setdefault("pipeline", {})
    resolved["pipeline"]["run_name"] = run_name
    resolved["pipeline"]["run_dir"] = str(run_dir)
    resolved["pipeline"]["config_path"] = str(config_path)
    resolved["pipeline"]["python_exe"] = str(python_exe)
    resolved["pipeline"]["git_commit"] = git_commit
    resolved["pipeline"]["timestamp"] = datetime.now().isoformat(timespec="seconds")

    # --- Crop ---
    crop_cfg = steps_cfg.get("crop", {})
    crop_run = _step_enabled(crop_cfg)
    crop_args = _normalize_args(
        crop_cfg.get("args", {}),
        path_keys=("input_xlsx", "echo_root", "output_root"),
    )
    cropped_root_base = (
        _expand_path(crop_cfg.get("output_root_base"))
        or _expand_path(paths_cfg.get("cropped_root_base"))
        or DEFAULT_CROPPED_ROOT_BASE
    )

    if crop_run:
        cropped_root = crop_args.get("output_root")
        if cropped_root is None:
            cropped_root = _unique_dir(cropped_root_base / run_name)
        crop_args["output_root"] = cropped_root
        crop_script = _step_script(crop_cfg, "ichilov_crop_dicoms.py", project_root)
        _ensure_script_exists(crop_script, "crop")
        crop_bool_flags = _merge_bool_flags(
            {"overwrite": ("--overwrite", None)},
            _parse_bool_flags(crop_cfg),
        )
        crop_cmd = _build_cmd(
            crop_script,
            crop_args,
            crop_bool_flags,
            python_exe,
        )
        _run_step("crop", crop_cmd)
    else:
        cropped_root = _expand_path(crop_cfg.get("use_if_skipped"))
        if cropped_root is None:
            cropped_root = _latest_dir(cropped_root_base)
        if cropped_root is None:
            raise FileNotFoundError(f"No cropped runs found under {cropped_root_base}")
        logger.info("Skipping crop step. Using cropped root: %s", cropped_root)
        crop_args["output_root"] = cropped_root

    # --- Frame Pretrain (Stage A) ---
    frame_pre_cfg = steps_cfg.get("frame_pretrain", {})
    frame_pre_run = _step_enabled(frame_pre_cfg)
    frame_pre_args = _normalize_args(
        frame_pre_cfg.get("args", {}),
        path_keys=("cropped_root", "weights", "output_dir", "input_xlsx", "echo_root"),
    )
    if pipeline_views and not frame_pre_args.get("views"):
        frame_pre_args["views"] = pipeline_views
    frame_pre_best: Optional[Path] = None
    frame_pre_output_dir: Optional[Path] = None

    if frame_pre_run:
        if frame_pre_args.get("views"):
            if frame_pre_args.get("input_xlsx") is None:
                frame_pre_args["input_xlsx"] = crop_args.get("input_xlsx")
            if frame_pre_args.get("echo_root") is None:
                frame_pre_args["echo_root"] = crop_args.get("echo_root")
        frame_pre_output_dir = frame_pre_args.get("output_dir") or (run_dir / "frame_pretrain")
        frame_pre_output_dir.mkdir(parents=True, exist_ok=True)
        frame_pre_args["output_dir"] = frame_pre_output_dir

        frame_pre_args["cropped_root"] = frame_pre_args.get("cropped_root") or cropped_root
        frame_pre_weights = frame_pre_args.get("weights") or default_weights
        frame_pre_args["weights"] = frame_pre_weights
        if frame_pre_weights and not frame_pre_weights.exists():
            raise FileNotFoundError(f"Frame-pretrain weights not found: {frame_pre_weights}")

        frame_pre_run_name = str(frame_pre_args.get("run_name") or "").strip()
        if not frame_pre_run_name:
            frame_pre_run_name = f"frame_mae_{run_name}"
            frame_pre_args["run_name"] = frame_pre_run_name

        best_pattern = str(frame_pre_cfg.get("best_weights_pattern") or "").strip()
        if best_pattern:
            frame_pre_best = _latest_file(frame_pre_output_dir, best_pattern)
        else:
            best_name = _default_best_name(frame_pre_cfg, f"{frame_pre_run_name}_best.pt")
            frame_pre_best = frame_pre_output_dir / best_name

        frame_pre_script = _step_script(frame_pre_cfg, "ichilov_pretrain_frame_mae.py", project_root)
        _ensure_script_exists(frame_pre_script, "frame_pretrain")
        frame_pre_bool_flags = _merge_bool_flags({}, _parse_bool_flags(frame_pre_cfg))
        frame_pre_cmd = _build_cmd(
            frame_pre_script,
            frame_pre_args,
            frame_pre_bool_flags,
            python_exe,
        )
        _run_step("frame_pretrain", frame_pre_cmd)
    else:
        frame_pre_best = _expand_path(frame_pre_cfg.get("use_if_skipped"))
        if frame_pre_best is None:
            pattern = str(frame_pre_cfg.get("best_weights_pattern") or "").strip() or "*frame_mae*best*.pt"
            frame_pre_best = _latest_file(experiments_root, pattern, dir_prefix=prefix)
        if frame_pre_best is None:
            logger.warning("No frame-pretrain weights found under %s", experiments_root)
        else:
            logger.info("Skipping frame pretrain. Using weights: %s", frame_pre_best)
            frame_pre_output_dir = frame_pre_best.parent

    # --- Frame Encode (Stage A output) ---
    frame_encode_cfg = steps_cfg.get("frame_encode", {})
    frame_encode_run = _step_enabled(frame_encode_cfg)
    frame_encode_args = _normalize_args(
        frame_encode_cfg.get("args", {}),
        path_keys=("input_xlsx", "echo_root", "cropped_root", "weights", "output_parquet"),
    )
    if pipeline_views and not frame_encode_args.get("views"):
        frame_encode_args["views"] = pipeline_views
    frame_embeddings_path: Optional[Path] = None
    frame_encode_weights: Optional[Path] = None

    if frame_encode_run:
        frame_encode_args["cropped_root"] = frame_encode_args.get("cropped_root") or cropped_root
        frame_encode_weights = frame_encode_args.get("weights")
        if frame_encode_weights is None:
            frame_encode_weights = frame_pre_best or frame_pre_args.get("weights")
        if frame_encode_weights is None:
            raise FileNotFoundError("Frame encode weights not specified and no frame-pretrain weights were found.")
        if not frame_encode_weights.exists():
            raise FileNotFoundError(f"Frame encode weights not found: {frame_encode_weights}")
        frame_encode_args["weights"] = frame_encode_weights

        frame_embeddings_path = frame_encode_args.get("output_parquet")
        if frame_embeddings_path is None:
            embeddings_dir = run_dir / "frame_embeddings"
            embeddings_dir.mkdir(parents=True, exist_ok=True)
            frame_embeddings_path = _unique_file(
                embeddings_dir / f"Ichilov_frame_embeddings_{run_name}.parquet"
            )
        frame_encode_args["output_parquet"] = frame_embeddings_path

        frame_encode_script = _step_script(frame_encode_cfg, "ichilov_encode_frames.py", project_root)
        _ensure_script_exists(frame_encode_script, "frame_encode")
        frame_encode_bool_flags = _merge_bool_flags({}, _parse_bool_flags(frame_encode_cfg))
        frame_encode_cmd = _build_cmd(
            frame_encode_script,
            frame_encode_args,
            frame_encode_bool_flags,
            python_exe,
        )
        _run_step("frame_encode", frame_encode_cmd)
    else:
        frame_embeddings_path = _expand_path(frame_encode_cfg.get("use_if_skipped"))
        if frame_embeddings_path is None:
            frame_embeddings_path = _latest_file(
                experiments_root, "Ichilov_frame_embeddings_*.parquet", dir_prefix=prefix
            )
        if frame_embeddings_path is None:
            logger.warning("No frame embeddings parquet found under %s", experiments_root)
        else:
            logger.info("Skipping frame encode. Using embeddings: %s", frame_embeddings_path)

    # --- Temporal Train (Stage B) ---
    temporal_cfg = steps_cfg.get("temporal_train", {})
    temporal_run = _step_enabled(temporal_cfg)
    temporal_args = _normalize_args(
        temporal_cfg.get("args", {}),
        path_keys=("input_embeddings", "report_xlsx", "output_dir", "log_dir"),
    )
    if pipeline_views and not temporal_args.get("views"):
        temporal_args["views"] = pipeline_views
    temporal_output_dir: Optional[Path] = None
    temporal_best: Optional[Path] = None

    if temporal_run:
        temporal_args["input_embeddings"] = temporal_args.get("input_embeddings") or frame_embeddings_path
        if temporal_args["input_embeddings"] is None:
            raise FileNotFoundError("Temporal training requires frame embeddings but none were found.")

        temporal_output_dir = temporal_args.get("output_dir") or (run_dir / "temporal")
        temporal_output_dir.mkdir(parents=True, exist_ok=True)
        temporal_args["output_dir"] = temporal_output_dir

        if temporal_args.get("log_dir") is None:
            temporal_args.pop("log_dir", None)

        temporal_run_name = str(temporal_args.get("run_name") or "").strip()
        if not temporal_run_name:
            temporal_run_name = f"temporal_{run_name}"
            temporal_args["run_name"] = temporal_run_name

        best_pattern = str(temporal_cfg.get("best_weights_pattern") or "").strip()
        if best_pattern:
            temporal_best = _latest_file(temporal_output_dir, best_pattern)
        else:
            best_name = _default_best_name(temporal_cfg, f"{temporal_run_name}_best.pt")
            temporal_best = temporal_output_dir / best_name

        temporal_script = _step_script(temporal_cfg, "ichilov_train_temporal_gls.py", project_root)
        _ensure_script_exists(temporal_script, "temporal_train")
        temporal_bool_flags = _merge_bool_flags({}, _parse_bool_flags(temporal_cfg))
        temporal_cmd = _build_cmd(
            temporal_script,
            temporal_args,
            temporal_bool_flags,
            python_exe,
        )
        _run_step("temporal_train", temporal_cmd)
    else:
        temporal_best = _expand_path(temporal_cfg.get("use_if_skipped"))
        if temporal_best is None:
            pattern = str(temporal_cfg.get("best_weights_pattern") or "").strip() or "*temporal*best*.pt"
            temporal_best = _latest_file(experiments_root, pattern, dir_prefix=prefix)
        if temporal_best is None:
            logger.warning("No temporal weights found under %s", experiments_root)
        else:
            logger.info("Skipping temporal train. Using weights: %s", temporal_best)
            temporal_output_dir = temporal_best.parent

    # --- Cine Encode (Stage B output) ---
    cine_encode_cfg = steps_cfg.get("cine_encode", {})
    cine_encode_run = _step_enabled(cine_encode_cfg)
    cine_encode_args = _normalize_args(
        cine_encode_cfg.get("args", {}),
        path_keys=("input_embeddings", "weights", "output_parquet", "report_xlsx"),
    )
    if pipeline_views and not cine_encode_args.get("views"):
        cine_encode_args["views"] = pipeline_views
    cine_embeddings_path: Optional[Path] = None
    cine_encode_weights: Optional[Path] = None

    if cine_encode_run:
        cine_encode_args["input_embeddings"] = cine_encode_args.get("input_embeddings") or frame_embeddings_path
        if cine_encode_args["input_embeddings"] is None:
            raise FileNotFoundError("Cine encoding requires frame embeddings but none were found.")

        cine_encode_weights = cine_encode_args.get("weights") or temporal_best
        if cine_encode_weights is None:
            raise FileNotFoundError("Cine encode weights not specified and no temporal weights were found.")
        if not cine_encode_weights.exists():
            raise FileNotFoundError(f"Cine encode weights not found: {cine_encode_weights}")
        cine_encode_args["weights"] = cine_encode_weights

        cine_embeddings_path = cine_encode_args.get("output_parquet")
        if cine_embeddings_path is None:
            cine_dir = run_dir / "cine_embeddings"
            cine_dir.mkdir(parents=True, exist_ok=True)
            cine_embeddings_path = _unique_file(
                cine_dir / f"Ichilov_cine_embeddings_{run_name}.parquet"
            )
        cine_encode_args["output_parquet"] = cine_embeddings_path

        cine_encode_script = _step_script(cine_encode_cfg, "ichilov_encode_cines.py", project_root)
        _ensure_script_exists(cine_encode_script, "cine_encode")
        cine_encode_bool_flags = _merge_bool_flags({}, _parse_bool_flags(cine_encode_cfg))
        cine_encode_cmd = _build_cmd(
            cine_encode_script,
            cine_encode_args,
            cine_encode_bool_flags,
            python_exe,
        )
        _run_step("cine_encode", cine_encode_cmd)
    else:
        cine_embeddings_path = _expand_path(cine_encode_cfg.get("use_if_skipped"))
        if cine_embeddings_path is None:
            cine_embeddings_path = _latest_file(
                experiments_root, "Ichilov_cine_embeddings_*.parquet", dir_prefix=prefix
            )
        if cine_embeddings_path is None:
            logger.warning("No cine embeddings parquet found under %s", experiments_root)
        else:
            logger.info("Skipping cine encode. Using embeddings: %s", cine_embeddings_path)

    # --- Visit Fusion (Stage C preparation) ---
    visit_fusion_cfg = steps_cfg.get("visit_fusion", {})
    visit_fusion_run = _step_enabled(visit_fusion_cfg)
    visit_fusion_args = _normalize_args(
        visit_fusion_cfg.get("args", {}),
        path_keys=("input_embeddings", "output_parquet", "report_xlsx"),
    )
    if pipeline_views and not visit_fusion_args.get("views"):
        visit_fusion_args["views"] = pipeline_views
    visit_embeddings_path: Optional[Path] = None

    if visit_fusion_run:
        visit_fusion_args["input_embeddings"] = visit_fusion_args.get("input_embeddings") or cine_embeddings_path
        if visit_fusion_args["input_embeddings"] is None:
            raise FileNotFoundError("Visit fusion requires cine embeddings but none were found.")

        visit_embeddings_path = visit_fusion_args.get("output_parquet")
        if visit_embeddings_path is None:
            visit_dir = run_dir / "visit_embeddings"
            visit_dir.mkdir(parents=True, exist_ok=True)
            visit_embeddings_path = _unique_file(
                visit_dir / f"Ichilov_visit_embeddings_{run_name}.parquet"
            )
        visit_fusion_args["output_parquet"] = visit_embeddings_path

        visit_fusion_script = _step_script(visit_fusion_cfg, "ichilov_fuse_visits.py", project_root)
        _ensure_script_exists(visit_fusion_script, "visit_fusion")
        visit_fusion_bool_flags = _merge_bool_flags({}, _parse_bool_flags(visit_fusion_cfg))
        visit_fusion_cmd = _build_cmd(
            visit_fusion_script,
            visit_fusion_args,
            visit_fusion_bool_flags,
            python_exe,
        )
        _run_step("visit_fusion", visit_fusion_cmd)
    else:
        visit_embeddings_path = _expand_path(visit_fusion_cfg.get("use_if_skipped"))
        if visit_embeddings_path is None:
            visit_embeddings_path = _latest_file(
                experiments_root, "Ichilov_visit_embeddings_*.parquet", dir_prefix=prefix
            )
        if visit_embeddings_path is None:
            logger.warning("No visit embeddings parquet found under %s", experiments_root)
        else:
            logger.info("Skipping visit fusion. Using embeddings: %s", visit_embeddings_path)

    # --- Longitudinal Train (Stage C) ---
    long_cfg = steps_cfg.get("longitudinal_train", {})
    long_run = _step_enabled(long_cfg)
    long_args = _normalize_args(
        long_cfg.get("args", {}),
        path_keys=("input_embeddings", "report_xlsx", "output_dir", "log_dir"),
    )
    if pipeline_views and not long_args.get("views"):
        long_args["views"] = pipeline_views
    long_output_dir: Optional[Path] = None
    long_best: Optional[Path] = None

    if long_run:
        long_args["input_embeddings"] = long_args.get("input_embeddings") or visit_embeddings_path
        if long_args["input_embeddings"] is None:
            raise FileNotFoundError("Longitudinal training requires visit embeddings but none were found.")

        long_output_dir = long_args.get("output_dir") or (run_dir / "longitudinal")
        long_output_dir.mkdir(parents=True, exist_ok=True)
        long_args["output_dir"] = long_output_dir

        if long_args.get("log_dir") is None:
            long_args.pop("log_dir", None)

        long_run_name = str(long_args.get("run_name") or "").strip()
        if not long_run_name:
            long_run_name = f"longitudinal_{run_name}"
            long_args["run_name"] = long_run_name

        best_pattern = str(long_cfg.get("best_weights_pattern") or "").strip()
        if best_pattern:
            long_best = _latest_file(long_output_dir, best_pattern)
        else:
            best_name = _default_best_name(long_cfg, f"{long_run_name}_best.pt")
            long_best = long_output_dir / best_name

        long_script = _step_script(long_cfg, "ichilov_train_longitudinal.py", project_root)
        _ensure_script_exists(long_script, "longitudinal_train")
        long_bool_flags = _merge_bool_flags({}, _parse_bool_flags(long_cfg))
        long_cmd = _build_cmd(
            long_script,
            long_args,
            long_bool_flags,
            python_exe,
        )
        _run_step("longitudinal_train", long_cmd)
    else:
        long_best = _expand_path(long_cfg.get("use_if_skipped"))
        if long_best is None:
            pattern = str(long_cfg.get("best_weights_pattern") or "").strip() or "*longitudinal*best*.pt"
            long_best = _latest_file(experiments_root, pattern, dir_prefix=prefix)
        if long_best is None:
            logger.info("Skipping longitudinal training step.")
        else:
            logger.info("Skipping longitudinal train. Using weights: %s", long_best)
            long_output_dir = long_best.parent

    artifacts = {
        "cropped_root": cropped_root,
        "frame_pretrain_output_dir": frame_pre_output_dir,
        "frame_pretrain_best_weights": frame_pre_best,
        "frame_encode_weights": frame_encode_weights,
        "frame_embeddings_path": frame_embeddings_path,
        "temporal_output_dir": temporal_output_dir,
        "temporal_best_weights": temporal_best,
        "cine_encode_weights": cine_encode_weights,
        "cine_embeddings_path": cine_embeddings_path,
        "visit_embeddings_path": visit_embeddings_path,
        "longitudinal_output_dir": long_output_dir,
        "longitudinal_best_weights": long_best,
    }
    resolved["artifacts"] = _stringify_paths(artifacts)

    resolved.setdefault("steps", {})
    resolved["steps"].setdefault("crop", {})
    resolved["steps"]["crop"]["run"] = crop_run
    resolved["steps"]["crop"]["args"] = _stringify_paths(crop_args)

    resolved["steps"].setdefault("frame_pretrain", {})
    resolved["steps"]["frame_pretrain"]["run"] = frame_pre_run
    resolved["steps"]["frame_pretrain"]["args"] = _stringify_paths(frame_pre_args)

    resolved["steps"].setdefault("frame_encode", {})
    resolved["steps"]["frame_encode"]["run"] = frame_encode_run
    resolved["steps"]["frame_encode"]["args"] = _stringify_paths(frame_encode_args)

    resolved["steps"].setdefault("temporal_train", {})
    resolved["steps"]["temporal_train"]["run"] = temporal_run
    resolved["steps"]["temporal_train"]["args"] = _stringify_paths(temporal_args)

    resolved["steps"].setdefault("cine_encode", {})
    resolved["steps"]["cine_encode"]["run"] = cine_encode_run
    resolved["steps"]["cine_encode"]["args"] = _stringify_paths(cine_encode_args)

    resolved["steps"].setdefault("visit_fusion", {})
    resolved["steps"]["visit_fusion"]["run"] = visit_fusion_run
    resolved["steps"]["visit_fusion"]["args"] = _stringify_paths(visit_fusion_args)

    resolved["steps"].setdefault("longitudinal_train", {})
    resolved["steps"]["longitudinal_train"]["run"] = long_run
    resolved["steps"]["longitudinal_train"]["args"] = _stringify_paths(long_args)

    resolved_path = run_dir / "resolved_config.yaml"
    with resolved_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(_stringify_paths(resolved), handle, sort_keys=False)

    logger.info("Pipeline2 complete. Run name: %s", run_name)
    logger.info("Run folder: %s", run_dir)
    logger.info("Cropped root: %s", cropped_root)
    logger.info("Frame-pretrain output: %s", frame_pre_output_dir)
    logger.info("Frame embeddings: %s", frame_embeddings_path)
    logger.info("Temporal output: %s", temporal_output_dir)
    logger.info("Cine embeddings: %s", cine_embeddings_path)
    logger.info("Visit embeddings: %s", visit_embeddings_path)
    logger.info("Longitudinal outputs: %s", long_output_dir)


if __name__ == "__main__":
    main()
