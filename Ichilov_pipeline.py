"""
Run the full Ichilov GLS pipeline from a YAML configuration file.

Config path (no CLI args):
  - default: ichilov_pipeline.yaml (next to this file)
  - override with env var: ICHILOV_PIPELINE_CONFIG

Outputs:
  - Each run writes outputs (except cropped DICOMs) into a timestamped
    folder under paths.experiments_root.
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
logger = logging.getLogger("Ichilov_pipeline")

CONFIG_ENV = "ICHILOV_PIPELINE_CONFIG"
DEFAULT_CONFIG_NAME = "ichilov_pipeline.yaml"

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
    subprocess.run(["git", "-C", str(root), "push", "origin", "main"], check=True)
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
            message = f"Ichilov pipeline run {run_name}"
            return _commit_and_push(root, message)
        return _get_git_commit(root)
    except Exception as exc:
        raise RuntimeError(f"Git commit/push failed: {exc}") from exc


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

    prefix = str(pipeline_cfg.get("run_name_prefix") or "ichilov_pipeline").strip()
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
        crop_bool_flags = {"overwrite": ("--overwrite", None)}
        crop_cmd = _build_cmd(
            project_root / "ichilov_crop_dicoms.py",
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

    # --- Pretrain ---
    pretrain_cfg = steps_cfg.get("pretrain", {})
    pretrain_run = _step_enabled(pretrain_cfg)
    pretrain_args = _normalize_args(
        pretrain_cfg.get("args", {}),
        path_keys=("cropped_root", "weights", "output_dir", "input_xlsx", "echo_root"),
    )
    if pipeline_views and not pretrain_args.get("views"):
        pretrain_args["views"] = pipeline_views
    pretrain_best: Optional[Path] = None
    pretrain_output_dir: Optional[Path] = None

    if pretrain_run:
        if pretrain_args.get("views"):
            if pretrain_args.get("input_xlsx") is None:
                pretrain_args["input_xlsx"] = crop_args.get("input_xlsx")
            if pretrain_args.get("echo_root") is None:
                pretrain_args["echo_root"] = crop_args.get("echo_root")
        pretrain_output_dir = pretrain_args.get("output_dir") or (run_dir / "pretrain")
        pretrain_output_dir.mkdir(parents=True, exist_ok=True)
        pretrain_args["output_dir"] = pretrain_output_dir

        pretrain_args["cropped_root"] = pretrain_args.get("cropped_root") or cropped_root
        pretrain_weights = pretrain_args.get("weights") or default_weights
        pretrain_args["weights"] = pretrain_weights
        if not pretrain_weights.exists():
            raise FileNotFoundError(f"Pretrain weights not found: {pretrain_weights}")

        pretrain_run_name = str(pretrain_args.get("run_name") or "").strip()
        if not pretrain_run_name:
            pretrain_run_name = f"echovisionfm_mae_{run_name}"
            pretrain_args["run_name"] = pretrain_run_name
        pretrain_best = pretrain_output_dir / f"{pretrain_run_name}_best_mae.pt"

        pretrain_bool_flags = {
            "include_last_window": ("--include-last-window", "--no-include-last-window"),
        }
        pretrain_cmd = _build_cmd(
            project_root / "ichilov_train_echovisionfm_last_layer.py",
            pretrain_args,
            pretrain_bool_flags,
            python_exe,
        )
        _run_step("pretrain", pretrain_cmd)
    else:
        pretrain_best = _expand_path(pretrain_cfg.get("use_if_skipped"))
        if pretrain_best is None:
            pretrain_best = _latest_file(experiments_root, "*_best_mae.pt", dir_prefix=prefix)
        if pretrain_best is None:
            logger.warning("No pretrain weights found under %s", experiments_root)
        else:
            logger.info("Skipping pretrain. Using weights: %s", pretrain_best)
            pretrain_output_dir = pretrain_best.parent

    # --- Encode ---
    encode_cfg = steps_cfg.get("encode", {})
    encode_run = _step_enabled(encode_cfg)
    encode_args = _normalize_args(
        encode_cfg.get("args", {}),
        path_keys=("input_xlsx", "echo_root", "cropped_root", "weights", "output_parquet"),
    )
    if pipeline_views and not encode_args.get("views"):
        encode_args["views"] = pipeline_views
    embeddings_path: Optional[Path] = None
    encode_weights: Optional[Path] = None

    if encode_run:
        encode_args["cropped_root"] = encode_args.get("cropped_root") or cropped_root
        encode_weights = encode_args.get("weights")
        if encode_weights is None:
            encode_weights = pretrain_best or pretrain_args.get("weights")
        if encode_weights is None:
            raise FileNotFoundError("Encode weights not specified and no pretrain weights were found.")
        if not encode_weights.exists():
            raise FileNotFoundError(f"Encode weights not found: {encode_weights}")
        encode_args["weights"] = encode_weights

        embeddings_path = encode_args.get("output_parquet")
        if embeddings_path is None:
            embeddings_dir = run_dir / "embeddings"
            embeddings_dir.mkdir(parents=True, exist_ok=True)
            embeddings_path = _unique_file(
                embeddings_dir / f"Ichilov_GLS_embeddings_{run_name}.parquet"
            )
        encode_args["output_parquet"] = embeddings_path

        encode_bool_flags = {
            "include_last_window": ("--include-last-window", "--no-include-last-window"),
            "use_stf": ("--use-stf", "--no-use-stf"),
            "safe_decode": ("--safe-decode", "--no-safe-decode"),
        }
        encode_cmd = _build_cmd(
            project_root / "ichilov_encode_dicoms.py",
            encode_args,
            encode_bool_flags,
            python_exe,
        )
        _run_step("encode", encode_cmd)
    else:
        embeddings_path = _expand_path(encode_cfg.get("use_if_skipped"))
        if embeddings_path is None:
            embeddings_path = _latest_file(
                experiments_root, "Ichilov_GLS_embeddings_*.parquet", dir_prefix=prefix
            )
        if embeddings_path is None:
            logger.warning("No embeddings parquet found under %s", experiments_root)
        else:
            logger.info("Skipping encode. Using embeddings: %s", embeddings_path)

    # --- Train GLS ---
    train_cfg = steps_cfg.get("train", {})
    train_run = _step_enabled(train_cfg)
    train_args = _normalize_args(
        train_cfg.get("args", {}),
        path_keys=("input_embeddings", "report_xlsx", "output_dir", "log_dir"),
    )
    if pipeline_views and not train_args.get("views"):
        train_args["views"] = pipeline_views
    gls_output_dir: Optional[Path] = None

    if train_run:
        train_args["input_embeddings"] = train_args.get("input_embeddings") or embeddings_path
        if train_args["input_embeddings"] is None:
            raise FileNotFoundError("Training requires embeddings but none were found.")

        gls_output_dir = train_args.get("output_dir") or (run_dir / "gls")
        gls_output_dir.mkdir(parents=True, exist_ok=True)
        train_args["output_dir"] = gls_output_dir

        if train_args.get("log_dir") is None:
            train_args.pop("log_dir", None)

        train_bool_flags = {
            "group_by_view": ("--group-by-view", "--no-group-by-view"),
            "run_baselines": ("--run-baselines", "--no-run-baselines"),
            "run_tree_baselines": ("--run-tree-baselines", "--no-run-tree-baselines"),
            "standardize": ("--standardize", "--no-standardize"),
        }
        train_cmd = _build_cmd(
            project_root / "ichilov_train_gls.py",
            train_args,
            train_bool_flags,
            python_exe,
        )
        _run_step("train_gls", train_cmd)
    else:
        logger.info("Skipping GLS training step.")

    artifacts = {
        "cropped_root": cropped_root,
        "pretrain_output_dir": pretrain_output_dir,
        "pretrain_best_weights": pretrain_best,
        "encode_weights": encode_weights,
        "embeddings_path": embeddings_path,
        "gls_output_dir": gls_output_dir,
    }
    resolved["artifacts"] = _stringify_paths(artifacts)

    resolved.setdefault("steps", {})
    resolved["steps"].setdefault("crop", {})
    resolved["steps"]["crop"]["run"] = crop_run
    resolved["steps"]["crop"]["args"] = _stringify_paths(crop_args)

    resolved["steps"].setdefault("pretrain", {})
    resolved["steps"]["pretrain"]["run"] = pretrain_run
    resolved["steps"]["pretrain"]["args"] = _stringify_paths(pretrain_args)

    resolved["steps"].setdefault("encode", {})
    resolved["steps"]["encode"]["run"] = encode_run
    resolved["steps"]["encode"]["args"] = _stringify_paths(encode_args)

    resolved["steps"].setdefault("train", {})
    resolved["steps"]["train"]["run"] = train_run
    resolved["steps"]["train"]["args"] = _stringify_paths(train_args)

    resolved_path = run_dir / "resolved_config.yaml"
    with resolved_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(_stringify_paths(resolved), handle, sort_keys=False)

    logger.info("Pipeline complete. Run name: %s", run_name)
    logger.info("Run folder: %s", run_dir)
    logger.info("Cropped root: %s", cropped_root)
    logger.info("Pretrain output: %s", pretrain_output_dir)
    logger.info("Embeddings: %s", embeddings_path)
    logger.info("GLS outputs: %s", gls_output_dir)


if __name__ == "__main__":
    main()
