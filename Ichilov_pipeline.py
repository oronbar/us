"""
Run the full Ichilov GLS pipeline with optional sliding-window sampling.

Steps:
  1) Crop DICOMs (ichilov_crop_dicoms.py)
  2) MAE pretraining (ichilov_train_echovisionfm_last_layer.py)
  3) Encode embeddings (ichilov_encode_dicoms.py)
  4) Train GLS models (ichilov_train_gls.py)
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("Ichilov_pipeline")


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


def _latest_file(root: Path, pattern: str) -> Optional[Path]:
    if not root.exists():
        return None
    candidates = [p for p in root.rglob(pattern) if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _run_step(label: str, args: List[str]) -> None:
    logger.info("Running %s: %s", label, " ".join(str(a) for a in args))
    subprocess.run(args, check=True)


def _resolve_user_home() -> Path:
    home = Path.home()
    return Path("F:\\") if home.name == "oronbar.RF" else home


def main() -> None:
    user_home = _resolve_user_home()
    project_root = Path(__file__).resolve().parent
    default_weights = user_home / "OneDrive - Technion" / "models" / "Encoder_weights" / "pytorch_model.bin"
    repo_weights = project_root / "Echo-Vison-FM" / "weights" / "pytorch_model.bin"
    if repo_weights.exists():
        default_weights = repo_weights

    parser = argparse.ArgumentParser(description="Run the full Ichilov GLS pipeline.")
    parser.add_argument("--run-name", type=str, default="", help="Optional run name (defaults to timestamp).")
    parser.add_argument(
        "--input-xlsx",
        type=Path,
        default=user_home / "OneDrive - Technion" / "DS" / "Report_Ichilov_GLS_and_Strain_oron.xlsx",
        help="Input Excel with GLS source DICOM paths.",
    )
    parser.add_argument(
        "--report-xlsx",
        type=Path,
        default=user_home / "OneDrive - Technion" / "DS" / "Report_Ichilov_GLS_and_Strain_oron.xlsx",
        help="Report Excel used for GLS targets in training.",
    )
    parser.add_argument(
        "--echo-root",
        type=Path,
        default=Path(r"D:\DS\Ichilov"),
        help="Root directory of original DICOMs.",
    )
    parser.add_argument(
        "--cropped-root-base",
        type=Path,
        default=Path(r"D:\DS\Ichilov_cropped"),
        help="Base directory for cropped DICOM outputs.",
    )
    parser.add_argument(
        "--pretrain-output-base",
        type=Path,
        default=user_home / "OneDrive - Technion" / "models" / "Encoder_weights",
        help="Base directory for MAE pretraining outputs.",
    )
    parser.add_argument(
        "--embeddings-output-base",
        type=Path,
        default=user_home / "OneDrive - Technion" / "DS",
        help="Base directory for embedding parquet outputs.",
    )
    parser.add_argument(
        "--gls-output-base",
        type=Path,
        default=user_home / "OneDrive - Technion" / "Experiments" / "Ichilov_embedding_GLS_models",
        help="Base directory for GLS training outputs.",
    )
    parser.add_argument(
        "--pretrain-weights",
        type=Path,
        default=default_weights,
        help="Initial weights (pytorch_model.bin) for MAE pretraining.",
    )
    parser.add_argument(
        "--encode-weights",
        type=Path,
        default=None,
        help="Optional weights for encoding (defaults to pretraining output if run).",
    )
    parser.add_argument(
        "--sampling-mode",
        type=str,
        choices=["window", "phase", "sliding_window"],
        default="sliding_window",
        help="Sampling mode for cropping (sliding_window keeps full clips).",
    )
    parser.add_argument("--clip-length", type=int, default=16, help="Frames per clip.")
    parser.add_argument("--clip-stride", type=int, default=4, help="Stride for sliding-window clips.")
    parser.add_argument("--window-sec", type=float, default=1.0, help="Window length in seconds (window mode).")
    parser.add_argument("--skip-crop", action="store_true", help="Skip cropping step.")
    parser.add_argument("--skip-pretrain", action="store_true", help="Skip MAE pretraining step.")
    parser.add_argument("--skip-encode", action="store_true", help="Skip encoding step.")
    parser.add_argument("--skip-train", action="store_true", help="Skip GLS training step.")
    parser.add_argument(
        "--gls-task",
        type=str,
        choices=["view", "visit"],
        default="visit",
        help="GLS training task (view or visit).",
    )
    parser.add_argument(
        "--gls-objective",
        type=str,
        choices=["regression", "ranking"],
        default="regression",
        help="GLS training objective.",
    )
    parser.add_argument(
        "--gls-clip-fusion",
        type=str,
        choices=["none", "mean", "max", "softmax", "attention"],
        default="attention",
        help="Clip-level fusion for sliding-window embeddings.",
    )
    parser.add_argument(
        "--gls-visit-fusion",
        type=str,
        choices=["softmax", "attention"],
        default="attention",
        help="Visit-level fusion across views.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=None,
        help="Optional embedding dimension filter (e.g., 768 or 1536).",
    )
    args = parser.parse_args()

    run_name = args.run_name.strip() or datetime.now().strftime("ichilov_%Y%m%d_%H%M%S")
    pretrain_run_name = f"echovisionfm_mae_{run_name}"
    python_exe = Path(sys.executable)

    cropped_root = args.cropped_root_base / run_name
    pretrain_output_dir = args.pretrain_output_base / run_name
    embeddings_path = _unique_file(args.embeddings_output_base / f"Ichilov_GLS_embeddings_{run_name}.parquet")
    gls_output_dir = _unique_dir(args.gls_output_base / run_name)

    args.skip_crop = True
    if args.skip_crop:
        latest_cropped = _latest_dir(args.cropped_root_base)
        if latest_cropped is not None:
            cropped_root = latest_cropped
            logger.info("Using latest cropped folder: %s", cropped_root)
        else:
            logger.warning("No cropped runs found under %s", args.cropped_root_base)
    else:
        cropped_root = _unique_dir(cropped_root)
    if args.skip_crop and not cropped_root.exists():
        logger.warning("Cropped root does not exist: %s", cropped_root)

    if not args.skip_crop:
        _run_step(
            "crop",
            [
                str(python_exe),
                str(project_root / "ichilov_crop_dicoms.py"),
                "--input-xlsx",
                str(args.input_xlsx),
                "--echo-root",
                str(args.echo_root),
                "--output-root",
                str(cropped_root),
                "--sampling-mode",
                args.sampling_mode,
                "--clip-length",
                str(args.clip_length),
                "--window-sec",
                str(args.window_sec),
            ],
        )
    else:
        logger.info("Skipping crop step.")
    pretrain_best: Optional[Path] = None
    args.skip_pretrain = True
    if not args.skip_pretrain:
        pretrain_output_dir = _unique_dir(pretrain_output_dir)
        pretrain_best = pretrain_output_dir / f"{pretrain_run_name}_best_mae.pt"
        if not args.pretrain_weights.exists():
            raise FileNotFoundError(f"Pretrain weights not found: {args.pretrain_weights}")
        pretrain_output_dir.mkdir(parents=True, exist_ok=True)
        pretrain_sampling = "sliding_window" if args.sampling_mode == "sliding_window" else "uniform"
        _run_step(
            "pretrain",
            [
                str(python_exe),
                str(project_root / "ichilov_train_echovisionfm_last_layer.py"),
                "--cropped-root",
                str(cropped_root),
                "--weights",
                str(args.pretrain_weights),
                "--output-dir",
                str(pretrain_output_dir),
                "--frames",
                str(args.clip_length),
                "--sampling-mode",
                pretrain_sampling,
                "--clip-stride",
                str(args.clip_stride),
                "--run-name",
                pretrain_run_name,
            ],
        )
    else:
        logger.info("Skipping pretrain step.")
        latest_pretrain = _latest_file(args.pretrain_output_base, "*_best_mae.pt")
        if latest_pretrain is not None:
            args.pretrain_weights = latest_pretrain
            pretrain_output_dir = latest_pretrain.parent
            pretrain_best = latest_pretrain
            logger.info("Using latest pretrain weights: %s", args.pretrain_weights)
        else:
            logger.warning("Pretrain weights not found under %s; using %s", args.pretrain_output_base, args.pretrain_weights)

    encode_weights = args.encode_weights
    if encode_weights is None:
        encode_weights = pretrain_best if not args.skip_pretrain else args.pretrain_weights
    if not args.skip_encode:
        if not encode_weights.exists():
            raise FileNotFoundError(f"Encode weights not found: {encode_weights}")
        encode_sampling = "sliding_window" if args.sampling_mode == "sliding_window" else "uniform"
        _run_step(
            "encode",
            [
                str(python_exe),
                str(project_root / "ichilov_encode_dicoms.py"),
                "--input-xlsx",
                str(args.input_xlsx),
                "--echo-root",
                str(args.echo_root),
                "--cropped-root",
                str(cropped_root),
                "--weights",
                str(encode_weights),
                "--output-parquet",
                str(embeddings_path),
                "--sampling-mode",
                encode_sampling,
                "--clip-length",
                str(args.clip_length),
                "--clip-stride",
                str(args.clip_stride),
            ],
        )
    else:
        logger.info("Skipping encoding step.")

    if not args.skip_train:
        if args.skip_encode and not embeddings_path.exists():
            logger.warning("Embedding parquet not found at %s", embeddings_path)
        gls_output_dir.mkdir(parents=True, exist_ok=True)
        train_cmd = [
            str(python_exe),
            str(project_root / "ichilov_train_gls.py"),
            "--input-embeddings",
            str(embeddings_path),
            "--report-xlsx",
            str(args.report_xlsx),
            "--output-dir",
            str(gls_output_dir),
            "--task",
            args.gls_task,
            "--objective",
            args.gls_objective,
            "--clip-fusion",
            args.gls_clip_fusion,
            "--visit-fusion",
            args.gls_visit_fusion,
        ]
        if args.embedding_dim is not None:
            train_cmd.extend(["--embedding-dim", str(args.embedding_dim)])
        _run_step("train_gls", train_cmd)
    else:
        logger.info("Skipping GLS training step.")

    logger.info("Pipeline complete. Run name: %s", run_name)
    logger.info("Cropped root: %s", cropped_root)
    logger.info("Pretrain output: %s", pretrain_output_dir)
    logger.info("Embeddings: %s", embeddings_path)
    logger.info("GLS outputs: %s", gls_output_dir)


if __name__ == "__main__":
    main()
