"""
Evaluate a trained pipeline5 strain curve model on phase-frame embeddings.

Example:
  python evaluate_strain_curve_model.py ^
    --input-parquet "C:\\path\\phase_embeddings.parquet" ^
    --checkpoint "C:\\path\\strain_curve_model_best.pt" ^
    --output-dir "C:\\path\\eval"
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from ichilov_strain_curve_utils import dataframe_preview_for_csv, write_json
from strain_curve_model import build_strain_curve_model
from train_strain_curve_model import (
    StrainEmbeddingDataset,
    _clean_training_df,
    _evaluate,
    _metrics_md,
    collate_batch,
    compute_metrics,
    save_diagnostics,
)


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_strain_curve_model")


def _torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate strain curve model.")
    parser.add_argument("--input-parquet", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--diagnostic-samples", type=int, default=80)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))

    logger.info("Loading embeddings: %s", args.input_parquet)
    df = _clean_training_df(pd.read_parquet(args.input_parquet))
    if df.empty:
        raise ValueError("No evaluable samples found.")
    logger.info("Loading checkpoint: %s", args.checkpoint)
    checkpoint = _torch_load(args.checkpoint, device)
    model_config = checkpoint.get("model_config")
    if not model_config:
        raise ValueError("Checkpoint missing model_config.")
    model = build_strain_curve_model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    loader = DataLoader(
        StrainEmbeddingDataset(df),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )
    preds = _evaluate(model, loader, df, device)
    metrics = compute_metrics(preds)

    pred_parquet = args.output_dir / "predictions.parquet"
    pred_csv = args.output_dir / "predictions.csv"
    preds.to_parquet(pred_parquet, index=False)
    dataframe_preview_for_csv(preds).to_csv(pred_csv, index=False)
    write_json(args.output_dir / "metrics.json", metrics)
    (args.output_dir / "metrics.md").write_text(_metrics_md(metrics), encoding="utf-8")
    save_diagnostics(preds, args.output_dir, args.diagnostic_samples)

    summary = {
        "input_parquet": str(args.input_parquet),
        "checkpoint": str(args.checkpoint),
        "predictions_parquet": str(pred_parquet),
        "predictions_csv": str(pred_csv),
        "metrics_json": str(args.output_dir / "metrics.json"),
        "n_samples": int(len(preds)),
        "n_patients": int(preds["patient_key"].dropna().nunique()) if "patient_key" in preds else 0,
    }
    write_json(args.output_dir / "evaluation_summary.json", summary)
    logger.info("Saved predictions: %s", pred_parquet)
    logger.info("Saved metrics: %s", args.output_dir / "metrics.json")


if __name__ == "__main__":
    main()
