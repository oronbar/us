"""
Encode per-frame embeddings into per-cine embeddings using a trained temporal model.
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ichilov_pipeline2_utils import add_gls_from_report, resolve_patient_column
from ichilov_train_temporal_gls import TemporalModel, _load_embeddings, _parse_embedding, _cine_key_column

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("ichilov_encode_cines")


def _prepare_cine_groups(df: pd.DataFrame, seq_len: int) -> List[dict]:
    cine_col = _cine_key_column(df)
    if "frame_index" not in df.columns:
        df = df.copy()
        df["frame_index"] = df.groupby(cine_col).cumcount()
    if "frame_time" not in df.columns:
        df = df.copy()
        df["frame_time"] = df["frame_index"].astype(float)

    groups = []
    for _, grp in df.groupby(cine_col):
        emb_list = []
        times = []
        for _, row in grp.iterrows():
            emb = _parse_embedding(row.get("embedding"))
            if emb is None:
                continue
            emb_list.append(emb)
            times.append(float(row.get("frame_time", 0.0)))
        if not emb_list:
            continue
        emb_arr = np.stack(emb_list, axis=0)
        times_arr = np.asarray(times, dtype=np.float32)
        n = emb_arr.shape[0]
        if n >= seq_len:
            sel = np.linspace(0, n - 1, seq_len)
            sel = np.clip(np.round(sel).astype(int), 0, n - 1)
            emb_arr = emb_arr[sel]
            times_arr = times_arr[sel]
            mask = np.ones(seq_len, dtype=np.bool_)
        else:
            pad = seq_len - n
            emb_arr = np.concatenate([emb_arr, np.zeros((pad, emb_arr.shape[1]), dtype=emb_arr.dtype)], axis=0)
            times_arr = np.concatenate([times_arr, np.zeros(pad, dtype=times_arr.dtype)], axis=0)
            mask = np.zeros(seq_len, dtype=np.bool_)
            mask[:n] = True

        if times_arr.max() > 0:
            times_norm = times_arr / times_arr.max()
        else:
            times_norm = times_arr
        delta_t = np.zeros_like(times_norm)
        delta_t[1:] = np.diff(times_norm)

        meta = grp.iloc[0]
        groups.append(
            {
                "x": emb_arr,
                "mask": mask,
                "time": times_norm,
                "delta": delta_t,
                "meta": meta,
            }
        )
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode per-frame embeddings into cine embeddings.")
    parser.add_argument("--input-embeddings", type=Path, required=True, help="Input frame embeddings parquet/csv.")
    parser.add_argument("--weights", type=Path, required=True, help="Temporal model weights.")
    parser.add_argument("--output-parquet", type=Path, required=True, help="Output parquet path.")
    parser.add_argument("--report-xlsx", type=Path, default=None, help="Optional report to attach GLS.")
    parser.add_argument("--views", type=str, default="", help="Optional view filter.")
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    df = _load_embeddings(args.input_embeddings)
    if args.report_xlsx and args.report_xlsx.exists():
        df = add_gls_from_report(df, args.report_xlsx)

    if "view" in df.columns and args.views:
        view_set = set(v.strip() for v in args.views.replace(";", ",").split(",") if v.strip())
        df = df[df["view"].isin(view_set)]

    ckpt = torch.load(args.weights, map_location="cpu")
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    input_dim = cfg.get("input_dim")
    if input_dim is None:
        sample = _parse_embedding(df["embedding"].iloc[0])
        input_dim = int(sample.shape[0])

    model = TemporalModel(
        input_dim=input_dim,
        model_dim=cfg.get("model_dim", args.embedding_dim),
        model=cfg.get("model", "transformer"),
        num_layers=cfg.get("num_layers", 2),
        num_heads=cfg.get("num_heads", 4),
        dropout=cfg.get("dropout", 0.1),
        use_delta_t=cfg.get("use_delta_t", True),
        use_phase=cfg.get("use_phase", False),
        time_embedding=cfg.get("time_embedding", "relative"),
    ).to(device)

    state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()

    groups = _prepare_cine_groups(df, seq_len=cfg.get("seq_len", args.seq_len))
    rows: List[dict] = []

    for item in tqdm(groups, desc="Encoding cines"):
        x = torch.tensor(item["x"], dtype=torch.float32).unsqueeze(0).to(device)
        mask = torch.tensor(item["mask"], dtype=torch.bool).unsqueeze(0).to(device)
        time_vals = torch.tensor(item["time"], dtype=torch.float32).unsqueeze(0).to(device)
        delta_t = torch.tensor(item["delta"], dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            _, pooled = model(x, mask, time_vals, delta_t, None)
        emb = pooled.squeeze(0).cpu().numpy()
        meta = item["meta"]
        rows.append(
            {
                "patient_num": meta.get("patient_num"),
                "patient_id": meta.get("patient_id"),
                "study_datetime": meta.get("study_datetime"),
                "view": meta.get("view"),
                "source_dicom": meta.get("source_dicom"),
                "cropped_dicom": meta.get("cropped_dicom"),
                "gls": meta.get("gls"),
                "embedding": emb.tolist(),
                "embedding_dim": int(emb.shape[0]),
            }
        )

    out_df = pd.DataFrame(rows)
    output_path = args.output_parquet
    if output_path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_path.with_name(f"{output_path.stem}_{stamp}{output_path.suffix}")
        logger.warning("Output exists; writing to %s", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path, index=False)
    logger.info("Saved cine embeddings: %s (%d rows)", output_path, len(out_df))


if __name__ == "__main__":
    main()
