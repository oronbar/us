"""
Fuse per-cine embeddings into per-visit embeddings (order-invariant).
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ichilov_pipeline2_utils import add_gls_from_report, resolve_patient_column, resolve_visit_column

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("ichilov_fuse_visits")


def _parse_embedding(val: object) -> np.ndarray:
    if isinstance(val, np.ndarray):
        return val.astype(np.float32)
    if isinstance(val, list):
        return np.asarray(val, dtype=np.float32)
    if isinstance(val, str):
        try:
            return np.asarray(eval(val, {"__builtins__": {}}), dtype=np.float32)
        except Exception:
            return None
    return None


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    exp = np.exp(x)
    return exp / (exp.sum() + 1e-8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse cine embeddings into visit embeddings.")
    parser.add_argument("--input-embeddings", type=Path, required=True, help="Input cine embeddings parquet/csv.")
    parser.add_argument("--output-parquet", type=Path, required=True, help="Output visit embeddings parquet.")
    parser.add_argument("--report-xlsx", type=Path, default=None, help="Optional report for GLS.")
    parser.add_argument("--views", type=str, default="", help="Optional view filter.")
    parser.add_argument("--fusion", type=str, choices=["attention", "set_transformer", "mean"], default="attention")
    parser.add_argument("--min-views", type=int, default=2)
    parser.add_argument("--use-view-embedding", action="store_true", default=True)
    parser.add_argument("--use-quality-embedding", action="store_true", default=False)
    args = parser.parse_args()

    df = pd.read_parquet(args.input_embeddings) if args.input_embeddings.suffix.lower() == ".parquet" else pd.read_csv(args.input_embeddings)
    if args.report_xlsx and args.report_xlsx.exists() and "gls" not in df.columns:
        df = add_gls_from_report(df, args.report_xlsx)

    if "view" in df.columns and args.views:
        view_set = set(v.strip() for v in args.views.replace(";", ",").split(",") if v.strip())
        df = df[df["view"].isin(view_set)]

    patient_col = resolve_patient_column(df)
    if patient_col is None:
        raise ValueError("No patient column found in embeddings.")

    visit_col = resolve_visit_column(df)
    if visit_col is None:
        raise ValueError("No visit/time column found in embeddings.")

    rng = np.random.default_rng(42)
    view_table: Dict[str, np.ndarray] = {}

    rows: List[dict] = []
    for (pid, visit_id), grp in df.groupby([patient_col, visit_col]):
        views = grp.get("view")
        view_list = list(views) if views is not None else []
        unique_views = set(v for v in view_list if isinstance(v, str))
        if args.min_views > 1 and len(unique_views) < args.min_views:
            continue

        emb_list = []
        view_list_for_emb = []
        for _, row in grp.iterrows():
            emb = _parse_embedding(row.get("embedding"))
            if emb is None:
                continue
            v = row.get("view")
            if args.use_view_embedding and isinstance(v, str):
                if v not in view_table:
                    view_table[v] = rng.normal(scale=0.01, size=emb.shape[0]).astype(np.float32)
                emb = emb + view_table[v]
            emb_list.append(emb)
            view_list_for_emb.append(v)
        if not emb_list:
            continue
        emb_arr = np.stack(emb_list, axis=0)

        if args.fusion == "mean":
            visit_emb = emb_arr.mean(axis=0)
        else:
            mean = emb_arr.mean(axis=0)
            scores = (emb_arr @ mean) / (np.linalg.norm(mean) + 1e-6)
            weights = _softmax(scores)
            attn = (weights[:, None] * emb_arr).sum(axis=0)
            if args.fusion == "set_transformer":
                max_pool = emb_arr.max(axis=0)
                visit_emb = 0.5 * attn + 0.5 * max_pool
            else:
                visit_emb = attn

        gls_val = None
        if "gls" in grp.columns:
            vals = pd.to_numeric(grp["gls"], errors="coerce").dropna().values
            gls_val = float(vals.mean()) if len(vals) else None

        rows.append(
            {
                "patient_id": pid,
                "visit_id": visit_id,
                "study_datetime": grp.get("study_datetime", pd.Series([None])).iloc[0] if "study_datetime" in grp.columns else None,
                "views": ",".join(sorted(unique_views)) if unique_views else None,
                "n_views": int(len(unique_views)),
                "gls": gls_val,
                "embedding": visit_emb.tolist(),
                "embedding_dim": int(visit_emb.shape[0]),
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
    logger.info("Saved visit embeddings: %s (%d rows)", output_path, len(out_df))


if __name__ == "__main__":
    main()
