"""
Cluster embeddings into two groups and summarize likely drivers (vendor/model/view, etc.),
optionally peeking into DICOM metadata for each cluster.

Defaults:
  Input embeddings: ~/OneDrive - Technion/DS/Ichilov_GLS_embeddings.parquet

Usage (PowerShell):
  .venv\\Scripts\\python analyze_cluster_metadata.py

Key outputs:
  - Console tables with top values per cluster for vendor/model/station/institution/view/series.
  - Optional DICOM metadata sampling per cluster to see what separates the groups.
  - Optional CSV with cluster assignments.
"""
from __future__ import annotations

import argparse
import ast
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("analyze_cluster_metadata")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def _find_column(df: pd.DataFrame, candidates: Sequence[str], required: bool = False) -> Optional[str]:
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in lower_map:
            return lower_map[key]
    for cand in candidates:
        key = cand.lower()
        for col in df.columns:
            if key in str(col).lower():
                return col
    if required:
        raise ValueError(f"Missing required column. Tried: {candidates}")
    return None


def _parse_embedding(val: object) -> Optional[np.ndarray]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, np.ndarray):
        return val.astype(np.float32)
    if isinstance(val, list):
        return np.asarray(val, dtype=np.float32)
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            return np.asarray(parsed, dtype=np.float32)
        except Exception:
            return None
    return None


def _load_embeddings(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.read_csv(path)


def _add_gls_from_report(df: pd.DataFrame, report_xlsx: Path) -> pd.DataFrame:
    rep = pd.read_excel(report_xlsx, engine="openpyxl")
    rep.columns = [str(c).strip() for c in rep.columns]

    mapping = {}
    for view in ("A2C", "A3C", "A4C"):
        gls_col = _find_column(rep, [f"{view}_GLS"], required=False)
        src_col = _find_column(rep, [f"{view}_GLS_SOURCE_DICOM"], required=False)
        if not gls_col or not src_col:
            continue
        for _, row in rep.iterrows():
            src = row.get(src_col)
            gls = row.get(gls_col)
            if pd.isna(src) or pd.isna(gls):
                continue
            try:
                gls_val = float(gls)
            except Exception:
                continue
            mapping[Path(str(src)).name.lower()] = gls_val

    if "source_dicom" not in df.columns:
        raise ValueError("Embeddings dataframe missing 'source_dicom' column needed for GLS join.")

    gls_vals: List[Optional[float]] = []
    for _, row in df.iterrows():
        src = row.get("source_dicom")
        base = Path(str(src)).name.lower()
        gls_vals.append(mapping.get(base))

    out = df.copy()
    out["gls"] = gls_vals
    return out


def _read_dicom_metadata(path: Path, fields: Sequence[str]) -> Dict[str, Optional[str]]:
    try:
        import pydicom
    except ImportError:
        logger.warning("pydicom not installed; skipping DICOM metadata parsing.")
        return {}

    out: Dict[str, Optional[str]] = {}
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    except Exception as e:
        logger.debug("Failed to read DICOM %s: %s", path, e)
        return out

    for f in fields:
        val = ds.get(f, None)
        if val is None:
            out[f] = None
            continue
        try:
            out[f] = str(val)
        except Exception:
            out[f] = None
    return out


def _summarize_counts(label: str, series: pd.Series, top_k: int = 5) -> None:
    counts = series.value_counts(dropna=False).head(top_k)
    logger.info("%s (top %d):", label, top_k)
    for v, c in counts.items():
        logger.info("  %s: %d", v, c)


def main() -> None:
    user_home = Path.home()
    default_embeddings = user_home / "OneDrive - Technion" / "DS" / "Ichilov_GLS_embeddings.parquet"

    parser = argparse.ArgumentParser(description="Cluster embeddings and summarize metadata per cluster.")
    parser.add_argument(
        "--input-embeddings",
        type=Path,
        default=default_embeddings,
        help=f"Embedding dataframe (parquet/csv) (default: {default_embeddings})",
    )
    parser.add_argument(
        "--report-xlsx",
        type=Path,
        default=user_home / "OneDrive - Technion" / "DS" / "Report_Ichilov_GLS_oron.xlsx",
        help="Optional report Excel to supply GLS targets if missing",
    )
    parser.add_argument("--target-col", type=str, default="", help="Optional target column name if not 'gls'")
    parser.add_argument("--embedding-dim", type=int, default=None, help="Filter to this embedding_dim if present")
    parser.add_argument("--clusters", type=int, default=2, help="Number of k-means clusters (default 2)")
    parser.add_argument("--pca-dim", type=int, default=30, help="PCA components before k-means")
    parser.add_argument("--max-samples", type=int, default=2000, help="Subsample rows before clustering (0 to disable)")
    parser.add_argument("--dicom-sample", type=int, default=200, help="Max DICOMs to read per cluster for metadata")
    parser.add_argument("--save-assignments", type=Path, default=None, help="Optional CSV to save cluster labels")
    args = parser.parse_args()

    df = _load_embeddings(args.input_embeddings)
    df.columns = [str(c).strip() for c in df.columns]

    if args.target_col:
        if args.target_col not in df.columns:
            raise ValueError(f"Target column '{args.target_col}' not found in embeddings df.")
        df["gls"] = df[args.target_col]
    elif "gls" not in df.columns and "GLS" in df.columns:
        df["gls"] = df["GLS"]

    if "gls" not in df.columns:
        if args.report_xlsx:
            df = _add_gls_from_report(df, args.report_xlsx)
        else:
            raise ValueError("No GLS target found. Provide --report-xlsx or --target-col.")

    emb_col = _find_column(df, ["embedding"], required=True)
    df["embedding_vec"] = df[emb_col].apply(_parse_embedding)
    df = df[df["embedding_vec"].notna()].copy()
    df["gls"] = pd.to_numeric(df["gls"], errors="coerce")
    df = df[df["gls"].notna()].copy()

    if df.empty:
        raise ValueError("No valid rows with embeddings and GLS targets.")

    if args.embedding_dim is not None:
        if "embedding_dim" in df.columns:
            df = df[df["embedding_dim"] == args.embedding_dim].copy()
            logger.info("Filtered to embedding_dim=%s; remaining rows=%d", args.embedding_dim, len(df))
        else:
            logger.warning("--embedding-dim provided but 'embedding_dim' column missing; proceeding without filtering.")

    emb_lengths = df["embedding_vec"].apply(lambda x: x.shape[0]).value_counts()
    if len(emb_lengths) > 1:
        target_dim = int(emb_lengths.idxmax())
        df = df[df["embedding_vec"].apply(lambda x: x.shape[0] == target_dim)].copy()
        logger.warning(
            "Mixed embedding dimensions detected; auto-selected dim=%d with %d samples.",
            target_dim,
            len(df),
        )

    if args.max_samples and len(df) > args.max_samples:
        df = df.sample(n=args.max_samples, random_state=42).copy()
        logger.info("Subsampled to %d samples for clustering.", len(df))

    X = np.stack(df["embedding_vec"].to_list()).astype(np.float32)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca_dim = min(args.pca_dim, X_scaled.shape[1])
    pca = PCA(n_components=pca_dim, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    km = KMeans(n_clusters=args.clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_pca)
    df["cluster"] = labels

    # Save assignments if requested.
    if args.save_assignments:
        out_path = args.save_assignments
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df[["cluster", "gls", "source_dicom"]].to_csv(out_path, index=False)
        logger.info("Saved cluster assignments: %s", out_path)

    # Summaries per cluster.
    meta_cols = {
        "vendor": ["vendor", "manufacturer", "manufacturer_name", "scanner_vendor"],
        "model": ["model", "manufacturer_model_name", "scanner_model", "device_model"],
        "station": ["station_name", "station"],
        "institution": ["institution_name", "institution"],
        "view": ["view", "view_key", "view_name", "view_type"],
        "series": ["series_description", "series_desc", "protocol_name"],
    }

    for c in range(args.clusters):
        sub = df[df["cluster"] == c]
        logger.info("=== Cluster %d | n=%d ===", c, len(sub))
        for label, candidates in meta_cols.items():
            col = _find_column(sub, candidates, required=False)
            if col:
                _summarize_counts(f"{label} ({col})", sub[col])
        # GLS distribution
        logger.info("GLS mean=%.3f std=%.3f min=%.3f max=%.3f", sub["gls"].mean(), sub["gls"].std(), sub["gls"].min(), sub["gls"].max())

    # Optional DICOM metadata sampling per cluster.
    dicom_fields = [
        "Manufacturer",
        "ManufacturerModelName",
        "StationName",
        "InstitutionName",
        "SeriesDescription",
        "StudyDescription",
        "BodyPartExamined",
        "ViewName",
        "ImageType",
    ]
    if "source_dicom" not in df.columns:
        logger.warning("source_dicom column missing; cannot read DICOM metadata.")
        return

    for c in range(args.clusters):
        sub = df[df["cluster"] == c]
        paths = [Path(p) for p in sub["source_dicom"] if isinstance(p, str)]
        if not paths:
            logger.warning("Cluster %d has no source_dicom paths.", c)
            continue
        sample_paths = paths[: args.dicom_sample]
        counts: Dict[str, Counter] = defaultdict(Counter)
        for p in sample_paths:
            meta = _read_dicom_metadata(p, dicom_fields)
            for k, v in meta.items():
                counts[k][v or "None"] += 1
        logger.info("--- DICOM metadata (cluster %d, %d sampled) ---", c, len(sample_paths))
        for field in dicom_fields:
            vals = counts.get(field, {})
            if not vals:
                continue
            logger.info("%s:", field)
            for val, cnt in Counter(vals).most_common(5):
                logger.info("  %s: %d", val, cnt)


if __name__ == "__main__":
    main()
