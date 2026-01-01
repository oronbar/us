"""
Plot sample strain curves from the Ichilov GLS+strain report.

Default input:
  F:\\OneDrive - Technion\\DS\\Report_Ichilov_GLS_and_Strain_oron.xlsx

Usage (PowerShell):
  .venv\\Scripts\\python plot_strain_curves.py
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for saving.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("plot_strain_curves")


def _find_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
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
    return None


def _is_number(val: object) -> bool:
    return isinstance(val, (int, float, np.number)) and not pd.isna(val)


def _curves_from_obj(obj: object) -> List[np.ndarray]:
    if obj is None:
        return []
    if isinstance(obj, np.ndarray):
        if obj.ndim == 1:
            return [obj.astype(float)]
        if obj.ndim == 2:
            if obj.shape[1] == 2 and obj.shape[0] > 2:
                return [obj.astype(float)]
            return [row.astype(float) for row in obj]
        return []
    if isinstance(obj, dict):
        for key in ("curves", "curve", "values", "value", "strain", "data"):
            if key in obj:
                return _curves_from_obj(obj[key])
        curves: List[np.ndarray] = []
        for val in obj.values():
            curves.extend(_curves_from_obj(val))
        return curves
    if isinstance(obj, (list, tuple)):
        if not obj:
            return []
        if all(_is_number(v) for v in obj):
            return [np.asarray(obj, dtype=float)]
        if all(isinstance(v, (list, tuple, np.ndarray)) for v in obj):
            try:
                lengths = [len(v) for v in obj]  # type: ignore[arg-type]
            except Exception:
                lengths = []
            if lengths and all(l == 2 for l in lengths):
                flat = []
                ok = True
                for pair in obj:
                    if not isinstance(pair, (list, tuple, np.ndarray)) or len(pair) != 2:
                        ok = False
                        break
                    if not (_is_number(pair[0]) and _is_number(pair[1])):
                        ok = False
                        break
                    flat.append([float(pair[0]), float(pair[1])])
                if ok and len(flat) > 2:
                    return [np.asarray(flat, dtype=float)]
        curves = []
        for item in obj:
            curves.extend(_curves_from_obj(item))
        return curves
    return []


def _parse_curves(val: object) -> List[np.ndarray]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    if isinstance(val, (list, tuple, dict, np.ndarray)):
        return _curves_from_obj(val)
    if isinstance(val, str):
        text = val.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except Exception:
            try:
                parsed = ast.literal_eval(text)
            except Exception:
                return []
        return _curves_from_obj(parsed)
    return []


def _collect_curves(
    df: pd.DataFrame,
    curves_col: str,
    label_col: Optional[str],
    max_curves: int,
    max_rows: int,
    seed: int,
) -> List[Tuple[str, np.ndarray]]:
    if curves_col not in df.columns:
        return []
    rows = df[df[curves_col].notna()].copy()
    if rows.empty:
        return []
    if max_rows > 0 and len(rows) > max_rows:
        rows = rows.sample(n=max_rows, random_state=seed)
    collected: List[Tuple[str, np.ndarray]] = []
    for idx, row in rows.iterrows():
        label_val = row.get(label_col) if label_col else None
        label = str(label_val) if label_val is not None and not pd.isna(label_val) else f"row_{idx}"
        curves = _parse_curves(row[curves_col])
        for curve in curves:
            if curve is None or len(curve) < 2:
                continue
            collected.append((label, np.asarray(curve, dtype=float)))
            if len(collected) >= max_curves:
                return collected
    return collected


def _plot_curves(
    curves: Iterable[Tuple[str, np.ndarray]],
    title: str,
    out_path: Path,
    show_legend: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, curve in curves:
        curve = np.asarray(curve, dtype=float)
        if curve.ndim == 2 and curve.shape[1] == 2:
            ax.plot(curve[:, 0], curve[:, 1], alpha=0.85, linewidth=1.5, label=label)
        else:
            ax.plot(np.arange(len(curve)), curve, alpha=0.85, linewidth=1.5, label=label)
    ax.set_title(title)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Strain")
    if show_legend:
        ax.legend(fontsize=7, loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    logger.info("Saved plot: %s", out_path)


def _reduce_curves(curves: List[np.ndarray]) -> Optional[np.ndarray]:
    if not curves:
        return None
    for curve in curves:
        arr = np.asarray(curve, dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] > 2:
            return arr
    one_d = [np.asarray(c, dtype=float) for c in curves if np.asarray(c).ndim == 1 and len(c) > 1]
    if not one_d:
        return None
    min_len = min(len(c) for c in one_d)
    if min_len < 2:
        return None
    stack = np.stack([c[:min_len] for c in one_d], axis=0)
    return stack.mean(axis=0)


def _sanitize_filename(text: str) -> str:
    cleaned = []
    for ch in str(text):
        cleaned.append(ch if (ch.isalnum() or ch in ("-", "_")) else "_")
    return "".join(cleaned).strip("_") or "visit"


def _collect_visit_curves(
    df: pd.DataFrame,
    view_cols: Dict[str, Optional[str]],
    patient_col: Optional[str],
    visit_col: Optional[str],
    max_visits: int,
    seed: int,
) -> List[Tuple[str, str, Dict[str, np.ndarray]]]:
    tmp = df.copy()
    if patient_col is None:
        tmp["_patient_key"] = "unknown"
    else:
        tmp["_patient_key"] = tmp[patient_col].astype(str).fillna("unknown")

    if visit_col is None:
        tmp["_visit_key"] = tmp.index.astype(str)
    else:
        dt = pd.to_datetime(tmp[visit_col], errors="coerce")
        tmp["_visit_key"] = dt.dt.strftime("%Y%m%d")
        missing = dt.isna()
        if missing.any():
            tmp.loc[missing, "_visit_key"] = tmp.loc[missing, visit_col].astype(str)

    tmp["_visit_id"] = tmp["_patient_key"] + "__" + tmp["_visit_key"]
    visit_ids = tmp["_visit_id"].dropna().unique().tolist()
    if max_visits > 0 and len(visit_ids) > max_visits:
        rng = np.random.default_rng(seed)
        visit_ids = rng.choice(visit_ids, size=max_visits, replace=False).tolist()

    results: List[Tuple[str, str, Dict[str, np.ndarray]]] = []
    for visit_id in visit_ids:
        group = tmp[tmp["_visit_id"] == visit_id]
        if group.empty:
            continue
        view_curves: Dict[str, np.ndarray] = {}
        for view_name, col in view_cols.items():
            if not col or col not in group.columns:
                continue
            curves: List[np.ndarray] = []
            for val in group[col].dropna().tolist():
                curves.extend(_parse_curves(val))
            reduced = _reduce_curves(curves)
            if reduced is not None:
                view_curves[view_name] = reduced
        if not view_curves:
            continue
        patient = group["_patient_key"].iloc[0]
        visit_key = group["_visit_key"].iloc[0]
        results.append((patient, visit_key, view_curves))
    return results


def _plot_visit_curves(
    patient: str,
    visit_key: str,
    view_curves: Dict[str, np.ndarray],
    out_dir: Path,
    show_legend: bool,
) -> Path:
    colors = {"A2C": "#1f77b4", "A3C": "#2ca02c", "A4C": "#d62728"}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for view, curve in view_curves.items():
        curve = np.asarray(curve, dtype=float)
        color = colors.get(view, None)
        if curve.ndim == 2 and curve.shape[1] == 2:
            ax.plot(curve[:, 0], curve[:, 1], alpha=0.9, linewidth=1.6, label=view, color=color)
        else:
            ax.plot(np.arange(len(curve)), curve, alpha=0.9, linewidth=1.6, label=view, color=color)
    ax.set_title(f"{patient} | {visit_key}")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Strain")
    if show_legend:
        ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fname = f"{_sanitize_filename(patient)}_{_sanitize_filename(visit_key)}_strain_views.png"
    out_path = out_dir / fname
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    logger.info("Saved plot: %s", out_path)
    return out_path


def main() -> None:
    default_input = Path(r"F:\OneDrive - Technion\DS\Report_Ichilov_GLS_and_Strain_oron.xlsx")
    parser = argparse.ArgumentParser(description="Plot sample A2C/A4C strain curves from report.")
    parser.add_argument("--input-xlsx", type=Path, default=default_input)
    parser.add_argument("--a4c-col", type=str, default="A4C_STRAIN_CURVES_JSON")
    parser.add_argument("--a3c-col", type=str, default="A3C_STRAIN_CURVES_JSON")
    parser.add_argument("--a2c-col", type=str, default="A2C_STRAIN_CURVES_JSON")
    parser.add_argument("--max-curves", type=int, default=12)
    parser.add_argument("--max-rows", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-legend", action="store_true", default=False)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--visit-col", type=str, default="", help="Optional visit date column name.")
    parser.add_argument("--per-visit", dest="per_visit", action="store_true", default=True, help="Plot per-visit curves.")
    parser.add_argument(
        "--no-per-visit",
        dest="per_visit",
        action="store_false",
        help="Disable per-visit curve plots.",
    )
    parser.add_argument("--max-visits", type=int, default=12, help="Max per-visit plots (0 = all).")
    args = parser.parse_args()

    df = pd.read_excel(args.input_xlsx, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]

    label_col = _find_column(df, ["patient_id", "patient_num", "patient", "subject", "id", "name"])
    if label_col is None:
        logger.warning("Patient identifier column not found; using row index labels.")

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = args.input_xlsx.parent / "strain_curve_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    visit_col = None
    if args.visit_col:
        visit_col = args.visit_col if args.visit_col in df.columns else None
        if visit_col is None:
            logger.warning("Visit column '%s' not found; falling back to auto-detection.", args.visit_col)
    if visit_col is None:
        visit_col = _find_column(
            df,
            [
                "study_datetime",
                "study_date",
                "visit_date",
                "exam_date",
                "acquisition_date",
                "scan_date",
                "date",
            ],
        )
        if visit_col is None:
            logger.warning("Visit date column not found; using row index per visit.")

    a4c_curves = _collect_curves(
        df,
        args.a4c_col,
        label_col,
        max_curves=args.max_curves,
        max_rows=args.max_rows,
        seed=args.seed,
    )
    if a4c_curves:
        _plot_curves(
            a4c_curves,
            title=f"A4C strain curves (n={len(a4c_curves)})",
            out_path=out_dir / "a4c_strain_curves.png",
            show_legend=not args.no_legend,
        )
    else:
        logger.warning("No A4C curves found for column '%s'.", args.a4c_col)

    a2c_curves = _collect_curves(
        df,
        args.a2c_col,
        label_col,
        max_curves=args.max_curves,
        max_rows=args.max_rows,
        seed=args.seed + 1,
    )
    if a2c_curves:
        _plot_curves(
            a2c_curves,
            title=f"A2C strain curves (n={len(a2c_curves)})",
            out_path=out_dir / "a2c_strain_curves.png",
            show_legend=not args.no_legend,
        )
    else:
        logger.warning("No A2C curves found for column '%s'.", args.a2c_col)

    if args.per_visit:
        view_cols = {"A2C": args.a2c_col, "A3C": args.a3c_col, "A4C": args.a4c_col}
        visits = _collect_visit_curves(
            df,
            view_cols=view_cols,
            patient_col=label_col,
            visit_col=visit_col,
            max_visits=args.max_visits,
            seed=args.seed,
        )
        if not visits:
            logger.warning("No per-visit curves found to plot.")
        else:
            visit_dir = out_dir / "per_visit"
            visit_dir.mkdir(parents=True, exist_ok=True)
            for patient, visit_key, view_curves in visits:
                _plot_visit_curves(
                    patient=patient,
                    visit_key=visit_key,
                    view_curves=view_curves,
                    out_dir=visit_dir,
                    show_legend=not args.no_legend,
                )


if __name__ == "__main__":
    main()
