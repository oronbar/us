import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ichilov_pipeline2_utils import add_gls_from_report


REQUIRED_COLS = [
    "patient_id",
    "study_datetime",
    "view",
    "source_dicom",
    "frame_index",
    "end_diastole",
    "end_systole",
    "embedding",
]


def _as_matrix(series: pd.Series) -> np.ndarray:
    return np.stack(series.apply(np.asarray).to_numpy()).astype(np.float32)


def _clean_patient_id(x: Any) -> str:
    s = str(x)
    if s.endswith(".0"):
        s = s[:-2]
    return s


def _run_linear_probe(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    test_size: float,
    alpha: float = 1e-4,
) -> Dict[str, Any]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    clf = make_pipeline(
        StandardScaler(),
        SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=alpha,
            max_iter=3000,
            tol=1e-4,
            n_jobs=-1,
            random_state=seed,
        ),
    )
    clf.fit(X_train, y_train)
    pred_train = clf.predict(X_train)
    pred_test = clf.predict(X_test)

    report = classification_report(y_test, pred_test, output_dict=True, zero_division=0)
    weighted_f1 = float(report.get("weighted avg", {}).get("f1-score", np.nan))
    macro_f1 = float(report.get("macro avg", {}).get("f1-score", np.nan))

    return {
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_classes": int(np.unique(y).size),
        "train_accuracy": float(accuracy_score(y_train, pred_train)),
        "test_accuracy": float(accuracy_score(y_test, pred_test)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, pred_test)),
        "test_weighted_f1": weighted_f1,
        "test_macro_f1": macro_f1,
    }


def _mean_embedding_by_visit(df: pd.DataFrame, edes_only: bool) -> pd.DataFrame:
    work = df.copy()
    if edes_only:
        work = work[(work["frame_index"] == work["end_diastole"]) | (work["frame_index"] == work["end_systole"])].copy()

    rows: List[Dict[str, Any]] = []
    for (pid, dt), grp in work.groupby(["patient_id", "study_datetime"], sort=False):
        emb = _as_matrix(grp["embedding"]).mean(axis=0)
        rows.append(
            {
                "patient_id": pid,
                "study_datetime": dt,
                "visit_embedding": emb,
                "n_frames": int(len(grp)),
            }
        )
    return pd.DataFrame(rows)


def _visit_gls(df: pd.DataFrame) -> pd.DataFrame:
    cine = (
        df.groupby(["patient_id", "study_datetime", "source_dicom"], as_index=False)["gls"]
        .mean()
        .dropna(subset=["gls"])
    )
    visit = cine.groupby(["patient_id", "study_datetime"], as_index=False)["gls"].mean()
    return visit


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    an = np.linalg.norm(a)
    bn = np.linalg.norm(b)
    if an == 0.0 or bn == 0.0:
        return np.nan
    return float(1.0 - (np.dot(a, b) / (an * bn)))


def _visit_change_table(visit_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for pid, grp in visit_df.groupby("patient_id"):
        grp = grp.sort_values("study_datetime")
        if len(grp) < 2:
            continue
        vals = grp.to_dict("records")
        for i in range(len(vals) - 1):
            cur = vals[i]
            nxt = vals[i + 1]
            emb_cur = np.asarray(cur["visit_embedding"], dtype=np.float32)
            emb_nxt = np.asarray(nxt["visit_embedding"], dtype=np.float32)
            delta_gls = float(nxt["gls"] - cur["gls"])
            rows.append(
                {
                    "patient_id": pid,
                    "study_datetime_prev": cur["study_datetime"],
                    "study_datetime_next": nxt["study_datetime"],
                    "days_between": float((nxt["study_datetime"] - cur["study_datetime"]).days),
                    "gls_prev": float(cur["gls"]),
                    "gls_next": float(nxt["gls"]),
                    "delta_gls": delta_gls,
                    "abs_delta_gls": abs(delta_gls),
                    "delta_emb_l2": float(np.linalg.norm(emb_nxt - emb_cur)),
                    "delta_emb_cosine": _cosine_distance(emb_cur, emb_nxt),
                }
            )
    return pd.DataFrame(rows)


def _corr_metrics(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        return {
            "n_pairs": int(len(x)),
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
        }
    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)
    return {
        "n_pairs": int(len(x)),
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_rho": float(sr),
        "spearman_p": float(sp),
    }


def _save_scatter(df: pd.DataFrame, xcol: str, ycol: str, out_png: Path, title: str) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df[xcol], df[ycol], s=18, alpha=0.75, color="#1f77b4")
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title(title)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def build_report(
    embeddings_parquet: Path,
    report_xlsx: Path,
    output_md: Path,
    output_json: Path,
    output_dir: Path,
    seed: int,
    test_size: float,
) -> Dict[str, Any]:
    df = pd.read_parquet(embeddings_parquet, columns=REQUIRED_COLS)
    df = df.copy()
    df["patient_id"] = df["patient_id"].map(_clean_patient_id)
    df["study_datetime"] = pd.to_datetime(df["study_datetime"], errors="coerce")
    df = df.dropna(subset=["study_datetime"]).reset_index(drop=True)

    X = _as_matrix(df["embedding"])

    # Probe 1: view classification.
    y_view = df["view"].astype(str).to_numpy()
    probe_view = _run_linear_probe(X, y_view, seed=seed, test_size=test_size)

    # Probe 2: patient ID classification.
    y_pid = df["patient_id"].astype(str).to_numpy()
    probe_pid = _run_linear_probe(X, y_pid, seed=seed, test_size=test_size)

    # Probe 3: visit-change signal.
    gls_df = add_gls_from_report(df, report_xlsx)
    gls_non_null = int(gls_df["gls"].notna().sum())

    visit_gls = _visit_gls(gls_df)

    visit_emb_all = _mean_embedding_by_visit(gls_df, edes_only=False)
    visit_emb_edes = _mean_embedding_by_visit(gls_df, edes_only=True)

    visit_all = visit_emb_all.merge(visit_gls, on=["patient_id", "study_datetime"], how="inner")
    visit_edes = visit_emb_edes.merge(visit_gls, on=["patient_id", "study_datetime"], how="inner")

    pairs_all = _visit_change_table(visit_all)
    pairs_edes = _visit_change_table(visit_edes)

    corr_all_abs_l2 = _corr_metrics(pairs_all["delta_emb_l2"].to_numpy(), pairs_all["abs_delta_gls"].to_numpy())
    corr_all_abs_cos = _corr_metrics(pairs_all["delta_emb_cosine"].to_numpy(), pairs_all["abs_delta_gls"].to_numpy())
    corr_all_signed_l2 = _corr_metrics(pairs_all["delta_emb_l2"].to_numpy(), pairs_all["delta_gls"].to_numpy())
    corr_all_signed_cos = _corr_metrics(pairs_all["delta_emb_cosine"].to_numpy(), pairs_all["delta_gls"].to_numpy())

    corr_edes_abs_l2 = _corr_metrics(pairs_edes["delta_emb_l2"].to_numpy(), pairs_edes["abs_delta_gls"].to_numpy())
    corr_edes_abs_cos = _corr_metrics(pairs_edes["delta_emb_cosine"].to_numpy(), pairs_edes["abs_delta_gls"].to_numpy())
    corr_edes_signed_l2 = _corr_metrics(pairs_edes["delta_emb_l2"].to_numpy(), pairs_edes["delta_gls"].to_numpy())
    corr_edes_signed_cos = _corr_metrics(pairs_edes["delta_emb_cosine"].to_numpy(), pairs_edes["delta_gls"].to_numpy())

    output_dir.mkdir(parents=True, exist_ok=True)
    scatter_all = output_dir / "visit_change_all_frames_l2_vs_abs_delta_gls.png"
    scatter_edes = output_dir / "visit_change_edes_frames_l2_vs_abs_delta_gls.png"
    _save_scatter(
        pairs_all,
        xcol="delta_emb_l2",
        ycol="abs_delta_gls",
        out_png=scatter_all,
        title="Visit change signal (all frames): L2(emb delta) vs |dGLS|",
    )
    _save_scatter(
        pairs_edes,
        xcol="delta_emb_l2",
        ycol="abs_delta_gls",
        out_png=scatter_edes,
        title="Visit change signal (ED/ES only): L2(emb delta) vs |dGLS|",
    )

    results: Dict[str, Any] = {
        "inputs": {
            "embeddings_parquet": str(embeddings_parquet),
            "report_xlsx": str(report_xlsx),
            "seed": int(seed),
            "test_size": float(test_size),
        },
        "dataset_summary": {
            "n_rows": int(len(df)),
            "n_patients": int(df["patient_id"].nunique()),
            "n_visits": int(df[["patient_id", "study_datetime"]].drop_duplicates().shape[0]),
            "n_views": int(df["view"].nunique()),
            "gls_labeled_rows": gls_non_null,
            "gls_labeled_row_fraction": float(gls_non_null / max(len(df), 1)),
            "gls_labeled_visits": int(visit_gls.shape[0]),
        },
        "probe_1_view_linear": probe_view,
        "probe_2_patient_id_linear": probe_pid,
        "probe_3_visit_change_signal": {
            "all_frames": {
                "n_visits_with_gls": int(visit_all.shape[0]),
                "n_pairs": int(pairs_all.shape[0]),
                "corr_l2_abs_delta_gls": corr_all_abs_l2,
                "corr_cosine_abs_delta_gls": corr_all_abs_cos,
                "corr_l2_signed_delta_gls": corr_all_signed_l2,
                "corr_cosine_signed_delta_gls": corr_all_signed_cos,
                "scatter_png": str(scatter_all),
            },
            "ed_es_only": {
                "n_visits_with_gls": int(visit_edes.shape[0]),
                "n_pairs": int(pairs_edes.shape[0]),
                "corr_l2_abs_delta_gls": corr_edes_abs_l2,
                "corr_cosine_abs_delta_gls": corr_edes_abs_cos,
                "corr_l2_signed_delta_gls": corr_edes_signed_l2,
                "corr_cosine_signed_delta_gls": corr_edes_signed_cos,
                "scatter_png": str(scatter_edes),
            },
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    def f(v: Any) -> str:
        if isinstance(v, float):
            if np.isnan(v):
                return "nan"
            return f"{v:.4f}"
        return str(v)

    md_lines = [
        "# Embedding Probe Report",
        "",
        "## Inputs",
        f"- Embeddings parquet: `{embeddings_parquet}`",
        f"- GLS report xlsx: `{report_xlsx}`",
        f"- Seed: `{seed}`",
        f"- Test size: `{test_size}`",
        "",
        "## Dataset Summary",
        f"- Rows: `{results['dataset_summary']['n_rows']}`",
        f"- Patients: `{results['dataset_summary']['n_patients']}`",
        f"- Visits (patient+datetime): `{results['dataset_summary']['n_visits']}`",
        f"- Views: `{results['dataset_summary']['n_views']}`",
        f"- GLS-labeled rows: `{results['dataset_summary']['gls_labeled_rows']}` ({100.0 * results['dataset_summary']['gls_labeled_row_fraction']:.2f}%)",
        f"- GLS-labeled visits: `{results['dataset_summary']['gls_labeled_visits']}`",
        "",
        "## Probe 1: Linear Probe for View",
        f"- Classes: `{results['probe_1_view_linear']['n_classes']}`",
        f"- Train size: `{results['probe_1_view_linear']['n_train']}`",
        f"- Test size: `{results['probe_1_view_linear']['n_test']}`",
        f"- Train accuracy: `{f(results['probe_1_view_linear']['train_accuracy'])}`",
        f"- Test accuracy: `{f(results['probe_1_view_linear']['test_accuracy'])}`",
        f"- Test balanced accuracy: `{f(results['probe_1_view_linear']['test_balanced_accuracy'])}`",
        f"- Test weighted F1: `{f(results['probe_1_view_linear']['test_weighted_f1'])}`",
        "",
        "## Probe 2: Linear Probe for Patient ID",
        f"- Classes: `{results['probe_2_patient_id_linear']['n_classes']}`",
        f"- Train size: `{results['probe_2_patient_id_linear']['n_train']}`",
        f"- Test size: `{results['probe_2_patient_id_linear']['n_test']}`",
        f"- Train accuracy: `{f(results['probe_2_patient_id_linear']['train_accuracy'])}`",
        f"- Test accuracy: `{f(results['probe_2_patient_id_linear']['test_accuracy'])}`",
        f"- Test balanced accuracy: `{f(results['probe_2_patient_id_linear']['test_balanced_accuracy'])}`",
        f"- Test weighted F1: `{f(results['probe_2_patient_id_linear']['test_weighted_f1'])}`",
        "",
        "## Probe 3: Visit-Change Signal vs dGLS",
        "### All-frames visit embedding",
        f"- Visits with GLS: `{results['probe_3_visit_change_signal']['all_frames']['n_visits_with_gls']}`",
        f"- Consecutive visit pairs: `{results['probe_3_visit_change_signal']['all_frames']['n_pairs']}`",
        f"- Corr[L2 emb delta, |dGLS|] Pearson r: `{f(results['probe_3_visit_change_signal']['all_frames']['corr_l2_abs_delta_gls']['pearson_r'])}` (p=`{f(results['probe_3_visit_change_signal']['all_frames']['corr_l2_abs_delta_gls']['pearson_p'])}`)",
        f"- Corr[L2 emb delta, |dGLS|] Spearman rho: `{f(results['probe_3_visit_change_signal']['all_frames']['corr_l2_abs_delta_gls']['spearman_rho'])}` (p=`{f(results['probe_3_visit_change_signal']['all_frames']['corr_l2_abs_delta_gls']['spearman_p'])}`)",
        f"- Corr[cosine emb delta, |dGLS|] Pearson r: `{f(results['probe_3_visit_change_signal']['all_frames']['corr_cosine_abs_delta_gls']['pearson_r'])}` (p=`{f(results['probe_3_visit_change_signal']['all_frames']['corr_cosine_abs_delta_gls']['pearson_p'])}`)",
        f"- Corr[L2 emb delta, signed dGLS] Pearson r: `{f(results['probe_3_visit_change_signal']['all_frames']['corr_l2_signed_delta_gls']['pearson_r'])}` (p=`{f(results['probe_3_visit_change_signal']['all_frames']['corr_l2_signed_delta_gls']['pearson_p'])}`)",
        f"- Scatter: `{results['probe_3_visit_change_signal']['all_frames']['scatter_png']}`",
        "",
        "### ED/ES-only visit embedding",
        f"- Visits with GLS: `{results['probe_3_visit_change_signal']['ed_es_only']['n_visits_with_gls']}`",
        f"- Consecutive visit pairs: `{results['probe_3_visit_change_signal']['ed_es_only']['n_pairs']}`",
        f"- Corr[L2 emb delta, |dGLS|] Pearson r: `{f(results['probe_3_visit_change_signal']['ed_es_only']['corr_l2_abs_delta_gls']['pearson_r'])}` (p=`{f(results['probe_3_visit_change_signal']['ed_es_only']['corr_l2_abs_delta_gls']['pearson_p'])}`)",
        f"- Corr[L2 emb delta, |dGLS|] Spearman rho: `{f(results['probe_3_visit_change_signal']['ed_es_only']['corr_l2_abs_delta_gls']['spearman_rho'])}` (p=`{f(results['probe_3_visit_change_signal']['ed_es_only']['corr_l2_abs_delta_gls']['spearman_p'])}`)",
        f"- Corr[cosine emb delta, |dGLS|] Pearson r: `{f(results['probe_3_visit_change_signal']['ed_es_only']['corr_cosine_abs_delta_gls']['pearson_r'])}` (p=`{f(results['probe_3_visit_change_signal']['ed_es_only']['corr_cosine_abs_delta_gls']['pearson_p'])}`)",
        f"- Corr[L2 emb delta, signed dGLS] Pearson r: `{f(results['probe_3_visit_change_signal']['ed_es_only']['corr_l2_signed_delta_gls']['pearson_r'])}` (p=`{f(results['probe_3_visit_change_signal']['ed_es_only']['corr_l2_signed_delta_gls']['pearson_p'])}`)",
        f"- Scatter: `{results['probe_3_visit_change_signal']['ed_es_only']['scatter_png']}`",
        "",
        "## Interpretation Guide",
        "- Probe 1 high score means view information is linearly recoverable from embeddings.",
        "- Probe 2 near-perfect score means embeddings strongly encode patient identity (high leakage risk for patient-overlapping splits).",
        "- Probe 3 near-zero correlations mean embedding change is weakly aligned with visit-level GLS change.",
    ]

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(md_lines), encoding="utf-8")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run view/patient probes and visit-change vs GLS report.")
    parser.add_argument(
        "--embeddings-parquet",
        type=Path,
        default=Path(
            r"C:\Users\oron\OneDrive - Technion\Experiments\DinoPipeline_21\frame_embeddings\Ichilov_frame_embeddings_DinoPipeline_21.parquet"
        ),
    )
    parser.add_argument(
        "--report-xlsx",
        type=Path,
        default=Path(r"C:\Users\oron\OneDrive - Technion\DS\Report_Ichilov_GLS_and_Strain_oron.xlsx"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(r"D:\us\embedding_probe_report_DinoPipeline_21.md"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(r"D:\us\embedding_probe_report_DinoPipeline_21.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"D:\us\embedding_probe_report_artifacts"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    results = build_report(
        embeddings_parquet=args.embeddings_parquet,
        report_xlsx=args.report_xlsx,
        output_md=args.output_md,
        output_json=args.output_json,
        output_dir=args.output_dir,
        seed=args.seed,
        test_size=args.test_size,
    )

    print(f"Saved report MD: {args.output_md}")
    print(f"Saved report JSON: {args.output_json}")
    print(
        "View probe test accuracy:",
        f"{results['probe_1_view_linear']['test_accuracy']:.4f}",
    )
    print(
        "Patient-ID probe test accuracy:",
        f"{results['probe_2_patient_id_linear']['test_accuracy']:.4f}",
    )
    print(
        "Visit-change Pearson r (all-frames, L2 vs |dGLS|):",
        f"{results['probe_3_visit_change_signal']['all_frames']['corr_l2_abs_delta_gls']['pearson_r']:.4f}",
    )


if __name__ == "__main__":
    main()
