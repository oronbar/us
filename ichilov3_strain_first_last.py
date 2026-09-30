"""Patient-held-out first/last strain classifier; labels are visit position proxies.

Train: .venv/Scripts/python ichilov3_strain_first_last.py
Infer: .venv/Scripts/python ichilov3_strain_first_last.py --predict curves.npz
NPZ inference input: curves float array [N,96], individual segment/layer
curves in native signed strain percent, normalized cardiac cycle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, roc_auc_score, brier_score_loss)
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / 'output/ichilov3_strain_first_last'


class CurveFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, mode='summary'):
        self.mode = mode

    def fit(self, X, y=None):
        self.transform(X)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1:] != (96,) or not np.isfinite(X).all():
            raise ValueError('Expected finite [N,96] individual strain curves')
        if self.mode == 'peak':
            return X.min(-1).reshape(-1, 1)
        if self.mode == 'waveform':
            # Fixed temporal subsampling, no population-fitted curve processing.
            return X[..., ::4].reshape(len(X), -1)
        if self.mode != 'summary':
            raise ValueError(self.mode)
        features = [X.min(-1), X.max(-1), X.mean(-1), X.std(-1),
                    X.argmin(-1) / 95., X[..., -1],
                    np.abs(np.diff(X, axis=-1)).mean(-1)]
        return np.stack(features, axis=-1).reshape(len(X), -1)


def cohort_from_tables(visits, curves, patient_ids, deterioration):
    from ichilov3_align_modalities import strain_tensors
    visits = visits[visits.patient_id.isin(patient_ids)].copy()
    if visits.visit_id.duplicated().any():
        raise ValueError('Duplicate visit ID')
    visits['study_datetime'] = pd.to_datetime(visits.study_datetime, errors='raise')
    if visits.study_datetime.isna().any():
        raise ValueError('Missing visit dates')
    tensors = strain_tensors(curves)
    rows, audit = [], []
    for patient, group in visits.groupby('patient_id'):
        group = group.sort_values('study_datetime')
        reason = 'included'
        if len(group) < 2:
            reason = 'fewer_than_two_visits'
        elif group.study_datetime.duplicated().any():
            reason = 'ambiguous_visit_times'
        elif not group.iloc[[0, -1]].visit_id.isin(tensors).all():
            reason = 'incomplete_true_endpoint_curves'
        first_gls = abs(float(group.iloc[0].gls_mid_peak_avg))
        last_gls = abs(float(group.iloc[-1].gls_mid_peak_avg))
        declines = 1 - group.gls_mid_peak_avg.abs().to_numpy(float) / first_gls if first_gls > 0 else np.full(len(group), np.nan)
        decline = declines[-1] if deterioration == 'last' else np.nanmax(declines)
        if reason == 'included':
            if not np.isfinite(first_gls) or first_gls <= 0 or not np.isfinite(decline):
                reason = 'missing_gls_for_deterioration'
            elif decline < .15:
                reason = 'no_qualifying_deterioration'
        audit.append(dict(patient_id=patient, n_visits=len(group), status=reason,
                          first_gls_magnitude=first_gls, last_gls_magnitude=last_gls,
                          qualifying_relative_decline=decline))
        if reason != 'included':
            continue
        for label, index in [(0, 0), (1, -1)]:
            r = group.iloc[index]
            rows.append(dict(patient_id=patient, visit_id=r.visit_id,
                             study_datetime=r.study_datetime, label=label,
                             label_name='bad_last' if label else 'good_first'))
    if not rows:
        raise ValueError('No eligible endpoint pairs')
    segment_rows, samples = [], []
    for row in rows:
        tensor = tensors[row['visit_id']]
        for segment in range(18):
            for layer_index, layer in enumerate(['endo', 'mid']):
                segment_rows.append(dict(**row, segment_number=segment + 1, layer=layer))
                samples.append(tensor[segment, layer_index])
    cohort = pd.DataFrame(segment_rows)
    X = np.stack(samples)
    return cohort, X, pd.DataFrame(audit)


def search_model():
    pipeline = Pipeline([('features', CurveFeatures()), ('scale', StandardScaler()),
                         ('model', LogisticRegression(max_iter=3000))])
    grid = [dict(features__mode=['summary', 'waveform'], model__C=[0.01, 0.1, 1.]),
            dict(features__mode=['summary'], scale=['passthrough'],
                 model=[ExtraTreesClassifier(n_estimators=250, random_state=42, n_jobs=1)],
                 model__min_samples_leaf=[3, 8])]
    return GridSearchCV(pipeline, grid, cv=GroupKFold(3, shuffle=True, random_state=43),
                        scoring='roc_auc', n_jobs=1, refit=True, error_score='raise')


def metrics(y, p):
    pred = p >= 0.5
    return dict(roc_auc=float(roc_auc_score(y, p)),
                accuracy=float(accuracy_score(y, pred)),
                balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                brier_score=float(brier_score_loss(y, p)),
                confusion_matrix=confusion_matrix(y, pred).tolist())


def run(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if args.predict:
        model = joblib.load(out / 'classifier.joblib')
        with np.load(args.predict, allow_pickle=False) as src:
            X = src['curves']
        p = model.predict_proba(X)[:, 1]
        pd.DataFrame(dict(probability_bad_last=p, prediction=np.where(p >= .5, 'bad_last', 'good_first'))).to_csv(out / 'inference_predictions.csv', index=False)
        return
    paths = {'visits': ROOT / 'amber_full_105_preprocessed/Ichilov_july_visits.parquet',
             'curves': ROOT / 'amber_full_105_preprocessed/Ichilov_july_dataset.parquet'}
    ids = {p.name for p in args.dataset.iterdir() if p.is_dir()}
    visits = pd.read_parquet(paths['visits'])
    curves = pd.read_parquet(paths['curves'], columns=['visit_id', 'curve_family', 'layer', 'segment_number', 'resampled_values'])
    if not args.deterioration:
        raise ValueError('Specify --deterioration last or any after agreeing on the cohort definition')
    cohort, X, audit = cohort_from_tables(visits, curves, ids, args.deterioration)
    cohort.to_csv(out / 'cohort.csv', index=False)
    audit.to_csv(out / 'cohort_audit.csv', index=False)
    y = cohort.label.to_numpy()
    groups = cohort.patient_id.to_numpy()
    if len(np.unique(groups)) < 8:
        raise ValueError('At least eight paired patients required for nested evaluation')
    p = np.full(len(y), np.nan)
    peak_p = np.full(len(y), np.nan)
    folds = np.full(len(y), -1)
    logs = []
    outer = GroupKFold(5, shuffle=True, random_state=42)
    for fold, (train, test) in enumerate(outer.split(X, y, groups)):
        assert not set(groups[train]) & set(groups[test])
        search = search_model()
        search.fit(X[train], y[train], groups=groups[train])
        p[test] = search.predict_proba(X[test])[:, 1]
        baseline = Pipeline([('features', CurveFeatures('peak')), ('scale', StandardScaler()),
                             ('model', LogisticRegression(C=1., max_iter=3000))])
        baseline.fit(X[train], y[train])
        peak_p[test] = baseline.predict_proba(X[test])[:, 1]
        folds[test] = fold
        log = dict(fold=fold, train_patients=len(set(groups[train])),
                   test_patients=len(set(groups[test])), inner_auc=search.best_score_,
                   selected={k: str(v) for k, v in search.best_params_.items()},
                   **metrics(y[test], p[test]))
        logs.append(log)
        print(json.dumps(log), flush=True)
    assert np.isfinite(p).all() and (folds >= 0).all()
    predictions = cohort.assign(fold=folds, probability_bad_last=p, peak_baseline_probability=peak_p,
                                prediction=(p >= .5).astype(int))
    predictions.to_csv(out / 'oof_predictions.csv', index=False)
    subgroups = []
    for (layer, segment), subset in predictions.groupby(['layer', 'segment_number']):
        subgroups.append(dict(layer=layer, segment_number=int(segment), n_curves=len(subset),
                              **metrics(subset.label, subset.probability_bad_last)))
    pd.DataFrame(subgroups).to_csv(out / 'segment_layer_metrics.csv', index=False)
    paired = predictions.groupby(['patient_id', 'label']).probability_bad_last.mean().unstack('label')
    paired_score = ((paired[1] > paired[0]).astype(float) + .5 * (paired[1] == paired[0])).to_numpy()
    # Resample patients with both visits together; never bootstrap individual curves.
    rng = np.random.default_rng(2026)
    blocks = [np.flatnonzero(groups == patient) for patient in paired.index]
    boot = []
    for _ in range(2000):
        chosen = rng.integers(0, len(blocks), len(blocks))
        idx = np.concatenate([blocks[i] for i in chosen])
        auc = roc_auc_score(y[idx], p[idx])
        boot.append([auc, accuracy_score(y[idx], p[idx] >= .5), paired_score[chosen].mean(),
                     auc - roc_auc_score(y[idx], peak_p[idx])])
    ci = np.quantile(boot, [.025, .975], axis=0)
    report = dict(task='first versus last visit; not validated clinical or technical quality',
                  n_patients=len(paired), n_visits=cohort.visit_id.nunique(), n_segment_curves=len(cohort),
                  deterioration=f'>=15% relative midwall GLS magnitude decline at {args.deterioration} follow-up',
                  labels={'0': 'good_first', '1': 'bad_last'},
                  evaluation='5 outer patient folds; 3 inner patient folds for model/hyperparameter selection',
                  metrics=metrics(y, p), paired_order_accuracy=float(paired_score.mean()),
                  peak_only_baseline_metrics=metrics(y, peak_p),
                  bootstrap_95_ci={key: ci[:, i].tolist() for i, key in enumerate(['roc_auc', 'accuracy', 'paired_order_accuracy', 'auc_improvement_vs_peak_only'])},
                  folds=logs, source_sha256={k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in paths.items()})
    (out / 'evaluation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    final = search_model()
    final.fit(X, y, groups=groups)
    joblib.dump(final.best_estimator_, out / 'classifier.joblib')
    report['final_model_parameters'] = {k: str(v) for k, v in final.best_params_.items()}
    (out / 'evaluation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    RocCurveDisplay.from_predictions(y, p, ax=axes[0], name='Patient-held-out', plot_chance_level=True)
    ConfusionMatrixDisplay.from_predictions(y, p >= .5, ax=axes[1], display_labels=['First', 'Last'], colorbar=False)
    fig.tight_layout()
    fig.savefig(out / 'evaluation.png', dpi=180)
    plt.close(fig)
    m = report['metrics']
    (out / 'README.md').write_text(f'''# Ichilov3 first/last strain classifier

{len(paired)} patients; {cohort.visit_id.nunique()} visits; {len(cohort)} individual segment/layer curves, balanced first/last classes. Intermediate visits excluded.
All report patients were checked against the ichilov3 patient folders.
Inclusion: {report['deterioration']}. See cohort_audit.csv for all exclusions.

Patient-held-out nested CV: ROC AUC {m['roc_auc']:.3f}, accuracy {m['accuracy']:.3f}.
Peak-only logistic baseline on identical outer folds: AUC {report['peak_only_baseline_metrics']['roc_auc']:.3f}.
Full model minus peak-only AUC paired-bootstrap 95% interval: {report['bootstrap_95_ci']['auc_improvement_vs_peak_only']}.
Within-patient mean last-visit score exceeds mean first-visit score in {paired_score.mean():.1%} of pairs (ties count half).
See evaluation.json for patient-bootstrap intervals and fold results. Intervals condition on this split and fitted models.

Each input is ONE longitudinal strain segment/layer curve, 96 phase-normalized values in native signed strain percent.
The 18 segments and endocardial/midwall layers are pooled; segment/layer identity is not a model input.
Technical reanalyses are averaged within visit/segment/layer using the existing deterministic preparation.
No dates, identifiers, visit number, future measurements, or precomputed learned embeddings enter the model.
Scaling and model selection are fitted inside patient-disjoint training folds.
True endpoints are chosen before checking curve completeness; missing endpoints never silently shift inward.

The labels are first/last visit proxies, not adjudicated good/bad quality, health, or cardiotoxicity.
Patient inclusion uses report-derived GLS deterioration, not clinician adjudication. Individual segments need not deteriorate.
Selection on GLS decline favors amplitude separation: results apply only to this selected cohort.
This model may capture longitudinal acquisition or treatment differences; it has no external validation.
The final classifier is refitted on all patients; evaluate performance using oof_predictions.csv, not training predictions.
The 0.5 threshold is fixed; scores have not been independently calibrated.

Run training: `.venv/Scripts/python ichilov3_strain_first_last.py --deterioration {args.deterioration}`
Run inference: `.venv/Scripts/python ichilov3_strain_first_last.py --predict curves.npz`
The NPZ must contain `curves` shaped [N,96], one segment/layer curve per row.
Load the joblib artifact only from a trusted source. Local cohort/prediction files contain patient IDs.
''', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ['folds', 'source_sha256']}, indent=2))


if __name__ == '__main__':
    # Keep the custom transformer importable when loading the saved artifact.
    from ichilov3_strain_first_last import run as main
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('E:/ichilov3'))
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--predict', type=Path)
    parser.add_argument('--deterioration', choices=['last', 'any'])
    with threadpool_limits(limits=2):
        main(parser.parse_args())
