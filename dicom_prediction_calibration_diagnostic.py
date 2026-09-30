"""Post-hoc diagnostic: can training-prevalence shrinkage explain Brier gains?

Added after the primary experiment; not a new primary model or a tuned blend.
Every held-out prior uses only patients in that repeat's training partition.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score

ROOT=Path('D:/us')
OUT=ROOT/'output/dicom_prediction'


def main():
    source=OUT/'paired_oof_predictions.parquet'
    predictions=pd.read_parquet(source)
    base=predictions[predictions.model.eq('retained_strain_clinical')].reset_index(drop=True)
    folds=pd.read_csv(ROOT/'cardiotoxicity_cnn_length_ablation_results/cnn_length_ablation_patient_folds.csv')
    folds=folds[folds.role.eq('test')]
    priors=np.full((3,len(base)),np.nan)
    for (repeat,fold),group in folds.groupby(['repeat','fold']):
        test=base.patient_id.isin(group.patient_id).to_numpy()
        assert not set(base.loc[test,'patient_id'])&set(base.loc[~test,'patient_id'])
        priors[int(repeat),test]=base.loc[~test,'label'].mean()
    assert np.isfinite(priors).all()
    y=base.label.to_numpy()
    shrink=.75*base.score.to_numpy()+.25*priors.mean(0)
    scores={'retained_strain_clinical':base.score.to_numpy(),'training_prior_blend':shrink}
    for name in ['strain_clinical_plus_echoprime','strain_clinical_plus_panecho']:
        scores[name]=base[['transition_id']].merge(predictions[predictions.model.eq(name)],on='transition_id',validate='one_to_one').score.to_numpy()
    results={name:dict(auc=roc_auc_score(y,p),ap=average_precision_score(y,p),brier=brier_score_loss(y,p)) for name,p in scores.items()}
    patients=base.patient_id.to_numpy();unique=np.unique(patients)
    indices={p:np.flatnonzero(patients==p) for p in unique};rng=np.random.default_rng(20260914)
    differences={name:[] for name in scores if name!='training_prior_blend'}
    losses={name:(y-p)**2 for name,p in scores.items()}
    for _ in range(2000):
        take=np.concatenate([indices[p] for p in rng.choice(unique,len(unique),replace=True)])
        for name in differences:
            differences[name].append(float((losses[name][take]-losses['training_prior_blend'][take]).mean()))
    for name,values in differences.items():
        results[name]['delta_brier_vs_prior']=results[name]['brier']-results['training_prior_blend']['brier']
        results[name]['delta_brier_vs_prior_ci']=np.quantile(values,[.025,.975]).tolist()
    result=dict(post_hoc=True,description='Fixed 25% outer-training prevalence / 75% retained OOF blend; no pixels, no tuning',
                paired_predictions_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),results=results)
    (OUT/'calibration_diagnostic.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
