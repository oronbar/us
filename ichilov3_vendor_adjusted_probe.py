"""Pre-specified additive manufacturer covariate in the patient-held-out EchoPrime GLS probe."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from ichilov3_train_late_fusion import video_preprocess
from ichilov3_train_fusion import atomic,hashfile

OUT=Path(r'D:\DS\ichilov3_temporal_trial_20260927\vendor_adjusted_probe')
OUT.mkdir(parents=True,exist_ok=True)
COHORT=Path(r'D:\DS\ichilov3_temporal_trial_20260927\physiology\cohort.parquet')
FOLDS=Path(r'D:\DS\ichilov3_aligned_20260927\patient_folds.parquet')
VIDEO=Path(r'D:\DS\ichilov3_temporal_trial_20260927\echoprime_visits.npz')
CLIPS=Path(r'D:\DS\ichilov3_preprocessing_audit_20260927\clip_audit.parquet')
BASE=Path(r'D:\DS\ichilov3_temporal_trial_20260927\physiology\oof_predictions.parquet')
ALPHAS=(.1,1.,10.,100.,1000.,10000.)
REP=('full_embedding','pca8_per_view')

cohort=pd.read_parquet(COHORT)
folds=pd.read_parquet(FOLDS)
clip=pd.read_parquet(CLIPS)
v=clip.groupby(['patient','visit_date']).manufacturer.agg(lambda x:x.iloc[0] if x.nunique()==1 else None).reset_index()
assert v.manufacturer.notna().all()
v=v.rename(columns={'patient':'patient_id'})
cohort=cohort.merge(v,on=['patient_id','visit_date'],validate='one_to_one')
vendor_names=sorted(cohort.manufacturer.unique())
vendor=np.stack([(cohort.manufacturer==m).to_numpy(float) for m in vendor_names],axis=1)
with np.load(VIDEO) as f:
    lookup={(str(p),str(d)):z for p,d,z in zip(f['patient_ids'],f['visit_dates'],f['embeddings'])}
    x=np.stack([lookup[(r.patient_id,r.visit_date)] for r in cohort.itertuples()])
assert np.isfinite(x).all()
groups=cohort.patient_id.to_numpy(str)
protocol=dict(input_hashes={k:hashfile(p) for k,p in dict(cohort=COHORT,folds=FOLDS,video=VIDEO,clip_audit=CLIPS,baseline_oof=BASE).items()},
    target='same-visit Mid and Endo GLS magnitude',features='EchoPrime three-view frozen video embeddings plus three one-hot manufacturer indicators',
    inner='3 grouped folds minimize MAE; ridge alpha and full vs PCA8 selected inside outer training only',
    outer='retained 3x5 patient-held-out folds',interpretation='exploratory vendor-aware probe, not causal evidence of vendor effect',
    code_sha256=hashfile(Path(__file__)))
atomic(OUT/'protocol.json',protocol)
records=[]
with threadpool_limits(limits=4):
    for (repeat,fold),assignment in folds.groupby(['repeat','fold']):
        heldout=np.isin(groups,assignment.patient_id)
        for target,column in [('mid_gls','gls_mid_magnitude'),('endo_gls','gls_endo_magnitude')]:
            y=cohort[column].to_numpy(float)
            train=np.flatnonzero(np.isfinite(y)&~heldout);test=np.flatnonzero(np.isfinite(y)&heldout)
            assert not set(groups[train])&set(groups[test])
            seed=20260929+int(repeat)*5+int(fold)
            splits=list(GroupKFold(n_splits=3,shuffle=True,random_state=seed).split(np.zeros(len(train)),y[train],groups[train]))
            candidates=[]
            for representation in REP:
                scores={a:[] for a in ALPHAS}
                for ia,ib in splits:
                    fit,val=train[ia],train[ib]
                    assert not set(groups[fit])&set(groups[val])
                    z,w,_=video_preprocess(x,fit,val,representation,seed)
                    z=np.concatenate([z,vendor[fit]],axis=1);w=np.concatenate([w,vendor[val]],axis=1)
                    for alpha in ALPHAS:
                        model=Ridge(alpha=alpha,solver='lsqr',tol=1e-5).fit(z,y[fit])
                        scores[alpha].append(mean_absolute_error(y[val],model.predict(w)))
                candidates.extend(dict(representation=representation,alpha=alpha,inner_mae=np.mean(errors)) for alpha,errors in scores.items())
            best=min(candidates,key=lambda q:q['inner_mae'])
            z,w,_=video_preprocess(x,train,test,best['representation'],seed)
            z=np.concatenate([z,vendor[train]],axis=1);w=np.concatenate([w,vendor[test]],axis=1)
            model=Ridge(alpha=best['alpha'],solver='lsqr',tol=1e-5).fit(z,y[train])
            for i,p in zip(test,model.predict(w)):
                records.append(dict(visit_id=cohort.visit_id.iloc[i],patient_id=groups[i],target=target,value=y[i],prediction=p,
                                    manufacturer=cohort.manufacturer.iloc[i],repeat=int(repeat),fold=int(fold)))
            print(target,repeat,fold,best['representation'],best['alpha'],flush=True)
raw=pd.DataFrame(records)
assert raw.groupby(['target','visit_id']).size().eq(3).all()
raw.to_parquet(OUT/'fold_predictions.parquet',index=False)
oof=raw.groupby(['target','visit_id','patient_id','manufacturer','value'],as_index=False).prediction.mean()
oof.to_parquet(OUT/'oof_predictions.parquet',index=False)
baseline=pd.read_parquet(BASE)
baseline=baseline[baseline.model.eq('echoprime')&baseline.target.isin(['mid_gls','endo_gls'])]
pair=oof.merge(baseline[['target','visit_id','patient_id','value','prediction']],on=['target','visit_id','patient_id','value'],suffixes=('_vendor','_baseline'),validate='one_to_one')
assert len(pair)==796
rng=np.random.default_rng(20261003);rows=[]
for target,g in pair.groupby('target'):
    for label,h in [('all',g),*[(m,g[g.manufacturer==m]) for m in vendor_names]]:
        y=h.value.to_numpy();a=h.prediction_baseline.to_numpy();b=h.prediction_vendor.to_numpy()
        pts=h.patient_id.unique();grp=h.patient_id.to_numpy();index={p:np.flatnonzero(grp==p) for p in pts}
        samples=[]
        for _ in range(2000):
            ix=np.concatenate([index[p] for p in rng.choice(pts,len(pts),replace=True)])
            if len(np.unique(y[ix]))<2:continue
            samples.append((mean_absolute_error(y[ix],b[ix])-mean_absolute_error(y[ix],a[ix]),
                            np.corrcoef(y[ix],b[ix])[0,1]-np.corrcoef(y[ix],a[ix])[0,1]))
        s=np.asarray(samples)
        rows.append(dict(target=target,manufacturer=label,n=len(h),patients=len(pts),
            baseline_mae=float(mean_absolute_error(y,a)),vendor_mae=float(mean_absolute_error(y,b)),
            delta_mae_ci=np.quantile(s[:,0],[.025,.975]).tolist(),
            baseline_r=float(np.corrcoef(y,a)[0,1]),vendor_r=float(np.corrcoef(y,b)[0,1]),
            delta_r_ci=np.quantile(s[:,1],[.025,.975]).tolist()))
pd.DataFrame(rows).to_json(OUT/'metrics.json',orient='records',indent=2)
lines=['# Additive manufacturer probe','',
    'Frozen heartbeat-sampled EchoPrime plus one-hot manufacturer; same visits and patient folds as the video-only baseline. Inner folds choose ridge regularization and representation. Exploratory post-hoc analysis.',
    '', '| Target | Manufacturer | N | MAE baseline → adjusted | Pearson r baseline → adjusted | 95% CI for MAE change | 95% CI for r change |',
    '|---|---|---:|---:|---:|---:|---:|']
for r in rows:
    lines.append(f'| {r["target"]} | {r["manufacturer"]} | {r["n"]} | {r["baseline_mae"]:.3f} → {r["vendor_mae"]:.3f} | {r["baseline_r"]:.3f} → {r["vendor_r"]:.3f} | {r["delta_mae_ci"][0]:+.3f} to {r["delta_mae_ci"][1]:+.3f} | {r["delta_r_ci"][0]:+.3f} to {r["delta_r_ci"][1]:+.3f} |')
lines+=['','Manufacturer differences can reflect patient mix, clip selection or label process. This probe cannot establish causal domain shift, and a gain selected post-hoc requires external or untouched validation.']
(OUT/'results.md').write_text('\n'.join(lines),encoding='utf-8')
print('Vendor-adjusted comparison saved to results.md')
