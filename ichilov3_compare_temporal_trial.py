"""Compare heartbeat-length and reference-window OOF predictions."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,roc_auc_score,average_precision_score

ROOT=Path(r'D:\DS\ichilov3_temporal_trial_20260927')
rows=[];rng=np.random.default_rng(20261001)
sampling=pd.read_parquet(ROOT/'sampling_audit.parquet')
eligible=sampling.groupby(['patient','visit_date']).resampled.all()
eligible=set(eligible[eligible].index)
for task,baseline,filename,keys in [
    ('physiology',Path(r'D:\DS\ichilov3_physiology_audit_20260927'),'oof_predictions.parquet',['target','model','visit_id','patient_id','value']),
    ('future',Path(r'D:\DS\ichilov3_late_fusion_20260927'),'paired_oof_predictions.parquet',['model','transition_id','patient_id','label'])]:
    folder='physiology' if task=='physiology' else 'late_fusion'
    a=pd.read_parquet(baseline/filename);b=pd.read_parquet(ROOT/folder/filename)
    value='prediction' if task=='physiology' else 'score'
    pair=a[keys+[value]].merge(b[keys+[value]],on=keys,suffixes=('_native','_heartbeat'),validate='one_to_one')
    assert len(pair)==len(a)==len(b)
    if task=='physiology':
        identity=pd.read_parquet(r'D:\DS\ichilov3_aligned_20260927\visit_alignment.parquet')[['visit_id','patient_id','visit_date']]
        pair=pair.merge(identity,on=['visit_id','patient_id'],validate='many_to_one')
    else:
        identity=pd.read_parquet(r'D:\DS\ichilov3_aligned_20260927\cohort.parquet')[['transition_id','current_visit_date']].rename(columns={'current_visit_date':'visit_date'})
        pair=pair.merge(identity,on='transition_id',validate='many_to_one')
    pair['all_views_resampled']=[(p,str(d)) in eligible for p,d in zip(pair.patient_id,pair.visit_date)]
    for key,g in pair.groupby(['target','model'] if task=='physiology' else ['model']):
        key=key if isinstance(key,tuple) else (key,);model=key[-1]
        if not any(m in model for m in ['echoprime','panecho']):continue
        for subgroup,h in [('all',g),('all_three_views_resampled',g[g.all_views_resampled]),('at_least_one_view_fallback',g[~g.all_views_resampled])]:
            if len(h)<2:continue
            y=h['value' if task=='physiology' else 'label'].to_numpy();p=h[value+'_native'].to_numpy();q=h[value+'_heartbeat'].to_numpy()
            groups=h.patient_id.to_numpy();pts=np.unique(groups);index={pt:np.flatnonzero(groups==pt) for pt in pts}
            functions={'mae':mean_absolute_error} if task=='physiology' else {'auc':roc_auc_score,'ap':average_precision_score}
            if task=='future' and len(np.unique(y))<2:continue
            for metric,fn in functions.items():
                samples=[]
                for draw in range(2000):
                    ix=np.concatenate([index[pt] for pt in rng.choice(pts,len(pts),replace=True)])
                    if task=='future' and len(np.unique(y[ix]))<2:continue
                    samples.append(fn(y[ix],q[ix])-fn(y[ix],p[ix]))
                lo,hi=np.quantile(samples,[.025,.975])
                rows.append(dict(task=task,target=key[0] if task=='physiology' else 'future_deterioration',model=model,subgroup=subgroup,
                    visits=len(h),patients=len(pts),metric=metric,native=fn(y,p),heartbeat=fn(y,q),delta=fn(y,q)-fn(y,p),lo=lo,hi=hi))
frame=pd.DataFrame(rows);frame.to_csv(ROOT/'paired_comparison.csv',index=False)
lines=['# Estimated-heartbeat temporal sampling trial','',
    'Native color, crop, checkpoint, normalization, input shape and patient folds are unchanged. Sixteen nearest native frames span one period estimated from recorded HR, with up to four evenly distributed starts. No ECG phase alignment or invented short-cine cycles.',
    'Fallback clips retain exact baseline embeddings. Refitting models can still change their predictions. This is an exploratory internal comparison, not external validation.',
    'Primary endpoint: AP for fixed 75% retained baseline + 25% EchoPrime. Other models, GLS probes and subgroup comparisons are secondary/exploratory.',
    '',f'Sampling counts: {sampling.reason.value_counts().to_dict()}',
    '', '| Target | Model | Subgroup | N | Metric | Reference | Heartbeat | Difference (95% patient-bootstrap CI) |',
    '|---|---|---|---:|---|---:|---:|---|']
for r in rows:
    lines.append(f'| {r["target"]} | {r["model"]} | {r["subgroup"]} | {r["visits"]} | {r["metric"]} | {r["native"]:.3f} | {r["heartbeat"]:.3f} | {r["delta"]:+.3f} ({r["lo"]:+.3f}, {r["hi"]:+.3f}) |')
lines+=['','Negative MAE differences and positive AUROC/AP differences favor heartbeat-length sampling. No multiplicity adjustment for exploratory comparisons. A negative result would not rule out ECG-aligned sampling; a positive result needs confirmation on unused patients.']
(ROOT/'results.md').write_text('\n'.join(lines),encoding='utf-8')
print(frame[frame.subgroup=='all'].to_string(index=False))
