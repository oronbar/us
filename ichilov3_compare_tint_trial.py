"""Paired patient-cluster bootstrap of saved native/tint OOF predictions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, roc_auc_score, average_precision_score

ROOT=Path(r'D:\DS\ichilov3_tint_trial_20260927')
changes=pd.read_parquet(ROOT/'color_changes.parquet')
rows=[]
rng=np.random.default_rng(20260930)
for task,native,new,keys in [
    ('physiology',Path(r'D:\DS\ichilov3_physiology_audit_20260927\oof_predictions.parquet'),ROOT/'physiology/oof_predictions.parquet',['target','model','visit_id','patient_id','value']),
    ('future',Path(r'D:\DS\ichilov3_late_fusion_20260927\paired_oof_predictions.parquet'),ROOT/'late_fusion/paired_oof_predictions.parquet',['model','transition_id','patient_id','label'])]:
    a=pd.read_parquet(native);b=pd.read_parquet(new)
    value='prediction' if task=='physiology' else 'score'
    pair=a[keys+[value]].merge(b[keys+[value]],on=keys,suffixes=('_native','_tint'),validate='one_to_one')
    assert len(pair)==len(a)==len(b)
    if task=='physiology':
        identity=pd.read_parquet(r'D:\DS\ichilov3_aligned_20260927\visit_alignment.parquet')[['visit_id','patient_id','visit_date']]
        pair=pair.merge(identity,on=['visit_id','patient_id'],validate='many_to_one')
    else:
        identity=pd.read_parquet(r'D:\DS\ichilov3_aligned_20260927\cohort.parquet')[['transition_id','current_visit_date']].rename(columns={'current_visit_date':'visit_date'})
        pair=pair.merge(identity,on='transition_id',validate='many_to_one')
    grouper=['target','model'] if task=='physiology' else ['model']
    for key,g in pair.groupby(grouper):
        key=key if isinstance(key,tuple) else (key,)
        model=key[-1]
        if not any(m in model for m in ['echoprime','panecho']):continue
        encoder='echoprime' if 'echoprime' in model else 'panecho'
        c=changes[(changes.model==encoder)&(changes.normalized_windows>0)]
        changed=set(zip(c.patient,c.visit_date))
        g=g.copy();g['changed']= [(p,str(d)) in changed for p,d in zip(g.patient_id,g.visit_date)]
        for subgroup,h in [('all',g),('tint_changed_visit',g[g.changed]),('untinted_control_visit',g[~g.changed])]:
            if len(h)<2:continue
            y=h['value' if task=='physiology' else 'label'].to_numpy()
            p=h[value+'_native'].to_numpy();q=h[value+'_tint'].to_numpy()
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
                                 visits=len(h),patients=len(pts),metric=metric,native=fn(y,p),tint=fn(y,q),delta=fn(y,q)-fn(y,p),lo=lo,hi=hi))
frame=pd.DataFrame(rows);frame.to_csv(ROOT/'paired_comparison.csv',index=False)
lines=['# Uniform-tint ablation','',
       'Exploratory paired comparison on identical patient-held-out cohorts and folds. Untinted controls reuse native embeddings exactly; model refitting can still change their predictions.',
       'JPEG-gated policy: only preview candidates are tested using exact sampled pixels. Uniform tints preserve the max-channel intensity; multihue Doppler is retained. This does not test all possible color normalization policies.',
       '', '| Task / target | Model | Subgroup | N | Metric | Native | Tint | Delta (95% patient-bootstrap CI) |',
       '|---|---|---|---:|---|---:|---:|---|']
for r in rows:
    lines.append(f'| {r["target"]} | {r["model"]} | {r["subgroup"]} | {r["visits"]} | {r["metric"]} | {r["native"]:.3f} | {r["tint"]:.3f} | {r["delta"]:+.3f} ({r["lo"]:+.3f}, {r["hi"]:+.3f}) |')
lines += ['', 'Negative MAE differences favor tint normalization; positive AUROC/AP differences favor tint normalization. Subgroups and multiple targets are exploratory, with no multiplicity adjustment. Do not select a production policy solely from outer-fold outcomes.']
(ROOT/'results.md').write_text('\n'.join(lines),encoding='utf-8')
print(frame[frame.subgroup=='all'].to_string(index=False))
