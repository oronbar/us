"""Read-only exploratory association of existing quality proxies with GLS errors."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT=Path(r'D:\DS\ichilov3_temporal_trial_20260927\quality_audit')
OUT.mkdir(parents=True,exist_ok=True)
v=pd.read_parquet(r'D:\us\amber_full_105_preprocessed\Ichilov_july_visits.parquet')
o=pd.read_parquet(r'D:\DS\ichilov3_temporal_trial_20260927\physiology\oof_predictions.parquet')
a=pd.read_parquet(r'D:\DS\ichilov3_aligned_20260927\visit_alignment.parquet')
clips=json.loads(Path(r'D:\DS\ichilov3_stage2_padded_20260926\full_manifest.json').read_text())
groups={}
for r in clips:groups.setdefault((r['patient'],r['visit_date']),[]).append(r)
meta=[]
for (patient,date),rs in groups.items():
    meta.append(dict(patient_id=patient,visit_date=date,
        selection_all_bookmark=all(r['selection_source']=='TOMTEC bookmark' for r in rs),
        selected_views_flagged=sum(r['status']=='needs_review' for r in rs),
        manufacturer=rs[0]['manufacturer']))
meta=pd.DataFrame(meta)
link=a[['visit_id','patient_id','visit_date']].merge(meta,on=['patient_id','visit_date'],validate='many_to_one')
base=v[['visit_id','technical_replicate_count','gls_mid_peak_a2c','gls_mid_peak_a3c','gls_mid_peak_a4c',
        'mid_curve_dispersion_rms','mid_shape_incoherence','mid_n_segments']].merge(link,on='visit_id',validate='one_to_one')
base['mid_view_spread']=base[['gls_mid_peak_a2c','gls_mid_peak_a3c','gls_mid_peak_a4c']].max(axis=1)-base[['gls_mid_peak_a2c','gls_mid_peak_a3c','gls_mid_peak_a4c']].min(axis=1)
pred=o[(o.target=='mid_gls')&(o.model=='echoprime')][['visit_id','patient_id','value','prediction']]
df=base.merge(pred,on=['visit_id','patient_id'],validate='one_to_one')
assert len(df)==398
df['absolute_error']=(df.value-df.prediction).abs()
df.to_parquet(OUT/'visit_quality_and_error.parquet',index=False)
proxies=['mid_view_spread','mid_curve_dispersion_rms','mid_shape_incoherence','selected_views_flagged']
lines=['# Retrospective quality-proxy audit','',
       '398 patient-held-out same-visit Mid-GLS predictions from the estimated-heartbeat EchoPrime model. Associations are exploratory and cannot establish label noise.',
       '', '| Proxy | Spearman rho with absolute error | Bottom quartile MAE | Top quartile MAE |',
       '|---|---:|---:|---:|']
results=[]
for key in proxies:
    x=df[key].astype(float);rho,p=spearmanr(x,df.absolute_error,nan_policy='omit')
    low=df[x<=x.quantile(.25)].absolute_error.mean();high=df[x>=x.quantile(.75)].absolute_error.mean()
    results.append(dict(proxy=key,rho=float(rho),p_unadjusted=float(p),bottom_quartile_mae=float(low),top_quartile_mae=float(high)))
    lines.append(f'| {key} | {rho:+.3f} | {low:.3f} | {high:.3f} |')
lines+=['','Categorical groups (not adjusted for patient, vendor, target range or multiple comparisons):','',
    '| Group | Visits | MAE |','|---|---:|---:|']
for key in ['technical_replicate_count','selection_all_bookmark','manufacturer']:
    for val,g in df.groupby(key):lines.append(f'| {key} = {val} | {len(g)} | {g.absolute_error.mean():.3f} |')
lines+=['','Three-view GLS spread and curve shape can reflect true regional disease, not necessarily a poor label. The original crop flags were overridden by the user; their presence is not a clinical invalidity judgment.',
        'The 16 duplicated reports come from 9 patients and are not proven identical-cine, independent rereads. Group differences are descriptive. Do not remove visits or set training weights from these results without grouped inner-fold testing and untouched external validation.',
        'No original data, labels, models or predictions were changed.']
(OUT/'results.md').write_text('\n'.join(lines),encoding='utf-8')
(OUT/'proxy_results.json').write_text(json.dumps(results,indent=2))
print('\n'.join(lines))
