"""Read-only label-disagreement audit and blinded reread sampling queue."""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd

OUT=Path(r'D:\DS\ichilov3_temporal_trial_20260927\label_review')
OUT.mkdir(exist_ok=True,parents=True)
RAW=Path(r'D:\us\amber_full_105_preprocessed\Ichilov_july_dataset.parquet')
VISITS=Path(r'D:\us\amber_full_105_preprocessed\Ichilov_july_visits.parquet')
OOF=Path(r'D:\DS\ichilov3_temporal_trial_20260927\physiology\oof_predictions.parquet')
ALIGN=Path(r'D:\DS\ichilov3_aligned_20260927\visit_alignment.parquet')
CROP=Path(r'D:\DS\ichilov3_stage2_padded_20260926\full_manifest.json')

fields=['visit_id','patient_id','study_uid','source_file','source_path','source_sha256','filename_export_datetime',
        'gls_mid_peak_avg','gls_endo_peak_avg','gls_mid_peak_a2c','gls_mid_peak_a3c','gls_mid_peak_a4c',
        'gls_endo_peak_a2c','gls_endo_peak_a3c','gls_endo_peak_a4c']
raw=pd.read_parquet(RAW,columns=fields).drop_duplicates()
counts=raw.groupby('visit_id').size()
assert len(counts)==400 and counts.value_counts().to_dict()=={1:384,2:16}
pair=raw[raw.visit_id.isin(counts[counts==2].index)].sort_values(['visit_id','source_file']).copy()
assert pair.groupby('visit_id').study_uid.nunique().eq(1).all()
assert pair.groupby('visit_id').source_sha256.nunique().eq(2).all()
cols=['gls_mid_peak_avg','gls_endo_peak_avg','gls_mid_peak_a2c','gls_mid_peak_a3c','gls_mid_peak_a4c',
      'gls_endo_peak_a2c','gls_endo_peak_a3c','gls_endo_peak_a4c']
differences=pair.groupby('visit_id')[cols].agg(lambda x:abs(x.iloc[1]-x.iloc[0]))
differences.columns=[c+'_pair_abs_difference' for c in differences.columns]
differences.to_parquet(OUT/'repeat_report_differences.parquet')

visits=pd.read_parquet(VISITS,columns=['visit_id','patient_id','gls_mid_magnitude','gls_endo_magnitude']).set_index('visit_id')
averages=pair.groupby('visit_id')[['gls_mid_peak_avg','gls_endo_peak_avg']].mean().abs()
assert np.allclose(visits.loc[averages.index,'gls_mid_magnitude'],averages.gls_mid_peak_avg)
assert np.allclose(visits.loc[averages.index,'gls_endo_magnitude'],averages.gls_endo_peak_avg)

oof=pd.read_parquet(OOF)
mid=oof[(oof.model=='echoprime')&(oof.target=='mid_gls')].set_index('visit_id')
endo=oof[(oof.model=='echoprime')&(oof.target=='endo_gls')].set_index('visit_id')
assert len(mid)==len(endo)==398
scores=pd.DataFrame(index=mid.index)
scores['patient_id']=mid.patient_id
scores['mid_label']=mid.value
scores['mid_prediction']=mid.prediction
scores['mid_abs_error']=abs(mid.prediction-mid.value)
scores['endo_label']=endo.value
scores['endo_prediction']=endo.prediction
scores['endo_abs_error']=abs(endo.prediction-endo.value)
scores['max_abs_error']=scores[['mid_abs_error','endo_abs_error']].max(axis=1)
scores['report_count']=counts.loc[scores.index].to_numpy()
scores=scores.join(differences,how='left')
scores.to_parquet(OUT/'analyst_visit_scores.parquet')

repeat_ids=list(scores[scores.report_count==2].index)
remaining=scores[scores.report_count==1].sort_values('max_abs_error',ascending=False)
high_ids=list(remaining.head(24).index)
pool=remaining.drop(index=high_ids).copy()
# Deterministic label-stratified random controls, no model error ranking.
pool['stratum']=pd.qcut(pool.mid_label,4,labels=False,duplicates='drop')
pool['random_order']=[hashlib.sha256(('review-control-20260928:'+v).encode()).hexdigest() for v in pool.index]
control_ids=[]
for _,g in pool.groupby('stratum'):
    control_ids.extend(g.sort_values('random_order').head(5).index.tolist())
selected=[('repeat_report',v) for v in repeat_ids]+[('large_oof_residual',v) for v in high_ids]+[('label_stratified_control',v) for v in control_ids]
assert len(selected)==60 and len({v for _,v in selected})==60

alignment=pd.read_parquet(ALIGN).set_index('visit_id')
clips=json.loads(CROP.read_text())
byvisit={(r['patient'],r['visit_date'],r['view']):r['source_path'] for r in clips}
assert len(byvisit)==len(clips)
records=[]
for stratum,visit_id in selected:
    visit=alignment.loc[visit_id]
    patient=str(visit.patient_id);date=str(visit.visit_date)
    reports=raw[raw.visit_id==visit_id].sort_values('source_file')
    r=dict(visit_id=visit_id,patient_id=patient,visit_date=date,study_uid=visit.study_uid,
           a2c_dicom=byvisit[(patient,date,'A2C')],a3c_dicom=byvisit[(patient,date,'A3C')],a4c_dicom=byvisit[(patient,date,'A4C')],
           report_1=str(reports.source_path.iloc[0]),report_2=str(reports.source_path.iloc[1]) if len(reports)==2 else '',
           reread_mid_gls='',reread_endo_gls='',reader_id='',same_three_cines_confirmed='',notes='')
    records.append((stratum,r))
full=pd.DataFrame([r for _,r in records])
blind=full.drop(columns=['report_1','report_2'])
blind.to_parquet(OUT/'blinded_review_queue.parquet',index=False)
analyst=full[['visit_id','patient_id','visit_date','report_1','report_2']].copy()
analyst['sampling_reason']=[s for s,_ in records]
analyst=analyst.join(scores.drop(columns='patient_id'),on='visit_id')
analyst.to_parquet(OUT/'analyst_queue.parquet',index=False)
metrics={
    'source_reports':len(raw),'visits':len(counts),'repeat_report_visits':len(pair)//2,
    'same_study_uid_in_all_pairs':True,'distinct_file_hashes_in_all_pairs':True,
    'visit_label_equals_mean_pair_for_all_repeats':True,
    'mid_pair_mean_abs_difference':float(differences.gls_mid_peak_avg_pair_abs_difference.mean()),
    'mid_pair_median_abs_difference':float(differences.gls_mid_peak_avg_pair_abs_difference.median()),
    'mid_pair_max_abs_difference':float(differences.gls_mid_peak_avg_pair_abs_difference.max()),
    'endo_pair_mean_abs_difference':float(differences.gls_endo_peak_avg_pair_abs_difference.mean()),
    'endo_pair_median_abs_difference':float(differences.gls_endo_peak_avg_pair_abs_difference.median()),
    'endo_pair_max_abs_difference':float(differences.gls_endo_peak_avg_pair_abs_difference.max()),
    'mid_model_mae_repeat_visits':float(scores.loc[repeat_ids,'mid_abs_error'].mean()),
    'mid_model_mae_single_report_visits':float(scores.loc[scores.report_count==1,'mid_abs_error'].mean()),
    'review_queue':dict(repeat_reports=len(repeat_ids),large_residuals=len(high_ids),stratified_controls=len(control_ids)),
    'limitations':['A pair shares Study UID, but reports do not identify the three SOP UIDs; identical source cines cannot be confirmed from the text exports.',
                   'Neither independent readers nor blinded rereads are documented; paired differences are evidence of export/analysis variation, not a formal interobserver reliability estimate.',
                   'The 16 repeat visits are too few for a precise noise ceiling; larger errors in this subset are descriptive, not causal.',
                   'Do not expose predictions or sampling reason to clinical rereaders; use the blinded queue only.']}
(OUT/'summary.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
lines=['# GLS label audit and blinded review sample','',
    f'{metrics["source_reports"]} report exports map to {metrics["visits"]} visits. Sixteen visits have two distinct report files with the same Study UID. The stored visit label is the arithmetic mean of those two exports.',
    f'Paired absolute Mid-GLS difference: mean {metrics["mid_pair_mean_abs_difference"]:.2f}, median {metrics["mid_pair_median_abs_difference"]:.2f}, maximum {metrics["mid_pair_max_abs_difference"]:.2f} percentage points. Endo: mean {metrics["endo_pair_mean_abs_difference"]:.2f}, median {metrics["endo_pair_median_abs_difference"]:.2f}, maximum {metrics["endo_pair_max_abs_difference"]:.2f}.',
    f'EchoPrime heartbeat Mid-GLS MAE is {metrics["mid_model_mae_repeat_visits"]:.2f} on those 16 visits versus {metrics["mid_model_mae_single_report_visits"]:.2f} on 382 single-report visits. This subgroup difference is descriptive.',
    '', 'The report text does not identify the three SOP UIDs or establish independent blinded rereads. The pair differences cannot by themselves quantify reader noise or the best achievable correlation.',
    '', 'The blinded review queue contains 60 visits: all 16 with repeat reports, 24 with large held-out errors and 20 deterministic random controls stratified across the Mid-GLS range. It gives reviewers the three selected DICOM paths while hiding existing reports, model predictions, labels and sampling reasons. The analyst table retains report paths separately.',
    '', 'Reviewers should confirm whether both exports used the same three cines and independently measure Mid/Endo GLS from those cines, blinded to the existing labels and predictions. Enter conventional signed GLS (for example, -18.0%), not magnitude. Then compute repeatability, adjudicated-label agreement, and patient-held-out prediction metrics on the prespecified complete sample.',
    '', 'No source data, labels or model predictions were changed.']
(OUT/'README.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(metrics,indent=2))
