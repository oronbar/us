"""Read-only check of repeat GLS exports and patient-cluster correlation uncertainty."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

root=Path(r'D:\DS\ichilov3_temporal_trial_20260927')
raw=pd.read_parquet(r'D:\us\amber_full_105_preprocessed\Ichilov_july_dataset.parquet')
reports=raw[['visit_id','source_file','gls_mid_peak_avg','gls_endo_peak_avg']].drop_duplicates()
paired=reports.groupby('visit_id').filter(lambda g:len(g)==2).sort_values(['visit_id','source_file'])
diff=paired.groupby('visit_id')[['gls_mid_peak_avg','gls_endo_peak_avg']].agg(lambda s:abs(s.iloc[1]-s.iloc[0]))
diff=diff.rename(columns={'gls_mid_peak_avg':'mid_report_abs_difference','gls_endo_peak_avg':'endo_report_abs_difference'})
diff.to_csv(root/'repeat_report_differences.csv')
ref=pd.read_parquet(r'D:\DS\ichilov3_physiology_audit_20260927\oof_predictions.parquet')
new=pd.read_parquet(root/'physiology/oof_predictions.parquet')
rows=[];rng=np.random.default_rng(20261002)
for target in ['mid_gls','endo_gls']:
    for model in ['echoprime','panecho']:
        a=ref[(ref.target==target)&(ref.model==model)]
        b=new[(new.target==target)&(new.model==model)]
        g=a[['visit_id','patient_id','value','prediction']].merge(b[['visit_id','patient_id','value','prediction']],on=['visit_id','patient_id','value'],suffixes=('_original','_heartbeat'),validate='one_to_one')
        pts=g.patient_id.unique();group=g.patient_id.to_numpy();ix={p:np.flatnonzero(group==p) for p in pts}
        y=g.value.to_numpy();old=g.prediction_original.to_numpy();pred=g.prediction_heartbeat.to_numpy()
        samples=[]
        for _ in range(2000):
            take=np.concatenate([ix[p] for p in rng.choice(pts,len(pts),replace=True)])
            samples.append((np.corrcoef(y[take],old[take])[0,1],np.corrcoef(y[take],pred[take])[0,1]))
        samples=np.asarray(samples)
        rows.append(dict(target=target,model=model,n=len(g),patients=len(pts),original_r=float(np.corrcoef(y,old)[0,1]),heartbeat_r=float(np.corrcoef(y,pred)[0,1]),
            heartbeat_r_ci=np.quantile(samples[:,1],[.025,.975]).tolist(),r_gain=float(np.corrcoef(y,pred)[0,1]-np.corrcoef(y,old)[0,1]),
            r_gain_ci=np.quantile(samples[:,1]-samples[:,0],[.025,.975]).tolist()))
out=dict(repeat_visits=len(diff),repeat_mid_mean_abs_difference=float(diff.mid_report_abs_difference.mean()),repeat_mid_median_abs_difference=float(diff.mid_report_abs_difference.median()),
    repeat_endo_mean_abs_difference=float(diff.endo_report_abs_difference.mean()),repeat_endo_median_abs_difference=float(diff.endo_report_abs_difference.median()),
    caveat='These exports are not verified independent blinded rereads of identical video; cannot estimate label reliability or an upper bound from them alone.',correlations=rows)
(root/'label_noise_check.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
