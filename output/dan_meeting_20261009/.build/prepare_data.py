import csv, json
from pathlib import Path

base = Path(r'D:\us\output\dan_meeting_20261009')
def rows(p):
    with open(p, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))
pred = rows(r'D:\DS\ichilov3_temporal_trial_20260927\vendor_failure_analysis\prediction_points.csv')
ge = [r for r in pred if r['target']=='mid_gls' and r['model']=='echoprime' and r['vendor_short']=='GE Healthcare']
assert len(ge)==230 and len({r['patient_id'] for r in ge})==79
vendors = [r for r in rows(r'D:\DS\ichilov3_temporal_trial_20260927\vendor_failure_analysis\vendor_metrics.csv') if r['target']=='mid_gls']
imp = rows(r'D:\us\cardiotoxicity_feature_importance_results\noncnn_feature_importance_top.csv')
keys = [('clinical_plus_transmural_sparse','d_tm_sd_gap_dct04'),('combined_extra_trees','cur_tm_mean_shape_gap_dct01'),('clinical_plus_variability_sparse','cur_var__mid_peak_abs_robust_sd')]
importance=[]
for model,key in keys:
    row=next(r for r in imp if r['model']==model and r['item']==key)
    importance.append({k:float(row[k]) for k in ['auc_drop_mean','auc_drop_boot_low','auc_drop_boot_high']})
data={'ge':{'gt':[float(r['value']) for r in ge],'pred':[float(r['prediction']) for r in ge]},'vendors':vendors,'importance':importance}
(base/'.build'/'data.json').write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8')
print('Prepared 230 anonymous prediction pairs and verified feature-importance values.')
