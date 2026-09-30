"""Summarize the current selection and paired-cohort availability, without fitting."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path('D:/us')
OUT = ROOT / 'output/dicom_prediction'
APICAL = ['A2C', 'A3C', 'A4C']


def main():
    spatial = pd.read_parquet(OUT / 'spatial_manifest.parquet')
    views = pd.read_parquet(OUT / 'view_manifest.parquet')
    selected = pd.read_parquet(OUT / 'selected_clips.parquet')
    visits = pd.read_parquet(ROOT / 'amber_full_105_preprocessed/Ichilov_july_visits.parquet')
    transitions = pd.read_parquet(ROOT / 'cardiotoxicity_next_visit_gpu_results/next_visit_transitions.parquet')
    transitions = transitions[transitions.mask__mid_first_rel15.astype(bool)].copy()
    stages = {
        'linked_multiframe': spatial,
        'tissue_preview': spatial[spatial.status.eq('ok')],
        'confident_apical': views[views.view.isin(APICAL)],
        'selected_apical': selected[selected.view.isin(APICAL)],
    }
    funnel = []
    for name, rows in stages.items():
        available = set(visits[visits.study_uid.isin(rows.study_uid)].visit_id)
        keep = transitions.current_visit_id.isin(available)
        funnel.append(dict(stage=name, clips=len(rows), studies=rows.study_uid.nunique(),
                           transitions=int(keep.sum()), patients=transitions.loc[keep, 'patient_id'].nunique(),
                           events=int(transitions.loc[keep, 'label__mid_first_rel15'].sum())))
        transitions[name] = keep
    pd.DataFrame(funnel).to_csv(OUT / 'apical_selection_funnel.csv', index=False)
    transitions[['transition_id', 'patient_id', 'label__mid_first_rel15', *stages]].to_csv(
        OUT / 'dicom_availability_audit.csv', index=False)
    retained = transitions.selected_apical
    summary = {'stages': funnel, 'excluded_transitions': int((~retained).sum()),
               'excluded_events': int(transitions.loc[~retained, 'label__mid_first_rel15'].sum()),
               'selected_appearance': selected.appearance.value_counts().to_dict(),
               'note': 'Patients can have both included and excluded transitions; these patient counts overlap.'}
    (OUT / 'coverage_summary.json').write_text(json.dumps(summary, indent=2))
    print(pd.DataFrame(funnel).to_string(index=False))


if __name__ == '__main__':
    main()
