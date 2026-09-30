"""Rank other cines within exactly the same visit/study and target view."""
import hashlib
import sys
from pathlib import Path
import pydicom
from ichilov3_prepare_selected import header


def ranked(record,all_records,output,progress=None):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from ichilov3_view_selector.suggestions import Suggester,MODEL_VIEWS,selection_score
    suggester=Suggester(output)
    forbidden={r['expected_sop'] for r in all_records if r['patient']==record['patient'] and r['visit_date']==record['visit_date']}
    sources=[];seen=set()
    for p in sorted(Path(record['source_path']).parent.iterdir()):
        if not p.is_file():continue
        try:
            ds=pydicom.dcmread(p,stop_before_pixels=True,force=True)
            sop=str(ds.SOPInstanceUID)
            if int(ds.get('NumberOfFrames',1))<16 or str(ds.StudyInstanceUID)!=record['expected_study'] or sop in forbidden or sop in seen:continue
            seen.add(sop)
            sources.append(dict(id=hashlib.sha256(str(p).casefold().encode()).hexdigest()[:24],path=str(p),sop=sop))
        except Exception:continue
    results=[]
    for i,source in enumerate(sources,1):
        if progress:progress(f'Checking cine {i} of {len(sources)}')
        prediction=suggester.analyze_file(source)
        if prediction['status']!='ok' or not prediction['bmode_candidate'] or prediction['predicted_view']!=record['view']:continue
        probability=prediction['probabilities'][MODEL_VIEWS.index(record['view'])]
        if probability<.45:continue
        score=selection_score(probability,prediction['frame_agreement'][record['view']],prediction['quality']['quality_proxy'])
        task=header(dict(file_id=record['file_id'],patient=record['patient'],visit_date=record['visit_date'],view=record['view'],source_path=source['path'],expected_sop=source['sop'],expected_study=record['expected_study'],selection_source='Reviewer-selected alternative cine',strain_report=record['strain_report']))
        if task['status']!='header_ok':continue
        warnings=[]
        duration=prediction['quality']['duration_seconds']
        beat=60/record['heart_rate'] if record.get('heart_rate') else .8
        if duration<beat*.85:warnings.append('Short cine: may not contain a full heartbeat')
        if task['frames']<31:warnings.append('EchoPrime needs edge-repeat padding')
        results.append(dict(candidate_id=source['id'],name=Path(source['path']).name,view=record['view'],probability=probability,score=score,quality=prediction['quality'],warnings=warnings,record=task))
    return sorted(results,key=lambda x:x['score'],reverse=True)[:5]
