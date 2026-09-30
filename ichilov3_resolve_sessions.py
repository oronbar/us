"""Conservatively corroborate conflicting bookmark sessions against report exports.

Inference is explicitly labelled: <=10 seconds and all six Endo/Mid per-view
GLS values within 0.05 percentage points. This is not a full waveform match.
"""
import collections,csv,datetime,json,re
from pathlib import Path
import pydicom
import xml.etree.ElementTree as ET
from ichilov3_audit import OUT

def main():
    visits=json.loads((OUT/'visits.json').read_text());links=json.loads((OUT/'links.json').read_text());cache={};decisions=[]
    for v in visits:
        v['session_selection']='Unique source references' if v['resolved_views'] else ''
        if 'conflicting source clips' not in v['notes'] or v['strain_report']!='Yes':continue
        grouped=collections.defaultdict(list)
        for l in links:
            if l['patient']==v['patient'] and l['date']==v['date'].replace('-','') and l['method']=='TOMTEC bookmark' and l['study_uid'] in v['report_uids'].splitlines():grouped[l['session']].append(l)
        report_matches=[];report_stats=[]
        for rp in v['report_files'].splitlines():
            entries=list(csv.reader(Path(rp).read_text(encoding='utf-8-sig').splitlines()));d={r[0]:r[1:] for r in entries if len(r)>1};export=d.get('Export Date and Time',[])
            if len(export)<2:report_matches.append([]);continue
            t=datetime.datetime.fromisoformat(export[0]+'T'+export[1]);possibles=[]
            for session,ls in grouped.items():
                try:st=datetime.datetime.strptime(session,'%Y-%m-%d_%H:%M:%S')
                except ValueError:continue
                seconds=abs((st-t).total_seconds())
                if seconds>10:continue
                byview=collections.defaultdict(list)
                for l in ls:byview[l['view']].append(l)
                if set(byview)!={'A2C','A3C','A4C'}:continue
                if any(len({l['source_uid'] for l in vv})!=1 for vv in byview.values()):continue
                diffs=[]
                for view,vv in byview.items():
                    for l in vv:
                        f=l['analysis_file']
                        if f not in cache:
                            ds=pydicom.dcmread(f);tree=ET.fromstring(ds[0x7fdf5051].value.rstrip(b'\x00'));values={}
                            for spline in tree.iter('AutoStrainSpline'):
                                tags=spline.findtext('DataNodeTags','')
                                if 'AUTOSTRAINSPLINE_LV' not in tags:continue
                                layer='Endo' if 'EndocardialContour' in tags else 'Mid' if 'MyocardialContour' in tags else ''
                                if layer and spline.findtext('Gls/UcumUnit')=='%':values[layer]=float(spline.findtext('Gls/Value'))
                            cache[f]=values
                        for layer in ['Endo','Mid']:
                            key=f'GLS {layer} Peak {view}'
                            if layer not in cache[f] or key not in d:diffs.append(float('inf'))
                            else:diffs.append(abs(cache[f][layer]-float(d[key][0])))
                if len(diffs)>=6 and max(diffs)<=0.05:
                    possibles.append({'session':session,'seconds':seconds,'max_gls_difference_pp':max(diffs),'uids':{view:vv[0]['source_uid'] for view,vv in byview.items()}})
            report_matches.append(possibles)
        # Every report export must support a unique, consistent source triplet.
        if not report_matches or any(not m for m in report_matches):continue
        signatures={tuple(sorted(p['uids'].items())) for m in report_matches for p in m}
        if len(signatures)!=1:continue
        sessions={p['session'] for m in report_matches for p in m};chosen=[l for session in sessions for l in grouped[session]]
        for view in ['A2C','A3C','A4C']:
            paths=sorted({p for l in chosen if l['view']==view for p in l['matched_files']});v[view]=paths[0] if paths else ''
        v['resolved_views']=sum(bool(v[view]) for view in ['A2C','A3C','A4C'])
        v['clip_source']='TOMTEC bookmark' if v['resolved_views'] else ''
        v['session_selection']='Report time + 6 GLS values support session'
        notes=[n for n in v['notes'].split('; ') if 'conflicting source clips' not in n and 'view(s) unresolved' not in n]
        notes.append('Session inferred from export time <=10s and 6 GLS values within 0.05 pp; waveform equivalence not tested')
        for view in ['A2C','A3C','A4C']:
            if not v[view]:notes.append(f'{view}: selected session references a video UID absent from the export')
        if v['resolved_views']<3:notes.append(f"{3-v['resolved_views']} view(s) unresolved")
        v['notes']='; '.join(notes)
        decisions.append({'patient':v['patient'],'date':v['date'],'reports':report_matches,'selected':{view:v[view] for view in ['A2C','A3C','A4C']}})
    summary=json.loads((OUT/'summary.json').read_text());summary['complete_triplets']=sum(v['resolved_views']==3 for v in visits);summary['resolved_view_counts']=dict(collections.Counter(v['resolved_views'] for v in visits));summary['report_supported_session_visits']=len(decisions)
    for name,data in [('visits',visits),('summary',summary),('session_decisions',decisions)]:
        (OUT/f'{name}.json').write_text(json.dumps(data,indent=2))
    print(json.dumps({'report_supported_session_visits':len(decisions),'complete_triplets':summary['complete_triplets']}))

if __name__=='__main__':main()
