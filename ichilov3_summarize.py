"""Resolve exact analysis references and create workbook-ready visit records."""
import collections,datetime,json,re
from pathlib import Path
from dicom_prediction_inventory import report_inventory
from ichilov3_audit import ROOT,OUT

def views(text):
    text=text.upper(); result=set()
    for n,w in [('2','TWO'),('3','THREE'),('4','FOUR')]:
        if re.search(rf'A{n}C|VIEW_PLANE_{n}CH|\b{n}[ -]?(?:CH|CHAMBER)|APICAL[ -]+{w}[ -]+CHAMBER',text):result.add('A'+n+'C')
    if 'APICAL LONG AXIS' in text or 'APLAX' in text:result.add('A3C')
    return result

def main():
    rows={}
    for line in (OUT/'inventory.jsonl').read_text(encoding='utf-8').splitlines():
        try:
            r=json.loads(line);rows[r['path']]=r
        except json.JSONDecodeError:pass
    rows=list(rows.values());valid=[r for r in rows if r['status']=='ok']
    # Re-read the actual report files rather than trusting the previous checklist.
    reports=report_inventory(Path('D:/DS/anonymized_reports')).to_dict('records')
    report_by=collections.defaultdict(list)
    for r in reports:
        r['patient']=re.sub(r'^AutoStrainCap_|_\d{8}_\d{6}\.txt$','',Path(r['report_path']).name)
        report_by[(r['patient'],r['study_datetime'][:10])].append(r)
    file_by=collections.defaultdict(list);folder_by=collections.defaultdict(list);physical_by=collections.defaultdict(list)
    for r in rows:
        physical_by[(r['patient_folder'],r['date_folder'].replace('_','-'))].append(r)
        date=r.get('StudyDate','')
        date=f'{date[:4]}-{date[4:6]}-{date[6:]}' if re.fullmatch(r'\d{8}',date) else r['date_folder'].replace('_','-')
        file_by[(r['patient_folder'],date)].append(r)
    for f in json.loads((OUT/'folders.json').read_text()):
        p=Path(f);folder_by[(p.parent.name,p.name.replace('_','-'))].append(f)
    by_uid=collections.defaultdict(list)
    for r in valid:
        if r.get('SOPInstanceUID'):by_uid[r['SOPInstanceUID']].append(r)
    links=[];candidates=[]
    def link(r,view,uid,method,session,extra=None):
        targets=[t for t in by_uid.get(uid,[]) if t['patient_folder']==r['patient_folder'] and t['StudyInstanceUID']==r['StudyInstanceUID'] and int(t.get('NumberOfFrames') or 1)>1]
        paths=sorted(set(t['path'] for t in targets))
        row={'patient':r['patient_folder'],'date':r['StudyDate'],'study_uid':r['StudyInstanceUID'],'view':view,'source_uid':uid,'analysis_file':r['path'],'method':method,'session':session,'matched_files':paths,'status':'Matched' if paths else 'Referenced video missing','extra':extra or {}}
        links.append(row)
    for r in valid:
        if r.get('container',{}).get('FileClass')=='Bookmark' and not r['bookmarks']:
            candidates.append({'file':r['path'],'reason':'Bookmark present, source-reference payload not decoded'})
        for b in r['bookmarks']:
            if b['strain_arrays']==0 or any('VIEW_PLANE_RV' in t or 'VIEW_PLANE_LA' in t.replace('VIEW_PLANE_LAX','') for t in b['view_tags']):
                candidates.append({'file':r['path'],'reason':'Bookmark is not a validated LV strain analysis'})
                continue
            label=r.get('container',{}).get('FileLabel','') or r.get('ImageComments','')
            tagged=views(' '.join(b['view_tags'])); labelled=views(label)
            v=tagged or labelled
            bad=bool(tagged and labelled and tagged!=labelled)
            related=r.get('container',{}).get('RelatedFileUID','')
            if related and set(b['source_uids'])!={related}:bad=True
            if b['source_studies'] and set(b['source_studies'])!={r['StudyInstanceUID']}:bad=True
            if not bad and len(v)==1 and len(b['source_uids'])==1:
                link(r,next(iter(v)),b['source_uids'][0],'TOMTEC bookmark',re.sub(r'_A[234]C$','',label),{'strain_arrays':b['strain_arrays'],'timing':b['timing']})
            else:candidates.append({'file':r['path'],'reason':'Bookmark view/reference conflict or ambiguity','view':sorted(v),'references':b['source_uids']})
        if 'tomtec' in r.get('Manufacturer','').lower():continue
        texts=r['text']+[{'location':'/private/'+p['tag'],'value':p['prefix']} for p in r['private'] if p['vr'] not in {'OB','OW','UN'}]
        strain=[t for t in texts if re.search(r'strain|\bAFI\b|GEMSAWMA',t['value'],re.I)]
        if not strain:continue
        found=set()
        # An SR strain measurement and its image/view must share a measurement subtree.
        for marker in strain:
            location=marker['location']
            if '/ContentSequence[' not in location:continue
            node=location.rsplit('/ConceptNameCodeSequence',1)[0] if '/ConceptNameCodeSequence' in location else location.rsplit('/ContentSequence[',1)[0]
            scoped=[t['value'] for t in texts if t['location'].startswith(node+'/')]
            v=views(' '.join(scoped))
            refs={t['uid'] for t in r['references'] if t['location'].startswith(node+'/') and '/ReferencedPerformedProcedureStepSequence' not in t['location']}
            if len(v)==1 and len(refs)==1:found.add((next(iter(v)),next(iter(refs))))
        for v,uid in sorted(found):link(r,v,uid,'Other vendor SR',r.get('ContentDate','')+' '+r.get('ContentTime',''))
        direct=set()
        if not r.get('SOPClassUID','').startswith('1.2.840.10008.5.1.4.1.1.88.') and any(re.search(r'strain|\bAFI\b',t['value'],re.I) for t in strain):
            vv=views(' '.join(t['value'] for t in texts))
            uu={t['uid'] for t in r['references'] if t['location'].startswith(('/SourceImageSequence[','/ReferencedImageSequence['))}
            if len(vv)==1 and len(uu)==1:
                direct={(next(iter(vv)),next(iter(uu)))}
                for v,uid in direct:link(r,v,uid,'Other vendor source image',r.get('ContentDate','')+' '+r.get('ContentTime',''))
        candidates.append({'file':r['path'],'reason':'Other vendor strain candidate','manufacturer':r.get('Manufacturer',''),'image_type':r.get('ImageType',''),'reference_count':len(r['references']),'extracted_links':len(found|direct),'markers':list(dict.fromkeys(t['value'] for t in strain))[:12]})
    link_by=collections.defaultdict(list)
    for l in links:link_by[(l['patient'],f"{l['date'][:4]}-{l['date'][4:6]}-{l['date'][6:]}")].append(l)
    visits=[]
    for key in sorted(set(report_by)|set(file_by)|set(folder_by),key=lambda k:([int(t) for t in re.findall(r'\d+',k[0])],k[1])):
        patient,date=key;rr=report_by[key];ff=file_by[key];good=[r for r in ff if r['status']=='ok'];ll=link_by[key]
        ruids={r['study_uid'] for r in rr};duids={r['StudyInstanceUID'] for r in good};issues=[]
        exact=bool(ruids&duids); match='Yes' if good and (not rr or exact) else 'Review - UID mismatch' if good else 'No'
        if rr and good and not exact:issues.append('Same patient/date but report Study UID not found')
        if folder_by[key] and not good:
            physical=physical_by[key]
            if physical:
                dates=sorted(set(r.get('StudyDate','unknown') for r in physical))
                issues.append('Named folder contains other study date(s): '+', '.join(dates))
            else:issues.append('Visit directory is empty')
        if len(duids)>1:issues.append('Multiple DICOM Study UIDs on this date')
        if any(r.get('PatientID','').replace('-','').strip()!=patient.replace('-','') for r in good):issues.append('DICOM PatientID differs from folder ID')
        if any(r['date_folder'].replace('_','-')!=date for r in good):issues.append('DICOM study date differs from folder date')
        bm=[r for r in good if r['bookmarks'] or r.get('container',{}).get('FileClass')=='Bookmark']
        vendor=[c for c in candidates if c['reason']=='Other vendor strain candidate' and c['file'] in {r['path'] for r in good}]
        # Do not mix source clips across conflicting saved analyses or report study UIDs.
        eligible=[l for l in ll if not rr or l['study_uid'] in ruids]
        preferred=[l for l in eligible if l['method']=='TOMTEC bookmark'] if bm else [l for l in eligible if l['method'].startswith('Other vendor')]
        selected={};methods=set();resolved=[]
        for view in ['A2C','A3C','A4C']:
            ls=[l for l in preferred if l['view']==view];uids={l['source_uid'] for l in ls}
            matched=[l for l in ls if l['matched_files']]
            if len(uids)==1 and matched:
                selected[view]=sorted(set(p for l in matched for p in l['matched_files']))[0];methods.update(l['method'] for l in matched);resolved+=matched
            else:
                selected[view]=''
                if len(uids)>1:issues.append(f'{view}: conflicting source clips across saved analyses')
                elif ls:issues.append(f'{view}: source UID found, video missing')
        # All three must occur together in at least one saved TOMTEC session to be a triplet.
        sessions=collections.defaultdict(set)
        for l in resolved:sessions[(l['study_uid'],l['session'])].add(l['view'])
        if all(selected.values()) and methods=={'TOMTEC bookmark'} and not any(len(v)==3 for v in sessions.values()):
            issues.append('Three views exist but do not share a saved analysis session')
            selected={v:'' for v in selected}
        missing=sum(not p for p in selected.values())
        if missing:issues.append(f'{missing} view(s) unresolved')
        if any(r['status']!='ok' for r in ff):issues.append('Unreadable or non-DICOM files present')
        visits.append({'patient':patient,'date':date,'strain_report':'Yes' if rr else 'No','matching_dicom_folder':match,'tomtec_bookmark':'Yes' if bm else 'No' if good else 'Unknown - no DICOM',
            'other_vendor_success':'Yes' if not bm and any(l['method'].startswith('Other vendor') for l in resolved) else 'Not needed - TOMTEC present' if bm else 'No',
            **selected,'clip_source':', '.join(sorted(methods)) if any(selected.values()) else '',
            'resolved_views':sum(bool(v) for v in selected.values()),'notes':'; '.join(issues),'report_exports':len(rr),
            'report_files':'\n'.join(r['report_path'] for r in rr),'dicom_folders':'\n'.join(sorted(set(str(Path(r['path']).parent) for r in ff)|set(folder_by[key]))),
            'dicom_files':len(good),'videos':sum(int(r.get('NumberOfFrames') or 1)>1 for r in good),
            'bookmark_files':'\n'.join(r['path'] for r in bm),'other_vendor_files':'\n'.join(c['file'] for c in vendor),
            'report_uids':'\n'.join(sorted(ruids)),'dicom_uids':'\n'.join(sorted(duids)),
            'analysis_sessions':len(set(l['session'] for l in eligible if l['method']=='TOMTEC bookmark'))})
    summary={'source_files':len(rows),'read_errors':sum(r['status']!='ok' for r in rows),'report_files':len(reports),'visit_rows':len(visits),'visits_with_report':sum(v['strain_report']=='Yes' for v in visits),'visits_with_dicom':sum(v['dicom_files']>0 for v in visits),'visits_with_tomtec':sum(v['tomtec_bookmark']=='Yes' for v in visits),'complete_triplets':sum(v['resolved_views']==3 for v in visits),'other_vendor_success_visits':sum(v['other_vendor_success']=='Yes' for v in visits),'resolved_view_counts':dict(collections.Counter(v['resolved_views'] for v in visits))}
    for name,data in [('visits',visits),('links',links),('candidates',candidates),('summary',summary),('reports',reports)]:
        (OUT/f'{name}.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
