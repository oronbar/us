"""Verify every selected clip and source inventory without writing to E:."""
import collections,json,os,xml.etree.ElementTree as ET
from pathlib import Path
import pydicom
from ichilov3_audit import ROOT,OUT

def main():
    inv={}
    for line in (OUT/'inventory.jsonl').read_text(encoding='utf-8').splitlines():
        try:r=json.loads(line);inv[r['path']]=r
        except json.JSONDecodeError:pass
    current={str(Path(d)/n) for d,_,names in os.walk(ROOT) for n in names}
    assert current==set(inv),(len(current),len(inv),'source listing changed or inventory incomplete')
    changes=[]
    for path,r in inv.items():
        st=Path(path).stat()
        if (st.st_size,st.st_mtime_ns)!=(r['size'],r['mtime_ns']):changes.append(path)
    assert not changes,changes
    visits=json.loads((OUT/'visits.json').read_text());links=json.loads((OUT/'links.json').read_text());reports=json.loads((OUT/'reports.json').read_text())
    assert len({(r['patient'],r['date']) for r in visits})==len(visits)
    assert sum(r['report_exports'] for r in visits)==len(reports)
    decisions=json.loads((OUT/'session_decisions.json').read_text())
    for decision in decisions:
        assert all(decision['reports'])
        matches=[m for rr in decision['reports'] for m in rr]
        assert all(m['seconds']<=10 and m['max_gls_difference_pp']<=0.05 for m in matches)
        assert len({tuple(sorted(m['uids'].items())) for m in matches})==1
    verified=0;cache={}
    for r in visits:
        assert r['resolved_views']==sum(bool(r[v]) for v in ['A2C','A3C','A4C'])
        for v in ['A2C','A3C','A4C']:
            p=r[v]
            if not p:continue
            clip=pydicom.dcmread(p,stop_before_pixels=True,specific_tags=['SOPInstanceUID','StudyInstanceUID','NumberOfFrames'])
            assert int(clip.NumberOfFrames)>1
            evidence=[l for l in links if l['patient']==r['patient'] and l['view']==v and p in l['matched_files']]
            assert evidence
            for l in evidence:
                assert str(clip.SOPInstanceUID)==l['source_uid'] and str(clip.StudyInstanceUID)==l['study_uid']
                if l['method']=='TOMTEC bookmark':
                    if l['analysis_file'] not in cache:
                        bm=pydicom.dcmread(l['analysis_file']); tree=ET.fromstring(bm[0x7fdf5051].value.rstrip(b'\x00'))
                        cache[l['analysis_file']]={n.text.strip() for n in tree.iter('SopInstanceUid') if n.text}
                    assert l['source_uid'] in cache[l['analysis_file']]
            verified+=1
    result={'inventory_files':len(inv),'source_size_mtime_unchanged':not changes,'verified_selected_views':verified,'report_files_accounted_for':len(reports),'unique_visit_rows':len(visits)}
    (OUT/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
