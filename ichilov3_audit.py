"""Read-only DICOM source-link audit. Outputs are isolated from the source tree."""
import collections, concurrent.futures, datetime, hashlib, json, os, re, sys
from pathlib import Path
import xml.etree.ElementTree as ET
import pydicom

ROOT=Path('E:/ichilov3')
OUT=Path('D:/us/output/dicom_prediction/ichilov3_20260923')
BULK={0x7fe00010,0x7fe00008,0x7fe00009,0x54001010}
FIELDS=['SOPClassUID','SOPInstanceUID','StudyInstanceUID','SeriesInstanceUID','StudyDate','ContentDate','ContentTime','InstanceNumber','Manufacturer','ManufacturerModelName','SoftwareVersions','SeriesDescription','ImageComments','NumberOfFrames','Rows','Columns','ImageType','PatientID']

def xml(value):
    if isinstance(value,bytes):value=value.decode('utf-8',errors='replace')
    return ET.fromstring(value.strip('\x00 \r\n\t'))

def inspect(path):
    st=path.stat();r={'path':str(path),'size':st.st_size,'mtime_ns':st.st_mtime_ns,'status':'ok','private':[],'references':[],'text':[],'bookmarks':[]}
    rel=path.relative_to(ROOT).parts;r['patient_folder']=rel[0] if len(rel)>1 else '';r['date_folder']=rel[1] if len(rel)>2 else ''
    try:
        ds=pydicom.dcmread(path,defer_size=1024)
        r.update({k:str(ds.get(k,'')) for k in FIELDS})
        def walk(d,location=''):
            for tag in d.keys():
                if int(tag) in BULK:continue
                e=d[tag];loc=f'{location}/{e.keyword or str(tag)}'
                if e.keyword=='ReferencedSOPInstanceUID' and str(e.value):r['references'].append({'uid':str(e.value),'location':loc})
                if e.VR=='SQ':
                    for i,item in enumerate(e.value):walk(item,loc+f'[{i}]')
                elif tag.is_private:
                    v=e.value; text=v[:8000].decode('utf-8',errors='replace') if isinstance(v,bytes) else str(v)
                    r['private'].append({'tag':str(tag),'vr':e.VR,'length':len(v) if hasattr(v,'__len__') else 0,'prefix':text[:500]})
                    if isinstance(v,bytes) and v.lstrip().startswith((b'<?xml',b'<TT_',b'<xtt',b'<reportData')):
                        try:
                            tree=xml(v)
                            if tree.tag=='xtt' and tree.find('.//SopInstanceUid') is not None:
                                r['bookmarks'].append({'tag':str(tag),'source_uids':sorted(set(n.text.strip() for n in tree.iter('SopInstanceUid') if n.text)),
                                    'source_studies':sorted(set(n.text.strip() for n in tree.iter('StudyInstanceUid') if n.text)),
                                    'view_tags':sorted(set((n.text or '').strip() for n in tree.iter('DataNodeTags') if re.search('PLANE|[234]CH',(n.text or '')))),
                                    'strain_arrays':len(list(tree.iter('StrainValues'))),'application':tree.findtext('.//ApplicationName',''),
                                    'timing':{n.tag:n.text for n in tree.find('.//HeartCycleInformation')} if tree.find('.//HeartCycleInformation') is not None else {}})
                            elif tree.tag=='TT_XML_DATACONTAINER_DOC_V2':
                                r.setdefault('container',{}).update({n.get('K'):n.get('V') for n in tree.iter('DI') if n.get('K') in ['FileClass','FileLabel','RelatedFileUID','GenAppl','Type']})
                            elif tree.tag=='reportData':
                                r['report_keys']=[n.get('key') for n in tree.iter('measuredFinding')]
                        except Exception as exc:r.setdefault('xml_errors',[]).append(str(exc))
                elif e.VR in {'LO','SH','ST','LT','UT','CS','UC'} and e.keyword not in {'PatientName','PatientID','InstitutionName','ReferringPhysicianName'}:
                    v=str(e.value)
                    if v:r['text'].append({'location':loc,'value':v[:1000]})
        walk(ds)
    except Exception as exc:r['status']='error';r['error']=str(exc)
    return r

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    target=OUT/'inventory.jsonl'
    completed=set()
    if target.exists():
        for line in target.read_text(encoding='utf-8').splitlines():
            try:completed.add(json.loads(line)['path'])
            except json.JSONDecodeError:pass
    paths=[];dirs=[]
    for patient in ROOT.iterdir():
        if patient.is_dir():
            for visit in patient.iterdir():
                if visit.is_dir():dirs.append(str(visit))
    for folder,sub,names in os.walk(ROOT):paths.extend(Path(folder)/name for name in names)
    if not (OUT/'folders.json').exists():(OUT/'folders.json').write_text(json.dumps(dirs,indent=2))
    paths=[p for p in paths if str(p) not in completed]
    print(f'{len(paths)} files; {len(dirs)} visit directories',flush=True)
    start=datetime.datetime.now();summary=collections.Counter()
    with target.open('a',encoding='utf-8') as f,concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        if completed:f.write('\n')
        for i,r in enumerate(pool.map(inspect,paths),1):
            f.write(json.dumps(r)+'\n');summary[r['status']]+=1
            if i%2000==0:print(f'{i}/{len(paths)} files, elapsed {datetime.datetime.now()-start}',flush=True)
    print(dict(summary),flush=True)

if __name__=='__main__':main()
