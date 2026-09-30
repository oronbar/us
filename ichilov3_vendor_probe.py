"""Search private vendor payloads in visits without TOMTEC for source references."""
import collections,json,re,zlib
from pathlib import Path
import pydicom
from ichilov3_audit import OUT,BULK

def main():
    inv={}
    for line in (OUT/'inventory.jsonl').read_text(encoding='utf-8').splitlines():
        try:r=json.loads(line);inv[r['path']]=r
        except json.JSONDecodeError:pass
    bmkeys={(r['patient_folder'],r.get('StudyInstanceUID')) for r in inv.values() if r['bookmarks'] or r.get('container',{}).get('FileClass')=='Bookmark'}
    uids={r.get('SOPInstanceUID') for r in inv.values() if int(r.get('NumberOfFrames') or 1)>1}
    outputs=[];examined=0
    for r in inv.values():
        if r['status']!='ok' or (r['patient_folder'],r.get('StudyInstanceUID')) in bmkeys or 'tomtec' in r.get('Manufacturer','').lower():continue
        payloads=[p for p in r['private'] if p['vr'] in ['OB','OW','UN'] and p['length']>64]
        if not payloads:continue
        ds=pydicom.dcmread(r['path'],defer_size=1024)
        # Traverse metadata only, never accessing pixel/waveform elements.
        def walk(d):
            for tag in d.keys():
                if int(tag) in BULK:continue
                e=d[tag]
                if e.VR=='SQ':
                    for item in e.value:yield from walk(item)
                elif tag.is_private and isinstance(e.value,bytes):yield e
        for e in walk(ds):
            data=e.value
            if len(data)<=64:continue
            examined+=1;variants=[data]
            if data[:2] in [b'\x78\x9c',b'\x78\xda',b'\x78\x01']:
                try:variants.append(zlib.decompressobj().decompress(data,20*1024*1024))
                except zlib.error:pass
            hits=set();markers=set()
            for b in variants:
                for text in [b.decode('utf-8',errors='ignore'),b.decode('utf-16-le',errors='ignore')]:
                    hits.update(u for u in re.findall(r'(?<![\d.])[12](?:\.\d+){5,}(?![\d.])',text) if u in uids)
                    markers.update(m.group(0) for m in re.finditer(r'(?i)strain|\bAFI\b|A[234]C|VIEW_PLANE_[234]CH',text))
            if hits or markers:outputs.append({'file':r['path'],'tag':str(e.tag),'length':len(data),'matching_video_uids':sorted(hits),'markers':sorted(markers)})
    result={'private_payloads_examined':examined,'findings':outputs,'limitation':'Binary string search and bounded zlib decoding; opaque vendor algorithms are not interpreted. No clip is selected from a string hit alone.'}
    (OUT/'vendor_private_probe.json').write_text(json.dumps(result,indent=2));print(json.dumps({'private_payloads_examined':examined,'finding_count':len(outputs),'examples':outputs[:3]},indent=2))

if __name__=='__main__':main()
