"""Download and verify the pinned public model artifacts; no patient data leaves the machine."""
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import requests

WEIGHTS={
 'model_data.zip':('https://github.com/echonet/EchoPrime/releases/download/v1.0.0/model_data.zip','b29362b6b40e8695b138d191f874ad92417b6da9fcdb2f0764ed1440be5272b3'),
 'panecho.pt':('https://github.com/CarDS-Yale/PanEcho/releases/download/v1.0/panecho.pt','896a279d2e5d669dd762adb2077e948ec38087006d2b4001657b87a132e2da35'),
}
SOURCES={
 'PanEcho_models.py':('https://raw.githubusercontent.com/CarDS-Yale/PanEcho/05d1a771fb6b4f187bb41167cfe53243437da92d/src/models.py','b211b3c81c76fcb6ed39cf258970e859c2bda74f2be6870f46556a4cd7873d43'),
 'EchoPrime_model.py':('https://raw.githubusercontent.com/echonet/EchoPrime/1d52e686c0e460b9c00c18ca92e968e486780733/echo_prime/model.py','04f22d9a2367a83edf0561463845fd45a995e8702ffd0936ecfc81cd0172e577'),
 'EchoPrime_utils.py':('https://raw.githubusercontent.com/echonet/EchoPrime/e634d7a8756aa17698eb58da12ffc25f71ab7321/utils/utils.py','fc976e24bb5dca61571452e0d3ee6e928b4320f572fe892fc69acf600313cb6f'),
}


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,default=Path('D:/us/output/dicom_prediction'))
    args=p.parse_args();out=args.output
    weights=out/'weights';sources=out/'vendor_sources'
    weights.mkdir(parents=True,exist_ok=True);sources.mkdir(exist_ok=True)
    for name,(url,expected) in WEIGHTS.items():
        path=weights/name
        if not path.exists():
            temporary=path.with_suffix(path.suffix+'.partial')
            with requests.get(url,stream=True,timeout=(30,120)) as response:
                response.raise_for_status()
                with temporary.open('wb') as stream:
                    for block in response.iter_content(8*1024*1024):stream.write(block)
            if digest(temporary)!=expected:raise RuntimeError(f'Checksum mismatch: {name}')
            temporary.replace(path)
        if digest(path)!=expected:raise RuntimeError(f'Checksum mismatch: {name}')
        print('Verified',name,flush=True)
    with zipfile.ZipFile(weights/'model_data.zip') as archive:
        for name in ['echo_prime_encoder.pt','view_classifier.pt']:
            member=next(x for x in archive.namelist() if Path(x).name==name and '/weights/' in x)
            with archive.open(member) as source,(weights/name).open('wb') as target:shutil.copyfileobj(source,target)
    for name,(url,expected) in SOURCES.items():
        response=requests.get(url,timeout=30);response.raise_for_status();text=response.text
        if hashlib.sha256(text.encode()).hexdigest()!=expected:raise RuntimeError(f'Source checksum mismatch: {name}')
        (sources/name).write_text(text,encoding='utf-8')
    for name,url in {
        'EchoPrime_LICENSE':'https://raw.githubusercontent.com/echonet/EchoPrime/1d52e686c0e460b9c00c18ca92e968e486780733/LICENSE',
        'PanEcho_LICENSES.md':'https://raw.githubusercontent.com/CarDS-Yale/PanEcho/05d1a771fb6b4f187bb41167cfe53243437da92d/LICENSES.md',
    }.items():
        response=requests.get(url,timeout=30);response.raise_for_status();(sources/name).write_text(response.text,encoding='utf-8')
    (out/'weights_manifest.json').write_text(json.dumps({'urls':{k:v[0] for k,v in WEIGHTS.items()},'sha256':{k:v[1] for k,v in WEIGHTS.items()}},indent=2))
    (out/'source_provenance.json').write_text(json.dumps({k:{'url':v[0],'text_sha256':v[1]} for k,v in SOURCES.items()},indent=2))


if __name__=='__main__':main()
