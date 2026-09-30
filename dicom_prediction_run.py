"""Run remaining DICOM experiment stages sequentially, with durable per-stage logs."""
import argparse
import datetime
import json
import subprocess
import sys
import os
from pathlib import Path


def main():
    stages={
        'inventory':['dicom_prediction_inventory.py'],
        'crop':['dicom_prediction_video.py','crop','--all'],
        'classify':['dicom_prediction_video.py','classify'],
        'encode':['dicom_prediction_encode.py'],
        'evaluate':['dicom_prediction_evaluate.py'],
        'report':['dicom_prediction_report.py'],
    }
    p=argparse.ArgumentParser(__doc__);p.add_argument('--from-stage',choices=list(stages),default='inventory')
    args=p.parse_args();root=Path(__file__).resolve().parent;out=root/'output/dicom_prediction'
    logs=out/'logs';logs.mkdir(parents=True,exist_ok=True)
    names=list(stages);status={'stages':{},'pid':os.getpid(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    for name in names[names.index(args.from_stage):]:
        status['active_stage']=name;status['stages'][name]='running'
        (out/'run_status.json').write_text(json.dumps(status,indent=2))
        print('Starting',name,flush=True)
        with (logs/f'{name}.log').open('a',encoding='utf-8') as log:
            log.write('\n'+datetime.datetime.now(datetime.timezone.utc).isoformat()+'\n');log.flush()
            result=subprocess.run([sys.executable,'-u',*stages[name]],cwd=root,stdout=log,stderr=subprocess.STDOUT)
        status['stages'][name]='complete' if result.returncode==0 else 'failed'
        status['last_exit_code']=result.returncode
        (out/'run_status.json').write_text(json.dumps(status,indent=2))
        if result.returncode:raise SystemExit(result.returncode)
        print('Completed',name,flush=True)
    status['active_stage']=None;status['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (out/'run_status.json').write_text(json.dumps(status,indent=2))


if __name__=='__main__':main()
