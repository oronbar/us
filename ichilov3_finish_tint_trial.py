"""Complete the paired report after the independently running encoding worker."""
import json, subprocess, sys, time
from pathlib import Path
import psutil
from ichilov3_train_fusion import atomic
root=Path(r'D:\DS\ichilov3_tint_trial_20260927')
deadline=time.time()+7200
while not (root/'run_complete.json').exists():
    if (root/'status.json').exists():
        status=json.loads((root/'status.json').read_text())
        if status.get('stage')=='error':raise RuntimeError(status['error'])
        if status.get('pid') and not psutil.pid_exists(status['pid']):raise RuntimeError('Encoding worker stopped before completion')
    if time.time()>deadline:raise TimeoutError('Trial exceeded two hours')
    time.sleep(15)
subprocess.run([sys.executable,r'D:\us\ichilov3_compare_tint_trial.py'],check=True)
atomic(root/'comparison_complete.json',dict(completed=True,report=str(root/'results.md')))
