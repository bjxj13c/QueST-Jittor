"""Bounded paired fine-tuning on frozen grouped official100 data; no final eval."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
data=ROOT/'outputs/official_reach_100_20261028/adapted'
output=ROOT/'outputs/official100_tokenizer_pair_20261028_v2'
output.mkdir(exist_ok=False)
started=time.monotonic()
report={'status':'running','epochs':20,'scope':'paired20epoch fine-tuning from old native best weights; single Reach,80train20val; no learned-policy evaluation',
        'started_utc':datetime.now(timezone.utc).isoformat(),'stages':[], 'server_shutdown':False}
env=os.environ.copy()
env.update(PYTHONPATH=str(ROOT/'src'),NVIDIA_TF32_OVERRIDE='0',OMP_NUM_THREADS='4')
def save():
    report['elapsed_seconds']=time.monotonic()-started
    (output/'workflow.json').write_text(json.dumps(report,indent=2))
def stage(name,command):
    record={'name':name,'status':'running','command':[str(x) for x in command]}
    report['stages'].append(record);save()
    with (output/f'{name}.log').open('wb') as log:
        child=subprocess.Popen(record['command'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            code=child.wait(timeout=max(1,7200-(time.monotonic()-started)))
        except BaseException:
            os.killpg(child.pid,signal.SIGTERM)
            try: child.wait(timeout=15)
            except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL);child.wait()
            record['status']='interrupted';save();raise
    record['exit_code']=code;record['status']='passed' if code==0 else 'failed';save()
    if code:raise RuntimeError(f'{name} failed; inspect {output}/{name}.log')
try:
    # Add missing metadata without changing split membership or data bytes.
    path=data/'records/reach_split_v1.json'
    manifest=json.loads(path.read_text())
    if any(key not in manifest for key in ('split_seed','stride','padding')):
        (output/'manifest_before_metadata.json').write_text(json.dumps(manifest,indent=2))
        manifest.update(split_seed=302900,split_unit='initial-state groups; no overlap',stride=1,padding='repeat final action within same episode')
        path.write_text(json.dumps(manifest,indent=2))
    torch=ROOT/'.venv-torch-linux/bin/python'
    native=[ROOT/'.venv-jittor-linux/bin/python','-u',ROOT/'scripts/run_jittor_linux.py','--device','cuda']
    plan=output/'batch_plan.npz'
    initial=ROOT/'outputs/gpu_20261028_104533_tokenizer_jittor/best_weights.npz'
    stage('batch_plan',[torch,ROOT/'scripts/create_batch_plan.py','--root',data,'--output',plan,'--epochs','20','--batch-size','16','--seed','303000'])
    common=['--root',data,'--device','cuda','--arch','full','--epochs','20','--batch-size','16','--seed','303000','--dropout','0','--initial-weights',initial,'--batch-plan',plan]
    stage('torch',[torch,'-u',ROOT/'scripts/train_tokenizer.py','--name','official100_tokenizer_torch20',*common])
    stage('jittor',native+[ROOT/'scripts/train_tokenizer_jittor.py','--name','official100_tokenizer_jittor20',*common])
    report['status']='completed'
except BaseException as error:
    report['status']='failed';report['error']=f'{type(error).__name__}: {error}'
    raise
finally:save()
