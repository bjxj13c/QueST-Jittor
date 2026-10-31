"""Automatic bounded tokenizer diagnosis -> paired policy -> dev10 -> evidence."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'outputs/official_reach_100_20261028/adapted'
TAG='official100_policy_20261028'
OUT=ROOT/'outputs'/TAG
resuming=OUT.exists()
if not resuming:OUT.mkdir()
TORCH=ROOT/'.venv-torch-linux/bin/python'
JT=ROOT/'.venv-jittor-linux/bin/python'
NATIVE=[JT,'-u',ROOT/'scripts/run_jittor_linux.py','--device','cuda']
started=time.monotonic()
env=os.environ.copy();env.update(PYTHONPATH=str(ROOT/'src'),NVIDIA_TF32_OVERRIDE='0',OMP_NUM_THREADS='4',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl')
report={'status':'running','run':TAG,'data_root':str(DATA),'stages':[],
    'scope':'single Reach,80train20validation; paired20epoch policies from same init;development2000-2009 only',
    'started_utc':datetime.now(timezone.utc).isoformat(),'server_shutdown':False}
if resuming:
    report=json.loads((OUT/'workflow.json').read_text())
    if report['status']!='failed':raise RuntimeError('Resume only a failed workflow; never duplicate running/completed jobs')
    shutil.copy2(OUT/'workflow.json',OUT/('workflow_failed_'+datetime.now(timezone.utc).strftime('%H%M%S')+'.json'))
    started-=report.get('elapsed_seconds',0)
    report.update(status='running',resumed_utc=datetime.now(timezone.utc).isoformat())
    report.pop('error',None)
def save():
    report['elapsed_seconds']=time.monotonic()-started
    tmp=OUT/'workflow.tmp';tmp.write_text(json.dumps(report,indent=2));tmp.replace(OUT/'workflow.json')
def stage(name,command,cap=3600):
    if any(record['name']==name and record['status']=='passed' for record in report['stages']):return
    record={'name':name,'status':'running','command':[str(x) for x in command]}
    report['stages'].append(record);save();tick=time.monotonic()
    if (OUT/f'{name}.log').exists():
        shutil.copy2(OUT/f'{name}.log',OUT/f'{name}_before_retry_{datetime.now(timezone.utc).strftime("%H%M%S")}.log')
    with (OUT/f'{name}.log').open('wb') as log:
        child=subprocess.Popen(record['command'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=child.wait(timeout=min(cap,max(1,10800-(time.monotonic()-started))))
        except BaseException:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            record['status']='interrupted';save();raise
    record.update(status='passed' if code==0 else 'failed',exit_code=code,elapsed_seconds=time.monotonic()-tick);save()
    if code:raise RuntimeError(f'{name} failed; see {OUT/name}.log')

try:
    assert json.loads((ROOT/'outputs/official100_tokenizer_pair_20261028_v2/workflow.json').read_text())['status']=='completed'
    new_token=DATA/'outputs/official100_tokenizer_jittor20'
    new_audit=OUT/'new_tokenizer_audit'
    stage('audit_new_tokenizer',NATIVE+[ROOT/'scripts/audit_tokenizer_transfer.py','--data-root',DATA,'--weights',new_token/'best_weights.npz','--config',new_token/'config.json','--output',new_audit,'--framework','jittor'],300)
    stage('audit_new_noise_reference',[TORCH,ROOT/'scripts/audit_noise_reference.py','--data-root',DATA,'--predictions',new_audit],120)
    old_noise=json.loads((ROOT/'outputs/tokenizer_transfer_20261028/noise_reference.json').read_text())
    new_noise=json.loads((new_audit/'noise_reference.json').read_text())
    old_error=old_noise['splits']['validation']['frozen_reconstruction_vs_clean_expert_mae']
    new_error=new_noise['splits']['validation']['frozen_reconstruction_vs_clean_expert_mae']
    chosen=new_token if new_error<old_error else ROOT/'outputs/gpu_20261028_104533_tokenizer_jittor'
    report['tokenizer_selection']={'criterion':'lower held-out reconstruction error versus noiseless expert reference at recorded states; no rollout seeds used',
        'old_clean_reference_mae':old_error,'new_clean_reference_mae':new_error,'chosen':str(chosen),
        'limitation':'extra offline diagnostic selection,not original paper selection protocol;clean references never policy inputs'}
    assets=DATA/'outputs'/f'{TAG}_assets';assets.mkdir(parents=True,exist_ok=resuming)
    weights=assets/'weights.npz';config=assets/'config.json'
    shutil.copy2(chosen/'best_weights.npz',weights);shutil.copy2(chosen/'config.json',config)
    common_token=['--tokenizer-weights',weights,'--tokenizer-config',config]
    shared=assets/'shared';token_pt=shared/'tokenizer_shared.pt'
    stage('shared_tokenizer',[TORCH,ROOT/'scripts/prepare_policy_stage.py','--root',DATA,'--framework','torch','--prepare-checkpoint',*common_token,'--output',shared],120)
    for name,command in [('torch',[TORCH]),('jittor',NATIVE)]:
        extra=['--tokenizer-checkpoint',token_pt] if name=='torch' else []
        stage('labels_'+name,command+[ROOT/'scripts/prepare_policy_stage.py','--root',DATA,'--framework',name,*common_token,*extra,'--output',OUT/f'labels_{name}'],600)
    for split in ('train','validation'):
        with np.load(OUT/'labels_torch'/f'{split}.npz') as a,np.load(OUT/'labels_jittor'/f'{split}.npz') as b:
            for key in a.files:np.testing.assert_array_equal(a[key],b[key])
    report['labels_exact']=True;save()
    plan=OUT/'batch_plan.npz'
    stage('batch_plan',[TORCH,ROOT/'scripts/create_batch_plan.py','--root',DATA,'--output',plan,'--epochs','20','--batch-size','16','--seed','303100'],60)
    initial=ROOT/'outputs/fixtures/policy_full_seed456/weights.npz'
    common=['--root',DATA,'--device','cuda','--batch-size','16','--dropout','0','--seed','20000','--epochs','20','--batch-plan',plan]
    stage('train_torch',[TORCH,'-u',ROOT/'scripts/train_policy.py','--tokenizer',token_pt,'--initial-weights',initial,'--name',TAG+'_torch',*common])
    stage('train_jittor',NATIVE+[ROOT/'scripts/train_policy_jittor.py',*common_token,'--initial-policy-weights',initial,'--name',TAG+'_jittor',*common])
    for name,command in [('torch',[TORCH]),('jittor',NATIVE)]:
        run=DATA/'outputs'/f'{TAG}_{name}'
        policy_args=['--policy',run/'best.pt','--tokenizer',token_pt] if name=='torch' else ['--policy-weights',run/'best_weights.npz','--policy-config',run/'config.json',*common_token]
        stage('development_'+name,command+[ROOT/'scripts'/f'evaluate_policy{"_jittor" if name=="jittor" else ""}.py','--root',DATA,*policy_args,'--device','cuda','--seed-set','development','--episodes','10','--videos','2','--action-horizon','8','--max-steps','150','--name',f'{TAG}_{name}_dev10'],1200)
        stage('plot_'+name,[JT,ROOT/'scripts/plot_policy_run.py',run],120)
    learned=DATA/'outputs'/f'{TAG}_jittor'
    for name,command in [('torch',[TORCH]),('jittor',NATIVE)]:
        stage('same_weights_'+name,command+[ROOT/'scripts/export_trained_policy_predictions.py','--root',DATA,'--framework',name,'--policy-weights',learned/'best_weights.npz','--policy-config',learned/'config.json',*common_token,'--output',OUT/f'predictions_{name}.npz'],180)
    with np.load(OUT/'predictions_torch.npz') as a,np.load(OUT/'predictions_jittor.npz') as b:
        comparison={key:{'max_abs':float(np.abs(a[key]-b[key]).max()),'allclose':bool(np.allclose(a[key],b[key],atol=2e-5,rtol=2e-4))} for key in a.files}
        comparison['tokens_exact']=bool(np.array_equal(a['tokens'],b['tokens']))
    (OUT/'same_weights_comparison.json').write_text(json.dumps(comparison,indent=2))
    report['same_weights_comparison']=comparison
    report['development']={name:json.loads((DATA/'outputs'/f'{TAG}_{name}_dev10/summary.json').read_text()) for name in ('torch','jittor')}
    report['status']='completed';save()
    archive=ROOT/'reports/official100_policy_completed_20261028.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for folder in [OUT,assets,*[DATA/'outputs'/f'{TAG}_{name}{suffix}' for name in ('torch','jittor') for suffix in ('','_dev10')]]:
            for path in folder.rglob('*'):
                if path.is_file() and path.name not in ('last.pt','last_optimizer.pkl'):
                    z.write(path,path.relative_to(ROOT))
    metadata={'path':str(archive),'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}
    (ROOT/'reports/official100_policy_archive.json').write_text(json.dumps(metadata,indent=2))
except BaseException as error:
    report.update(status='failed',error=f'{type(error).__name__}: {error}');raise
finally:save()
