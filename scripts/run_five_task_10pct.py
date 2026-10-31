"""Bounded five-task450-demo workflow;preserves old results and no shutdown."""
from datetime import datetime,timezone
import fcntl
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

ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'outputs/five_task_10pct_20261029';OUT=RUN/'experiments';DATA=RUN/'dataset'
OUT.mkdir(parents=True,exist_ok=True)
lock=(RUN/'workflow.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
protocol=json.loads((ROOT/'records/five_task_10pct_20261029.json').read_text())
TORCH=ROOT/'.venv-torch-linux/bin/python';COLLECT=ROOT/'.venv-collect-linux/bin/python'
NATIVE=[ROOT/'.venv-jittor-linux/bin/python','-u',ROOT/'scripts/run_jittor_linux.py','--device','cuda']
env=os.environ.copy();env.update(PYTHONPATH=str(ROOT/'src'),NVIDIA_TF32_OVERRIDE='0',OMP_NUM_THREADS='4',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl')
started=time.monotonic();previous_elapsed=0
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
source_paths=[ROOT/'records/five_task_10pct_20261029.json',*list((ROOT/'scripts').glob('*five_task_10pct.py')),
    ROOT/'scripts/collect_official_reach.py',ROOT/'scripts/train_controlled_policy.py',ROOT/'scripts/train_tokenizer.py',ROOT/'scripts/train_tokenizer_jittor.py',
    ROOT/'scripts/evaluate_policy.py',ROOT/'scripts/evaluate_policy_jittor.py',*list((ROOT/'src/quest_repro').glob('*.py'))]
hashes={str(p.relative_to(ROOT)):digest(p) for p in source_paths}
report={'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),'protocol':protocol,'source_hashes':hashes,'stages':[],'server_shutdown':False}
if (OUT/'workflow.json').exists():
    report=json.loads((OUT/'workflow.json').read_text())
    assert report['status']=='failed','Refuse duplicate running/completed workflow'
    assert report['source_hashes']==hashes,'Do not silently change code/protocol while resuming'
    shutil.copy2(OUT/'workflow.json',OUT/f'failed_{int(time.time())}.json')
    previous_elapsed=report.get('elapsed_seconds',0);report.update(status='running');report.pop('error',None)

def save():
    report['elapsed_seconds']=previous_elapsed+time.monotonic()-started
    temp=OUT/'workflow.tmp';temp.write_text(json.dumps(report,indent=2));temp.replace(OUT/'workflow.json')

def stage(name,command,cap=3600):
    if any(s['name']==name and s['status']=='passed' for s in report['stages']):return
    remaining=protocol['time_limit_hours']*3600-(previous_elapsed+time.monotonic()-started)
    if remaining<=0:raise TimeoutError('Six-hour workflow time limit reached;server remains ON')
    record={'name':name,'status':'running','command':[str(v) for v in command],'started_utc':datetime.now(timezone.utc).isoformat()}
    report['stages'].append(record);save();log=OUT/f'{name}.log'
    if log.exists():shutil.copy2(log,OUT/f'{name}_before_retry_{int(time.time())}.log')
    tick=time.monotonic()
    with log.open('wb') as f:
        child=subprocess.Popen(record['command'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        record['pid']=child.pid;save()
        try:code=child.wait(timeout=min(cap,remaining))
        except BaseException:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            record['status']='interrupted';save();raise
    record.update(status='passed' if code==0 else 'failed',exit_code=code,elapsed_seconds=time.monotonic()-tick);save()
    if code:raise RuntimeError(f'{name} failed;preserved log:{log}')

def train(framework,initial,smoke=False):
    settings=protocol['policy'];name=('smoke_' if smoke else 'policy_')+framework
    cmd=([TORCH,'-u'] if framework=='torch' else NATIVE)+[ROOT/'scripts/train_controlled_policy.py','--root',DATA,'--output',OUT/name,
        '--framework',framework,'--tokenizer-weights',DATA/'assets/weights.npz','--tokenizer-config',DATA/'assets/config.json',
        '--initial-weights',initial,'--canonical-labels',DATA/'assets/canonical_labels',
        '--updates',str(4 if smoke else settings['updates']),'--eval-every',str(4 if smoke else settings['eval_every']),
        '--lr',str(settings['lr']),'--weight-decay',str(settings['weight_decay']),'--dropout',str(settings['dropout']),
        '--shift',str(settings['shift']),'--seed',str(settings['seed'])]
    if framework=='torch':cmd+=['--tokenizer-checkpoint',DATA/'assets/shared/tokenizer_shared.pt']
    stage(name,cmd,5400)

try:
    save()
    assert shutil.disk_usage(ROOT).free>12*1024**3,'Need12GiB free before new data/model outputs'
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
    report['gpu_preflight']=gpu;save()
    assert int(gpu.split(',')[1].strip().split()[0])<512,'Another GPU workload appears active;do not overlap'
    for name,tasks,n,seed in [('collect_new_two',protocol['tasks'][3:],100,protocol['collection_seeds']['new_two']),
        ('collect_extra_ten',protocol['tasks'],10,protocol['collection_seeds']['additional_ten'])]:
        stage(name,[COLLECT,'-u',ROOT/'scripts/collect_official_reach.py','--output',RUN/name,'--episodes',str(n),'--seed',str(seed),'--tasks',*tasks],7200)
    stage('prepare_data',[TORCH,ROOT/'scripts/prepare_five_task_10pct.py'],600)
    stage('goal_isolation',[TORCH,ROOT/'scripts/audit_five_task_10pct.py','--phase','isolation'],180)
    initial_token=ROOT/'outputs/gpu_20261028_104533_tokenizer_jittor/best_weights.npz'
    token=protocol['tokenizer'];plan=OUT/'tokenizer_batch_plan.npz'
    stage('tokenizer_plan',[TORCH,ROOT/'scripts/create_batch_plan.py','--root',DATA,'--output',plan,'--epochs',str(token['epochs']),
        '--batch-size','16','--seed',str(token['seed'])],60)
    for framework in ('torch','jittor'):
        command=([TORCH,'-u'] if framework=='torch' else list(NATIVE))
        command+=[ROOT/'scripts'/('train_tokenizer.py' if framework=='torch' else 'train_tokenizer_jittor.py'),'--root',DATA,
            '--name','five_tokenizer_'+framework,'--device','cuda','--epochs',str(token['epochs']),'--batch-size','16',
            '--seed',str(token['seed']),'--dropout','0','--initial-weights',initial_token,'--batch-plan',plan]
        stage('tokenizer_'+framework,command,3600)
    assets=DATA/'assets';assets.mkdir(exist_ok=True)
    native_token=DATA/'outputs/five_tokenizer_jittor'
    for source,dest in [(native_token/'best_weights.npz',assets/'weights.npz'),(native_token/'config.json',assets/'config.json')]:
        if dest.exists():assert digest(source)==digest(dest)
        else:shutil.copy2(source,dest)
    stage('bridge',[TORCH,ROOT/'scripts/prepare_policy_stage.py','--root',DATA,'--framework','torch','--prepare-checkpoint',
        '--tokenizer-weights',assets/'weights.npz','--tokenizer-config',assets/'config.json','--output',assets/'shared'],120)
    stage('canonical_labels',NATIVE+[ROOT/'scripts/prepare_policy_stage.py','--root',DATA,'--framework','jittor',
        '--tokenizer-weights',assets/'weights.npz','--tokenizer-config',assets/'config.json','--output',assets/'canonical_labels'],600)
    initial=OUT/'initial_five.npz'
    if not initial.exists():
        with np.load(ROOT/'outputs/fixtures/policy_full_seed456/weights.npz') as old:values={k:old[k].copy() for k in old.files}
        key='task_encoder.weight';old=values[key];new=np.random.default_rng(310456).normal(size=(5,old.shape[1])).astype(old.dtype);new[0]=old[0];values[key]=new
        np.savez_compressed(initial,**values)
    for framework in ('torch','jittor'):train(framework,initial,smoke=True)
    for framework in ('torch','jittor'):train(framework,initial)
    summaries=[json.loads((OUT/('policy_'+f)/'summary.json').read_text()) for f in ('torch','jittor')]
    assert summaries[0]['batch_stream_sha256']==summaries[1]['batch_stream_sha256']
    for framework in ('torch','jittor'):
        stage('fixture_'+framework,([TORCH] if framework=='torch' else NATIVE)+[ROOT/'scripts/export_trained_policy_predictions.py',
            '--root',DATA,'--framework',framework,'--policy-weights',OUT/'policy_jittor/best_weights.npz',
            '--policy-config',OUT/'policy_jittor/config.json','--tokenizer-weights',assets/'weights.npz',
            '--tokenizer-config',assets/'config.json','--output',OUT/f'fixture_{framework}.npz'],300)
    with np.load(OUT/'fixture_torch.npz') as a,np.load(OUT/'fixture_jittor.npz') as b:
        comparison={}
        for key in a.files:
            np.testing.assert_allclose(a[key],b[key],atol=2e-5,rtol=2e-4);comparison[key]={'max_abs':float(np.abs(a[key]-b[key]).max())}
        np.testing.assert_array_equal(a['tokens'],b['tokens']);comparison['tokens_exact']=True
    (OUT/'same_weights.json').write_text(json.dumps(comparison,indent=2))
    freeze={'utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':digest(ROOT/'records/five_task_10pct_20261029.json'),
        'dataset_sha256':digest(DATA/'data/demonstrations.hdf5'),'tokenizer_sha256':digest(assets/'weights.npz'),
        'models':{f:digest(OUT/('policy_'+f)/'best_weights.npz') for f in ('torch','jittor')},'test':protocol['test']}
    path=OUT/'test_freeze.json'
    if path.exists():
        prior=json.loads(path.read_text());assert all(prior[k]==v for k,v in freeze.items() if k!='utc')
    else:path.write_text(json.dumps(freeze,indent=2))
    for framework in ('torch','jittor'):
        for task_id,task in enumerate(protocol['tasks']):
            model=OUT/('policy_'+framework);name=f'test_{framework}_{task}';test=protocol['test']
            if framework=='torch':cmd=[TORCH,ROOT/'scripts/evaluate_policy.py','--policy',model/'best.pt','--tokenizer',assets/'shared/tokenizer_shared.pt','--sampling-backend','numpy']
            else:cmd=NATIVE+[ROOT/'scripts/evaluate_policy_jittor.py','--policy-weights',model/'best_weights.npz','--policy-config',model/'config.json','--tokenizer-weights',assets/'weights.npz','--tokenizer-config',assets/'config.json']
            stage(name,cmd+['--root',DATA,'--device','cuda','--seed-set','experiment','--seed-start',str(test['seed_start']),
                '--episodes',str(test['episodes_per_task']),'--env-name',task+'-goal-observable','--task-id',str(task_id),'--per-episode-sampling',
                '--top-k',str(test['top_k']),'--sampling-seed',str(test['sampling_seed']),'--action-horizon',str(test['action_horizon']),
                '--max-steps',str(test['max_steps']),'--videos','0','--name',name],900)
    stage('completion_report',[TORCH,ROOT/'scripts/audit_five_task_10pct.py','--phase','report'],240)
    # Archive data,weights,logs,source and reports before declaring completion.
    archive=ROOT/'reports/five_task_10pct_20261029.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for path in RUN.rglob('*'):
            if path.is_file() and path.name not in ('last.pt','last_weights.npz','last_optimizer.pkl','workflow.lock','workflow.tmp') and 'videos' not in path.relative_to(RUN).parts:
                z.write(path,path.relative_to(ROOT))
        for path in source_paths:z.write(path,path.relative_to(ROOT))
        for path in (ROOT/'reports').glob('FIVE_TASK_10PCT_RESULTS_20261029*'):z.write(path,path.relative_to(ROOT))
    report.update(status='completed',archive={'path':str(archive),'bytes':archive.stat().st_size,'sha256':digest(archive)})
except BaseException as error:
    report.update(status='failed',error=f'{type(error).__name__}: {error}');raise
finally:save()
