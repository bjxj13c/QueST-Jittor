"""Bounded automatic2x2 data/tuning experiments and3task Torch/Jittor run."""
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

ROOT=Path(__file__).resolve().parents[1];NIGHT=ROOT/'outputs/overnight_20261029';OUT=NIGHT/'experiments'
resuming=OUT.exists()
if not resuming:OUT.mkdir()
protocol=json.loads((ROOT/'records/overnight_protocol_20261029.json').read_text())
report={'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),'protocol':protocol,'stages':[],'server_shutdown':False}
started=time.monotonic()
if resuming:
    report=json.loads((OUT/'workflow.json').read_text())
    if report['status']!='failed':raise RuntimeError('Refuse duplicate running/completed workflow')
    shutil.copy2(OUT/'workflow.json',OUT/('failed_'+datetime.now(timezone.utc).strftime('%H%M%S')+'.json'))
    started-=report.get('elapsed_seconds',0);report.update(status='running');report.pop('error',None)
TORCH=ROOT/'.venv-torch-linux/bin/python';JT=ROOT/'.venv-jittor-linux/bin/python'
NATIVE=[JT,'-u',ROOT/'scripts/run_jittor_linux.py','--device','cuda']
env=os.environ.copy();env.update(PYTHONPATH=str(ROOT/'src'),NVIDIA_TF32_OVERRIDE='0',OMP_NUM_THREADS='4',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl')
def save():
    report['elapsed_seconds']=time.monotonic()-started
    tmp=OUT/'workflow.tmp';tmp.write_text(json.dumps(report,indent=2));tmp.replace(OUT/'workflow.json')
def stage(name,command,cap=3600):
    if any(s['name']==name and s['status']=='passed' for s in report['stages']):return
    r={'name':name,'status':'running','command':[str(x) for x in command]};report['stages'].append(r);save()
    log=OUT/f'{name}.log'
    if log.exists():shutil.copy2(log,OUT/f'{name}_before_retry_{int(time.time())}.log')
    tick=time.monotonic()
    with log.open('wb') as f:
        process=subprocess.Popen(r['command'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=process.wait(timeout=min(cap,max(1,21000-(time.monotonic()-started))))
        except BaseException:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            r['status']='interrupted';save();raise
    r.update(status='passed' if code==0 else 'failed',exit_code=code,elapsed_seconds=time.monotonic()-tick);save()
    if code:raise RuntimeError(f'{name} failed;preserved log:{log}')
def copy_same(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():assert target.read_bytes()==source.read_bytes()
    else:shutil.copy2(source,target)
def setup_assets(label,token_folder):
    data=NIGHT/'datasets'/label;assets=data/'assets';assets.mkdir(exist_ok=True)
    copy_same(token_folder/'best_weights.npz',assets/'weights.npz');copy_same(token_folder/'config.json',assets/'config.json')
    stage('bridge_'+label,[TORCH,ROOT/'scripts/prepare_policy_stage.py','--root',data,'--framework','torch','--prepare-checkpoint',
        '--tokenizer-weights',assets/'weights.npz','--tokenizer-config',assets/'config.json','--output',assets/'shared'],120)
    stage('labels_'+label,NATIVE+[ROOT/'scripts/prepare_policy_stage.py','--root',data,'--framework','jittor',
        '--tokenizer-weights',assets/'weights.npz','--tokenizer-config',assets/'config.json','--output',assets/'canonical_labels'],600)
    return data,assets
def train(name,framework,data,assets,initial,settings,updates=4000,every=400):
    destination=OUT/name
    cmd=([TORCH,'-u'] if framework=='torch' else NATIVE)+[ROOT/'scripts/train_controlled_policy.py','--root',data,'--output',destination,
        '--framework',framework,'--tokenizer-weights',assets/'weights.npz','--tokenizer-config',assets/'config.json',
        '--initial-weights',initial,'--canonical-labels',assets/'canonical_labels','--updates',str(updates),'--eval-every',str(every),
        '--lr',str(settings['lr']),'--dropout',str(settings['dropout']),'--weight-decay',str(settings['weight_decay']),'--shift',str(settings['shift'])]
    if framework=='torch':cmd+=['--tokenizer-checkpoint',assets/'shared/tokenizer_shared.pt']
    stage(name,cmd)
    return destination
def evaluate(name,framework,data,assets,model,seed_start,episodes,task='reach-v2',task_id=0):
    if framework=='torch':
        cmd=[TORCH,ROOT/'scripts/evaluate_policy.py','--policy',model/'best.pt','--tokenizer',assets/'shared/tokenizer_shared.pt','--sampling-backend','numpy']
    else:
        cmd=NATIVE+[ROOT/'scripts/evaluate_policy_jittor.py','--policy-weights',model/'best_weights.npz','--policy-config',model/'config.json',
            '--tokenizer-weights',assets/'weights.npz','--tokenizer-config',assets/'config.json']
    stage(name,cmd+['--root',data,'--device','cuda','--seed-set','experiment','--seed-start',str(seed_start),'--episodes',str(episodes),
        '--env-name',task+'-goal-observable','--task-id',str(task_id),'--per-episode-sampling','--top-k','5','--sampling-seed','30000',
        '--action-horizon','8','--max-steps','150','--videos','0','--name',name],600)
    return json.loads((data/'outputs'/name/'summary.json').read_text())
def compare_weights(label,data,assets,model):
    for name,command in [('torch',[TORCH]),('jittor',NATIVE)]:
        stage(label+'_fixture_'+name,command+[ROOT/'scripts/export_trained_policy_predictions.py','--root',data,'--framework',name,
            '--policy-weights',model/'best_weights.npz','--policy-config',model/'config.json','--tokenizer-weights',assets/'weights.npz',
            '--tokenizer-config',assets/'config.json','--output',OUT/f'{label}_fixture_{name}.npz'],300)
    with np.load(OUT/f'{label}_fixture_torch.npz') as a,np.load(OUT/f'{label}_fixture_jittor.npz') as b:
        comparison={k:{'max_abs':float(np.abs(a[k]-b[k]).max()),'allclose':bool(np.allclose(a[k],b[k],atol=2e-5,rtol=2e-4))} for k in a.files}
        comparison['tokens_exact']=bool(np.array_equal(a['tokens'],b['tokens']))
    (OUT/f'{label}_same_weights.json').write_text(json.dumps(comparison,indent=2));return comparison

try:
    assert json.loads((NIGHT/'data_preparation_completed.json').read_text())['status']=='completed'
    old_token=ROOT/'outputs/gpu_20261028_104533_tokenizer_jittor';initial=ROOT/'outputs/fixtures/policy_full_seed456/weights.npz'
    roots={label:setup_assets(label,old_token) for label in ['reach80','reach280']}
    base=protocol['four_cases']['small_base'];d,a=roots['reach80']
    for framework in ['torch','jittor']:train('smoke_'+framework,framework,d,a,initial,base,updates=4,every=4)
    for name,settings in protocol['four_cases'].items():
        d,a=roots[settings['data']];train(name,'torch',d,a,initial,settings)
    selected=min(protocol['four_cases'],key=lambda name:json.loads((OUT/name/'summary.json').read_text())['best_validation_ce'])
    report['selected_single_case_by_validation']=selected;save()
    setting=protocol['four_cases'][selected];d,a=roots[setting['data']]
    native=train('selected_native','jittor',d,a,initial,setting)
    assert json.loads((native/'summary.json').read_text())['batch_stream_sha256']==json.loads((OUT/selected/'summary.json').read_text())['batch_stream_sha256']
    report['single_same_weights']=compare_weights('single',d,a,native);save()
    # Freeze every candidate checkpoint before evaluating any new test scene.
    freeze={'utc':datetime.now(timezone.utc).isoformat(),'selected_by_validation':selected,'test_seeds':list(range(8000,8050)),
        'candidates':{name:hashlib.sha256((OUT/name/'best_weights.npz').read_bytes()).hexdigest() for name in protocol['four_cases']},
        'native_sha256':hashlib.sha256((native/'best_weights.npz').read_bytes()).hexdigest(),
        'commitment':'no post-test changes or selection based on8000..8049;all four factorial outcomes reported'}
    frozen=OUT/'single_test_freeze.json'
    if not frozen.exists():frozen.write_text(json.dumps(freeze,indent=2))
    report['single_results']={}
    for name,settings in protocol['four_cases'].items():
        d,a=roots[settings['data']];report['single_results'][name]=evaluate('test_'+name,'torch',d,a,OUT/name,8000,50)
    d,a=roots[setting['data']];report['single_results']['selected_native']=evaluate('test_selected_native','jittor',d,a,native,8000,50);save()
    # Genuine3task first stage and task-conditioned policy,not three single-task models.
    multi=NIGHT/'datasets/multi240'
    stage('multi_tokenizer5',NATIVE+[ROOT/'scripts/train_tokenizer_jittor.py','--root',multi,'--name','overnight_multi_tokenizer5','--device','cuda',
        '--epochs','5','--batch-size','16','--dropout','0','--seed','304800','--initial-weights',old_token/'best_weights.npz'],1800)
    d,a=setup_assets('multi240',multi/'outputs/overnight_multi_tokenizer5')
    init_multi=OUT/'initial_multi3.npz'
    if not init_multi.exists():
        with np.load(initial) as original:values={k:original[k].copy() for k in original.files}
        key='task_encoder.weight';old=values[key];new=np.random.default_rng(304904).normal(size=(3,old.shape[1])).astype(old.dtype);new[0]=old[0];values[key]=new
        np.savez_compressed(init_multi,**values)
    for framework in ['torch','jittor']:train('multi_'+framework,framework,d,a,init_multi,base)
    report['multi_same_weights']=compare_weights('multi',d,a,OUT/'multi_jittor');save()
    report['multi_results']={}
    for framework in ['torch','jittor']:
        for task_id,task in enumerate(protocol['multi_task_tasks']):
            key=framework+'_'+task
            report['multi_results'][key]=evaluate('test_multi_'+key,framework,d,a,OUT/('multi_'+framework),9000,20,task,task_id)
    report['status']='completed';save()
    archive=ROOT/'reports/overnight_experiments_20261029.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for path in OUT.rglob('*'):
            if path.is_file() and path.name not in ['last.pt','last_weights.npz']:z.write(path,path.relative_to(ROOT))
        for label in ['reach80','reach280','multi240']:
            folder=NIGHT/'datasets'/label
            for sub in ['records','assets','outputs']:
                for path in (folder/sub).rglob('*'):
                    if path.is_file() and path.suffix not in ['.pkl'] and path.name not in ['last_weights.npz']:z.write(path,path.relative_to(ROOT))
            z.write(folder/'data_summary.json',(folder/'data_summary.json').relative_to(ROOT))
    (ROOT/'reports/overnight_archive_20261029.json').write_text(json.dumps({'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()},indent=2))
except BaseException as error:
    report.update(status='failed',error=f'{type(error).__name__}: {error}');raise
finally:save()
