"""Complete all4 nativeJittor factorial replications after mainGPU workflow."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];NIGHT=ROOT/'outputs/overnight_20261029';OUT=NIGHT/'experiments'
STATE=OUT/'native_factorial_workflow.json';BASE=OUT/'native_factorial';BASE.mkdir(exist_ok=True)
if STATE.exists():
    report=json.loads(STATE.read_text())
    if report['status']!='failed':raise RuntimeError('Refuse duplicate native extension')
    report.update(status='waiting_main');report.pop('error',None)
else:report={'status':'waiting_main','registered_utc':datetime.now(timezone.utc).isoformat(),'stages':[],
    'extension':json.loads((ROOT/'records/overnight_native_factorial_extension_20261029.json').read_text())}
def save():STATE.write_text(json.dumps(report,indent=2))
save();started=time.monotonic()
env=os.environ.copy();env.update(PYTHONPATH=str(ROOT/'src'),NVIDIA_TF32_OVERRIDE='0',OMP_NUM_THREADS='4',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl')
JT=ROOT/'.venv-jittor-linux/bin/python';NATIVE=[JT,'-u',ROOT/'scripts/run_jittor_linux.py','--device','cuda']
def stage(name,command):
    if any(s['name']==name and s['status']=='passed' for s in report['stages']):return
    r={'name':name,'status':'running','command':[str(c) for c in command]};report['stages'].append(r);save()
    path=BASE/f'{name}.log'
    if path.exists():shutil.copy2(path,BASE/f'{name}_previous_{int(time.time())}.log')
    with path.open('wb') as f:
        p=subprocess.Popen(r['command'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=p.wait(timeout=3600)
        except BaseException:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            r['status']='interrupted';save();raise
    r.update(status='passed' if code==0 else 'failed',exit_code=code);save()
    if code:raise RuntimeError(f'{name} failed,see{path}')
try:
    while True:
        main=json.loads((OUT/'workflow.json').read_text())
        metadata=ROOT/'reports/overnight_archive_20261029.json'
        if main['status']=='completed' and metadata.exists():
            archive_info=json.loads(metadata.read_text())
            if (ROOT/'reports/overnight_experiments_20261029.zip').stat().st_size==archive_info['bytes']:break
        if main['status']=='failed':raise RuntimeError('Main workflow failed;repair it before continuing extension')
        if subprocess.run(['tmux','has-session','-t','quest_overnight_experiments'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:
            # Confirm state after checking handle to avoid a completion race.
            if json.loads((OUT/'workflow.json').read_text())['status']!='completed':raise RuntimeError('Main handle missing while state nonterminal;inspect before resuming')
        if time.monotonic()-started>18000:raise TimeoutError('Main wait exceeded5hours')
        time.sleep(20)
    report['status']='running';selected=main['selected_single_case_by_validation'];protocol=main['protocol'];save()
    models={selected:OUT/'selected_native'}
    results={selected:{'model':str(models[selected]),'evaluation':main['single_results']['selected_native']}}
    for name,setting in protocol['four_cases'].items():
        if name==selected:continue
        data=NIGHT/'datasets'/setting['data'];assets=data/'assets';model=BASE/name
        stage('train_'+name,NATIVE+[ROOT/'scripts/train_controlled_policy.py','--framework','jittor','--root',data,'--output',model,
            '--tokenizer-weights',assets/'weights.npz','--tokenizer-config',assets/'config.json','--initial-weights',OUT/'common_initial_policy.npz',
            '--canonical-labels',assets/'canonical_labels','--updates','4000','--eval-every','400','--lr',str(setting['lr']),
            '--dropout',str(setting['dropout']),'--weight-decay',str(setting['weight_decay']),'--shift',str(setting['shift'])])
        assert json.loads((model/'summary.json').read_text())['batch_stream_sha256']==json.loads((OUT/name/'summary.json').read_text())['batch_stream_sha256']
        models[name]=model
    import hashlib
    frozen={'utc':datetime.now(timezone.utc).isoformat(),'cases':{name:hashlib.sha256((model/'best_weights.npz').read_bytes()).hexdigest() for name,model in models.items()},
        'settings_fixed_before_tests':True,'selected_case_reuses_existing_result':selected}
    if not (BASE/'test_freeze.json').exists():(BASE/'test_freeze.json').write_text(json.dumps(frozen,indent=2))
    for name,setting in protocol['four_cases'].items():
        if name==selected:continue
        data=NIGHT/'datasets'/setting['data'];assets=data/'assets';model=models[name];evalname='overnight_native_'+name+'_test50'
        stage('test_'+name,NATIVE+[ROOT/'scripts/evaluate_policy_jittor.py','--root',data,'--policy-weights',model/'best_weights.npz',
            '--policy-config',model/'config.json','--tokenizer-weights',assets/'weights.npz','--tokenizer-config',assets/'config.json',
            '--device','cuda','--seed-set','experiment','--seed-start','8000','--episodes','50','--per-episode-sampling',
            '--top-k','5','--sampling-seed','30000','--action-horizon','8','--max-steps','150','--videos','0','--name',evalname])
        results[name]={'model':str(model),'evaluation':json.loads((data/'outputs'/evalname/'summary.json').read_text())}
    for name,model in models.items():assert hashlib.sha256((model/'best_weights.npz').read_bytes()).hexdigest()==frozen['cases'][name]
    report.update(status='completed',results=results,elapsed_seconds=time.monotonic()-started);save()
    print('NATIVE_FACTORIAL',{name:sum(ep['success'] for ep in row['evaluation']['episodes']) for name,row in results.items()},flush=True)
except BaseException as error:
    report.update(status='failed',error=f'{type(error).__name__}:{error}');save();raise
