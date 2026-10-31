"""Paired normalized-random vs ImageNet initialization; isolated queued GPU run."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import zipfile
import numpy as np

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main(root,allow_concurrent=False):
    import fcntl
    out=root/'outputs/pretrained_vision_20261029';p=read(root/'records/pretrained_vision_protocol_20261029.json')
    lock=(out/'workflow.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'workflow.json').exists():
        prior=read(out/'workflow.json')
        assert allow_concurrent and prior['status']=='waiting_for_sampling' and not prior['stages'] and not (out/'freeze.json').exists(), 'Only resume an untouched waiting scheduler'
        (out/'workflow.json').rename(out/f'workflow_waiting_before_parallel_{int(time.time())}.json')
    state={'status':'waiting_for_sampling','started_utc':datetime.now(timezone.utc).isoformat(),'stages':[],'server_shutdown':False}
    def save():
        tmp=out/'workflow.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(out/'workflow.json')
    state['execution_scheduling']='concurrent_gpu_trial' if allow_concurrent else 'sequential'
    save();deadline=time.monotonic()+6*3600
    while not allow_concurrent:
        previous=read(root/'outputs/task_balanced_20261029/workflow.json')
        if previous['status']=='completed':break
        if previous['status']=='failed':state.update(status='blocked',reason='sampling workflow failed;no automatic GPU launch');save();return
        if time.monotonic()>deadline:state.update(status='expired_wait');save();return
        time.sleep(20)
    state.update(status='running');save();started=time.monotonic()
    data=root/'outputs/five_task_10pct_20261029/dataset';old=root/'outputs/five_task_10pct_20261029/experiments'
    torch=root/'.venv-torch-linux/bin/python';native=[root/'.venv-jittor-linux/bin/python',root/'scripts/run_jittor_linux.py','--device','cuda']
    env=os.environ.copy();env.update(PYTHONPATH=str(root/'src'),NVIDIA_TF32_OVERRIDE='0',OMP_NUM_THREADS='4',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl')
    files=[root/'records/pretrained_vision_protocol_20261029.json',Path(__file__).resolve(),root/'scripts/run_imagenet_normalized.py',
           root/'scripts/train_vision_policy.py',root/'scripts/train_sampling_policy.py',out/'assets/imagenet_initial.npz',out/'assets/manifest.json',
           old/'initial_five.npz',data/'assets/weights.npz',data/'assets/shared/tokenizer_shared.pt',root/'outputs/task_balanced_20261029/batch_plan.npz']
    files+=list((root/'src/quest_repro').glob('*.py'))
    frozen={f.relative_to(root).as_posix():sha(f) for f in files};(out/'freeze.json').write_text(json.dumps(frozen,indent=2))
    def stage(name,cmd,cap=3600):
        remaining=4*3600-(time.monotonic()-started)
        if remaining<=0:raise TimeoutError('Four-hour experiment bound;server stays on')
        row={'name':name,'status':'running','command':[str(x) for x in cmd]};state['stages'].append(row);save()
        with (out/f'{name}.log').open('xb') as log:
            child=subprocess.Popen(row['command'],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            row['pid']=child.pid;save()
            try:code=child.wait(timeout=min(cap,remaining))
            except BaseException:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise
        row.update(exit_code=code,status='passed' if code==0 else 'failed');save()
        if code:raise RuntimeError(f'{name} failed;preserved logs')
    def command(framework,script):
        return ([torch] if framework=='torch' else list(native))+[root/'scripts/run_imagenet_normalized.py','--framework',framework,root/'scripts'/script]
    def fixture(framework,weights,config,dest):
        return command(framework,'export_trained_policy_predictions.py')+['--root',data,'--framework',framework,'--policy-weights',weights,
            '--policy-config',config,'--tokenizer-weights',data/'assets/weights.npz','--tokenizer-config',data/'assets/config.json','--output',dest]
    def aligned(a,b):
        with np.load(a) as x,np.load(b) as y:
            assert set(x.files)==set(y.files)
            for k in x.files:np.testing.assert_allclose(x[k],y[k],atol=2e-5,rtol=2e-4)
            np.testing.assert_array_equal(x['tokens'],y['tokens'])
    try:
        memory=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip());assert memory<(8192 if allow_concurrent else 512)
        assert sha(data/read(data/'records/reach_split_v1.json')['dataset'])==p['dataset_sha256']
        assert sha(out/'assets/imagenet_initial.npz')==p['pretrained_initial_sha256']
        assert sha(root/'outputs/task_balanced_20261029/batch_plan.npz')==p['plan_sha256']
        initial_config=read(old/'policy_jittor/config.json');initial_config['normalization']='RGB/255 then ImageNet mean/std;normalized entry point REQUIRED'
        (out/'initial_config.json').write_text(json.dumps(initial_config,indent=2))
        for f in ('torch','jittor'):
            stage('initial_fixture_'+f,fixture(f,out/'assets/imagenet_initial.npz',out/'initial_config.json',out/f'initial_fixture_{f}.npz'),300)
        aligned(out/'initial_fixture_torch.npz',out/'initial_fixture_jittor.npz')
        for condition in ('random_normalized','imagenet_finetune'):
            initial=old/'initial_five.npz' if condition=='random_normalized' else out/'assets/imagenet_initial.npz'
            for smoke in (True,False):
                for f in ('torch','jittor'):
                    name=('smoke_' if smoke else '')+condition+'_'+f
                    cmd=command(f,'train_vision_policy.py')+['--root',data,'--output',out/name,'--framework',f,
                        '--tokenizer-weights',data/'assets/weights.npz','--tokenizer-config',data/'assets/config.json','--initial-weights',initial,
                        '--canonical-labels',data/'assets/canonical_labels','--batch-plan',root/'outputs/task_balanced_20261029/batch_plan.npz',
                        '--updates',str(4 if smoke else 12000),'--eval-every',str(4 if smoke else 1000),
                        '--lr','0.0001','--weight-decay','0.0001','--dropout','0','--shift','0','--seed','310500']
                    if f=='torch':cmd+=['--tokenizer-checkpoint',data/'assets/shared/tokenizer_shared.pt']
                    stage(name,cmd)
            for f in ('torch','jittor'):
                stage('fixture_'+condition+'_'+f,fixture(f,out/(condition+'_jittor')/'best_weights.npz',out/(condition+'_jittor')/'config.json',out/f'fixture_{condition}_{f}.npz'),300)
            aligned(out/f'fixture_{condition}_torch.npz',out/f'fixture_{condition}_jittor.npz')
        selected={c+'_'+f:sha(out/(c+'_'+f)/'best_weights.npz') for c in p['conditions'] for f in ('torch','jittor')}
        (out/'evaluation_freeze.json').write_text(json.dumps(selected,indent=2))
        results={}
        for condition in p['conditions']:
            results[condition]={}
            for f in ('torch','jittor'):
                model=out/(condition+'_'+f);cases=[]
                for task_id,task in enumerate(p['tasks']):
                    name=f'vision_{condition}_{f}_{task}_20261029'
                    if f=='torch':cmd=command(f,'evaluate_policy.py')+['--policy',model/'best.pt','--tokenizer',data/'assets/shared/tokenizer_shared.pt','--sampling-backend','numpy']
                    else:cmd=command(f,'evaluate_policy_jittor.py')+['--policy-weights',model/'best_weights.npz','--policy-config',model/'config.json','--tokenizer-weights',data/'assets/weights.npz','--tokenizer-config',data/'assets/config.json']
                    stage(name,cmd+['--root',data,'--device','cuda','--seed-set','experiment','--seed-start','14000','--episodes','20',
                        '--env-name',task+'-goal-observable','--task-id',str(task_id),'--per-episode-sampling','--top-k','5','--sampling-seed','30000',
                        '--max-steps','150','--action-horizon','8','--videos','0','--name',name],900)
                    s=read(data/'outputs'/name/'summary.json');assert len(s['episodes'])==20
                    assert [v['seed'] for v in s['episodes']]==list(range(14000,14020))
                    if f=='jittor':assert not s['torch_imported']
                    cases.append([int(v['success']) for v in s['episodes']])
                results[condition][f]={'outcomes':cases,'successes':int(np.sum(cases)),'episodes':100}
        for relative,digest in frozen.items():assert sha(root/relative)==digest,relative
        for name,digest in selected.items():assert sha(out/name/'best_weights.npz')==digest
        audits={}
        for condition in p['conditions']:
            for f in ('torch','jittor'):
                folder=out/(condition+'_'+f);s=read(folder/'summary.json');cfg=read(folder/'config.json')
                assert s['status']=='completed' and s['updates']==12000 and not s['nonfinite_loss_or_gradient']
                assert s['batch_stream_sha256']==p['batch_stream_sha256']
                gradients=[json.loads(v) for v in (folder/'gradients.jsonl').read_text().splitlines()]
                assert [v['step'] for v in gradients]==list(range(1,12001)) and np.isfinite([v['preclip_norm'] for v in gradients]).all()
                assert cfg['normalization'].startswith('RGB/255 then ImageNet')
                if f=='jittor':assert not s['torch_imported']
                audits[condition+'_'+f]={'updates':s['updates'],'best_step':s['best_step'],'stream_sha':s['batch_stream_sha256']}
        effects={};rng=np.random.default_rng(323000)
        for f in ('torch','jittor'):
            delta=np.array(results['imagenet_finetune'][f]['outcomes'])-np.array(results['random_normalized'][f]['outcomes'])
            draws=np.stack([delta[i,rng.integers(0,20,(5000,20))].mean(1) for i in range(5)]).mean(0)
            effects[f]={'difference':float(delta.mean()),'paired_task_stratified_bootstrap95':np.percentile(draws,[2.5,97.5]).tolist()}
        report={'status':'audited','protocol':p,'training_audit':audits,'results':results,'pretraining_effect':effects,
            'limitations':['one training seed','development scenes previously used;not independent final benchmark','partial ResNet18 only;128x128 no center crop','all backbone parameters fine-tuned;not frozen features']}
        target=root/'reports/PRETRAINED_VISION_RESULTS_20261029.json';target.write_text(json.dumps(report,indent=2))
        archive=root/'reports/pretrained_vision_20261029.zip'
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
            for path in out.rglob('*'):
                if path.is_file() and path.name not in ('last.pt','last_weights.npz','workflow.lock','workflow.tmp'):z.write(path,path.relative_to(root))
            for folder in (data/'outputs').glob('vision_*20261029'):
                for path in folder.rglob('*'):
                    if path.is_file():z.write(path,path.relative_to(root))
            for path in files[:5]+[target,root/'scripts/prepare_pretrained_vision.py']:z.write(path,path.relative_to(root))
        state.update(status='completed',archive={'path':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size})
    except BaseException as e:state.update(status='failed',error=f'{type(e).__name__}: {e}');raise
    finally:state['elapsed_seconds']=time.monotonic()-started;save()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--allow-concurrent',action='store_true');a=p.parse_args();main(a.root.resolve(),a.allow_concurrent)
