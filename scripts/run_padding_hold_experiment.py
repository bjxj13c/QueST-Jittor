"""Last scoped Reach study:padding reconstruction mask and real holding demos."""
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
import h5py

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main(root):
    import fcntl
    out=root/'outputs/padding_hold_20261029';p=read(root/'records/padding_hold_protocol_20261029.json')
    lock=(out/'workflow.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (out/'workflow.json').exists(),'No duplicate/restart'
    state={'status':'waiting_for_collection','stages':[],'started_utc':datetime.now(timezone.utc).isoformat(),'server_shutdown':False}
    def save():
        q=out/'workflow.tmp';q.write_text(json.dumps(state,indent=2));q.replace(out/'workflow.json')
    save();deadline=time.monotonic()+1800
    while not (out/'collection/collection_summary.json').exists():
        if time.monotonic()>deadline:state.update(status='failed',error='Collection wait timed out;inspect collection logs');save();return
        time.sleep(10)
    state['status']='running';save();started=time.monotonic();torch=root/'.venv-torch-linux/bin/python';native=[root/'.venv-jittor-linux/bin/python',root/'scripts/run_jittor_linux.py','--device','cuda']
    old=root/'outputs/five_task_10pct_20261029/dataset/assets';env=os.environ.copy();env.update(PYTHONPATH=str(root/'src'),NVIDIA_TF32_OVERRIDE='0',OMP_NUM_THREADS='4',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl')
    files=[Path(__file__).resolve(),root/'records/padding_hold_protocol_20261029.json',old/'weights.npz',old/'config.json']
    files+=[root/'scripts'/n for n in ['collect_padding_hold.py','train_padding_tokenizer.py','prepare_padding_assets.py','evaluate_reach_holding.py','train_controlled_policy.py']]
    files+=list((root/'src/quest_repro').glob('*.py'))
    for c in ('repeat','holding'):files+=[out/'collection'/c/'data/reach.hdf5',out/'collection'/c/'records/reach_split_v1.json']
    frozen={f.relative_to(root).as_posix():sha(f) for f in files};(out/'freeze.json').write_text(json.dumps(frozen,indent=2))
    def stage(name,cmd,cap=3600):
        remaining=5*3600-(time.monotonic()-started)
        if remaining<=0:raise TimeoutError('Five-hour workflow bound;server stays on')
        row={'name':name,'status':'running','command':[str(x) for x in cmd]};state['stages'].append(row);save()
        with (out/(name+'.log')).open('xb') as log:
            child=subprocess.Popen(row['command'],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);row['pid']=child.pid;save()
            try:code=child.wait(timeout=min(cap,remaining))
            except BaseException:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise
        row.update(status='passed' if code==0 else 'failed',exit_code=code);save()
        if code:raise RuntimeError(f'{name} failed;logs preserved')
    def prefix(f):return [torch] if f=='torch' else list(native)
    try:
        assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<512
        collection=read(out/'collection/collection_summary.json');assert len(collection['episodes'])==110 and collection['paired_prefixes_identical']
        stage('initial_policy',[torch,root/'scripts/prepare_padding_assets.py','--root',root,'--condition','initial'],120)
        for c in p['conditions']:
            data=out/'collection'/('holding' if c=='holding' else 'repeat');folder=out/c;folder.mkdir(exist_ok=False)
            for smoke in (True,False):
                for f in ('torch','jittor'):
                    name=('smoke_' if smoke else '')+f'tokenizer_{f}';cmd=prefix(f)+[root/'scripts/train_padding_tokenizer.py','--root',data,'--output',folder/name,'--framework',f,'--initial',old/'weights.npz','--config',old/'config.json','--updates',str(4 if smoke else 2000)]
                    if c=='masked':cmd+=['--masked']
                    stage(c+'_'+name,cmd,1800)
                if smoke:
                    a=read(folder/'smoke_tokenizer_torch/summary.json');b=read(folder/'smoke_tokenizer_jittor/summary.json');assert a['batch_stream_sha256']==b['batch_stream_sha256']
                    x=read_first(folder/'smoke_tokenizer_torch/gradients.jsonl');y=read_first(folder/'smoke_tokenizer_jittor/gradients.jsonl')
                    np.testing.assert_allclose(x['loss'],y['loss'],atol=1e-4,rtol=1e-3);np.testing.assert_allclose(x['gradient_norm'],y['gradient_norm'],atol=1e-3,rtol=1e-3)
            stage(c+'_assets',[torch,root/'scripts/prepare_padding_assets.py','--root',root,'--condition',c],300)
            assets=folder/'assets'
            for f in ('torch','jittor'):
                cmd=prefix(f)+[root/'scripts/train_controlled_policy.py','--root',data,'--output',folder/('policy_'+f),'--framework',f,
                    '--tokenizer-weights',assets/'weights.npz','--tokenizer-config',assets/'config.json','--initial-weights',out/'initial_policy.npz',
                    '--canonical-labels',assets/'canonical_labels','--updates','6000','--eval-every','1000','--lr','0.0001','--weight-decay','0.0001','--dropout','0','--shift','0','--seed','510500']
                if f=='torch':cmd+=['--tokenizer-checkpoint',assets/'tokenizer_shared.pt']
                stage(c+'_policy_'+f,cmd)
            for f in ('torch','jittor'):
                stage(c+'_fixture_'+f,prefix(f)+[root/'scripts/export_trained_policy_predictions.py','--root',data,'--framework',f,
                    '--policy-weights',folder/'policy_jittor/best_weights.npz','--policy-config',folder/'policy_jittor/config.json',
                    '--tokenizer-weights',assets/'weights.npz','--tokenizer-config',assets/'config.json','--output',folder/f'fixture_{f}.npz'],300)
            with np.load(folder/'fixture_torch.npz') as a,np.load(folder/'fixture_jittor.npz') as b:
                for k in a.files:np.testing.assert_allclose(a[k],b[k],atol=2e-5,rtol=2e-4)
                np.testing.assert_array_equal(a['tokens'],b['tokens'])
        selected={}
        for c in p['conditions']:
            for f in ('torch','jittor'):
                q=out/c/('policy_'+f)/'best_weights.npz';selected[q.relative_to(root).as_posix()]=sha(q)
        (out/'evaluation_freeze.json').write_text(json.dumps(selected,indent=2))
        results={}
        for c in p['conditions']:
            data=out/'collection'/('holding' if c=='holding' else 'repeat');folder=out/c;assets=folder/'assets';results[c]={}
            for f in ('torch','jittor'):
                model=folder/('policy_'+f);name=f'padding_{c}_{f}_standard_20261029'
                if f=='torch':cmd=[torch,root/'scripts/evaluate_policy.py','--policy',model/'best.pt','--tokenizer',assets/'tokenizer_shared.pt','--sampling-backend','numpy']
                else:cmd=list(native)+[root/'scripts/evaluate_policy_jittor.py','--policy-weights',model/'best_weights.npz','--policy-config',model/'config.json','--tokenizer-weights',assets/'weights.npz','--tokenizer-config',assets/'config.json']
                stage(name,cmd+['--root',data,'--device','cuda','--seed-set','experiment','--seed-start','80000','--episodes','30','--env-name','reach-v2-goal-observable','--task-id','0','--per-episode-sampling','--top-k','5','--sampling-seed','90000','--max-steps','150','--action-horizon','8','--videos','1','--name',name],900)
                stage(c+'_holding_diagnostic_'+f,prefix(f)+[root/'scripts/evaluate_reach_holding.py','--framework',f,'--policy-dir',model,'--assets',assets,'--output',folder/('holding_diagnostic_'+f)],1200)
                standard=read(data/'outputs'/name/'summary.json');diagnostic=read(folder/('holding_diagnostic_'+f)/'summary.json')
                assert [e['seed'] for e in standard['episodes']]==list(range(80000,80030))
                assert [e['success'] for e in standard['episodes']]==[e['success_within150'] for e in diagnostic['episodes']],'Standard and continuation must agree before first success'
                if f=='jittor':assert not diagnostic['torch_imported'] and not standard['torch_imported']
                reached=[e for e in diagnostic['episodes'] if e['success_within150']];complete=[e for e in reached if e['post_steps']==60]
                results[c][f]={'standard_successes':sum(e['success'] for e in standard['episodes']),'episodes':30,'hold_all60_unconditional':diagnostic['hold_all60_unconditional'],
                    'successes_with_full60_followup':len(complete),'mean_post_success_fraction_conditional':float(np.mean([e['post_success_fraction'] for e in complete])) if complete else None,
                    'mean_tcp_drift60_conditional':float(np.mean([e['tcp_drift_60'] for e in complete])) if complete else None}
        for path,digest in frozen.items():assert sha(root/path)==digest,path
        for path,digest in selected.items():assert sha(root/path)==digest,path
        for c in p['conditions']:
            folder=out/c;streams=[]
            for f in ('torch','jittor'):
                t=read(folder/f'tokenizer_{f}/summary.json');s=read(folder/f'policy_{f}/summary.json');assert t['updates']==2000 and t['finite'] and s['updates']==6000 and not s['nonfinite_loss_or_gradient'];streams.append(t['batch_stream_sha256'])
                if f=='jittor':assert not t['torch_imported'] and not s['torch_imported']
            assert len(set(streams))==1
        report={'status':'audited','protocol':p,'results':results,'limitations':['single training seed and single Reach task','new paired dataset;not directly comparable to five-task scores','mask removes reconstruction penalty on padding but encoder and token labels still contain padded blocks','holding changes training data length/distribution despite same updates','conditional holding metrics involve different successful scenes;also report unconditional hold rate','not proof of causal explanation for earlier drift']}
        target=root/'reports/PADDING_HOLD_RESULTS_20261029.json';target.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
        archive=root/'reports/padding_hold_20261029.zip'
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
            for q in out.rglob('*'):
                if q.is_file() and q.name not in ('last.pt','last_weights.npz','workflow.tmp','workflow.lock') and not str(q.relative_to(out)).startswith('smoke_collection'):z.write(q,q.relative_to(root))
            for q in files[:2]+[target]+[root/'scripts'/n for n in ['collect_padding_hold.py','train_padding_tokenizer.py','prepare_padding_assets.py','evaluate_reach_holding.py']]:z.write(q,q.relative_to(root))
        state.update(status='completed',archive={'path':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive)})
    except BaseException as e:state.update(status='failed',error=f'{type(e).__name__}: {e}');raise
    finally:state['elapsed_seconds']=time.monotonic()-started;save()

def read_first(path):return json.loads(path.read_text().splitlines()[0])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args();main(a.root.resolve())
