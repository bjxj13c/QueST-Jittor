"""Three independent paired training runs;fresh frozen-scene stability check.

First run reuses checkpoints already selected by validation. Two additional
initializations retrain both sampling conditions in both native frameworks.
Training seeds,not individual rollouts,are the units of the primary interval.
"""
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
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def summarize(root):
    out=root/'outputs/sampling_stability_20261029';p=read(root/'records/sampling_stability_protocol_20261029.json')
    data=root/'outputs/five_task_10pct_20261029/dataset';results={};training={}
    for f in ('torch','jittor'):
        pairs=[];all_groups=[]
        for seed in p['training_seeds']:
            groups={}
            for condition in ('uniform','balanced'):
                episodes=[]
                for i,task in enumerate(p['tasks']):
                    q=data/'outputs'/f'stability_{seed}_{condition}_{f}_{task}_20261029'
                    s=read(q/'summary.json');assert s['horizon']==8 and s['top_k']==5 and s['task_id']==i and s['max_steps']==150
                    assert [v['seed'] for v in s['episodes']]==list(range(16000,16030))
                    if f=='jittor':assert not s['torch_imported']
                    episodes.append([int(v['success']) for v in s['episodes']])
                groups[condition]={'outcomes':episodes,'successes':int(np.sum(episodes)),'episodes':150,'rate':float(np.mean(episodes))}
            difference=groups['balanced']['rate']-groups['uniform']['rate'];pairs.append(difference)
            all_groups.append({'training_seed':seed,'conditions':groups,'paired_difference':difference})
        a=np.array(pairs);mean=float(a.mean());std=float(a.std(ddof=1));half=4.3026527299*std/np.sqrt(3)
        results[f]={'runs':all_groups,'mean_paired_difference':mean,'seed_difference_sample_std':std,
            'paired_seed_t95_interval':[max(-1.,mean-half),min(1.,mean+half)],'positive_runs':int((a>0).sum()),
            'mean_success_rates':{c:float(np.mean([v['conditions'][c]['rate'] for v in all_groups])) for c in ('uniform','balanced')},
            'interpretation':'positive across all3seeds and lower interval>0' if (a>0).all() and mean-half>0 else 'not enough evidence of stable improvement across training seeds'}
    frozen=read(out/'freeze.json')
    for path,digest in frozen.items():assert sha(root/path)==digest,path
    selected=read(out/'evaluation_freeze.json')
    for path,digest in selected.items():assert sha(root/path)==digest,path
    for seed in p['training_seeds'][1:]:
        streams={}
        for condition in ('uniform','balanced'):
            group=[]
            for f in ('torch','jittor'):
                folder=out/f'{seed}_{condition}_{f}';s=read(folder/'summary.json');c=read(folder/'config.json')
                assert s['status']=='completed' and s['updates']==12000 and not s['nonfinite_loss_or_gradient']
                assert c['dataset_sha256']==p['dataset_sha256'] and c['initial_policy_npz_sha256']==sha(out/f'initial_{seed}.npz')
                assert c['tokenizer_weights_sha256']==p['tokenizer_sha256'];group.append(s['batch_stream_sha256'])
                if condition=='balanced':assert c['sampling_plan_sha256']==sha(out/f'balanced_plan_{seed}.npz')
                gradients=[json.loads(v) for v in (folder/'gradients.jsonl').read_text().splitlines()]
                assert [v['step'] for v in gradients]==list(range(1,12001)) and np.isfinite([v['preclip_norm'] for v in gradients]).all()
                if f=='jittor':assert not s['torch_imported']
                training[f'{seed}_{condition}_{f}']={'updates':s['updates'],'best_step':s['best_step'],'gradient_norm_max':s['preclip_gradient_norm_max']}
            assert len(set(group))==1;streams[condition]=group[0]
            with np.load(out/f'fixture_{seed}_{condition}_torch.npz') as a,np.load(out/f'fixture_{seed}_{condition}_jittor.npz') as b:
                for k in a.files:np.testing.assert_allclose(a[k],b[k],atol=2e-5,rtol=2e-4)
                np.testing.assert_array_equal(a['tokens'],b['tokens'])
        assert streams['uniform']!=streams['balanced']
    report={'status':'audited','protocol':p,'results':results,'training_audit':training,
        'limitations':['only3trainingseeds;interval is approximate t interval with df2','tokenizer/data split fixed;not full end-to-end training variance',
        'prior run reused;two new paired initializations','only5tasks,not full45task paper benchmark','concurrent execution timing not standalone performance','evaluation scenes frozen;do not retune on these results']}
    dest=root/'reports/SAMPLING_STABILITY_RESULTS_20261029';dest.with_suffix('.json').write_text(json.dumps(report,indent=2))
    lines=['# 均衡采样：三随机种子稳定性检验','','固定数据和分词器，两组配对初始化及训练预算；新评估种子16000–16029，每任务30局。','',
        '| 框架 | 训练种子 | 原采样成功/150 | 均衡采样成功/150 | 差值 |','|---|---:|---:|---:|---:|']
    for f,r in results.items():
        for v in r['runs']:lines.append(f"| {f} | {v['training_seed']} | {v['conditions']['uniform']['successes']} | {v['conditions']['balanced']['successes']} | {100*v['paired_difference']:+.2f}pp |")
    lines+=['','以训练种子作为统计单位，报告配对差值的均值、样本标准差及df=2的近似t区间。不能把1800次rollout当作1800次独立训练。',json.dumps({f:{k:v for k,v in r.items() if k!='runs'} for f,r in results.items()},ensure_ascii=False,indent=2)]
    dest.with_suffix('.md').write_text('\n'.join(lines),encoding='utf-8');print(json.dumps(results,indent=2))

def run(root):
    import fcntl
    out=root/'outputs/sampling_stability_20261029';out.mkdir(parents=True,exist_ok=True)
    lock=(out/'workflow.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (out/'workflow.json').exists(),'Preserve prior state;no blind restart'
    p=read(root/'records/sampling_stability_protocol_20261029.json');data=root/'outputs/five_task_10pct_20261029/dataset';old=root/'outputs/five_task_10pct_20261029/experiments'
    state={'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),'stages':[],'server_shutdown':False};started=time.monotonic()
    def save():
        state['elapsed_seconds']=time.monotonic()-started;q=out/'workflow.tmp';q.write_text(json.dumps(state,indent=2));q.replace(out/'workflow.json')
    torch=root/'.venv-torch-linux/bin/python';native=[root/'.venv-jittor-linux/bin/python',root/'scripts/run_jittor_linux.py','--device','cuda']
    env=os.environ.copy();env.update(PYTHONPATH=str(root/'src'),NVIDIA_TF32_OVERRIDE='0',OMP_NUM_THREADS='4',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl')
    def stage(name,cmd,cap=3600):
        remaining=5*3600-(time.monotonic()-started)
        if remaining<=0:raise TimeoutError('Five-hour run limit;server remains ON')
        row={'name':name,'status':'running','command':[str(v) for v in cmd]};state['stages'].append(row);save()
        with (out/f'{name}.log').open('xb') as log:
            child=subprocess.Popen(row['command'],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);row['pid']=child.pid;save()
            try:code=child.wait(timeout=min(cap,remaining))
            except BaseException:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise
        row.update(status='passed' if code==0 else 'failed',exit_code=code);save()
        if code:raise RuntimeError(f'{name} failed;inspect retained logs')
    def modeldir(seed,condition,f):
        if seed==310500:return (old if condition=='uniform' else root/'outputs/task_balanced_20261029')/('policy_'+f)
        return out/f'{seed}_{condition}_{f}'
    try:
        save();assert read(root/'outputs/task_balanced_20261029/workflow.json')['status']=='completed'
        assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<8192
        assert sha(data/read(data/'records/reach_split_v1.json')['dataset'])==p['dataset_sha256']
        for previous in (data/'outputs').glob('*/summary.json'):
            seen=read(previous)
            events=seen.get('episodes',[])
            used={v.get('seed') for v in events if isinstance(v,dict)} if isinstance(events,list) else set()
            assert not used.intersection(range(16000,16030)),f'Fresh scene range already consumed: {previous}'
        (out/'fresh_scene_precheck.json').write_text(json.dumps({'range':[16000,16029],'not_in_prior_rollout_summaries':True}))
        files=[root/'records/sampling_stability_protocol_20261029.json',Path(__file__).resolve(),root/'scripts/train_sampling_policy.py',root/'scripts/train_controlled_policy.py',root/'scripts/prepare_task_balanced_plan.py',data/'assets/weights.npz']
        files+=list((root/'src/quest_repro').glob('*.py'))
        for c in ('uniform','balanced'):
            for f in ('torch','jittor'):files+=[modeldir(310500,c,f)/'best_weights.npz']
        (out/'freeze.json').write_text(json.dumps({f.relative_to(root).as_posix():sha(f) for f in files},indent=2))
        stage('prepare_initializations',[torch,root/'scripts/prepare_stability_initializations.py','--root',root],120)
        for seed in p['training_seeds'][1:]:
            plan=out/f'balanced_plan_{seed}.npz'
            stage(f'plan_{seed}',[torch,root/'scripts/prepare_task_balanced_plan.py','--root',data,'--output',plan,'--updates','12000','--seed',str(seed+10000)],120)
            for condition in ('uniform','balanced'):
                for f in ('torch','jittor'):
                    script='train_controlled_policy.py' if condition=='uniform' else 'train_sampling_policy.py'
                    cmd=([torch] if f=='torch' else list(native))+[root/'scripts'/script,'--root',data,'--output',modeldir(seed,condition,f),'--framework',f,
                        '--tokenizer-weights',data/'assets/weights.npz','--tokenizer-config',data/'assets/config.json','--initial-weights',out/f'initial_{seed}.npz',
                        '--canonical-labels',data/'assets/canonical_labels','--updates','12000','--eval-every','1000','--lr','0.0001',
                        '--weight-decay','0.0001','--dropout','0','--shift','0','--seed',str(seed)]
                    if condition=='balanced':cmd+=['--batch-plan',plan]
                    if f=='torch':cmd+=['--tokenizer-checkpoint',data/'assets/shared/tokenizer_shared.pt']
                    stage(f'train_{seed}_{condition}_{f}',cmd)
                for f in ('torch','jittor'):
                    model=modeldir(seed,condition,'jittor')
                    stage(f'fixture_{seed}_{condition}_{f}',([torch] if f=='torch' else list(native))+[root/'scripts/export_trained_policy_predictions.py',
                        '--root',data,'--framework',f,'--policy-weights',model/'best_weights.npz','--policy-config',model/'config.json',
                        '--tokenizer-weights',data/'assets/weights.npz','--tokenizer-config',data/'assets/config.json','--output',out/f'fixture_{seed}_{condition}_{f}.npz'],300)
                with np.load(out/f'fixture_{seed}_{condition}_torch.npz') as a,np.load(out/f'fixture_{seed}_{condition}_jittor.npz') as b:
                    for k in a.files:np.testing.assert_allclose(a[k],b[k],atol=2e-5,rtol=2e-4)
                    np.testing.assert_array_equal(a['tokens'],b['tokens'])
        selected={}
        for seed in p['training_seeds']:
            for c in ('uniform','balanced'):
                for f in ('torch','jittor'):
                    q=modeldir(seed,c,f)/'best_weights.npz';selected[q.relative_to(root).as_posix()]=sha(q)
        (out/'evaluation_freeze.json').write_text(json.dumps(selected,indent=2))
        for seed in p['training_seeds']:
            for condition in ('uniform','balanced'):
                for f in ('torch','jittor'):
                    model=modeldir(seed,condition,f)
                    for task_id,task in enumerate(p['tasks']):
                        name=f'stability_{seed}_{condition}_{f}_{task}_20261029'
                        if f=='torch':cmd=[torch,root/'scripts/evaluate_policy.py','--policy',model/'best.pt','--tokenizer',data/'assets/shared/tokenizer_shared.pt','--sampling-backend','numpy']
                        else:cmd=list(native)+[root/'scripts/evaluate_policy_jittor.py','--policy-weights',model/'best_weights.npz','--policy-config',model/'config.json','--tokenizer-weights',data/'assets/weights.npz','--tokenizer-config',data/'assets/config.json']
                        stage(name,cmd+['--root',data,'--device','cuda','--seed-set','experiment','--seed-start','16000','--episodes','30','--env-name',task+'-goal-observable',
                            '--task-id',str(task_id),'--per-episode-sampling','--top-k','5','--sampling-seed','50000','--max-steps','150','--action-horizon','8','--videos','0','--name',name],900)
        stage('summary_audit',[torch,Path(__file__).resolve(),'--root',root,'--phase','summarize'],240)
        archive=root/'reports/sampling_stability_20261029.zip'
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
            for q in out.rglob('*'):
                if q.is_file() and q.name not in ('last.pt','last_weights.npz','workflow.tmp','workflow.lock'):z.write(q,q.relative_to(root))
            for folder in (data/'outputs').glob('stability_*20261029'):
                for q in folder.rglob('*'):
                    if q.is_file():z.write(q,q.relative_to(root))
            for q in [Path(__file__).resolve(),root/'scripts/prepare_stability_initializations.py',root/'records/sampling_stability_protocol_20261029.json']+list((root/'reports').glob('SAMPLING_STABILITY_RESULTS_20261029.*')):z.write(q,q.relative_to(root))
        state.update(status='completed',archive={'path':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive)})
    except BaseException as e:state.update(status='failed',error=f'{type(e).__name__}: {e}');raise
    finally:save()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--phase',choices=['run','summarize'],default='run');a=p.parse_args();(run if a.phase=='run' else summarize)(a.root.resolve())
