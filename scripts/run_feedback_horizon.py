"""Frozen-policy4/8/16execution-interval development ablation,no training."""
import argparse
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
import h5py
import numpy as np

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def goal(value):return value.astype(np.float32).astype(np.float64).round(7).tobytes()
def paths(root):
    return root/'outputs/feedback_horizon_20261029',root/'outputs/five_task_10pct_20261029/dataset',root/'outputs/five_task_10pct_20261029/experiments'
def case_name(framework,horizon,task):return f'feedback_h{horizon}_{framework}_{task}_20261029'

def isolation(root,p):
    os.environ['MUJOCO_GL']='egl';os.environ['PYOPENGL_PLATFORM']='egl'
    from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
    out,data,_=paths(root);seen={t:set() for t in p['tasks']}
    roots=[root,root/'outputs/official_reach_100_20261028/adapted',
        root/'outputs/overnight_20261029/datasets/reach280',root/'outputs/overnight_20261029/datasets/multi240',data]
    for folder in roots:
        m=read(folder/'records/reach_split_v1.json');names=m.get('task_names',['reach-v2']);ids=m.get('episode_task_ids',{})
        with h5py.File(folder/m['dataset'],'r') as f:
            for ep in m['train']+m['validation']:
                task=names[ids.get(ep,0)]
                if task in seen:seen[task].add(goal(f[ep]['observations'][0][-3:]))
    rows=[]
    for task in p['tasks']:
        overlaps=[];unique=set()
        for seed in range(p['seed_start'],p['seed_start']+p['episodes_per_task']):
            env=ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task+'-goal-observable'](seed=seed)
            try:
                obs,_=env.reset();key=goal(obs[-3:]);unique.add(key)
                if key in seen[task]:overlaps.append(seed)
            finally:env.close()
        rows.append({'task':task,'known_goals':len(seen[task]),'development_unique_goals':len(unique),'overlap_seeds':overlaps})
    (out/'goal_isolation.json').write_text(json.dumps({'scope':'goal-only audit before any new policy outcomes','rows':rows},indent=2))
    assert all(not row['overlap_seeds'] for row in rows),'Register alternate unused development seeds before testing'
    print(json.dumps(rows,indent=2),flush=True)

def summarize(root,p):
    out,data,models=paths(root);freeze=read(out/'freeze.json');records={};audit={}
    assert all(not row['overlap_seeds'] for row in read(out/'goal_isolation.json')['rows'])
    assert sha(data/'data/demonstrations.hdf5')==p['dataset_sha256']
    for name,digest in freeze['files'].items():assert sha(root/name)==digest,name
    for framework in p['frameworks']:
        records[framework]={}
        for horizon in p['horizons']:
            groups={}
            for task_id,task in enumerate(p['tasks']):
                folder=data/'outputs'/case_name(framework,horizon,task);s=read(folder/'summary.json');rows=s['episodes']
                assert s['horizon']==horizon and s['task_id']==task_id and s['environment']==task+'-goal-observable'
                assert s['top_k']==p['top_k'] and s['max_steps']==p['max_steps'] and s['per_episode_sampling']
                assert [r['seed'] for r in rows]==list(range(p['seed_start'],p['seed_start']+p['episodes_per_task']))
                if framework=='jittor':
                    assert not s['torch_imported'] and s['policy_weights_sha256']==p['weights'][framework]['npz_sha256']
                    assert s['tokenizer_weights_sha256']==p['tokenizer_npz_sha256']
                else:
                    assert s['policy_sha256']==freeze['files'][(models/'policy_torch/best.pt').relative_to(root).as_posix()]
                    assert s['tokenizer_sha256']==freeze['files'][(data/'assets/shared/tokenizer_shared.pt').relative_to(root).as_posix()]
                decisions=[]
                for row in rows:
                    with np.load(folder/f"seed_{row['seed']}.npz") as a:
                        assert a['predicted_actions'].shape==(row['steps'],4)
                        assert a['generated_tokens'].shape==((row['steps']+horizon-1)//horizon,8)
                        assert np.isfinite(a['predicted_actions']).all();decisions.append(len(a['generated_tokens']))
                successes=sum(bool(r['success']) for r in rows);n=len(rows)
                assert successes/n==s['success_rate']
                groups[task]={'successes':successes,'episodes':n,'rate':successes/n,
                    'outcomes':[int(r['success']) for r in rows],'mean_decisions':float(np.mean(decisions)),
                    'mean_control_steps':float(np.mean([r['steps'] for r in rows])),
                    'median_decision_ms':s['median_decision_ms'],'p95_decision_ms':s['p95_decision_ms'],
                    'performance_scope':s['performance_scope']}
            records[framework][str(horizon)]={'tasks':groups,'successes':sum(g['successes'] for g in groups.values()),
                'episodes':sum(g['episodes'] for g in groups.values()),'rate':float(np.mean([g['rate'] for g in groups.values()])),
                'mean_decisions':float(np.mean([g['mean_decisions'] for g in groups.values()]))}
    rng=np.random.default_rng(321000);effects={}
    for framework in p['frameworks']:
        effects[framework]={};base=records[framework]['8']
        for horizon in (4,16):
            current=records[framework][str(horizon)]
            delta=np.array([np.array(current['tasks'][t]['outcomes'])-np.array(base['tasks'][t]['outcomes']) for t in p['tasks']])
            samples=np.stack([delta[i,rng.integers(0,p['episodes_per_task'],size=(5000,p['episodes_per_task']))].mean(1) for i in range(len(p['tasks']))],axis=1).mean(1)
            effects[framework][str(horizon)]={'rate_difference_vs8':float(delta.mean()),
                'paired_task_stratified_bootstrap95':np.percentile(samples,[2.5,97.5]).tolist(),
                'improved_scenes':int((delta==1).sum()),'worsened_scenes':int((delta==-1).sum())}
    audit.update(frozen_files_unchanged=True,new_goal_isolation=True,all600_rollouts_checked=True,
        actual_action_horizons_verified=True,native_no_torch=True)
    result={'status':'audited','protocol':p,'audit':audit,'records':records,'effects':effects,
        'limitations':['development comparison,not finalbenchmark','one frozen checkpoint/framework','top5 stochastic sampling','timing includes host transfer/cold start,notstandardized','no direct test of padding or dataquality hypotheses']}
    dest=root/'reports/FEEDBACK_HORIZON_RESULTS_20261029';dest.with_suffix('.json').write_text(json.dumps(result,indent=2))
    lines=['# 固定权重的动作执行间隔对照','',
        '预测动作块始终32步，仅改变每次执行4、8、16步后重新观察。固定五任务模型和分词器，没有重新训练。新开发场景14000–14019，每任务20次，两框架合计600次评估。','',
        '| 框架 | 执行间隔 | 成功/100 | 平均每局策略调用次数 |','|---|---:|---:|---:|']
    for f in p['frameworks']:
        for h in (4,8,16):
            r=records[f][str(h)];lines.append(f"| {f} | {h} | {r['successes']}/100 | {r['mean_decisions']:.2f} |")
    lines+=['','## 分任务结果','','| 框架/间隔 | Reach | Reach-wall | Push | Door-open | Window-open |','|---|---:|---:|---:|---:|---:|']
    for f in p['frameworks']:
        for h in (4,8,16):lines.append('| '+f+'/'+str(h)+' | '+' | '.join(str(records[f][str(h)]['tasks'][t]['successes'])+'/20' for t in p['tasks'])+' |')
    lines+=['','执行4步增加反馈及策略调用次数，执行16步延长开环区间；效果须看实际结果。开发场景可用于选择下一步方案，不能作为改进后的独立最终成绩。未更改补齐、损失、成功标准、训练数据或模型参数。',
        '成功率差的配对分层bootstrap区间见JSON；只反映当前检查点与场景抽样，不包含重新训练的方差。耗时包含冷启动和传输，不能直接换算为完整控制频率。']
    dest.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,f in zip(axes,p['frameworks']):
        for j,h in enumerate((4,8,16)):ax.bar(np.arange(5)+(j-1)*.25,[records[f][str(h)]['tasks'][t]['rate'] for t in p['tasks']],width=.25,label=f'execute{h}')
        ax.set_xticks(range(5),['Reach','Wall','Push','Door','Window']);ax.set_ylim(0,1.05);ax.set_title(f);ax.legend();ax.set_ylabel('Success rate;20scenes/task');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Frozen five-task policies:feedback interval development comparison');fig.tight_layout();fig.savefig(dest.with_suffix('.png'),dpi=170);plt.close(fig)
    print(json.dumps({'audit':audit,'overall':{f:{h:r['successes'] for h,r in v.items()} for f,v in records.items()},'effects':effects},indent=2),flush=True)

def run(root,p):
    import fcntl
    out,data,models=paths(root);out.mkdir(parents=True,exist_ok=True)
    lock=(out/'workflow.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    files=[models/'policy_torch/best.pt',models/'policy_torch/best_weights.npz',models/'policy_jittor/best_weights.npz',
        models/'policy_jittor/config.json',data/'assets/weights.npz',data/'assets/config.json',data/'assets/shared/tokenizer_shared.pt',
        root/'scripts/evaluate_policy.py',root/'scripts/evaluate_policy_jittor.py',Path(__file__).resolve(),root/'records/feedback_horizon_protocol_20261029.json']
    freeze={'utc':datetime.now(timezone.utc).isoformat(),'files':{path.relative_to(root).as_posix():sha(path) for path in files},'protocol':p}
    state={'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),'protocol':p,'stages':[],'server_shutdown':False}
    if (out/'workflow.json').exists():
        state=read(out/'workflow.json');assert state['status']=='failed','Refuse duplicate running/completed workflow'
        old=read(out/'freeze.json');assert old['files']==freeze['files'] and old['protocol']==p,'Frozen inputs changed'
        shutil.copy2(out/'workflow.json',out/f'failed_{int(time.time())}.json');state.update(status='running');state.pop('error',None)
    else:(out/'freeze.json').write_text(json.dumps(freeze,indent=2))
    started=time.monotonic();previous=state.get('elapsed_seconds',0)
    env=os.environ.copy();env.update(PYTHONPATH=str(root/'src'),NVIDIA_TF32_OVERRIDE='0',OMP_NUM_THREADS='4',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl')
    torch=root/'.venv-torch-linux/bin/python';native=[root/'.venv-jittor-linux/bin/python','-u',root/'scripts/run_jittor_linux.py','--device','cuda']
    def save():
        state['elapsed_seconds']=previous+time.monotonic()-started
        temp=out/'workflow.tmp';temp.write_text(json.dumps(state,indent=2));temp.replace(out/'workflow.json')
    def stage(name,cmd,cap=900):
        if any(r['name']==name and r['status']=='passed' for r in state['stages']):return
        remaining=p['time_limit_hours']*3600-(previous+time.monotonic()-started)
        if remaining<=0:raise TimeoutError('Workflow time bound reached;server remains ON')
        row={'name':name,'status':'running','command':[str(v) for v in cmd]};state['stages'].append(row);save()
        log=out/f'{name}.log'
        if log.exists():shutil.copy2(log,out/f'{name}_before_retry_{int(time.time())}.log')
        with log.open('wb') as f:
            child=subprocess.Popen(row['command'],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            row['pid']=child.pid;save()
            try:code=child.wait(timeout=min(cap,remaining))
            except BaseException:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                row['status']='interrupted';save();raise
        row.update(status='passed' if code==0 else 'failed',exit_code=code);save()
        if code:raise RuntimeError(f'{name} failed;preserved log:{log}')
    try:
        save();assert read(models/'workflow.json')['status']=='completed'
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
        assert int(gpu.strip())<512,'GPU appears occupied;do not overlap workloads'
        assert shutil.disk_usage(root).free>2*1024**3
        assert sha(data/'data/demonstrations.hdf5')==p['dataset_sha256']
        for f in p['frameworks']:assert sha(models/f'policy_{f}/best_weights.npz')==p['weights'][f]['npz_sha256']
        assert sha(data/'assets/weights.npz')==p['tokenizer_npz_sha256']
        stage('goal_isolation',[torch,Path(__file__),'--phase','isolation','--root',root],180)
        for horizon in p['horizons']:
            for framework in p['frameworks']:
                for task_id,task in enumerate(p['tasks']):
                    name=case_name(framework,horizon,task)
                    if framework=='torch':cmd=[torch,root/'scripts/evaluate_policy.py','--policy',models/'policy_torch/best.pt','--tokenizer',data/'assets/shared/tokenizer_shared.pt','--sampling-backend','numpy']
                    else:cmd=native+[root/'scripts/evaluate_policy_jittor.py','--policy-weights',models/'policy_jittor/best_weights.npz','--policy-config',models/'policy_jittor/config.json','--tokenizer-weights',data/'assets/weights.npz','--tokenizer-config',data/'assets/config.json']
                    stage(name,cmd+['--root',data,'--device','cuda','--seed-set','experiment','--seed-start',str(p['seed_start']),
                        '--episodes',str(p['episodes_per_task']),'--env-name',task+'-goal-observable','--task-id',str(task_id),'--per-episode-sampling',
                        '--top-k',str(p['top_k']),'--sampling-seed',str(p['sampling_seed']),'--max-steps',str(p['max_steps']),
                        '--action-horizon',str(horizon),'--videos','0','--name',name])
        stage('completion_report',[torch,Path(__file__),'--phase','summarize','--root',root],240)
        archive=root/'reports/feedback_horizon_20261029.zip'
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
            for path in out.rglob('*'):
                if path.is_file() and path.name not in ('workflow.lock','workflow.tmp'):z.write(path,path.relative_to(root))
            for folder in (data/'outputs').glob('feedback_h*20261029'):
                for path in folder.rglob('*'):
                    if path.is_file():z.write(path,path.relative_to(root))
            for path in (root/'reports').glob('FEEDBACK_HORIZON_RESULTS_20261029*'):z.write(path,path.relative_to(root))
            for path in (Path(__file__).resolve(),root/'records/feedback_horizon_protocol_20261029.json'):z.write(path,path.relative_to(root))
        state.update(status='completed',archive={'path':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive)})
    except BaseException as error:state.update(status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:save()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--phase',choices=['run','isolation','summarize'],default='run');args=parser.parse_args()
    root=args.root.resolve();protocol=read(root/'records/feedback_horizon_protocol_20261029.json')
    {'run':run,'isolation':isolation,'summarize':summarize}[args.phase](root,protocol)
