"""Isolate task-balanced sampling;reuse immutable five-task models/data as control."""
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
def paths(root):return root/'outputs/task_balanced_20261029',root/'outputs/five_task_10pct_20261029/dataset',root/'outputs/five_task_10pct_20261029/experiments'

def audit(root,p):
    out,data,old=paths(root);freeze=read(out/'freeze.json');m=read(data/'records/reach_split_v1.json')
    assert len(m['train'])==450 and len(m['validation'])==100 and m['sha256']==p['dataset_sha256']
    assert sha(data/m['dataset'])==m['sha256']
    for file,digest in freeze['files'].items():assert sha(root/file)==digest,file
    with np.load(out/'batch_plan.npz') as a:
        plan=a['indices'];info=json.loads(str(a['metadata']));assert plan.shape==(12000,16)
    assert info['train_episodes']==m['train'] and info['dataset_sha256']==m['sha256']
    with h5py.File(data/m['dataset'],'r') as file:
        window_tasks=np.concatenate([np.full(len(file[e]['actions']),m['episode_task_ids'][e],dtype=np.int64) for e in m['train']])
    np.testing.assert_array_equal(np.unique(plan),np.arange(len(window_tasks)))
    np.testing.assert_array_equal(np.bincount(window_tasks[plan].ravel(),minlength=5),np.full(5,38400))
    assert info['all_train_windows_covered'] and all(v==38400 for v in info['planned_task_exposures'].values())
    results={};streams=[];test=p['test']
    for framework in ('torch','jittor'):
        folder=out/('policy_'+framework);s=read(folder/'summary.json');c=read(folder/'config.json')
        assert s['status']=='completed' and s['updates']==p['updates'] and s['seen_examples']==192000
        assert c['initial_policy_npz_sha256']==p['initial_sha256'] and c['dataset_sha256']==p['dataset_sha256']
        assert c['tokenizer_weights_sha256']==p['tokenizer_npz_sha256'] and c['sampling_plan_sha256']==p['plan_sha256']
        assert s['batch_stream_sha256']==info['batch_stream_sha256'];streams.append(s['batch_stream_sha256'])
        gradients=[json.loads(v) for v in (folder/'gradients.jsonl').read_text().splitlines()]
        assert [v['step'] for v in gradients]==list(range(1,12001))
        assert np.isfinite([v['preclip_norm'] for v in gradients]).all() and not s['nonfinite_loss_or_gradient']
        if framework=='jittor':assert not s['torch_imported']
        groups={}
        for task_id,task in enumerate(p['tasks']):
            baseline_folder=data/'outputs'/f'feedback_h8_{framework}_{task}_20261029'
            current_folder=data/'outputs'/f'task_balanced_{framework}_{task}_20261029'
            controls=[]
            for condition,where in [('baseline',baseline_folder),('balanced',current_folder)]:
                value=read(where/'summary.json');episodes=value['episodes']
                assert value['horizon']==8 and value['top_k']==5 and value['task_id']==task_id and value['max_steps']==150
                assert [v['seed'] for v in episodes]==list(range(test['seed_start'],test['seed_start']+20))
                if framework=='jittor':assert not value['torch_imported']
                counts=sum(bool(v['success']) for v in episodes)
                controls.append({'successes':counts,'episodes':20,'rate':counts/20,'outcomes':[int(v['success']) for v in episodes]})
            groups[task]={'baseline':controls[0],'balanced':controls[1]}
        results[framework]={'training':s,'tasks':groups,'baseline_successes':sum(v['baseline']['successes'] for v in groups.values()),
            'balanced_successes':sum(v['balanced']['successes'] for v in groups.values())}
    assert len(set(streams))==1
    with np.load(out/'fixture_torch.npz') as a,np.load(out/'fixture_jittor.npz') as b:
        for k in a.files:np.testing.assert_allclose(a[k],b[k],atol=2e-5,rtol=2e-4)
        np.testing.assert_array_equal(a['tokens'],b['tokens'])
    freeze_test=read(out/'test_freeze.json')
    for framework in ('torch','jittor'):assert sha(out/('policy_'+framework)/'best_weights.npz')==freeze_test['models'][framework]
    rng=np.random.default_rng(322000)
    for framework in results:
        delta=np.array([np.array(v['balanced']['outcomes'])-np.array(v['baseline']['outcomes']) for v in results[framework]['tasks'].values()])
        draws=np.stack([delta[i,rng.integers(0,20,(5000,20))].mean(1) for i in range(5)],axis=1).mean(1)
        results[framework]['effect']={'difference':float(delta.mean()),'paired_task_stratified_bootstrap95':np.percentile(draws,[2.5,97.5]).tolist(),
            'improved_scenes':int((delta==1).sum()),'worsened_scenes':int((delta==-1).sum())}
    gates={'same450train100val':True,'all_train_windows_in_balanced_schedule':True,'same_initialization_tokenizer_and_budget':True,
        '24000_finite_policy_updates':True,'same_native_learned_weights_align':True,'200_new_and200_cached_matched_rollouts':True,'frozen_inputs_preserved':True}
    report={'status':'audited','audit':gates,'protocol':p,'results':results,
        'limitations':['developmentcomparison,notnewfinalbenchmark','one training seed','failuredataretained,nofiltering','samplingeffectonly,nopretrainedvisionorpaddingchange']}
    dest=root/'reports/TASK_BALANCED_RESULTS_20261029';dest.with_suffix('.json').write_text(json.dumps(report,indent=2))
    lines=['# 五任务均衡采样对照','',
        '450条训练、100条验证及分词器均不变，五任务训练窗口曝光各20%。两框架各12000次更新、同一初始化与共同批次计划；动作执行间隔固定8，按验证CE选检查点。','',
        '| 任务 | Torch原采样/20 | Torch均衡/20 | Jittor原采样/20 | Jittor均衡/20 |','|---|---:|---:|---:|---:|']
    for task in p['tasks']:
        lines.append('| '+task+' | '+' | '.join(str(results[f]['tasks'][task][c]['successes'])+'/20' for f in ('torch','jittor') for c in ('baseline','balanced'))+' |')
    lines+=['','新开发场景为14000–14019。控制组复用同场景的既有8步反馈记录，没有重跑旧测试。逐步梯度、相同权重对齐与实际窗口计划已检查。',
        '本实验只改变任务采样比例，失败轨迹仍保留。改善或退步不能归因于筛除了失败数据、使用了预训练视觉模型或修改了末尾补齐。',
        '只训练一个随机种子，成功率差及配对分层bootstrap区间见JSON。这些是开发结果，后续方案确定后还需新的最终测试。']
    dest.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,f in zip(axes,('torch','jittor')):
        for j,c in enumerate(('baseline','balanced')):ax.bar(np.arange(5)+(j-.5)*.36,[results[f]['tasks'][t][c]['rate'] for t in p['tasks']],width=.36,label=c)
        ax.set_xticks(range(5),['Reach','Wall','Push','Door','Window']);ax.set_ylim(0,1.05);ax.set_title(f);ax.legend();ax.grid(axis='y',alpha=.2)
    fig.suptitle('Task-balanced sampling;matched development scenes,one training seed');fig.tight_layout();fig.savefig(dest.with_suffix('.png'),dpi=170);plt.close(fig)
    print(json.dumps({'audit':gates,'results':{f:{k:v for k,v in r.items() if k not in ('tasks','training')} for f,r in results.items()}},indent=2),flush=True)

def run(root,p):
    import fcntl
    out,data,old=paths(root);out.mkdir(parents=True,exist_ok=True)
    lock=(out/'workflow.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    files=[root/'records/task_balanced_protocol_20261029.json',Path(__file__).resolve(),root/'scripts/train_sampling_policy.py',
        root/'scripts/train_controlled_policy.py',out/'batch_plan.npz',old/'initial_five.npz',data/'assets/weights.npz',data/'assets/shared/tokenizer_shared.pt',
        old/'policy_torch/best_weights.npz',old/'policy_jittor/best_weights.npz']
    freeze={'utc':datetime.now(timezone.utc).isoformat(),'files':{f.relative_to(root).as_posix():sha(f) for f in files}}
    state={'status':'running','protocol':p,'started_utc':datetime.now(timezone.utc).isoformat(),'stages':[],'server_shutdown':False}
    if (out/'workflow.json').exists():
        state=read(out/'workflow.json');assert state['status']=='failed','Refuse duplicate active/completed run'
        assert read(out/'freeze.json')['files']==freeze['files'];shutil.copy2(out/'workflow.json',out/f'failed_{int(time.time())}.json')
        state.update(status='running');state.pop('error',None)
    else:(out/'freeze.json').write_text(json.dumps(freeze,indent=2))
    previous=state.get('elapsed_seconds',0);started=time.monotonic();torch=root/'.venv-torch-linux/bin/python'
    native=[root/'.venv-jittor-linux/bin/python','-u',root/'scripts/run_jittor_linux.py','--device','cuda']
    env=os.environ.copy();env.update(PYTHONPATH=str(root/'src'),NVIDIA_TF32_OVERRIDE='0',OMP_NUM_THREADS='4',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl')
    def save():
        state['elapsed_seconds']=previous+time.monotonic()-started;tmp=out/'workflow.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(out/'workflow.json')
    def stage(name,cmd,cap=3600):
        if any(v['name']==name and v['status']=='passed' for v in state['stages']):return
        remaining=p['time_limit_hours']*3600-state.get('elapsed_seconds',0)
        if remaining<=0:raise TimeoutError('Three-hour bound reached;server remains ON')
        row={'name':name,'status':'running','command':[str(v) for v in cmd]};state['stages'].append(row);save();log=out/f'{name}.log'
        if log.exists():shutil.copy2(log,out/f'{name}_before_retry_{int(time.time())}.log')
        with log.open('wb') as f:
            child=subprocess.Popen(row['command'],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);row['pid']=child.pid;save()
            try:code=child.wait(timeout=min(cap,remaining))
            except BaseException:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                row['status']='interrupted';save();raise
        row.update(status='passed' if code==0 else 'failed',exit_code=code);save()
        if code:raise RuntimeError(f'{name} failed;preserved log:{log}')
    def train(f,smoke=False):
        name=('smoke_' if smoke else 'policy_')+f
        cmd=([torch,'-u'] if f=='torch' else native)+[root/'scripts/train_sampling_policy.py','--root',data,'--output',out/name,
            '--framework',f,'--tokenizer-weights',data/'assets/weights.npz','--tokenizer-config',data/'assets/config.json',
            '--initial-weights',old/'initial_five.npz','--canonical-labels',data/'assets/canonical_labels','--batch-plan',out/'batch_plan.npz',
            '--updates',str(4 if smoke else p['updates']),'--eval-every',str(4 if smoke else p['eval_every']),
            '--lr',str(p['lr']),'--weight-decay',str(p['weight_decay']),'--dropout','0','--shift','0','--seed',str(p['model_seed'])]
        if f=='torch':cmd+=['--tokenizer-checkpoint',data/'assets/shared/tokenizer_shared.pt']
        stage(name,cmd,3600)
    try:
        save();assert read(root/'outputs/feedback_horizon_20261029/workflow.json')['status']=='completed','Wait for feedback experiment before GPU work'
        memory=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        assert memory<512,'Another GPU workload is active';assert shutil.disk_usage(root).free>3*1024**3
        assert sha(out/'batch_plan.npz')==p['plan_sha256'] and sha(old/'initial_five.npz')==p['initial_sha256']
        assert read(data/'records/reach_split_v1.json')['sha256']==p['dataset_sha256']
        for f in ('torch','jittor'):train(f,smoke=True)
        for f in ('torch','jittor'):train(f)
        for f in ('torch','jittor'):
            stage('fixture_'+f,([torch] if f=='torch' else native)+[root/'scripts/export_trained_policy_predictions.py','--root',data,'--framework',f,
                '--policy-weights',out/'policy_jittor/best_weights.npz','--policy-config',out/'policy_jittor/config.json',
                '--tokenizer-weights',data/'assets/weights.npz','--tokenizer-config',data/'assets/config.json','--output',out/f'fixture_{f}.npz'],300)
        with np.load(out/'fixture_torch.npz') as a,np.load(out/'fixture_jittor.npz') as b:
            for k in a.files:np.testing.assert_allclose(a[k],b[k],atol=2e-5,rtol=2e-4)
            np.testing.assert_array_equal(a['tokens'],b['tokens'])
        frozen={'utc':datetime.now(timezone.utc).isoformat(),'models':{f:sha(out/('policy_'+f)/'best_weights.npz') for f in ('torch','jittor')}}
        if (out/'test_freeze.json').exists():assert read(out/'test_freeze.json')['models']==frozen['models']
        else:(out/'test_freeze.json').write_text(json.dumps(frozen,indent=2))
        for f in ('torch','jittor'):
            model=out/('policy_'+f)
            for task_id,task in enumerate(p['tasks']):
                name=f'task_balanced_{f}_{task}_20261029'
                if f=='torch':cmd=[torch,root/'scripts/evaluate_policy.py','--policy',model/'best.pt','--tokenizer',data/'assets/shared/tokenizer_shared.pt','--sampling-backend','numpy']
                else:cmd=native+[root/'scripts/evaluate_policy_jittor.py','--policy-weights',model/'best_weights.npz','--policy-config',model/'config.json','--tokenizer-weights',data/'assets/weights.npz','--tokenizer-config',data/'assets/config.json']
                stage(name,cmd+['--root',data,'--device','cuda','--seed-set','experiment','--seed-start','14000','--episodes','20','--env-name',task+'-goal-observable',
                    '--task-id',str(task_id),'--per-episode-sampling','--top-k','5','--sampling-seed','30000','--max-steps','150','--action-horizon','8','--videos','0','--name',name],900)
        stage('completion_report',[torch,Path(__file__),'--root',root,'--phase','audit'],240)
        archive=root/'reports/task_balanced_20261029.zip'
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
            for path in out.rglob('*'):
                if path.is_file() and path.name not in ('last.pt','last_weights.npz','workflow.tmp','workflow.lock'):z.write(path,path.relative_to(root))
            for folder in (data/'outputs').glob('task_balanced_*20261029'):
                for path in folder.rglob('*'):
                    if path.is_file():z.write(path,path.relative_to(root))
            for path in (root/'reports').glob('TASK_BALANCED_RESULTS_20261029*'):z.write(path,path.relative_to(root))
            for path in [Path(__file__).resolve(),root/'scripts/train_sampling_policy.py',root/'scripts/prepare_task_balanced_plan.py',root/'records/task_balanced_protocol_20261029.json']:z.write(path,path.relative_to(root))
        state.update(status='completed',archive={'bytes':archive.stat().st_size,'sha256':sha(archive),'path':str(archive)})
    except BaseException as error:state.update(status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:save()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);parser.add_argument('--phase',choices=['run','audit'],default='run');args=parser.parse_args()
    root=args.root.resolve();p=read(root/'records/task_balanced_protocol_20261029.json');(run if args.phase=='run' else audit)(root,p)
