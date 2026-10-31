"""Twelve-cell activation factorial;protected job first,four resource-gated workers.

Independent cases continue after a failed cell;all failed logs remain visible.
No model/data/source changes to the protected padding/holding workflow.
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
def write(p,v):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2));q.replace(p)
def environment(root):
    env=os.environ.copy();env.update(PYTHONPATH=str(root/'src'),NVIDIA_TF32_OVERRIDE='0',OMP_NUM_THREADS='4',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl');return env
def runtime(root,f):return [root/'.venv-torch-linux/bin/python'] if f=='torch' else [root/'.venv-jittor-linux/bin/python',root/'scripts/run_jittor_linux.py','--device','cuda']
def variant(root,f,conv,ffn,script):return runtime(root,f)+[root/'scripts/run_activation_variant.py','--framework',f,'--conv',conv,'--ffn',ffn,root/'scripts'/script]
def aligned(a,b):
    with np.load(a) as x,np.load(b) as y:
        assert set(x.files)==set(y.files)
        for key in x.files:
            if key in ('indices','tokens'):np.testing.assert_array_equal(x[key],y[key])
            else:np.testing.assert_allclose(x[key],y[key],atol=2e-5,rtol=2e-4,err_msg=key)

def worker(root,name):
    out=root/'outputs/activation_grid_20261029';case=out/'cases'/name;data=case/'dataset';shared=out/'shared';p=read(root/'records/activation_grid_protocol_20261029.json');conv,ffn=name.split('_')
    import fcntl
    lock=(case/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (case/'workflow.json').exists(),'No blind worker restart'
    state={'case':name,'status':'running','stages':[],'started_utc':datetime.now(timezone.utc).isoformat()};started=time.monotonic();env=environment(root)
    def save():state['elapsed_seconds']=time.monotonic()-started;write(case/'workflow.json',state)
    def stage(label,cmd,cap=7200):
        row={'name':label,'status':'running','command':[str(x) for x in cmd]};state['stages'].append(row);save()
        with (case/(label+'.log')).open('xb') as log:
            child=subprocess.Popen(row['command'],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);row['pid']=child.pid;save()
            try:code=child.wait(timeout=cap)
            except BaseException:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise
        row.update(status='passed' if code==0 else 'failed',exit_code=code);save()
        if code:raise RuntimeError(f'{label} failed;preserved {case/(label+".log")}')
    def tokenizer_probe(label,weights,config):
        for f in ('torch','jittor'):
            stage(label+'_'+f,variant(root,f,conv,ffn,'activation_grid_probe.py')+['--framework',f,'--data',data,'--weights',weights,'--config',config,'--output',case/f'{label}_{f}.npz'],600)
        aligned(case/f'{label}_torch.npz',case/f'{label}_jittor.npz')
    try:
        save();tokenizer_probe('initial_probe',shared/'initial_tokenizer.npz',shared/'tokenizer_config.json')
        for smoke in (True,False):
            for f in ('torch','jittor'):
                folder=case/(('smoke_' if smoke else '')+'tokenizer_'+f)
                stage(folder.name,variant(root,f,conv,ffn,'train_activation_tokenizer.py')+['--root',data,'--output',folder,'--framework',f,
                    '--initial',shared/'initial_tokenizer.npz','--config',shared/'tokenizer_config.json','--updates',str(4 if smoke else p['tokenizer_updates']),
                    '--eval-every',str(4 if smoke else p['tokenizer_eval_every']),'--seed',str(p['seed'])])
            if smoke:
                x=read_first(case/'smoke_tokenizer_torch/gradients.jsonl');y=read_first(case/'smoke_tokenizer_jittor/gradients.jsonl')
                np.testing.assert_allclose(x['loss'],y['loss'],atol=1e-4,rtol=1e-3);np.testing.assert_allclose(x['gradient_norm'],y['gradient_norm'],atol=1e-3,rtol=1e-3)
        tokenizer_probe('learned_tokenizer_probe',case/'tokenizer_jittor/best_weights.npz',case/'tokenizer_jittor/config.json')
        stage('assets',variant(root,'torch',conv,ffn,'activation_grid_assets.py')+['--root',root,'--case',name],900);assets=case/'assets'
        for smoke in (True,False):
            for f in ('torch','jittor'):
                folder=case/(('smoke_' if smoke else '')+'policy_'+f)
                cmd=variant(root,f,conv,ffn,'train_activation_policy.py')+['--root',data,'--output',folder,'--framework',f,'--tokenizer-weights',assets/'weights.npz',
                    '--tokenizer-config',assets/'config.json','--initial-weights',shared/'initial_policy.npz','--canonical-labels',assets/'canonical_labels',
                    '--updates',str(4 if smoke else p['policy_updates']),'--eval-every',str(4 if smoke else 1000),'--lr','0.0001','--weight-decay','0.0001','--dropout','0','--shift','0','--seed',str(p['seed'])]
                if f=='torch':cmd+=['--tokenizer-checkpoint',assets/'tokenizer_shared.pt']
                stage(folder.name,cmd,3600)
        for f in ('torch','jittor'):
            stage('policy_fixture_'+f,variant(root,f,conv,ffn,'export_trained_policy_predictions.py')+['--root',data,'--framework',f,
                '--policy-weights',case/'policy_jittor/best_weights.npz','--policy-config',case/'policy_jittor/config.json',
                '--tokenizer-weights',assets/'weights.npz','--tokenizer-config',assets/'config.json','--output',case/f'policy_fixture_{f}.npz'],600)
        aligned(case/'policy_fixture_torch.npz',case/'policy_fixture_jittor.npz')
        freeze={'models':{f:sha(case/f'policy_{f}/best_weights.npz') for f in ('torch','jittor')},'tokenizer':sha(assets/'weights.npz')}
        write(case/'evaluation_freeze.json',freeze)
        for f in ('torch','jittor'):
            for task_id,task in enumerate(p['tasks']):
                label=f'activation_{name}_{f}_{task}'
                if f=='torch':cmd=variant(root,f,conv,ffn,'evaluate_policy.py')+['--policy',case/'policy_torch/best.pt','--tokenizer',assets/'tokenizer_shared.pt','--sampling-backend','numpy']
                else:cmd=variant(root,f,conv,ffn,'evaluate_policy_jittor.py')+['--policy-weights',case/'policy_jittor/best_weights.npz','--policy-config',case/'policy_jittor/config.json','--tokenizer-weights',assets/'weights.npz','--tokenizer-config',assets/'config.json']
                stage(label,cmd+['--root',data,'--device','cuda','--seed-set','experiment','--seed-start','100000','--episodes','30','--env-name',task+'-goal-observable',
                    '--task-id',str(task_id),'--per-episode-sampling','--top-k','5','--sampling-seed','110000','--max-steps','150','--action-horizon','8','--videos','0','--name',label],1800)
        verify_case(root,name);state['status']='completed'
    except BaseException as e:state.update(status='failed',error=f'{type(e).__name__}: {e}');raise
    finally:save()

def read_first(p):return json.loads(p.read_text().splitlines()[0])

def verify_case(root,name):
    out=root/'outputs/activation_grid_20261029';case=out/'cases'/name;p=read(root/'records/activation_grid_protocol_20261029.json');conv,ffn=name.split('_');streams={'tokenizer':[],'policy':[]}
    for f in ('torch','jittor'):
        for kind,budget in [('tokenizer',p['tokenizer_updates']),('policy',p['policy_updates'])]:
            folder=case/f'{kind}_{f}';s=read(folder/'summary.json');c=read(folder/'config.json');assert s['status']=='completed' and s['updates']==budget
            assert c['conv_activation']==conv and c['ffn_activation']==ffn and c['dataset_sha256']==p['dataset_sha256']
            if kind=='tokenizer':assert c['initial_sha256']==sha(out/'shared/initial_tokenizer.npz') and s['finite']
            else:assert c['initial_policy_npz_sha256']==sha(out/'shared/initial_policy.npz') and not s['nonfinite_loss_or_gradient']
            gradients=[json.loads(v) for v in (folder/'gradients.jsonl').read_text().splitlines()];assert [v['step'] for v in gradients]==list(range(1,budget+1))
            key='gradient_norm' if kind=='tokenizer' else 'preclip_norm';assert np.isfinite([v[key] for v in gradients]).all();streams[kind].append(s['batch_stream_sha256'])
            if f=='jittor':assert not s['torch_imported']
        for task_id,task in enumerate(p['tasks']):
            s=read(case/'dataset/outputs'/f'activation_{name}_{f}_{task}'/'summary.json')
            assert s['task_id']==task_id and s['horizon']==8 and s['top_k']==5 and s['max_steps']==150 and s['sampling_seed']==110000
            assert [v['seed'] for v in s['episodes']]==list(range(100000,100030))
            if f=='jittor':assert not s['torch_imported']
    for values in streams.values():assert len(set(values))==1
    for stem in ('initial_probe','learned_tokenizer_probe','policy_fixture'):aligned(case/f'{stem}_torch.npz',case/f'{stem}_jittor.npz')
    freeze=read(case/'evaluation_freeze.json')
    for f,digest in freeze['models'].items():assert sha(case/f'policy_{f}/best_weights.npz')==digest
    assert sha(case/'assets/weights.npz')==freeze['tokenizer']
    write(case/'audit.json',{'status':'passed','same_initializations':True,'finite_complete_training':True,'same_batch_streams':streams,'all_alignment_probes_passed':True,'actual_rollouts':300,'frozen_checkpoints_preserved':True})

def gpu_resource():
    values=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
    import shutil
    return {'free_vram_mib':int(values[0]),'gpu_util':int(values[1]),'disk_free_gib':shutil.disk_usage('.').free/1024**3}

def supervisor(root,resume_scheduler=False):
    import fcntl
    out=root/'outputs/activation_grid_20261029';out.mkdir(parents=True,exist_ok=True);lock=(out/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'workflow.json').exists():
        prior=read(out/'workflow.json')
        assert resume_scheduler and prior['status']=='waiting_for_protected_job' and not prior.get('active') and all(v=='queued' for v in prior['cases'].values()) and not list((out/'cases').glob('*/workflow.json')), 'Only resume an untouched waiting supervisor'
        backup=out/'scheduler_revisions';backup.mkdir(exist_ok=True)
        (out/'workflow.json').rename(backup/f'workflow_before_four_{int(time.time())}.json')
        if (out/'freeze.json').exists():(out/'freeze.json').rename(backup/f'freeze_before_four_{int(time.time())}.json')
    p=read(root/'records/activation_grid_protocol_20261029.json');state={'status':'preparing','started_utc':datetime.now(timezone.utc).isoformat(),'cases':{n:'queued' for n in p['cases']},'protected_workflow':'padding_hold_20261029','max_workers':4,'server_shutdown':False};write(out/'workflow.json',state)
    torch=root/'.venv-torch-linux/bin/python';env=environment(root)
    try:
        if not resume_scheduler:
            with (out/'initialization.log').open('xb') as log:subprocess.run([str(torch),str(root/'scripts/activation_grid_assets.py'),'--root',str(root)],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
        else:
            manifest=read(out/'shared/initialization_manifest.json')
            for name,digest in manifest.items():assert sha(out/'shared'/f'initial_{name}.npz')==digest, 'Shared initialization changed'
        inputs=[root/'records/activation_grid_protocol_20261029.json']+[root/'scripts'/n for n in p['scripts']]+list((root/'src/quest_repro').glob('*.py'))
        inputs+=[out/'shared/initial_tokenizer.npz',out/'shared/initial_policy.npz',root/'outputs/five_task_10pct_20261029/dataset/records/reach_split_v1.json']
        write(out/'freeze.json',{q.relative_to(root).as_posix():sha(q) for q in inputs})
        pending=list(p['cases']);children={};started=time.monotonic();protected=root/'outputs/padding_hold_20261029/workflow.json'
        while pending or children:
            if time.monotonic()-started>36*3600:raise TimeoutError('36hour scheduling limit;preserve jobs for inspection')
            for name,entry in list(children.items()):
                code=entry['process'].poll()
                if code is not None:
                    entry['log'].close();state['cases'][name]='completed' if code==0 else 'failed';del children[name]
            busy=protected.exists() and read(protected)['status'] in ('running','waiting_for_collection')
            resource=gpu_resource();state.update(resource=resource,protected_running=busy,active=list(children),status='waiting_for_protected_job' if busy else 'running')
            if not busy and pending and len(children)<4 and resource['free_vram_mib']>=6144 and resource['disk_free_gib']>=3:
                name=pending.pop(0);log=(out/'cases'/name/'worker_launcher.log').open('xb')
                child=subprocess.Popen(['nice','-n','10',str(torch),str(Path(__file__).resolve()),'--root',str(root),'--case',name],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                children[name]={'process':child,'log':log};state['cases'][name]='running'
            with (out/'resource_samples.jsonl').open('a') as f:f.write(json.dumps({'unix_time':time.time(),**resource,'protected_running':busy,'active':list(children)})+'\n')
            write(out/'workflow.json',state);time.sleep(10)
        state['status']='isolated_benchmarks';write(out/'workflow.json',state)
        for name in p['cases']:
            if state['cases'][name]!='completed':continue
            conv,ffn=name.split('_')
            for framework in ('torch','jittor'):
                # Workers are finished;do not mix these timings with concurrent training.
                case=out/'cases'/name;log=case/f'isolated_benchmark_{framework}.log'
                with log.open('xb') as f:
                    result=subprocess.run([str(v) for v in variant(root,framework,conv,ffn,'benchmark_activation_case.py')+['--framework',framework,'--case-root',case]],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=600)
                if result.returncode:state.setdefault('benchmark_failures',[]).append(name+'_'+framework)
        summarize(root);state['status']='completed' if all(v=='completed' for v in state['cases'].values()) and not state.get('benchmark_failures') else 'completed_with_failures'
        state['finished_utc']=datetime.now(timezone.utc).isoformat();write(out/'workflow.json',state)
        archive=root/'reports/activation_grid_20261029.zip'
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
            for q in out.rglob('*'):
                if q.is_file() and q.name not in ('supervisor.lock','worker.lock','workflow.tmp'):z.write(q,q.relative_to(root))
            for q in [root/'records/activation_grid_protocol_20261029.json']+[root/'scripts'/n for n in p['scripts']]+list((root/'reports').glob('ACTIVATION_GRID_RESULTS_20261029.*')):z.write(q,q.relative_to(root))
        state['archive']={'path':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size};write(out/'workflow.json',state)
    except BaseException as e:state.update(status='failed',error=f'{type(e).__name__}: {e}');write(out/'workflow.json',state);raise

def summarize(root):
    out=root/'outputs/activation_grid_20261029';p=read(root/'records/activation_grid_protocol_20261029.json');rows=[];raw={};curves={}
    for name in p['cases']:
        case=out/'cases'/name;wf=read(case/'workflow.json') if (case/'workflow.json').exists() else {'status':'not_started'};conv,ffn=name.split('_')
        if wf['status']!='completed':rows.append({'case':name,'status':wf['status'],'error':wf.get('error')});continue
        verify_case(root,name);raw[name]={};curves[name]={}
        for f in ('torch','jittor'):
            outcomes=[];task_success={}
            for task in p['tasks']:
                s=read(case/'dataset/outputs'/f'activation_{name}_{f}_{task}'/'summary.json');values=[int(v['success']) for v in s['episodes']];outcomes.append(values);task_success[task]=sum(values)
            tok=read(case/f'tokenizer_{f}/best_validation.json');training=read(case/f'policy_{f}/summary.json');ts=read(case/f'tokenizer_{f}/summary.json');benchmark=read(case/f'isolated_benchmark_{f}.json') if (case/f'isolated_benchmark_{f}.json').exists() else None
            row={'case':name,'conv':conv,'ffn':ffn,'framework':f,'status':'completed','successes':int(np.sum(outcomes)),'episodes':150,'success_rate':float(np.mean(outcomes)),'task_successes':task_success,
                'tokenizer_mae_all':tok['mae_all'],'tokenizer_mae_valid':tok['mae_valid'],'tokenizer_selected_step':tok['step'],'policy_best_step':training['best_step'],'policy_validation_ce':training['best_validation_ce'],
                'tokenizer_training_seconds_contended':ts['elapsed_seconds'],'policy_training_seconds_contended':training['elapsed_seconds'],'isolated_benchmark':benchmark}
            rows.append(row);raw[name][f]=outcomes
            curves[name][f]={'tokenizer':[json.loads(x) for x in (case/f'tokenizer_{f}/metrics.jsonl').read_text().splitlines()],
                'policy':[json.loads(x) for x in (case/f'policy_{f}/metrics.jsonl').read_text().splitlines()]}
    for relative,digest in read(out/'freeze.json').items():assert sha(root/relative)==digest,relative
    original=read(root/'outputs/five_task_10pct_20261029/dataset/records/reach_split_v1.json')
    assert sha(root/'outputs/five_task_10pct_20261029/dataset'/original['dataset'])==p['dataset_sha256']
    for kind in ('tokenizer','policy'):
        streams={read(out/'cases'/r['case']/f'{kind}_{r["framework"]}/summary.json')['batch_stream_sha256'] for r in rows if r['status']=='completed'}
        assert len(streams)<=1,f'Across-case batch order differs for {kind}'
    effects={};rng=np.random.default_rng(520510)
    for row in rows:
        if row['status']!='completed':continue
        f=row['framework'];name=row['case']
        if 'mish_gelu' in raw:
            delta=np.asarray(raw[name][f])-np.asarray(raw['mish_gelu'][f]);draws=np.stack([delta[i,rng.integers(0,30,(5000,30))].mean(1) for i in range(5)]).mean(0)
            row['difference_from_mish_gelu']=float(delta.mean());row['exploratory_paired_scene_bootstrap95']=np.percentile(draws,[2.5,97.5]).tolist()
    for f in ('torch','jittor'):
        successful=[r for r in rows if r.get('framework')==f and r['status']=='completed']
        if len(successful)!=12:
            effects[f]={'status':'not calculated for incomplete factorial','completed_cells':len(successful)};continue
        effects[f]={'conv_means':{c:float(np.mean([r['success_rate'] for r in successful if r['conv']==c])) for c in p['conv_activations'] if any(r['conv']==c for r in successful)},
            'ffn_means':{c:float(np.mean([r['success_rate'] for r in successful if r['ffn']==c])) for c in p['ffn_activations'] if any(r['ffn']==c for r in successful)}}
    trained=[r for r in rows if r['status']=='completed']
    complete_status='audited_complete_grid' if len(trained)==24 and all(r['isolated_benchmark'] is not None for r in trained) else 'complete_training_with_benchmark_failures' if len(trained)==24 else 'partial_grid_with_failures'
    report={'status':complete_status,'protocol':p,'rows':rows,'outcomes':raw,'curves':curves,'descriptive_factor_effects':effects,
        'limitations':['one training seed;factor means descriptive,not inferential factorial ANOVA','3600matched rollouts do not equal3600 independent trainings','multiple comparisons;bootstrap intervals exploratory and unadjusted','paired policies share this cells learned native tokenizer;Torch tokenizer training is recorded separately','absolute policy CE across different learned token vocabularies is not directly equivalent','contended wall times are not standalone speed rankings;use isolated benchmark','not full45task paper benchmark;winner needs fresh confirmation']}
    target=root/'reports/ACTIVATION_GRID_RESULTS_20261029';target.with_suffix('.json').write_text(json.dumps(report,indent=2))
    lines=['# QueST激活函数完整4×3对照','','| 卷积 | FFN | 框架 | 成功/150 | 重建MAE(有效位置) | 策略最佳步 |','|---|---|---|---:|---:|---:|']
    for r in rows:
        if r['status']=='completed':lines.append(f"| {r['conv']} | {r['ffn']} | {r['framework']} | {r['successes']} | {r['tokenizer_mae_valid']:.6f} | {r['policy_best_step']} |")
        else:lines.append(f"| {r['case']} | — | — | {r['status']} | — | — |")
    lines+=['','所有失败组保留日志。单训练种子，不能把最优组合直接称为稳定提升。收敛曲线、分任务结果和耗时边界见JSON。','ResNet的ReLU及FSQ有界变换不变，Transformer激活同时用于分词器编码器/解码器与条件策略。']
    target.with_suffix('.md').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,6))
    for ax,f in zip(axes,('torch','jittor')):
        matrix=np.full((4,3),np.nan)
        for r in rows:
            if r.get('framework')==f and r['status']=='completed':matrix[p['conv_activations'].index(r['conv']),p['ffn_activations'].index(r['ffn'])]=r['success_rate']
        ax.imshow(matrix,vmin=0,vmax=1,cmap='viridis');ax.set_xticks(range(3),p['ffn_activations']);ax.set_yticks(range(4),p['conv_activations']);ax.set_title(f);ax.set_xlabel('Transformer FFN');ax.set_ylabel('Temporal convolution')
        for i in range(4):
            for j in range(3):ax.text(j,i,'failed' if np.isnan(matrix[i,j]) else f'{matrix[i,j]:.1%}',ha='center',va='center',color='white')
    fig.tight_layout();fig.savefig(target.with_suffix('.png'),dpi=180);plt.close(fig)
    print(json.dumps({'status':report['status'],'rows':len(rows),'descriptive_effects':effects}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--case');p.add_argument('--summarize',action='store_true');p.add_argument('--resume-scheduler',action='store_true');a=p.parse_args();root=a.root.resolve()
    if a.case:worker(root,a.case)
    elif a.summarize:summarize(root)
    else:supervisor(root,a.resume_scheduler)
