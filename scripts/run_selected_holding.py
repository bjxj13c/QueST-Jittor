"""After grid completion,top3+baseline paired real-holding fine-tuning,max4workers."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import zipfile
import numpy as np
from run_activation_grid import read,sha,write,variant,runtime,environment,aligned,gpu_resource

def worker(root,name):
    out=root/'outputs/selected_activation_holding_20261030';p=read(root/'records/selected_holding_protocol_20261030.json');selection=read(out/'selection.json');unit=out/'units'/name;combo,arm=name.split('__');conv,ffn=combo.split('_');initial=selection['initializations'][combo]
    import fcntl
    lock=(unit/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (unit/'workflow.json').exists()
    state={'unit':name,'status':'running','stages':[],'started_utc':datetime.now(timezone.utc).isoformat()};started=time.monotonic();env=environment(root);data=unit/'dataset'
    def save():state['elapsed_seconds']=time.monotonic()-started;write(unit/'workflow.json',state)
    def stage(label,cmd,cap=3600):
        row={'name':label,'status':'running','command':[str(v) for v in cmd]};state['stages'].append(row);save()
        with (unit/(label+'.log')).open('xb') as f:
            child=subprocess.Popen(row['command'],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);row['pid']=child.pid;save()
            try:code=child.wait(timeout=cap)
            except BaseException:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise
        row.update(status='passed' if code==0 else 'failed',exit_code=code);save()
        if code:raise RuntimeError(f'{label} failed;preserved all logs')
    try:
        save()
        for key,digest in initial['hashes'].items():assert sha(root/initial[key])==digest
        for f in ('torch','jittor'):
            stage('tokenizer_'+f,variant(root,f,conv,ffn,'train_activation_tokenizer.py')+['--root',data,'--output',unit/f'tokenizer_{f}','--framework',f,
                '--initial',root/initial['tokenizer'],'--config',root/initial['tokenizer_config'],'--updates','2000','--eval-every','500','--seed','530500'])
        for f in ('torch','jittor'):
            stage('tokenizer_probe_'+f,variant(root,f,conv,ffn,'activation_grid_probe.py')+['--framework',f,'--data',data,'--weights',unit/'tokenizer_jittor/best_weights.npz',
                '--config',unit/'tokenizer_jittor/config.json','--output',unit/f'tokenizer_probe_{f}.npz'],600)
        aligned(unit/'tokenizer_probe_torch.npz',unit/'tokenizer_probe_jittor.npz')
        stage('assets',variant(root,'torch',conv,ffn,'prepare_selected_holding.py')+['--root',root,'--unit',name],900);assets=unit/'assets'
        for f in ('torch','jittor'):
            cmd=variant(root,f,conv,ffn,'train_activation_policy.py')+['--root',data,'--output',unit/f'policy_{f}','--framework',f,'--tokenizer-weights',assets/'weights.npz',
                '--tokenizer-config',assets/'config.json','--initial-weights',root/initial['policy'],'--canonical-labels',assets/'canonical_labels','--updates','4000','--eval-every','1000',
                '--lr','0.0001','--weight-decay','0.0001','--dropout','0','--shift','0','--seed','530500']
            if f=='torch':cmd+=['--tokenizer-checkpoint',assets/'tokenizer_shared.pt']
            stage('policy_'+f,cmd)
        for f in ('torch','jittor'):
            stage('policy_fixture_'+f,variant(root,f,conv,ffn,'export_trained_policy_predictions.py')+['--root',data,'--framework',f,'--policy-weights',unit/'policy_jittor/best_weights.npz',
                '--policy-config',unit/'policy_jittor/config.json','--tokenizer-weights',assets/'weights.npz','--tokenizer-config',assets/'config.json','--output',unit/f'policy_fixture_{f}.npz'],600)
        aligned(unit/'policy_fixture_torch.npz',unit/'policy_fixture_jittor.npz')
        write(unit/'evaluation_freeze.json',{'policies':{f:sha(unit/f'policy_{f}/best_weights.npz') for f in ('torch','jittor')},'tokenizer':sha(assets/'weights.npz')})
        for f in ('torch','jittor'):
            for task_id,task in enumerate(p['tasks']):
                label=f'selected_{name}_{f}_{task}'
                if f=='torch':cmd=variant(root,f,conv,ffn,'evaluate_policy.py')+['--policy',unit/'policy_torch/best.pt','--tokenizer',assets/'tokenizer_shared.pt','--sampling-backend','numpy']
                else:cmd=variant(root,f,conv,ffn,'evaluate_policy_jittor.py')+['--policy-weights',unit/'policy_jittor/best_weights.npz','--policy-config',unit/'policy_jittor/config.json','--tokenizer-weights',assets/'weights.npz','--tokenizer-config',assets/'config.json']
                stage(label,cmd+['--root',data,'--device','cuda','--seed-set','experiment','--seed-start','120000','--episodes','30','--env-name',task+'-goal-observable','--task-id',str(task_id),
                    '--per-episode-sampling','--top-k','5','--sampling-seed','130000','--max-steps','150','--action-horizon','8','--videos','0','--name',label],1800)
            stage('reach_holding_'+f,variant(root,f,conv,ffn,'evaluate_selected_holding.py')+['--framework',f,'--policy-dir',unit/f'policy_{f}','--assets',assets,'--output',unit/f'holding_diagnostic_{f}','--seed-start','120000','--sampling-seed','130000'],1800)
        audit_unit(root,name);state['status']='completed'
    except BaseException as e:state.update(status='failed',error=f'{type(e).__name__}: {e}');raise
    finally:save()

def audit_unit(root,name):
    out=root/'outputs/selected_activation_holding_20261030';unit=out/'units'/name;selection=read(out/'selection.json');combo,arm=name.split('__');conv,ffn=combo.split('_');initial=selection['initializations'][combo];p=read(root/'records/selected_holding_protocol_20261030.json');m=read(unit/'dataset/records/reach_split_v1.json')
    assert len(m['train'])==450 and len(m['validation'])==100 and not set(m['train']).intersection(m['validation']);assert sha(unit/'dataset'/m['dataset'])==m['sha256']
    for kind,budget in [('tokenizer',2000),('policy',4000)]:
        streams=[]
        for f in ('torch','jittor'):
            folder=unit/f'{kind}_{f}';s=read(folder/'summary.json');c=read(folder/'config.json');assert s['status']=='completed' and s['updates']==budget and c['conv_activation']==conv and c['ffn_activation']==ffn and c['dataset_sha256']==m['sha256']
            assert (c['initial_sha256'] if kind=='tokenizer' else c['initial_policy_npz_sha256'])==initial['hashes'][kind]
            assert s.get('finite',not s.get('nonfinite_loss_or_gradient',False));streams.append(s['batch_stream_sha256'])
            values=[json.loads(x) for x in (folder/'gradients.jsonl').read_text().splitlines()];assert [v['step'] for v in values]==list(range(1,budget+1));key='gradient_norm' if kind=='tokenizer' else 'preclip_norm';assert np.isfinite([v[key] for v in values]).all()
            if f=='jittor':assert not s['torch_imported']
        assert len(set(streams))==1
    for label in ('tokenizer_probe','policy_fixture'):aligned(unit/f'{label}_torch.npz',unit/f'{label}_jittor.npz')
    frozen=read(unit/'evaluation_freeze.json')
    for f,digest in frozen['policies'].items():assert sha(unit/f'policy_{f}/best_weights.npz')==digest
    assert sha(unit/'assets/weights.npz')==frozen['tokenizer']
    for f in ('torch','jittor'):
        for task_id,task in enumerate(p['tasks']):
            s=read(unit/'dataset/outputs'/f'selected_{name}_{f}_{task}'/'summary.json');assert [v['seed'] for v in s['episodes']]==list(range(120000,120030)) and s['horizon']==8 and s['top_k']==5 and s['sampling_seed']==130000 and s['task_id']==task_id
            if f=='jittor':assert not s['torch_imported']
        standard=read(unit/'dataset/outputs'/f'selected_{name}_{f}_reach-v2'/'summary.json');diagnostic=read(unit/f'holding_diagnostic_{f}/summary.json')
        assert [v['success'] for v in standard['episodes']]==[v['success_within150'] for v in diagnostic['episodes']]
        if f=='jittor':assert not diagnostic['torch_imported']
    write(unit/'audit.json',{'status':'passed','paired_warm_starts_verified':True,'finite_updates':12000,'actual_rollouts':360,'learned_weight_alignment_passed':True,'first_hit_diagnostic_consistent':True})

def supervisor(root):
    import fcntl
    out=root/'outputs/selected_activation_holding_20261030';out.mkdir(parents=True,exist_ok=True);lock=(out/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (out/'workflow.json').exists()
    p=read(root/'records/selected_holding_protocol_20261030.json');state={'status':'waiting_for_complete_activation_grid','started_utc':datetime.now(timezone.utc).isoformat(),'units':{},'max_workers':4,'server_shutdown':False,'09:00_stop':False};write(out/'workflow.json',state)
    started=time.monotonic();grid=root/'outputs/activation_grid_20261029/workflow.json';torch=root/'.venv-torch-linux/bin/python';env=environment(root)
    try:
        while True:
            source=read(grid)
            if source['status']=='completed' and (root/'reports/ACTIVATION_GRID_RESULTS_20261029.json').exists():break
            if source['status'] in ('failed','completed_with_failures'):state.update(status='blocked_on_incomplete_grid',reason='Need complete audited12cell results before selection');write(out/'workflow.json',state);return
            if time.monotonic()-started>48*3600:raise TimeoutError('48hwaitlimit;no shutdown')
            time.sleep(30)
        with (out/'selection_and_data.log').open('xb') as log:subprocess.run([str(torch),str(root/'scripts/prepare_selected_holding.py'),'--root',str(root)],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600,check=True)
        selection=read(out/'selection.json');files=[root/'records/selected_holding_protocol_20261030.json',out/'selection.json']+[root/'scripts'/n for n in p['scripts']]
        files+=list((root/'src/quest_repro').glob('*.py'));frozen={q.relative_to(root).as_posix():sha(q) for q in files};frozen.update(selection['external_sources'])
        for initial in selection['initializations'].values():
            for key,digest in initial['hashes'].items():frozen[initial[key]]=digest
        write(out/'freeze.json',frozen);pending=list(selection['units']);state.update(status='running',selected=selection['selected'],units={n:'queued' for n in pending});write(out/'workflow.json',state);children={}
        while pending or children:
            for name,entry in list(children.items()):
                code=entry['process'].poll()
                if code is not None:entry['log'].close();state['units'][name]='completed' if code==0 else 'failed';del children[name]
            resource=gpu_resource();state.update(resource=resource,active=list(children))
            if pending and len(children)<4 and resource['free_vram_mib']>=6144 and resource['disk_free_gib']>=3:
                name=pending.pop(0);log=(out/'units'/name/'worker_launcher.log').open('xb');child=subprocess.Popen(['nice','-n','10',str(torch),str(Path(__file__).resolve()),'--root',str(root),'--unit',name],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                children[name]={'process':child,'log':log};state['units'][name]='running'
            with (out/'resources.jsonl').open('a') as f:f.write(json.dumps({'unix_time':time.time(),**resource,'active':list(children)})+'\n')
            write(out/'workflow.json',state);time.sleep(10)
        summarize(root);state['status']='completed' if all(v=='completed' for v in state['units'].values()) else 'completed_with_failures';state['finished_utc']=datetime.now(timezone.utc).isoformat();write(out/'workflow.json',state)
        archive=root/'reports/selected_activation_holding_20261030.zip'
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
            for q in out.rglob('*'):
                if q.is_file() and q.name not in ('supervisor.lock','worker.lock','workflow.tmp'):z.write(q,q.relative_to(root))
            for q in [root/'records/selected_holding_protocol_20261030.json']+[root/'scripts'/n for n in p['scripts']]+list((root/'reports').glob('SELECTED_HOLDING_RESULTS_20261030.*')):z.write(q,q.relative_to(root))
        state['archive']={'path':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive)};write(out/'workflow.json',state)
    except BaseException as e:state.update(status='failed',error=f'{type(e).__name__}: {e}');write(out/'workflow.json',state);raise

def summarize(root):
    out=root/'outputs/selected_activation_holding_20261030';selection=read(out/'selection.json');p=read(root/'records/selected_holding_protocol_20261030.json');results={};failures=[]
    for name in selection['units']:
        unit=out/'units'/name;w=read(unit/'workflow.json') if (unit/'workflow.json').exists() else {'status':'not_started'}
        if w['status']!='completed':failures.append({'unit':name,**w});continue
        audit_unit(root,name);combo,arm=name.split('__');results.setdefault(combo,{})[arm]={}
        for f in ('torch','jittor'):
            outcomes=[]
            for task in p['tasks']:
                s=read(unit/'dataset/outputs'/f'selected_{name}_{f}_{task}'/'summary.json');outcomes.append([int(v['success']) for v in s['episodes']])
            d=read(unit/f'holding_diagnostic_{f}/summary.json');post=[v for v in d['episodes'] if v['post_steps']==60]
            results[combo][arm][f]={'successes':int(np.sum(outcomes)),'episodes':150,'rate':float(np.mean(outcomes)),'task_successes':[sum(x) for x in outcomes],'outcomes':outcomes,
                'reach_hold_all60_rate':d['hold_all60_unconditional'],'reach_post_success_fraction_conditional':float(np.mean([v['post_success_fraction'] for v in post])) if post else None,
                'reach_tcp_drift60_conditional':float(np.mean([v['tcp_drift_60'] for v in post])) if post else None,'tokenizer_reconstruction':read(unit/f'tokenizer_{f}/best_validation.json')}
    rng=np.random.default_rng(530510)
    for combo,arms in results.items():
        if not all(a in arms for a in ('repeat','holding')):continue
        arms['paired_effect']={}
        for f in ('torch','jittor'):
            delta=np.asarray(arms['holding'][f]['outcomes'])-np.asarray(arms['repeat'][f]['outcomes']);draws=np.stack([delta[i,rng.integers(0,30,(5000,30))].mean(1) for i in range(5)]).mean(0)
            arms['paired_effect'][f]={'success_rate_difference':float(delta.mean()),'exploratory_task_stratified_bootstrap95':np.percentile(draws,[2.5,97.5]).tolist(),
                'reach_hold60_rate_difference':arms['holding'][f]['reach_hold_all60_rate']-arms['repeat'][f]['reach_hold_all60_rate']}
    for path,digest in read(out/'freeze.json').items():assert sha(root/path)==digest,path
    report={'status':'audited' if not failures else 'partial_with_failures','selection':selection,'protocol':p,'results':results,'failures':failures,
        'limitations':['single fine-tuning seed;not stable improvement evidence','selected on earlier development grid;new scenes here do not retroactively validate all choices','paired warm-start botharms fromsame native checkpoints;not fresh from-scratchtraining','Reach replaces oldReachinbotharms;other4tasks unchanged','holding changes trainingdata quantity/distribution;fixedupdatesonly','conditionalholding comparedondifferent successfulscenes;unconditionalrateincluded','external HDF5links require preservedsourcefiles;allsourceSHAchecked']}
    target=root/'reports/SELECTED_HOLDING_RESULTS_20261030';target.with_suffix('.json').write_text(json.dumps(report,indent=2))
    lines=['# 优选激活组合＋真实保持示范配对实验','','| 组合 | 框架 | 无保持/150 | 有保持/150 | 无保持：Reach连续保持率 | 有保持：Reach连续保持率 |','|---|---|---:|---:|---:|---:|']
    for combo,arms in results.items():
        if not all(a in arms for a in ('repeat','holding')):continue
        for f in ('torch','jittor'):
            a,b=arms['repeat'][f],arms['holding'][f];lines.append(f"| {combo} | {f} | {a['successes']} | {b['successes']} | {a['reach_hold_all60_rate']:.1%} | {b['reach_hold_all60_rate']:.1%} |")
    lines+=['','按首轮两框架五任务平均成功率选前三组，额外保留Mish+GELU基线。五任务数据中仅Reach替换成配对新示范，其他四任务不变。','两组从同一已训练native检查点继续训练相同步数；新场景120000–120029，仅验证集选择检查点。单次微调种子，不能称为稳定提升。',f'未完成或失败单元数：{len(failures)}，完整错误保留于JSON。']
    target.with_suffix('.md').write_text('\n'.join(lines),encoding='utf-8');print(json.dumps({'status':report['status'],'selected':selection['selected'],'failures':len(failures)}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--unit');p.add_argument('--summarize',action='store_true');a=p.parse_args();root=a.root.resolve()
    if a.unit:worker(root,a.unit)
    elif a.summarize:summarize(root)
    else:supervisor(root)
