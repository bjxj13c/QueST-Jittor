"""Close the one pending baseline holding branch using preserved checkpoints.

Recheck original implementation/tolerances first. No training,no overwrites.
If strict check fails,this workflow records it and does not bypass the gate.
"""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import zipfile
import numpy as np
from run_activation_grid import read,sha,write,variant

def main():
    import fcntl
    root=Path(__file__).resolve().parents[1];out=root/'outputs/holding_completion_20261030';out.mkdir(exist_ok=False)
    lock=(out/'workflow.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    unit=root/'outputs/selected_activation_holding_20261030/units/mish_gelu__holding';assets=unit/'assets';data=unit/'dataset';p=read(root/'records/holding_completion_inputs_20261030.json')
    state={'status':'validating_clone','started_utc':datetime.now(timezone.utc).isoformat(),'stages':[],'no_retraining':True,'server_shutdown':False};started=time.monotonic()
    env=os.environ.copy();env.update(PYTHONPATH=str(root/'src'),MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',NVIDIA_TF32_OVERRIDE='0',OMP_NUM_THREADS='4')
    def save():state['elapsed_seconds']=time.monotonic()-started;write(out/'workflow.json',state)
    def stage(name,cmd,cap=1800):
        row={'name':name,'status':'running','command':[str(v) for v in cmd]};state['stages'].append(row);save()
        with (out/(name+'.log')).open('xb') as log:
            child=subprocess.Popen(row['command'],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);row['pid']=child.pid;save()
            try:code=child.wait(timeout=cap)
            except BaseException:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise
        row.update(status='passed' if code==0 else 'failed',exit_code=code);save()
        if code:raise RuntimeError(f'{name} failed;preserved logs')
    try:
        save();assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<512
        for path,digest in p['file_hashes'].items():assert sha(unit/path)==digest,path
        for path,digest in p['external_sources'].items():assert sha(root/path)==digest,path
        write(out/'inputs_verified.json',p);state['status']='strict_original_alignment';save()
        for f in ('torch','jittor'):
            stage('original_fixture_'+f,variant(root,f,'mish','gelu','export_trained_policy_predictions.py')+['--root',data,'--framework',f,
                '--policy-weights',unit/'policy_jittor/best_weights.npz','--policy-config',unit/'policy_jittor/config.json',
                '--tokenizer-weights',assets/'weights.npz','--tokenizer-config',assets/'config.json','--output',out/f'original_fixture_{f}.npz'],1800)
        comparison={};failed=False
        with np.load(out/'original_fixture_torch.npz') as a,np.load(out/'original_fixture_jittor.npz') as b:
            for k in a.files:
                error=np.abs(a[k]-b[k]);bad=error>(p['atol']+p['rtol']*np.abs(b[k]));comparison[k]={'max_abs':float(error.max()),'mismatches':int(bad.sum()),'exact':bool(np.array_equal(a[k],b[k]))}
                failed|=bool(bad.any())
            failed|=not np.array_equal(a['tokens'],b['tokens'])
        write(out/'strict_original_comparison.json',{'atol':p['atol'],'rtol':p['rtol'],'results':comparison,'passed':not failed,'original_implementation_unchanged':True})
        if failed:state.update(status='needs_precision_repair',reason='Original pairwise tolerance failed;no evaluation launched');save();return
        state['status']='evaluating';save();names={};tasks=['reach-v2','reach-wall-v2','push-v2','door-open-v2','window-open-v2']
        for f in ('torch','jittor'):
            names[f]=[]
            for task_id,task in enumerate(tasks):
                name=f'completion_mish_gelu_holding_{f}_{task}_20261030';names[f].append(name)
                if f=='torch':cmd=variant(root,f,'mish','gelu','evaluate_policy.py')+['--policy',unit/'policy_torch/best.pt','--tokenizer',assets/'tokenizer_shared.pt','--sampling-backend','numpy']
                else:cmd=variant(root,f,'mish','gelu','evaluate_policy_jittor.py')+['--policy-weights',unit/'policy_jittor/best_weights.npz','--policy-config',unit/'policy_jittor/config.json','--tokenizer-weights',assets/'weights.npz','--tokenizer-config',assets/'config.json']
                stage(name,cmd+['--root',data,'--device','cuda','--seed-set','experiment','--seed-start','120000','--episodes','30','--env-name',task+'-goal-observable','--task-id',str(task_id),
                    '--per-episode-sampling','--top-k','5','--sampling-seed','130000','--max-steps','150','--action-horizon','8','--videos','0','--name',name],1800)
            stage('holding_diagnostic_'+f,variant(root,f,'mish','gelu','evaluate_selected_holding.py')+['--framework',f,'--policy-dir',unit/f'policy_{f}','--assets',assets,
                '--output',out/f'holding_diagnostic_{f}','--seed-start','120000','--sampling-seed','130000'],1800)
        original=read(root/'reports/SELECTED_HOLDING_RESULTS_20261030.json');updated=read(root/'reports/SELECTED_HOLDING_RESULTS_20261030.json');result={}
        for f in ('torch','jittor'):
            outcomes=[]
            for task_id,name in enumerate(names[f]):
                s=read(data/'outputs'/name/'summary.json');assert [e['seed'] for e in s['episodes']]==list(range(120000,120030)) and s['task_id']==task_id
                if f=='jittor':assert not s['torch_imported']
                outcomes.append([int(e['success']) for e in s['episodes']])
            d=read(out/f'holding_diagnostic_{f}/summary.json');assert outcomes[0]==[int(e['success_within150']) for e in d['episodes']];post=[e for e in d['episodes'] if e['post_steps']==60]
            if f=='jittor':assert not d['torch_imported']
            result[f]={'successes':int(np.sum(outcomes)),'episodes':150,'rate':float(np.mean(outcomes)),'outcomes':outcomes,'task_successes':[sum(v) for v in outcomes],
                'reach_hold_all60_rate':d['hold_all60_unconditional'],'reach_post_success_fraction_conditional':float(np.mean([e['post_success_fraction'] for e in post])) if post else None,
                'reach_tcp_drift60_conditional':float(np.mean([e['tcp_drift_60'] for e in post])) if post else None,'tokenizer_reconstruction':read(unit/f'tokenizer_{f}/best_validation.json')}
        for path,digest in p['file_hashes'].items():assert sha(unit/path)==digest,path
        updated['results']['mish_gelu']['holding']=result;updated['failures']=[];updated['status']='audited_with_separate_completion'
        updated['completion']={'protocol':p,'strict_alignment':comparison,'actual_added_rollouts':360,'no_retraining':True,'old_failed_report_sha256':sha(root/'reports/SELECTED_HOLDING_RESULTS_20261030.json'),'source':'new clone original implementation recheck;original failure and report preserved'}
        target=root/'reports/SELECTED_HOLDING_RESULTS_COMPLETED_20261030.json';write(target,updated)
        lines=['# 原始激活保持分支补测','','原权重、原实现、原容差通过新实例复查后，完成两框架360次评估；没有重训，也没有覆盖原失败记录。','','|框架|无保持/150|有保持/150|Reach持续保持率：无→有|','|---|---:|---:|---:|']
        for f in ('torch','jittor'):
            a=original['results']['mish_gelu']['repeat'][f];b=result[f];lines.append(f"|{f}|{a['successes']}|{b['successes']}|{a['reach_hold_all60_rate']:.1%}→{b['reach_hold_all60_rate']:.1%}|")
        lines+=['','仍为单个微调种子；原GPU实例与克隆实例的数值行为可能不同，不据此宣称稳定提升。']
        (root/'reports/HOLDING_COMPLETION_20261030.md').write_text('\n'.join(lines),encoding='utf-8')
        archive=root/'reports/holding_completion_20261030.zip'
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
            for q in out.rglob('*'):
                if q.is_file() and q.name not in ('workflow.lock','workflow.tmp'):z.write(q,q.relative_to(root))
            for f in ('torch','jittor'):
                for name in names[f]:
                    for q in (data/'outputs'/name).rglob('*'):
                        if q.is_file():z.write(q,q.relative_to(root))
            for q in [target,root/'reports/HOLDING_COMPLETION_20261030.md',Path(__file__).resolve(),root/'records/holding_completion_inputs_20261030.json']:z.write(q,q.relative_to(root))
        state.update(status='completed',archive={'path':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive)})
    except BaseException as e:state.update(status='failed',error=f'{type(e).__name__}: {e}');raise
    finally:save()
if __name__=='__main__':main()
