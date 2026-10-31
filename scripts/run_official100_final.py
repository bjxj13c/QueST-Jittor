"""Freeze using DEVELOPMENT evidence, then evaluate reserved seeds once."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'outputs/official_reach_100_20261028/adapted'
OUT=ROOT/'outputs/official100_policy_20261028'
PREFIX='official100_policy_20261028'
assert json.loads((OUT/'workflow.json').read_text())['status']=='completed'
assert json.loads((OUT/'sampling5_comparison.json').read_text())['tokens_exact']
candidates=[]
for label,horizon,k,suffix_t,suffix_j in [('greedy_h8',8,1,'_dev10','_dev10'),('greedy_h4',4,1,'_dev10_h4','_dev10_h4'),('top5_h8',8,5,'_dev10_top5_numpy','_dev10_top5')]:
    counts={}
    for framework,suffix in [('torch',suffix_t),('jittor',suffix_j)]:
        report=json.loads((DATA/'outputs'/f'{PREFIX}_{framework}{suffix}'/'summary.json').read_text())
        assert report['seed_set']=='development' and len(report['episodes'])==10
        counts[framework]=sum(ep['success'] for ep in report['episodes'])
    candidates.append({'label':label,'horizon':horizon,'top_k':k,'development_successes':counts,'combined_successes':sum(counts.values())})
choice=max(candidates,key=lambda c:(c['combined_successes'],c['label']=='top5_h8'))
assets=DATA/'outputs'/f'{PREFIX}_assets'
models={'torch':DATA/'outputs'/f'{PREFIX}_torch/best.pt','jittor':DATA/'outputs'/f'{PREFIX}_jittor/best_weights.npz'}
sources=['src/quest_repro/sampling.py','src/quest_repro/policy_torch.py','src/quest_repro/policy_jittor.py','scripts/evaluate_policy.py','scripts/evaluate_policy_jittor.py']
protocol={'frozen_utc':datetime.now(timezone.utc).isoformat(),'selection':'exploratory development comparison,highest combined successes;tie prefers author-style top5/h8;all choices before final tests',
    'candidates':candidates,'choice':choice,'seeds':list(range(1000,1020)),
    'sampler':'shared NumPy PCG64 inverse CDF,seed30000;temperature1;greedy iftopk1',
    'max_steps':150,'tokenizer_weights_sha256':hashlib.sha256((assets/'weights.npz').read_bytes()).hexdigest(),
    'policy_sha256':{name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in models.items()},
    'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},
    'commitment':'no tuning,checkpoint changes,seed changes or protocol selection based on these final outcomes'}
with (OUT/'final_protocol.json').open('x') as f:json.dump(protocol,f,indent=2)
print('FROZEN_PROTOCOL',json.dumps(protocol),flush=True)
env=os.environ.copy();env.update(PYTHONPATH=str(ROOT/'src'),NVIDIA_TF32_OVERRIDE='0',OMP_NUM_THREADS='4',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl')
started=time.monotonic();workflow={'status':'running','protocol':protocol,'stages':[]}
def save():
    workflow['elapsed_seconds']=time.monotonic()-started
    (OUT/'final_workflow.json').write_text(json.dumps(workflow,indent=2))
try:
    for name in ('torch','jittor'):
        interpreter=ROOT/('.venv-torch-linux/bin/python' if name=='torch' else '.venv-jittor-linux/bin/python')
        if name=='torch':
            cmd=[interpreter,ROOT/'scripts/evaluate_policy.py','--policy',models[name],'--tokenizer',assets/'shared/tokenizer_shared.pt','--sampling-backend','numpy']
        else:
            cmd=[interpreter,ROOT/'scripts/run_jittor_linux.py','--device','cuda',ROOT/'scripts/evaluate_policy_jittor.py',
                '--policy-weights',models[name],'--policy-config',models[name].parent/'config.json',
                '--tokenizer-weights',assets/'weights.npz','--tokenizer-config',assets/'config.json']
        cmd+=['--root',DATA,'--device','cuda','--seed-set','final','--episodes','20','--videos','0',
            '--action-horizon',str(choice['horizon']),'--top-k',str(choice['top_k']),'--sampling-seed','30000','--max-steps','150','--name',f'{PREFIX}_{name}_final20']
        stage={'name':name,'status':'running','command':[str(x) for x in cmd]};workflow['stages'].append(stage);save()
        with (OUT/f'final_{name}.log').open('xb') as log:
            code=subprocess.run(stage['command'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=240).returncode
        stage.update(status='passed' if code==0 else 'failed',exit_code=code);save()
        if code:raise RuntimeError(f'{name} final evaluation failed;preserve partial evidence')
    for name,path in models.items():assert hashlib.sha256(path.read_bytes()).hexdigest()==protocol['policy_sha256'][name]
    for name,value in protocol['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value
    workflow['results']={name:json.loads((DATA/'outputs'/f'{PREFIX}_{name}_final20/summary.json').read_text()) for name in models}
    workflow['status']='completed';save()
    archive=ROOT/'reports/official100_final_and_ablations_20261028.zip'
    extra_dirs=[OUT/'torch_tcp_diagnostics',OUT/'jittor_tcp_diagnostics']
    for name in models:
        for suffix in ['_dev10_h4','_dev10_top5','_final20']:
            extra_dirs.append(DATA/'outputs'/f'{PREFIX}_{name}{suffix}')
    extra_dirs.append(DATA/'outputs'/f'{PREFIX}_torch_dev10_top5_numpy')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for folder in extra_dirs:
            for path in folder.rglob('*'):
                if path.is_file():z.write(path,path.relative_to(ROOT))
        for pattern in ['final*','sampling5*']:
            for path in OUT.glob(pattern):
                if path.is_file():z.write(path,path.relative_to(ROOT))
        for name in sources:z.write(ROOT/name,name)
    metadata={'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}
    (ROOT/'reports/official100_final_archive.json').write_text(json.dumps(metadata,indent=2))
    print('FINAL_RESULTS',{name:report['success_rate'] for name,report in workflow['results'].items()},flush=True)
except BaseException as error:
    workflow.update(status='failed',error=f'{type(error).__name__}: {error}');raise
finally:save()
