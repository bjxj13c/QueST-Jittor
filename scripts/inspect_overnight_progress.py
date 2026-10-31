"""Compact live-process plus experiment-state snapshot,without credentials."""
from pathlib import Path
import json
import subprocess
ROOT=Path(__file__).resolve().parents[1];folder=ROOT/'outputs/overnight_20261029/experiments'
x=json.loads((folder/'workflow.json').read_text())
print('STATUS',x['status'],'elapsed',x.get('elapsed_seconds'))
print('STAGES',[(s['name'],s['status']) for s in x['stages']]);print('ERROR',x.get('error'))
print('SELECTED_BY_VALIDATION',x.get('selected_single_case_by_validation'))
extension=folder/'native_factorial_workflow.json'
if extension.exists():
    e=json.loads(extension.read_text());print('NATIVE_MATRIX',e['status'],[(s['name'],s['status']) for s in e['stages']],e.get('error'))
    if e.get('results'):print('NATIVE_MATRIX_COUNTS',{name:sum(ep['success'] for ep in result['evaluation']['episodes']) for name,result in e['results'].items()})
for field in ['single_results','multi_results']:
    if field in x:print(field,{name:sum(ep['success'] for ep in result['episodes']) for name,result in x[field].items()})
for name in ['small_base','large_base','small_tuned','large_tuned','selected_native','multi_tokenizer5','multi_torch','multi_jittor']:
    p=folder/(name+'.log')
    if p.exists():print(name,'\n'.join(p.read_text(errors='replace').splitlines()[-2:]))
targets=['run_overnight_experiments.py','run_native_factorial_extension.py','train_controlled_policy.py','train_tokenizer_jittor.py','evaluate_policy.py','evaluate_policy_jittor.py']
processes=[]
for line in subprocess.check_output(['ps','-eo','pid,etime,args'],text=True).splitlines():
    fields=line.strip().split(None,2)
    if len(fields)==3 and '.venv-' in fields[2].split()[0] and 'python' in fields[2].split()[0]:
        matches=[name for name in targets if name in fields[2]]
        if matches:processes.append({'pid':int(fields[0]),'elapsed':fields[1],'script':matches})
print('LIVE_PROCESSES',json.dumps(processes))
