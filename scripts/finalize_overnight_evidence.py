"""Wait for both live workflows,then audit,render and archive full evidence."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1];NIGHT=ROOT/'outputs/overnight_20261029';OUT=NIGHT/'experiments'
state=ROOT/'reports/overnight_closeout_20261029.json'
report={'status':'waiting','stages':[]};state.write_text(json.dumps(report));start=time.monotonic()
def save():state.write_text(json.dumps(report,indent=2))
def stage(name,command,timeout):
    entry={'name':name,'status':'running','command':[str(x) for x in command]};report['stages'].append(entry);save()
    with (OUT/(name+'.log')).open('wb') as f:code=subprocess.run(entry['command'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=timeout).returncode
    entry.update(status='passed' if code==0 else 'failed',exit_code=code);save()
    if code:raise RuntimeError(f'{name} failed;inspect{OUT/name}.log')
try:
    while True:
        a=json.loads((OUT/'workflow.json').read_text());b=json.loads((OUT/'native_factorial_workflow.json').read_text())
        if a['status']=='failed' or b['status']=='failed':raise RuntimeError('An experiment workflow failed;repair before closeout')
        if a['status']=='completed' and b['status']=='completed':break
        if time.monotonic()-start>18000:raise TimeoutError('Closeoutwait exceeded5hours')
        time.sleep(20)
    report['status']='running';save()
    stage('completion_audit_report',[ROOT/'.venv-jittor-linux/bin/python',ROOT/'scripts/summarize_overnight_experiments.py','--root',ROOT,'--output',ROOT/'reports/OVERNIGHT_RESULTS_20261029.md'],240)
    stage('saved_prediction_media',[ROOT/'.venv-jittor-linux/bin/python',ROOT/'scripts/replay_overnight_media.py'],240)
    archive=ROOT/'reports/overnight_all_results_20261029.zip';files=[]
    for path in NIGHT.rglob('*'):
        if not path.is_file():continue
        relative=path.relative_to(NIGHT)
        if path.suffix in ['.hdf5','.pkl']:continue
        if 'raw' in relative.parts or path.name in ['last.pt','last_weights.npz']:continue
        if relative.parts[0].startswith('collect_') and 'videos' in relative.parts:continue
        files.append(path)
    for folder in ['src/quest_repro','requirements','licenses']:
        files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    files.extend(p for p in (ROOT/'scripts').glob('*.py') if p.name!='remote_session.py')
    files.extend((ROOT/'reports').glob('OVERNIGHT_RESULTS_20261029*'))
    for name in ['records/overnight_protocol_20261029.json','records/overnight_native_factorial_extension_20261029.json','LICENSE','THIRD_PARTY_NOTICES.md','pyproject.toml']:files.append(ROOT/name)
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for path in sorted(set(files)):z.write(path,path.relative_to(ROOT))
    digest=hashlib.sha256()
    with archive.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):digest.update(block)
    report.update(status='completed',archive={'bytes':archive.stat().st_size,'sha256':digest.hexdigest()},elapsed_seconds=time.monotonic()-start);save()
    print(json.dumps(report,indent=2),flush=True)
except BaseException as error:
    report.update(status='failed',error=f'{type(error).__name__}:{error}');save();raise
