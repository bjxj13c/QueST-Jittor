"""Local CPU-only backup verifier;waits for authenticated transfer completion.

Does not connect remotely,request credentials,start training,or modify archives.
Existing evidence files must match byte-for-byte;never overwrite differences.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
TARGET=(ROOT/'reports/five_task_evidence_20261029').resolve()
STATE=ROOT/'reports/CLOSEOUT_BACKUP_STATUS_20261030.json'
ITEMS=[
 ('390_padding_full_backup_download','padding_hold_20261029.zip',1012181959,'d88c957d6263508d5654b0273373f218d290da43b89a60ecd208238cd9a9be6e'),
 ('391_activation_grid_full_backup_download','activation_grid_20261029.zip',5135599219,'f7b6dc611fcaa35942ff2b677cf1f7542d92010befbc6ae3f2e07cd7bcd7e6cc'),
 ('392_selected_holding_full_backup_download','selected_activation_holding_20261030.zip',1901325626,'3c4d858382a41bd1d57b9e149012b3925c760c58e2b269e5f6cfe10952c8fb39')]
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def save(v):
    v['updated_local']=time.strftime('%Y-%m-%d %H:%M:%S');q=STATE.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2),encoding='utf-8');q.replace(STATE)
def verify_merge(path,expected):
    assert digest(path)==expected,'Archive SHA mismatch';added=overlaps=0
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            dest=(TARGET/info.filename).resolve();assert dest.is_relative_to(TARGET),info.filename
        assert z.testzip() is None,'Archive CRC failure'
        for info in z.infolist():
            if info.is_dir():continue
            dest=(TARGET/info.filename).resolve()
            if dest.exists():
                h=hashlib.sha256()
                with z.open(info) as f:
                    for b in iter(lambda:f.read(1048576),b''):h.update(b)
                assert dest.stat().st_size==info.file_size and digest(dest)==h.hexdigest(),f'Preserve differing overlap:{dest}'
                overlaps+=1;continue
            dest.parent.mkdir(parents=True,exist_ok=True);temp=dest.with_name(dest.name+'.extracting');assert not temp.exists(),str(temp)
            with z.open(info) as source,temp.open('xb') as target:shutil.copyfileobj(source,target,1048576)
            assert not dest.exists();temp.replace(dest);added+=1
    return {'sha256':expected,'crc':'passed','safe_paths':'passed','added_files':added,'identical_overlaps':overlaps}
def padding_audit():
    base=TARGET/'outputs/padding_hold_20261029';p=json.loads((TARGET/'records/padding_hold_protocol_20261029.json').read_text());total=0
    for relative,expected in json.loads((base/'freeze.json').read_text()).items():assert digest(TARGET/relative)==expected,relative
    for c in p['conditions']:
        data=base/'collection'/('holding' if c=='holding' else 'repeat')
        for f in ('torch','jittor'):
            for kind,budget in [('tokenizer',2000),('policy',6000)]:
                folder=base/c/f'{kind}_{f}';s=json.loads((folder/'summary.json').read_text());assert s['updates']==budget
                rows=[json.loads(v) for v in (folder/'gradients.jsonl').read_text().splitlines()];assert [v['step'] for v in rows]==list(range(1,budget+1));key='gradient_norm' if kind=='tokenizer' else 'preclip_norm';assert np.isfinite([v[key] for v in rows]).all()
                if f=='jittor':assert not s['torch_imported']
            standard=json.loads((data/'outputs'/f'padding_{c}_{f}_standard_20261029/summary.json').read_text());diag=json.loads((base/c/f'holding_diagnostic_{f}/summary.json').read_text())
            assert len(standard['episodes'])==len(diag['episodes'])==30;assert [v['success'] for v in standard['episodes']]==[v['success_within150'] for v in diag['episodes']];total+=60
    assert total==360;return {'status':'passed','actual_rollouts':total,'finite_training_updates':48000}
def main():
    assert not STATE.exists(),'Verifier already has state;inspect before rerun';started=time.monotonic();state={'status':'waiting_for_transfers','archives':{},'audits':{},'no_gpu_or_training':True};save(state)
    try:
        for request,name,size,expected in ITEMS:
            result=ROOT/'records/remote_session/results'/(request+'.json');path=ROOT/'reports'/name
            while not result.exists():
                if time.monotonic()-started>6*3600:raise TimeoutError('Six-hour transfer wait expired')
                state.update(status='waiting_for_transfer',current_archive=name,downloaded_bytes=path.stat().st_size if path.exists() else 0,expected_bytes=size);save(state);time.sleep(15)
            transfer=json.loads(result.read_text());assert transfer['status']=='completed',transfer
            assert path.stat().st_size==size,(name,path.stat().st_size,size);state.update(status='verifying_archive',current_archive=name);save(state)
            state['archives'][name]=verify_merge(path,expected);save(state);print('Verified:',name,flush=True)
        state['status']='independent_cpu_audits';save(state);state['audits']['padding']=padding_audit();save(state)
        python=ROOT/'.venv-torch/Scripts/python.exe';env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='-1'
        for label,script,phase in [('grid','run_activation_grid.py','--summarize'),('followup','run_selected_holding.py','--summarize')]:
            log=ROOT/'records'/f'closeout_{label}_local_audit_20261030.log'
            with log.open('wb') as f:result=subprocess.run([str(python),str(TARGET/'scripts'/script),'--root',str(TARGET),phase],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1800)
            assert result.returncode==0,f'{label}audit failed;inspect{log}';state['audits'][label]={'status':'passed','log':str(log)};save(state)
        state.update(status='verified_all_archives_and_cpu_audits',known_unfinished_unit='mish_gelu__holding:strictGPUalignmentfailed,simulationnotrun',elapsed_seconds=time.monotonic()-started);save(state)
    except BaseException as e:state.update(status='failed',error=f'{type(e).__name__}: {e}');save(state);raise
if __name__=='__main__':main()
