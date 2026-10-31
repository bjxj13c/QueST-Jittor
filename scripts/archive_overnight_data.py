"""Stream immutable collected/prepared inputs into a separate verifiable archive."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1];NIGHT=ROOT/'outputs/overnight_20261029'
files=[]
for label in ['reach80','reach280','multi240']:
    folder=NIGHT/'datasets'/label
    files+=[folder/'data/demonstrations.hdf5',folder/'data_summary.json']
    files+=list((folder/'records').glob('*.json'))
for name in ['collect_reach_seed304001','collect_reach_seed304002','collect_multi_seed304003']:
    folder=NIGHT/name;files+=list((folder/'raw').rglob('*.hdf5'))
    files+=[folder/'provenance.json',folder/'collection_config.yaml']
files+=[NIGHT/'test_isolation.json',NIGHT/'data_preparation_completed.json']
assert all(p.is_file() and p.resolve().is_relative_to(NIGHT) for p in files)
size=sum(p.stat().st_size for p in files)
if shutil.disk_usage(ROOT).free<size+1024**3:raise RuntimeError('Insufficient free disk for safe dataset backup')
archive=ROOT/'reports/overnight_data_20261029.zip';manifest=[]
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
    for path in files:
        digest=hashlib.sha256()
        with path.open('rb') as src,z.open(path.relative_to(ROOT).as_posix(),'w',force_zip64=True) as dst:
            while True:
                block=src.read(4*1024*1024)
                if not block:break
                digest.update(block);dst.write(block)
        manifest.append({'path':path.relative_to(ROOT).as_posix(),'bytes':path.stat().st_size,'sha256':digest.hexdigest()})
        print('ARCHIVED',path.relative_to(ROOT),flush=True)
    z.writestr('overnight_data_manifest.json',json.dumps(manifest,indent=2))
digest=hashlib.sha256()
with archive.open('rb') as f:
    for block in iter(lambda:f.read(4*1024*1024),b''):digest.update(block)
record={'status':'completed','bytes':archive.stat().st_size,'sha256':digest.hexdigest(),'files':len(files),'input_bytes':size}
(ROOT/'reports/overnight_data_archive_20261029.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
