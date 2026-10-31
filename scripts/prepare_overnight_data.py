"""Prepare nested data controls and3task data,keeping old split and all failures."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import h5py
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
NIGHT=ROOT/'outputs/overnight_20261029'
OLD=ROOT/'outputs/official_reach_100_20261028/adapted'
old_manifest=json.loads((OLD/'records/reach_split_v1.json').read_text())
old_path=OLD/old_manifest['dataset']

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def convert_group(source,target):
    actions=source['actions'][:].astype(np.float32);n=len(actions)
    obs=source['obs/obs_gt'][:];states=source['obs/robot_states'][:]
    images=source['obs/corner_rgb'][:]
    assert actions.shape==(n,4) and images.shape==(n+1,128,128,3) and images.dtype==np.uint8
    assert obs.shape==(n+1,39) and states.shape==(n+1,8)
    np.testing.assert_allclose(states,np.concatenate((obs[:,:4],obs[:,18:22]),axis=1),atol=1e-7)
    assert np.isfinite(actions).all() and np.abs(actions).max()<=1
    target.create_dataset('images',data=images[:-1],compression='lzf',chunks=(1,128,128,3))
    for key,value in {'actions':actions,'observations':obs[:-1],'robot_states':states[:-1].astype(np.float32),
        'rewards':source['reward'][:],'success':source['success'][:],
        'terminated':source['terminated'][:],'truncated':source['truncated'][:]}.items():
        assert len(value)==n and np.isfinite(value).all()
        target.create_dataset(key,data=value,compression='lzf')
    target.attrs['episode_success']=bool(source['success'][:].any())

def initial_key(group):
    # Hold out goals,not merely noisy duplicate executions of identical starts.
    return group['observations'][0][-3:].astype(np.float64).round(7).tobytes()

def finish(folder,train,val,tasks,task_ids,provenance):
    path=folder/'data/demonstrations.hdf5';records=folder/'records';records.mkdir()
    assert not set(train)&set(val)
    summary={};stats={'source':'training episodes only','applied':False}
    with h5py.File(path,'r') as f:
        assert set(f)==set(train+val)
        for task_id,task in enumerate(tasks):
            a=[ep for ep in train if task_ids[ep]==task_id];b=[ep for ep in val if task_ids[ep]==task_id]
            assert not {initial_key(f[ep]) for ep in a}&{initial_key(f[ep]) for ep in b}
            summary[task]={'train_episodes':len(a),'validation_episodes':len(b),
                'train_goals':len({initial_key(f[ep]) for ep in a}),
                'validation_goals':len({initial_key(f[ep]) for ep in b}),
                'train_steps':sum(len(f[ep]['actions']) for ep in a),
                'validation_steps':sum(len(f[ep]['actions']) for ep in b),
                'expert_successes':sum(bool(f[ep]['success'][:].any()) for ep in a+b)}
        for key in ('actions','robot_states'):
            values=np.concatenate([f[ep][key][:] for ep in train])
            stats[key]={stat:getattr(values,stat)(axis=0).tolist() for stat in ('mean','std','min','max')}
    manifest={'dataset':'data/demonstrations.hdf5','sha256':digest(path),'train':train,'validation':val,
        'chunk_length':32,'stride':1,'padding':'repeat final action within same episode',
        'split_seed':304090,'task_names':tasks,'episode_task_ids':task_ids,'provenance':provenance,
        'scope':'overnight controlled data expansion/multitask experiments;not paper45tasks'}
    (records/'reach_split_v1.json').write_text(json.dumps(manifest,indent=2))
    (records/'reach_train_statistics_v1.json').write_text(json.dumps(stats,indent=2))
    (folder/'data_summary.json').write_text(json.dumps(summary,indent=2));print(folder.name,json.dumps(summary),flush=True)

base=NIGHT/'datasets/reach80';(base/'data').mkdir(parents=True,exist_ok=False)
os.link(old_path,base/'data/demonstrations.hdf5')
ids={ep:0 for ep in old_manifest['train']+old_manifest['validation']}
finish(base,old_manifest['train'],old_manifest['validation'],['reach-v2'],ids,{'old_root':str(OLD),'reuse':'hardlink,read-only inputs;old split unchanged'})
for label,tasks,extra_sources in [
    ('reach280',['reach-v2'],[(NIGHT/'collect_reach_seed304001','reach-v2'),(NIGHT/'collect_reach_seed304002','reach-v2')]),
    ('multi240',['reach-v2','reach-wall-v2','push-v2'],[(NIGHT/'collect_multi_seed304003','reach-wall-v2'),(NIGHT/'collect_multi_seed304003','push-v2')])]:
    folder=NIGHT/'datasets'/label;(folder/'data').mkdir(parents=True,exist_ok=False)
    task_ids=dict(ids);train=list(old_manifest['train']);val=list(old_manifest['validation']);sources=[]
    with h5py.File(folder/'data/demonstrations.hdf5','x') as out:
        with h5py.File(old_path,'r') as original:
            for name in original:original.copy(name,out,name=name)
        for source_root,task in extra_sources:
            source=source_root/f'raw/metaworld/ML45_PRISE/train/{task}.hdf5'
            sources.append({'path':str(source),'sha256':digest(source)})
            new_names=[]
            with h5py.File(source,'r') as f:
                for old_name in sorted(f['data'],key=lambda s:int(s.split('_')[-1])):
                    name=f'demo_{len(out)}';convert_group(f['data'][old_name],out.create_group(name))
                    task_ids[name]=tasks.index(task);new_names.append(name)
            if label=='reach280':train.extend(new_names)
            else:
                groups={}
                for name in new_names:groups.setdefault(initial_key(out[name]),[]).append(name)
                if len(groups)!=50 or any(len(g)!=2 for g in groups.values()):raise ValueError('Review unexpected goal repetition before splitting')
                keys=list(groups);order=np.random.default_rng(304090+tasks.index(task)).permutation(len(keys))
                train.extend(name for i in order[:40] for name in groups[keys[i]])
                val.extend(name for i in order[40:] for name in groups[keys[i]])
    finish(folder,train,val,tasks,task_ids,{'old_root':str(OLD),'new_sources':sources,'old_reach_split_preserved':True})
(NIGHT/'data_preparation_completed.json').write_text(json.dumps({'status':'completed','datasets':['reach80','reach280','multi240']}))
