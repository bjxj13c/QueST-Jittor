"""Prepare exactly450training/100validation episodes without changing old data."""
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'outputs/five_task_10pct_20261029'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def goal(group):
    return group['observations'][0][-3:].astype(np.float32).astype(np.float64).round(7).tobytes()

def convert(source,target):
    actions=source['actions'][:].astype(np.float32);n=len(actions)
    obs=source['obs/obs_gt'][:];states=source['obs/robot_states'][:];images=source['obs/corner_rgb'][:]
    assert n>0 and actions.shape==(n,4) and images.shape==(n+1,128,128,3) and images.dtype==np.uint8
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

def main():
    p=json.loads((ROOT/'records/five_task_10pct_20261029.json').read_text());tasks=p['tasks']
    old=ROOT/'outputs/overnight_20261029/datasets/multi240'
    m=json.loads((old/'records/reach_split_v1.json').read_text());assert m['task_names']==tasks[:3]
    folder=RUN/'dataset';(folder/'data').mkdir(parents=True,exist_ok=False);(folder/'records').mkdir()
    path=folder/'data/demonstrations.hdf5'
    train=list(m['train']);val=list(m['validation']);ids=dict(m['episode_task_ids']);sources=[]
    assert digest(old/m['dataset'])==m['sha256']
    with h5py.File(path,'x') as out:
        with h5py.File(old/m['dataset'],'r') as f:
            for name in f:f.copy(name,out,name=name)
        for collection,task_list,extra in [('collect_new_two',tasks[3:],False),('collect_extra_ten',tasks,True)]:
            for task in task_list:
                source=RUN/collection/f'raw/metaworld/ML45_PRISE/train/{task}.hdf5'
                sources.append({'path':str(source),'sha256':digest(source),'task':task})
                new=[]
                with h5py.File(source,'r') as f:
                    assert len(f['data'])==(10 if extra else 100)
                    for name in sorted(f['data'],key=lambda s:int(s.split('_')[-1])):
                        dest=f'demo_{len(out)}';convert(f['data'][name],out.create_group(dest))
                        ids[dest]=tasks.index(task);new.append(dest)
                        out[dest].attrs['source_path']=str(source);out[dest].attrs['source_episode']=name
                if extra:
                    # Do not silently drop/replace conflicting data to reach the target count.
                    held={goal(out[name]) for name in val if ids[name]==tasks.index(task)}
                    assert not held & {goal(out[name]) for name in new},f'New train goal overlaps validation:{task}'
                    train.extend(new)
                else:
                    groups={}
                    for name in new:groups.setdefault(goal(out[name]),[]).append(name)
                    assert len(groups)==50 and all(len(v)==2 for v in groups.values()),f'Unexpected goal grouping:{task}'
                    keys=list(groups);order=np.random.default_rng(p['split_seed']+tasks.index(task)).permutation(50)
                    train.extend(name for i in order[:40] for name in groups[keys[i]])
                    val.extend(name for i in order[40:] for name in groups[keys[i]])
        assert len(train)==450 and len(val)==100 and not set(train)&set(val)
        summary={}
        for i,task in enumerate(tasks):
            a=[ep for ep in train if ids[ep]==i];b=[ep for ep in val if ids[ep]==i]
            assert len(a)==90 and len(b)==20 and not {goal(out[e]) for e in a}&{goal(out[e]) for e in b}
            failed=[ep for ep in a+b if not bool(out[ep]['success'][:].any())]
            summary[task]={'train_episodes':90,'validation_episodes':20,
                'train_goals':len({goal(out[e]) for e in a}),'validation_goals':len({goal(out[e]) for e in b}),
                'train_steps':sum(len(out[e]['actions']) for e in a),'validation_steps':sum(len(out[e]['actions']) for e in b),
                'failed_episodes':failed,'failed_steps':sum(len(out[e]['actions']) for e in failed)}
        stats={'source':'training episodes only','applied':False}
        for key in ('actions','robot_states'):
            values=np.concatenate([out[e][key][:] for e in train])
            stats[key]={s:getattr(values,s)(axis=0).tolist() for s in ('mean','std','min','max')}
    manifest={'dataset':'data/demonstrations.hdf5','sha256':digest(path),'train':train,'validation':val,
        'task_names':tasks,'episode_task_ids':ids,'chunk_length':32,'stride':1,
        'padding':'repeat final action within same episode','split_seed':p['split_seed'],
        'provenance':{'old_root':str(old),'old_sha256':m['sha256'],'new_sources':sources,'old_split_preserved':True},
        'scope':'450actualtraining episodes=10%author4500;notfullpaperbenchmark'}
    for name,data in [('records/reach_split_v1.json',manifest),('records/reach_train_statistics_v1.json',stats),('data_summary.json',summary)]:
        (folder/name).write_text(json.dumps(data,indent=2))
    (RUN/'data_preparation_completed.json').write_text(json.dumps({'status':'completed','train':450,'validation':100,'fraction_of_author_pretraining':.1,'sha256':manifest['sha256']},indent=2))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
