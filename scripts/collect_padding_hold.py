"""Paired Reach prefixes and genuine expert-controlled post-success continuation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import h5py

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from quest_repro.simulation import ReachSimulation

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--episodes',type=int,default=110);a=p.parse_args()
    assert a.episodes in (2,110);a.output.mkdir(parents=True,exist_ok=False)
    from metaworld.policies import SawyerReachV2Policy
    destinations={}
    for condition in ('repeat','holding'):
        root=a.output/condition;(root/'data').mkdir(parents=True);(root/'records').mkdir();destinations[condition]=h5py.File(root/'data/reach.hdf5','x')
    rows=[]
    try:
        for i in range(a.episodes):
            seed=60000+i;rng=np.random.default_rng(70000+i);records={k:[] for k in ('images','robot_states','observations','actions','success')};first=None
            with ReachSimulation(seed) as env:
                observation,_=env.reset();expert=SawyerReachV2Policy();goal=np.asarray(env.env._target_pos).copy()
                for step in range(210):
                    raw=env.observation.copy();action=np.clip(expert.get_action(raw)+rng.normal(0,.1,4),-1,1).astype(np.float32)
                    records['images'].append(observation['images'][0,0].transpose(1,2,0).copy());records['robot_states'].append(observation['robot_states'][0,0].copy())
                    records['observations'].append(raw);records['actions'].append(action)
                    observation,_,terminated,truncated,info=env.step(action);hit=bool(info.get('success',False));records['success'].append(hit)
                    if hit and first is None:first=step+1
                    if terminated or truncated:break
                    if first is not None and step+1>=first+60:break
                    if first is None and step+1>=150:break
            prefix=first if first is not None else len(records['actions']);name=f'demo_{i}'
            for condition,file in destinations.items():
                size=prefix if condition=='repeat' else len(records['actions']);g=file.create_group(name)
                g.attrs.update(seed=seed,episode_success=first is not None,goal=goal,first_success_step=first or -1)
                for key,value in records.items():g.create_dataset(key,data=np.asarray(value[:size]),compression='gzip')
            row={'episode':name,'seed':seed,'first_success_step':first,'prefix_steps':prefix,'continued_steps':len(records['actions']),
                 'post_success_fraction':float(np.mean(records['success'][prefix:])) if len(records['success'])>prefix else None,'goal':goal.tolist()}
            rows.append(row);print(json.dumps(row),flush=True)
    finally:
        for f in destinations.values():f.close()
    for condition in destinations:
        root=a.output/condition;path=root/'data/reach.hdf5';names=[f'demo_{i}' for i in range(a.episodes)];ntrain=90 if a.episodes==110 else 1
        manifest={'dataset':'data/reach.hdf5','sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'chunk_length':32,'split_seed':60000,
            'train':names[:ntrain],'validation':names[ntrain:],'task_names':['reach-v2'],'episode_task_ids':{n:0 for n in names},
            'scope':'paired expert Reach prefixes vs60genuine post-success control steps;not official dataset','noise_std':.1,'collection_seeds':[60000,60000+a.episodes-1]}
        (root/'records/reach_split_v1.json').write_text(json.dumps(manifest,indent=2))
    with h5py.File(a.output/'repeat/data/reach.hdf5') as x,h5py.File(a.output/'holding/data/reach.hdf5') as y:
        for name in x:
            n=len(x[name]['actions'])
            for key in ('images','robot_states','actions','success'):np.testing.assert_array_equal(x[name][key][:],y[name][key][:n])
    (a.output/'collection_summary.json').write_text(json.dumps({'episodes':rows,'paired_prefixes_identical':True,'failures_kept':True},indent=2))
    print('Paired data collection and prefix checks passed',flush=True)

if __name__=='__main__':main()
