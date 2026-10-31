"""Audit author's T+1 observations, convert pre-action frames, smoke shared loader."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--expected-episodes',type=int,default=5)
    parser.add_argument('--formal',action='store_true')
    args = parser.parse_args()
    run = args.run.resolve()
    source = run/'raw/metaworld/ML45_PRISE/train/reach-v2.hdf5'
    target_root = run/'adapted'
    target_root.mkdir(exist_ok=False)
    (target_root/'data').mkdir()
    (target_root/'records').mkdir()
    target = target_root/('data/reach_official_100_v1.hdf5' if args.formal else 'data/reach_official_smoke_v1.hdf5')
    summary = {'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'scope': '5 episode collection and data-interface smoke; not training/evaluation',
        'alignment': 'action[t] paired with obs[t]; terminal obs[T] retained in original only',
        'episodes': []}
    with h5py.File(source,'r') as original, h5py.File(target,'x') as adapted:
        episodes = sorted(original['data'], key=lambda k:int(k.split('_')[-1]))
        if len(episodes)!=args.expected_episodes:
            raise ValueError(f'Expected {args.expected_episodes} episodes, got {len(episodes)}')
        for name in episodes:
            group = original['data'][name]
            actions = group['actions'][:]
            images = group['obs/corner_rgb'][:]
            states = group['obs/robot_states'][:]
            privileged = group['obs/obs_gt'][:]
            length = len(actions)
            assert length > 0 and actions.shape==(length,4)
            assert images.shape==(length+1,128,128,3) and images.dtype==np.uint8
            assert states.shape==(length+1,8) and privileged.shape==(length+1,39)
            assert np.isfinite(actions).all() and np.abs(actions).max()<=1
            assert np.isfinite(states).all() and np.isfinite(privileged).all()
            np.testing.assert_allclose(states,np.concatenate((privileged[:,:4],privileged[:,18:22]),axis=1),atol=1e-7)
            output = adapted.create_group(name)
            # No additional vertical flip: author's wrapper already flipped images.
            values = {'actions':actions.astype(np.float32),'images':images[:-1],
                'robot_states':states[:-1].astype(np.float32),'observations':privileged[:-1]}
            for key in ('success','terminated','truncated','reward'):
                value=group[key][:]
                assert len(value)==length and np.isfinite(value).all()
                values['rewards' if key=='reward' else key]=value
            for key,value in values.items():
                output.create_dataset(key,data=value,compression='gzip')
            success=bool(np.any(values['success']))
            output.attrs['episode_success']=success
            summary['episodes'].append({'name':name,'steps':length,'success':success,
                'action_min':float(actions.min()),'action_max':float(actions.max()),
                'terminated_last':bool(values['terminated'][-1]),'truncated_last':bool(values['truncated'][-1])})
            np.testing.assert_array_equal(output['images'][0],images[0])
    if args.formal:
        if args.expected_episodes!=100: raise ValueError('Frozen formal protocol requires 100 episodes')
        protocol=json.loads((Path(__file__).resolve().parents[1]/'records/official_reach_100_protocol_v1.json').read_text())
        # Official task goals repeat after50episodes. Keep matching starts
        # together so the held-out set does not share initial states.
        groups={}
        with h5py.File(target,'r') as file:
            for name in episodes:
                key=hashlib.sha256(file[name]['observations'][0].tobytes()).hexdigest()
                groups.setdefault(key,[]).append(name)
        if len(groups)!=50 or any(len(group)!=2 for group in groups.values()):
            raise ValueError('Expected50initial-state groups with2trajectories each; review split protocol')
        group_ids=np.random.default_rng(protocol['split_seed']).permutation(list(groups)).tolist()
        train=[name for key in group_ids[:40] for name in groups[key]]
        validation=[name for key in group_ids[40:] for name in groups[key]]
        summary['initial_state_groups']={'total':50,'train':40,'validation':10,'overlap':0}
        scope='single-task100demo fixed official-collection protocol; no training result yet'
    else:
        train,validation=episodes[:4],episodes[4:]
        scope='interface smoke split only; no model selection or final evaluation'
    manifest={'dataset':target.relative_to(target_root).as_posix(),
        'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
        'chunk_length':32,'train':train,'validation':validation,
        'scope':scope,
        'collection':json.loads((run/'provenance.json').read_text())}
    (target_root/'records/reach_split_v1.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    if args.formal:
        statistics={'source':'training episodes only','applied':False}
        with h5py.File(target,'r') as file:
            for key in ('actions','robot_states'):
                values=np.concatenate([file[name][key][:] for name in train])
                statistics[key]={stat:getattr(values,stat)(axis=0).tolist() for stat in ('mean','std','min','max')}
        (target_root/'records/reach_train_statistics_v1.json').write_text(json.dumps(statistics,indent=2))
        nested=dict(manifest,train=train[:24],scope='nested24 training demos with same20validation episodes')
        (target_root/'records/nested24_split_v1.json').write_text(json.dumps(nested,indent=2))
        summary['scope']=scope
        summary['protocol']=protocol
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    from quest_repro.data import ActionChunkDataset, load_manifest
    load_manifest(target_root,verify_hash=True)
    dataset=ActionChunkDataset(target_root,include_observations=True)
    samples=[dataset[0],dataset[len(dataset)-1]]
    for sample in samples:
        assert sample['actions'].shape==(32,4)
        assert sample['obs']['corner_rgb'].shape==(1,3,128,128)
        assert sample['obs']['robot_states'].shape==(1,8)
        assert set(sample['obs'])=={'corner_rgb','robot_states'}
    summary['shared_numpy_loader']={'passed':True,'train_windows':len(dataset),
        'validation_windows':len(ActionChunkDataset(target_root,'validation')),
        'privileged_state_excluded':True,'rgb_uint8_CHW':True}
    summary['success_count']=sum(ep['success'] for ep in summary['episodes'])
    (run/'validation.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
