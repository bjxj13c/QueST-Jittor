"""Offline clean expert reference diagnostic; never a policy input."""
import argparse
import os
os.environ['MUJOCO_GL']='egl'
os.environ['PYOPENGL_PLATFORM']='egl'
from pathlib import Path
import json
import h5py
import numpy as np
from metaworld.policies import SawyerReachV2Policy

parser=argparse.ArgumentParser()
parser.add_argument('--data-root',type=Path,required=True)
parser.add_argument('--predictions',type=Path,required=True)
args=parser.parse_args()
manifest=json.loads((args.data_root/'records/reach_split_v1.json').read_text())
expert=SawyerReachV2Policy()
with h5py.File(args.data_root/manifest['dataset'],'r') as f:
    clean={name:np.stack([np.clip(expert.get_action(obs),-1,1) for obs in f[name]['observations'][:]]).astype(np.float32) for name in manifest['train']+manifest['validation']}
report={'scope':'offline reference at recorded states;39D expert state excluded from policy inputs;not rollout success','splits':{}}
for split in ('train','validation'):
    with np.load(args.predictions/f'{split}.npz') as x:
        refs=[]
        for name,start in zip(x['episode'],x['start']):
            values=clean[str(name)][int(start):int(start)+32]
            if len(values)<32:values=np.pad(values,((0,32-len(values)),(0,0)),mode='edge')
            refs.append(values)
        refs=np.stack(refs);mask=np.broadcast_to(x['mask'],refs.shape)
        report['splits'][split]={'noisy_action_vs_clean_expert_mae':float(np.abs(x['target']-refs)[mask].mean(dtype=np.float64)),
            'frozen_reconstruction_vs_clean_expert_mae':float(np.abs(x['prediction']-refs)[mask].mean(dtype=np.float64))}
(args.predictions/'noise_reference.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
