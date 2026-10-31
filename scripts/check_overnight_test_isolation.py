"""Audit proposed new seeds against all known training/validation goals;no policy."""
import os
os.environ['MUJOCO_GL']='egl';os.environ['PYOPENGL_PLATFORM']='egl'
import json
from pathlib import Path
import h5py
import numpy as np
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

ROOT=Path(__file__).resolve().parents[1];NIGHT=ROOT/'outputs/overnight_20261029'
report={'scope':'goal-overlap audit before model testing,no policy outcomes inspected','datasets':{}}
for label,start,count in [('reach280',8000,50),('multi240',9000,20)]:
    root=NIGHT/'datasets'/label;manifest=json.loads((root/'records/reach_split_v1.json').read_text())
    result={}
    with h5py.File(root/manifest['dataset'],'r') as f:
        for task_id,task in enumerate(manifest['task_names']):
            seen={f[name]['observations'][0][-3:].astype(np.float64).round(7).tobytes() for name in manifest['train']+manifest['validation'] if manifest['episode_task_ids'][name]==task_id}
            overlaps=[]
            for seed in range(start,start+count):
                env=ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task+'-goal-observable'](seed=seed)
                try:
                    obs,_=env.reset();key=obs[-3:].astype(np.float32).astype(np.float64).round(7).tobytes()
                    if key in seen:overlaps.append(seed)
                finally:env.close()
            result[task]={'test_seed_start':start,'test_episodes':count,'training_validation_goal_count':len(seen),'overlap_seeds':overlaps}
            assert not overlaps,f'New test range overlaps seen goal:{label},{task}'
    report['datasets'][label]=result
(NIGHT/'test_isolation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
