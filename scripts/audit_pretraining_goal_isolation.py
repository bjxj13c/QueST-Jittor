"""Additional goal-only exposure audit,including original tokenizer pretraining."""
import os
os.environ['MUJOCO_GL']='egl';os.environ['PYOPENGL_PLATFORM']='egl'
from pathlib import Path
import json
import h5py
import numpy as np
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

ROOT=Path(__file__).resolve().parents[1];NIGHT=ROOT/'outputs/overnight_20261029'
seen={task:set() for task in ['reach-v2','reach-wall-v2','push-v2']}
for root in [ROOT,NIGHT/'datasets/reach280',NIGHT/'datasets/multi240']:
    m=json.loads((root/'records/reach_split_v1.json').read_text());names=m.get('task_names',['reach-v2']);ids=m.get('episode_task_ids',{})
    with h5py.File(root/m['dataset'],'r') as f:
        for ep in m['train']+m['validation']:
            seen[names[ids.get(ep,0)]].add(f[ep]['observations'][0][-3:].astype(np.float64).round(7).tobytes())
records=[]
for task,start,n in [('reach-v2',8000,50),('reach-v2',9000,20),('reach-wall-v2',9000,20),('push-v2',9000,20)]:
    overlaps=[]
    for seed in range(start,start+n):
        env=ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task+'-goal-observable'](seed=seed)
        try:
            obs,_=env.reset();np.testing.assert_allclose(obs[-3:],env._target_pos,atol=1e-7)
            key=obs[-3:].astype(np.float32).astype(np.float64).round(7).tobytes()
            if key in seen[task]:overlaps.append(seed)
        finally:env.close()
    assert not overlaps
    records.append({'task':task,'seed_start':start,'episodes':n,'all_known_goal_count':len(seen[task]),'overlaps':overlaps})
report={'scope':'additional goal-coordinate audit including original30demo action-tokenizer corpus;no model predictions inspected or changed','records':records}
(NIGHT/'pretraining_goal_isolation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
