"""Replay recorded learned-policy controls on SAME seeds; no new policy fitting."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from quest_repro.simulation import set_render_environment
set_render_environment()
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

parser=argparse.ArgumentParser()
parser.add_argument('--evaluation',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=False)
source=json.loads((args.evaluation/'summary.json').read_text())
assert source['seed_set']=='development'
records=[]
for row in source['episodes']:
    seed=row['seed'];actions=np.load(args.evaluation/f'seed_{seed}.npz')['predicted_actions']
    env=ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE['reach-v2-goal-observable'](seed=seed)
    try:
        obs,_=env.reset();distances=[float(np.linalg.norm(env.tcp_center-env._target_pos))]
        np.testing.assert_allclose(obs[-3:],env._target_pos,atol=1e-7)
        success=False
        for action in actions:
            obs,_,_,_,info=env.step(np.clip(np.asarray(action,dtype=np.float32),-1,1))
            distance=float(np.linalg.norm(env.tcp_center-env._target_pos))
            np.testing.assert_allclose(distance,info['obj_to_target'],atol=1e-7)
            distances.append(distance)
            success=success or bool(info.get('success',False))
        assert success==row['success'],f'Replay differs for{seed}'
        records.append({'seed':seed,'success':success,'steps':len(actions),'initial_distance_m':distances[0],
            'minimum_distance_m':min(distances),'final_distance_m':distances[-1]})
    finally:env.close()
report={'scope':'deterministic replay of recorded development policy actions;goal state used only for distance diagnostics,never policy input',
        'distance_definition':'environment tcp_center to target,verified against info.obj_to_target;NOT observation hand-origin position',
        'replay_success_matches_all':True,'episodes':records}
(args.output/'distances.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
