"""HD replay of saved native policy controls; does not rerun/tune the policy."""
import argparse
import json
import os
from pathlib import Path
import sys
os.environ['MUJOCO_GL']='egl';os.environ['PYOPENGL_PLATFORM']='egl'
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

parser=argparse.ArgumentParser()
parser.add_argument('--evaluation',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=False)
summary=json.loads((args.evaluation/'summary.json').read_text())
assert summary['seed_set']=='final' and summary['framework']=='jittor' and not summary['torch_imported']
selected=[next(row for row in summary['episodes'] if row['success']),next(row for row in summary['episodes'] if not row['success'])]
records=[]
for row in selected:
    seed=row['seed'];label='success' if row['success'] else 'failure'
    actions=np.load(args.evaluation/f'seed_{seed}.npz')['predicted_actions']
    env=ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE['reach-v2-goal-observable'](seed=seed)
    env.model.cam_pos[2]=[.75,.075,.7]
    env.model.vis.global_.offwidth=1280;env.model.vis.global_.offheight=720
    renderer=mujoco.Renderer(env.model,height=720,width=1280)
    path=args.output/f'jittor_final_{label}_seed{seed}_720p.mp4'
    distances=[];success=False
    try:
        env.reset()
        with imageio.get_writer(str(path),fps=20,codec='libx264',quality=7) as writer:
            for index,action in enumerate(actions):
                renderer.update_scene(env.data,camera='corner2');frame=renderer.render()[::-1].copy();writer.append_data(frame)
                if index==0:Image.fromarray(frame).save(args.output/f'{label}_initial.png')
                _,_,_,_,info=env.step(np.clip(action.astype(np.float32),-1,1))
                distances.append(float(info['obj_to_target']));success=success or bool(info['success'])
            renderer.update_scene(env.data,camera='corner2');frame=renderer.render()[::-1].copy();writer.append_data(frame)
            Image.fromarray(frame).save(args.output/f'{label}_final.png')
        assert success==row['success']
        records.append({'seed':seed,'success':success,'steps':len(actions),'video':path.name,
            'minimum_tcp_target_distance_m':min(distances),'final_tcp_target_distance_m':distances[-1]})
    finally:
        if hasattr(renderer,'close'):renderer.close()
        else:renderer._mjr_context.free();renderer._gl_context.free()
        env.close()
assert 'torch' not in sys.modules
report={'scope':'720p replay of previously recorded final native policy actions;no new model inference or protocol tuning',
    'selection':'first successful and first failed final episode;both outcomes retained',
    'replay_success_matches_original':True,'torch_imported':False,'episodes':records}
(args.output/'provenance.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
