"""Authentic720p replays of saved native predictions,not new policy tests."""
import os
os.environ['MUJOCO_GL']='egl';os.environ['PYOPENGL_PLATFORM']='egl'
from pathlib import Path
import json
import sys
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

ROOT=Path(__file__).resolve().parents[1];NIGHT=ROOT/'outputs/overnight_20261029';OUT=NIGHT/'experiments'
workflow=json.loads((OUT/'workflow.json').read_text());assert workflow['status']=='completed'
settings=workflow['protocol']['four_cases'][workflow['selected_single_case_by_validation']]
cases=[('single_selected',NIGHT/'datasets'/settings['data']/'outputs/test_selected_native')]
for task in workflow['protocol']['multi_task_tasks']:
    cases.append(('multi_'+task,NIGHT/'datasets/multi240/outputs'/('test_multi_jittor_'+task)))
destination=OUT/'media';destination.mkdir(exist_ok=False);records=[]
for label,folder in cases:
    result=json.loads((folder/'summary.json').read_text())
    assert result['framework']=='jittor' and result['seed_set']=='experiment' and not result['torch_imported']
    task=result['environment']
    for outcome in [True,False]:
        choices=[row for row in result['episodes'] if row['success']==outcome]
        if not choices:continue
        row=choices[0];seed=row['seed'];suffix='success' if outcome else 'failure'
        actions=np.load(folder/f'seed_{seed}.npz')['predicted_actions']
        env=ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task](seed=seed);env.model.cam_pos[2]=[.75,.075,.7]
        env.model.vis.global_.offwidth=1280;env.model.vis.global_.offheight=720
        renderer=mujoco.Renderer(env.model,height=720,width=1280)
        path=destination/f'{label}_{suffix}_seed{seed}_720p.mp4';success=False
        try:
            env.reset()
            with imageio.get_writer(str(path),fps=20,codec='libx264',quality=7) as writer:
                for i,action in enumerate(actions):
                    renderer.update_scene(env.data,camera='corner2');frame=renderer.render()[::-1].copy();writer.append_data(frame)
                    if i==0:Image.fromarray(frame).save(destination/f'{path.stem}_initial.png')
                    _,_,_,_,info=env.step(np.clip(action.astype(np.float32),-1,1));success=success or bool(info['success'])
                renderer.update_scene(env.data,camera='corner2');frame=renderer.render()[::-1].copy();writer.append_data(frame)
                Image.fromarray(frame).save(destination/f'{path.stem}_final.png')
            assert success==outcome
            records.append({'case':label,'environment':task,'seed':seed,'success':success,'steps':len(actions),'video':path.name})
        finally:
            if hasattr(renderer,'close'):renderer.close()
            else:renderer._mjr_context.free();renderer._gl_context.free()
            env.close()
assert 'torch' not in sys.modules
(destination/'provenance.json').write_text(json.dumps({'scope':'saved action replay,no new predictions/tuning;20fps visual playback does not establish real-time control speed',
    'selection':'first success and first failure for validation-selected single native and each multi task','torch_imported':False,'replay_outcomes_match':True,'clips':records},indent=2))
print('REPLAY_CLIPS',len(records),flush=True)
