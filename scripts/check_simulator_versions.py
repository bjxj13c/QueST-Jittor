"""Bounded paired simulator audit; development seeds only, no learned policy."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ['MUJOCO_GL']='egl'
os.environ['PYOPENGL_PLATFORM']='egl'
import numpy as np
import gymnasium
import mujoco
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from metaworld.policies import SawyerReachV2Policy


def close_renderer(renderer):
    if hasattr(renderer,'close'): renderer.close()
    else:
        renderer._mjr_context.free()
        renderer._gl_context.free()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--author-renderer',action='store_true')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    arrays={}
    summary={'gymnasium':gymnasium.__version__,'mujoco':mujoco.__version__,
        'seeds':[9100,9101,9102],'scope':'simulator compatibility audit, not policy evaluation',
        'cases':[]}
    for i,seed in enumerate(summary['seeds']):
        np.random.seed(seed)
        env=ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE['reach-v2-goal-observable'](seed=seed)
        env.model.cam_pos[2]=[0.75,0.075,0.7]
        renderer=mujoco.Renderer(env.model,height=128,width=128)
        rng=np.random.default_rng(seed+200)
        expert=SawyerReachV2Policy()
        try:
            obs,_=env.reset()
            states=[obs.copy()]; actions=[]; rewards=[]; flags=[]; images=[]
            # Include a timeout probe independent of expert success.
            count=510 if i==2 else 150
            for step in range(count):
                if step in (0,25,149,499):
                    renderer.update_scene(env.data,camera='corner2')
                    images.append(renderer.render()[::-1].copy())
                action=np.zeros(4,dtype=np.float32) if i==2 else np.clip(expert.get_action(obs)+rng.normal(0,.2,4),-1,1).astype(np.float32)
                obs,reward,terminated,truncated,info=env.step(action)
                actions.append(action); states.append(obs.copy()); rewards.append(reward)
                flags.append((bool(terminated),bool(truncated),bool(info.get('success',False))))
                if terminated or truncated: break
            for key,value in [('states',states),('actions',actions),('rewards',rewards),('flags',flags),('images',images)]:
                arrays[f'{seed}_{key}']=np.asarray(value)
            summary['cases'].append({'seed':seed,'steps':len(actions),'any_success':bool(np.asarray(flags)[:,2].any()),'last_flags':list(flags[-1])})
        finally:
            close_renderer(renderer); env.close()
    if args.author_renderer:
        upstream=Path(__file__).resolve().parents[1]/'upstream/QueST'
        sys.path.insert(0,str(upstream))
        from quest.utils.metaworld_utils import MetaWorldWrapper
        meta={'observation':{'rgb':{'corner_rgb':[3,128,128]},'lowdim':{'robot_states':8}}}
        env=MetaWorldWrapper('reach-v2',meta,env_kwargs={'seed':9200})
        renderer=mujoco.Renderer(env.env.model,height=128,width=128)
        errors=[]; state_errors=[]
        try:
            obs,_=env.reset()
            for step in range(12):
                renderer.update_scene(env.env.data,camera='corner2')
                canonical=renderer.render()[::-1].copy()
                errors.append(int(np.abs(canonical.astype(np.int16)-obs['corner_rgb'].astype(np.int16)).max()))
                expected=np.concatenate((obs['obs_gt'][:4],obs['obs_gt'][18:22]))
                state_errors.append(float(np.abs(expected-obs['robot_states']).max()))
                if step==0:
                    arrays['author_image']=obs['corner_rgb'].copy(); arrays['canonical_image']=canonical
                obs,_,_,_,_=env.step(np.zeros(4,dtype=np.float32))
            summary['author_renderer_comparison']={'max_pixel_abs':max(errors),'max_state_abs':max(state_errors),'frames':len(errors)}
        finally:
            close_renderer(renderer); env.close()
    np.savez_compressed(args.output/'fixtures.npz',**arrays)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__': main()
