"""Render saved native-policy actions;verify original success and return on replay."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from quest_repro.simulation import ReachSimulation

def main():
    import mujoco
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw,ImageFont
    out=ROOT/'outputs/success_videos_20261030';out.mkdir(parents=True,exist_ok=False)
    data=ROOT/'outputs/five_task_10pct_20261029/dataset/outputs';proof={}
    fontpath=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf');font=ImageFont.truetype(str(fontpath),25) if fontpath.exists() else ImageFont.load_default()
    for task,title in [('door-open-v2','Door opening'),('window-open-v2','Window opening')]:
        folder=data/('test_jittor_'+task);summary=json.loads((folder/'summary.json').read_text());episode=next(e for e in summary['episodes'] if e['success']);seed=episode['seed'];source=folder/f'seed_{seed}.npz'
        with np.load(source) as z:actions=z['predicted_actions'].copy()
        assert len(actions)==episode['steps'];target=out/f'QueST_Jittor_{task}_success_seed{seed}_720p.mp4';returns=0.;first=None
        def annotate(frame,caption):
            im=Image.fromarray(frame);draw=ImageDraw.Draw(im);draw.rectangle((0,0,1280,54),fill=(20,24,30));draw.text((20,12),f'QueST / Jittor | {title} | {caption}',fill='white',font=font);return np.asarray(im)
        with ReachSimulation(seed,env_name=task+'-goal-observable') as sim:
            sim.env.model.vis.global_.offwidth=1280;sim.env.model.vis.global_.offheight=720
            sim.video_renderer=mujoco.Renderer(sim.env.model,height=720,width=1280);sim.reset()
            with imageio.get_writer(str(target),fps=20,codec='libx264',quality=8,macro_block_size=16,pixelformat='yuv420p') as writer:
                writer.append_data(annotate(sim.video_frame(),'saved policy rollout'))
                for i,action in enumerate(actions):
                    _,reward,terminated,truncated,info=sim.step(action);returns+=reward
                    if info.get('success',False) and first is None:first=i+1
                    frame=sim.video_frame();writer.append_data(annotate(frame,f'step {i+1}/{len(actions)}'))
                assert first is not None,'Saved trajectory did not reproduce task success'
                assert first==len(actions),(first,len(actions))
                assert abs(returns-episode['return'])<1e-4,(returns,episode['return'])
                final=annotate(frame,'SUCCESS verified / final frame paused')
                for _ in range(20):writer.append_data(final)
        proof[task]={'file':str(target),'seed':seed,'framework':'native Jittor saved actions','source_actions_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'source_policy_weights_sha256':summary['policy_weights_sha256'],'original_episode':episode,'replayed_success':True,'first_success_step':first,'replayed_return':returns,
            'resolution':[1280,720],'fps':20,'final_pause_frames':20,'video_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'bytes':target.stat().st_size,
            'scope':'replay of original learned-policy actions,not new inference/training;last1second is explicitly labelled frozen frame,not learned holding'}
        print(json.dumps(proof[task]),flush=True)
    (out/'verification.json').write_text(json.dumps(proof,indent=2))
if __name__=='__main__':main()
