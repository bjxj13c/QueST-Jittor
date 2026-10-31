"""Five matched-scene learned-policy replays and MetaWorld scripted experts.

No model inference, training, or benchmark changes. Every replay verifies the
original first-success step and return. Final pause is explicitly labelled.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from quest_repro.simulation import ReachSimulation

TASKS={'reach-v2':'SawyerReachV2Policy','reach-wall-v2':'SawyerReachWallV2Policy',
       'push-v2':'SawyerPushV2Policy','door-open-v2':'SawyerDoorOpenV2Policy',
       'window-open-v2':'SawyerWindowOpenV2Policy'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(root,out,platform_replay=False,resume=False):
    # Set platform-appropriate GL before importing MuJoCo/MetaWorld.
    from quest_repro.simulation import set_render_environment
    set_render_environment()
    import mujoco
    import metaworld.policies as policies
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw,ImageFont
    out.mkdir(parents=True,exist_ok=resume)
    fontfile=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    font=ImageFont.truetype(str(fontfile),22) if fontfile.exists() else ImageFont.load_default()
    report={'status':'running','scope':'illustrative successes,not new benchmark; learned videos replay saved native Jittor baseline actions; experts are MetaWorld scripted policies',
            'selection':'first saved successful baseline Jittor episode for each task; expert uses the same scene seed',
            'fps':20,'playback_note':'playback rate is not environment control frequency','videos':[]}
    if resume:
        previous=json.loads((out/'verification.json').read_text(encoding='utf-8'))
        assert previous['status']=='failed','Only resume a preserved incomplete video export'
        (out/f'verification_before_resume_{time.time_ns()}.json').write_text(json.dumps(previous,indent=2),encoding='utf-8')
        report['videos']=previous['videos']
        for row in report['videos']:
            assert sha(out/row['video'])==row['video_sha256']
            assert sha(out/row['proof'])==row['proof_sha256']
    import platform
    report['render_platform']=platform.system()
    report['cross_platform_video_mode']=platform_replay
    report['return_validation_note']='Cross-platform video replay requires exact saved actions and original first-success step; reward differences are reported, not substituted into original benchmark' if platform_replay else 'Original success step and reward sum must match'
    def save():
        (out/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    save()
    def caption(frame,text):
        im=Image.fromarray(frame);d=ImageDraw.Draw(im);d.rectangle((0,0,1280,48),fill=(18,23,30));d.text((16,11),text,fill='white',font=font)
        return np.asarray(im)
    try:
        for task,expert_name in TASKS.items():
            folder=root/'outputs/five_task_10pct_20261029/dataset/outputs'/('test_jittor_'+task)
            summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
            episode=next(e for e in summary['episodes'] if e['success'])
            seed=episode['seed'];source=folder/f'seed_{seed}.npz'
            with np.load(source) as z:actions=z['predicted_actions'].copy()
            assert actions.shape==(episode['steps'],4) and np.isfinite(actions).all()
            for kind in ('learned_jittor','scripted_expert'):
                if any(r['task']==task and r['kind']==kind for r in report['videos']):continue
                target=out/f'{task}__{kind}__seed{seed}__720p.mp4'
                if target.exists():
                    assert resume
                    preserved=out/(target.stem+f'__failed_attempt_{time.time_ns()}.mp4')
                    assert preserved.resolve().is_relative_to(out.resolve())
                    target.rename(preserved)
                label='LEARNED Jittor / saved actions' if kind=='learned_jittor' else 'SCRIPTED EXPERT / MetaWorld'
                rewards=0.;first=None;recorded=[];states=[];successes=[]
                with ReachSimulation(seed,env_name=task+'-goal-observable') as sim:
                    sim.env.model.vis.global_.offwidth=1280;sim.env.model.vis.global_.offheight=720
                    sim.video_renderer=mujoco.Renderer(sim.env.model,height=720,width=1280)
                    sim.reset();expert=getattr(policies,expert_name)() if kind=='scripted_expert' else None
                    initial_obs=sim.observation.copy()
                    with imageio.get_writer(str(target),fps=20,codec='libx264',quality=8,macro_block_size=16,pixelformat='yuv420p') as writer:
                        writer.append_data(caption(sim.video_frame(),f'{label} | {task} | seed {seed}'))
                        budget=len(actions) if kind=='learned_jittor' else 500
                        for i in range(budget):
                            a=actions[i] if expert is None else np.clip(expert.get_action(sim.observation),-1,1).astype(np.float32)
                            states.append(sim.observation.copy());recorded.append(a.copy())
                            _,reward,terminated,truncated,info=sim.step(a);rewards+=float(reward)
                            hit=bool(info.get('success',False));successes.append(hit)
                            if hit and first is None:first=i+1
                            frame=sim.video_frame();writer.append_data(caption(frame,f'{label} | {task} | step {i+1} | success={hit}'))
                            if expert is not None and (hit or terminated or truncated):break
                        assert first is not None,f'{kind} did not succeed: {task} seed {seed}'
                        if expert is None:
                            assert first==len(actions)
                            if not platform_replay:assert abs(rewards-episode['return'])<1e-4,(task,rewards,episode['return'])
                        # No fictitious holding controls: only labelled frozen frames.
                        paused=caption(frame,f'{label} | SUCCESS | final frame paused (not holding control)')
                        for _ in range(20):writer.append_data(paused)
                proof=out/f'{task}__{kind}__seed{seed}__actions.npz'
                if proof.exists():
                    assert resume
                    proof.rename(out/(proof.stem+f'__failed_attempt_{time.time_ns()}.npz'))
                np.savez_compressed(proof,actions=np.asarray(recorded),observations=np.asarray(states),success=np.asarray(successes),initial_observation=initial_obs)
                row={'task':task,'kind':kind,'seed':seed,'video':target.name,'video_sha256':sha(target),'bytes':target.stat().st_size,
                     'success':True,'first_success_step':first,'steps':len(recorded),'return':rewards,'resolution':[1280,720],'fps':20,'paused_final_frames':20,
                     'proof':proof.name,'proof_sha256':sha(proof)}
                if expert is None:
                    np.testing.assert_array_equal(np.asarray(recorded),actions)
                    row.update(source_actions_sha256=sha(source),source_policy_weights_sha256=summary['policy_weights_sha256'],original_episode=episode,
                               source_actions_exact=True,success_step_matches_original=first==episode['steps'],return_delta=rewards-episode['return'],
                               strict_return_match=bool(abs(rewards-episode['return'])<1e-4),replay_matches_original=bool(abs(rewards-episode['return'])<1e-4))
                else:row.update(expert_class=expert_name,expert_is_trained_neural_model=False,scene_matches_learned_seed=True)
                report['videos'].append(row);save();print(json.dumps(row),flush=True)
        assert len(report['videos'])==10
        for task in TASKS:
            pair=[r for r in report['videos'] if r['task']==task]
            with np.load(out/pair[0]['proof']) as x,np.load(out/pair[1]['proof']) as y:np.testing.assert_array_equal(x['initial_observation'],y['initial_observation'])
        report.update(status='completed',matched_initial_states_verified=True,all_ten_successes_verified=True)
        save()
        lines=['# 五任务：学习策略与专家策略视频','','每个任务两段720p视频，使用同一初始场景。学习策略重放已记录的Jittor主基线成功动作；专家为MetaWorld规则脚本，不是训练出的神经网络。','片尾1秒是明确标注的定格，不代表学会了保持。20fps为播放速率，不是控制频率。','这些是成功案例演示，不表示所有场景都成功，也不是新一轮评估。','本次在Windows本机重放Linux记录。保存动作与首次成功步数已核对；接触任务可能产生累计奖励差异，详见verification.json，原始基准分数不替换。' if platform_replay else '与原记录核对成功步数及累计奖励。','','|任务|Jittor学习策略|MetaWorld专家|','|---|---|---|']
        for task in TASKS:
            a,b=[next(r for r in report['videos'] if r['task']==task and r['kind']==k) for k in ('learned_jittor','scripted_expert')]
            lines.append(f"|{task}|[{a['first_success_step']}步成功]({a['video']})|[{b['first_success_step']}步成功]({b['video']})|")
        (out/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    except BaseException as e:
        report.update(status='failed',error=f'{type(e).__name__}: {e}');save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output',type=Path,required=True);p.add_argument('--platform-replay',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
    main(a.root.resolve(),a.output.resolve(),a.platform_replay,a.resume)
