"""Behavior diagnostic:continue learned control60steps after first true success.

No target-distance oracle changes actions,stops the robot,or clears action
chunks. Standard first-hit evaluation is run separately with original scripts.
"""
import argparse
from collections import deque
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from quest_repro.config import PolicyConfig,TokenizerConfig
from quest_repro.simulation import ReachSimulation

def main():
    p=argparse.ArgumentParser();p.add_argument('--framework',choices=['torch','jittor'],required=True);p.add_argument('--policy-dir',type=Path,required=True);p.add_argument('--assets',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed-start',type=int,default=120000);p.add_argument('--sampling-seed',type=int,default=130000);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    pc=json.loads((a.policy_dir/'config.json').read_text());tc=json.loads((a.assets/'config.json').read_text())
    if a.framework=='torch':
        import torch
        from quest_repro.policy_torch import QueSTPolicy
        from quest_repro.tokenizer_torch import SkillVAE
        from quest_repro.training import load_npz_state
        torch.set_num_threads(4);model=QueSTPolicy(PolicyConfig(**pc['config'])).cuda().eval();tokenizer=SkillVAE(TokenizerConfig(**tc['config'])).cuda().eval();load_npz_state(model,a.policy_dir/'best_weights.npz');load_npz_state(tokenizer,a.assets/'weights.npz')
        def predict(obs,rng):
            with torch.no_grad():
                context=model.get_context(**{k:torch.from_numpy(v).cuda() for k,v in obs.items()});tokens=model.policy_prior.generate_sampled(context,top_k=5,rng=rng);return tokenizer.decode_actions(tokens)[0].cpu().numpy(),tokens[0].cpu().numpy()
    else:
        import jittor as jt
        from quest_repro.policy_jittor import QueSTPolicy
        from quest_repro.tokenizer_jittor import SkillVAE,strict_load_npz
        jt.flags.use_cuda=1;model=QueSTPolicy(PolicyConfig(**pc['config']));tokenizer=SkillVAE(TokenizerConfig(**tc['config']));strict_load_npz(model,a.policy_dir/'best_weights.npz');strict_load_npz(tokenizer,a.assets/'weights.npz');model.eval();tokenizer.eval()
        def predict(obs,rng):
            with jt.no_grad():
                context=model.get_context(jt.array(obs['images']),jt.array(obs['robot_states']),jt.array(obs['task_id']).int32());tokens=model.policy_prior.generate_sampled(context,top_k=5,rng=rng);return tokenizer.decode_actions(tokens)[0].numpy(),tokens[0].numpy()
    episodes=[]
    for seed in range(a.seed_start,a.seed_start+30):
        rng=np.random.default_rng(a.sampling_seed+seed);queue=deque();first=None;successes=[];distances=[];positions=[];actions=[];tokens=[]
        with ReachSimulation(seed,video=seed==a.seed_start) as sim:
            obs,_=sim.reset();writer=None
            if seed==a.seed_start:
                import imageio.v2 as imageio
                writer=imageio.get_writer(str(a.output/f'seed_{a.seed_start}.mp4'),fps=20,codec='libx264',quality=7)
            try:
                for step in range(210):
                    if writer:writer.append_data(sim.video_frame())
                    if not queue:
                        block,ids=predict(obs,rng);assert np.isfinite(block).all();queue.extend(block[:8]);tokens.append(ids)
                    action=queue.popleft();actions.append(action.copy());obs,_,terminated,truncated,info=sim.step(action)
                    hit=bool(info.get('success',False));successes.append(hit);positions.append(np.asarray(sim.env.tcp_center).copy());distances.append(float(np.linalg.norm(sim.env.tcp_center-sim.env._target_pos)))
                    if hit and first is None:first=step+1
                    if terminated or truncated:break
                    if first is None and step+1>=150:break
                    if first is not None and step+1>=first+60:break
                if writer:writer.append_data(sim.video_frame())
            finally:
                if writer:writer.close()
        post=successes[first:first+60] if first is not None else [];pos=np.asarray(positions)
        row={'seed':seed,'first_success_step':first,'success_within150':first is not None and first<=150,'steps':len(actions),'post_steps':len(post),
            'post_success_fraction':float(np.mean(post)) if post else None,'held_all60':bool(len(post)==60 and all(post)),
            'tcp_drift_60':float(np.linalg.norm(pos[first+59]-pos[first-1])) if len(post)==60 else None,'final_tcp_distance':distances[-1]}
        np.savez_compressed(a.output/f'seed_{seed}.npz',actions=np.asarray(actions),tokens=np.asarray(tokens),success=np.asarray(successes),tcp=np.asarray(positions),distance=np.asarray(distances))
        episodes.append(row);print(json.dumps(row),flush=True)
    result={'framework':a.framework,'episodes':episodes,'scope':'diagnostic continuation;not a replacement for standard success metric','action_horizon':8,'top_k':5,'seed_start':a.seed_start,'sampling_seed':a.sampling_seed,
        'first_hit_rate':float(np.mean([v['success_within150'] for v in episodes])),'hold_all60_unconditional':float(np.mean([v['held_all60'] for v in episodes])),
        'torch_imported':'torch' in sys.modules}
    if a.framework=='jittor':assert not result['torch_imported']
    (a.output/'summary.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
