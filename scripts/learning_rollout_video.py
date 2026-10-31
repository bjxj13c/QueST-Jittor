"""Record the fixed seed0 teaching snapshot,not a new benchmark or tuning run."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from quest_repro.config import PolicyConfig,TokenizerConfig
from quest_repro.policy_torch import QueSTPolicy
from quest_repro.tokenizer_torch import SkillVAE
from quest_repro.simulation import ReachSimulation

def main():
    torch.set_num_threads(2)
    policy_path=ROOT/'outputs/learning/policy_context_snapshot_20261029.pt'
    token_path=ROOT/'outputs/five_task_10pct_20261029/dataset/assets/shared/tokenizer_shared.pt'
    policy_state=torch.load(policy_path,map_location='cpu',weights_only=False)
    token_hash=hashlib.sha256(token_path.read_bytes()).hexdigest()
    assert policy_state['extra']['tokenizer_sha256']==token_hash
    token_state=torch.load(token_path,map_location='cpu',weights_only=False)
    policy=QueSTPolicy(PolicyConfig(**policy_state['config']))
    policy.load_state_dict(policy_state['model']);policy.eval()
    tokenizer=SkillVAE(TokenizerConfig(**token_state['config']))
    tokenizer.load_state_dict(token_state['model']);tokenizer.eval()
    folder=ROOT/'outputs/learning'/('policy_video_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    folder.mkdir(parents=True,exist_ok=False)
    simulation=ReachSimulation(seed=0,video=True)
    steps=decisions=0;success=ended=False;predicted=[];executed=[];tokens=[];distances=[]
    try:
        obs,_=simulation.reset()
        with imageio.get_writer(str(folder/'policy.mp4'),fps=20,codec='libx264',quality=7) as writer,torch.no_grad():
            while steps<150 and not ended:
                actions,ids=policy.predict_actions(torch.from_numpy(obs['images']),torch.from_numpy(obs['robot_states']),
                    torch.from_numpy(obs['task_id']),tokenizer,top_k=1)
                decisions+=1;tokens.append(ids[0].numpy())
                for action in actions[0,:8].numpy():
                    writer.append_data(simulation.video_frame())
                    predicted.append(action.copy());executed.append(np.clip(action,-1,1))
                    obs,_,terminated,truncated,info=simulation.step(action)
                    steps+=1;success=bool(info['success']);ended=success or terminated or truncated
                    distances.append(float(np.linalg.norm(simulation.observation[:3]-simulation.observation[-3:])))
                    if ended or steps>=150:break
                print(f'Decision {decisions},steps {steps},success {success}',flush=True)
            writer.append_data(simulation.video_frame())
        np.savez_compressed(folder/'rollout.npz',predicted_actions=np.asarray(predicted),executed_actions=np.asarray(executed),
            generated_tokens=np.asarray(tokens),reach_distances=np.asarray(distances))
        report={'scope':'seed0fixedgreedy teaching snapshot;not finalbenchmark;20fps playback is not control frequency',
            'framework':'torch','device':'cpu','seed':0,'environment':'reach-v2-goal-observable','top_k':1,
            'snapshot_step':policy_state['step'],'action_horizon':8,'max_steps':150,'steps':steps,'decisions':decisions,
            'success':success,'final_distance':distances[-1],
            'policy_sha256':hashlib.sha256(policy_path.read_bytes()).hexdigest(),'tokenizer_sha256':token_hash,
            'video':'policy.mp4','actions':'rollout.npz'}
        (folder/'summary.json').write_text(json.dumps(report,indent=2))
        print('SUCCESS:',success);print('VIDEO:',folder/'policy.mp4');print('SUMMARY:',folder/'summary.json')
    finally:simulation.close()

if __name__=='__main__':main()
