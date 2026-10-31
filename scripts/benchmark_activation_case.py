"""Isolated synchronized timings after all grid workers have finished."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from quest_repro.config import TokenizerConfig,PolicyConfig
from quest_repro.data import ActionChunkDataset
def main():
    p=argparse.ArgumentParser();p.add_argument('--framework',choices=['torch','jittor'],required=True);p.add_argument('--case-root',type=Path,required=True);a=p.parse_args();case=a.case_root
    assets=case/'assets';pc=json.loads((case/f'policy_{a.framework}/config.json').read_text());tc=json.loads((assets/'config.json').read_text());data=ActionChunkDataset(case/'dataset','validation',include_observations=True)
    inputs=[data[i] for i in range(16)];x=np.stack([v['actions'] for v in inputs]);one=inputs[0];obs=one['obs'];images=obs['corner_rgb'][None];states=obs['robot_states'][None];tasks=np.array([one['task_id']])
    if a.framework=='torch':
        import torch
        from quest_repro.tokenizer_torch import SkillVAE
        from quest_repro.policy_torch import QueSTPolicy
        from quest_repro.training import load_npz_state
        torch.set_num_threads(2);tok=SkillVAE(TokenizerConfig(**tc['config'])).cuda().eval();policy=QueSTPolicy(PolicyConfig(**pc['config'])).cuda().eval();load_npz_state(tok,assets/'weights.npz');load_npz_state(policy,case/f'policy_{a.framework}/best_weights.npz');xx=torch.from_numpy(x).cuda();ii=torch.from_numpy(images).cuda();ss=torch.from_numpy(states).cuda();tt=torch.from_numpy(tasks).cuda()
        def tokenizer():
            with torch.no_grad():return tok.decode(tok.vq(tok.encode(xx))[0]).cpu().numpy()
        def predict():
            with torch.no_grad():
                ctx=policy.get_context(ii,ss,tt);indices=policy.policy_prior.generate(ctx,top_k=1);return tok.decode_actions(indices).cpu().numpy()
        def sync():torch.cuda.synchronize()
    else:
        import jittor as jt
        from quest_repro.tokenizer_jittor import SkillVAE,strict_load_npz
        from quest_repro.policy_jittor import QueSTPolicy
        jt.flags.use_cuda=1;tok=SkillVAE(TokenizerConfig(**tc['config']));policy=QueSTPolicy(PolicyConfig(**pc['config']));strict_load_npz(tok,assets/'weights.npz');strict_load_npz(policy,case/f'policy_{a.framework}/best_weights.npz');tok.eval();policy.eval();xx=jt.array(x);ii=jt.array(images);ss=jt.array(states);tt=jt.array(tasks).int32()
        def tokenizer():
            with jt.no_grad():return tok.decode(tok.vq(tok.encode(xx))[0]).numpy()
        def predict():
            with jt.no_grad():return tok.decode_actions(policy.policy_prior.generate_greedy(policy.get_context(ii,ss,tt))).numpy()
        def sync():jt.sync_all(True)
        assert 'torch' not in sys.modules
    report={}
    for name,fn in [('tokenizer_batch16',tokenizer),('policy_greedy_batch1',predict)]:
        for _ in range(10):result=fn();sync()
        assert np.isfinite(result).all();times=[]
        for _ in range(50):
            sync();start=time.perf_counter();result=fn();sync();times.append((time.perf_counter()-start)*1000)
        report[name]={'median_ms':float(np.median(times)),'p95_ms':float(np.percentile(times,95)),'raw_ms':times}
    report.update(framework=a.framework,scope='isolated GPU forward timings,host transfer included,no rendering;greedy timing differs from stochastic online evaluation',warmup=10,repeats=50,torch_imported='torch' in sys.modules)
    (case/f'isolated_benchmark_{a.framework}.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
