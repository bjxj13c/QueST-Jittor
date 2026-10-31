"""Measured scalar/gradient and real-chunk tokenizer alignment,isolated process."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from quest_repro.config import TokenizerConfig
from quest_repro.data import ActionChunkDataset

def main():
    p=argparse.ArgumentParser();p.add_argument('--framework',choices=['torch','jittor'],required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--weights',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists();conv=os.environ['QUEST_ACTIVATION_CONV'];ffn=os.environ['QUEST_ACTIVATION_FFN'];cfg=TokenizerConfig(**json.loads(a.config.read_text())['config'])
    data=ActionChunkDataset(a.data,'validation');positions=np.linspace(0,len(data)-1,16,dtype=int);x=np.stack([data[i]['actions'] for i in positions]);scalars=np.linspace(-9.95,9.95,200,dtype=np.float32)
    if a.framework=='torch':
        import torch
        from quest_repro.tokenizer_torch import SkillVAE,FSQ
        from quest_repro.training import load_npz_state
        import torch.nn.functional as f
        torch.set_num_threads(2);model=SkillVAE(cfg).cuda().eval();load_npz_state(model,a.weights)
        def act(v,n):return f.mish(v) if n=='mish' else f.silu(v) if n=='silu' else f.gelu(v,approximate='none') if n=='gelu' else f.relu(v)
        v=torch.tensor(scalars,device='cuda',requires_grad=True);c=act(v,conv);g=torch.autograd.grad(c.sum(),v)[0];v=torch.tensor(scalars,device='cuda',requires_grad=True);ff=act(v,ffn);fg=torch.autograd.grad(ff.sum(),v)[0]
        with torch.no_grad():
            z=torch.from_numpy(x).cuda();encoded=model.encode(z);prediction=model(z)[0];indices=model.get_indices(z)
        arrays={k:v.detach().cpu().numpy() for k,v in {'conv_scalar':c,'conv_gradient':g,'ffn_scalar':ff,'ffn_gradient':fg,'encoded':encoded,'prediction':prediction,'indices':indices}.items()}
    else:
        import jittor as jt
        from jittor import nn
        from quest_repro.tokenizer_jittor import SkillVAE,strict_load_npz
        jt.flags.use_cuda=1;model=SkillVAE(cfg);strict_load_npz(model,a.weights);model.eval()
        def act(v,n):return v*nn.softplus(v).tanh() if n=='mish' else v*jt.sigmoid(v) if n=='silu' else nn.gelu(v) if n=='gelu' else nn.relu(v)
        v=jt.array(scalars);v.start_grad();c=act(v,conv);g=jt.grad(c.sum(),v);v=jt.array(scalars);v.start_grad();ff=act(v,ffn);fg=jt.grad(ff.sum(),v)
        with jt.no_grad():
            z=jt.array(x);encoded=model.encode(z);prediction=model.forward_details(z)['prediction'];indices=model.get_indices(z)
        arrays={k:v.numpy() for k,v in {'conv_scalar':c,'conv_gradient':g,'ffn_scalar':ff,'ffn_gradient':fg,'encoded':encoded,'prediction':prediction,'indices':indices}.items()};assert 'torch' not in sys.modules
    np.savez_compressed(a.output,**arrays);print(json.dumps({'status':'exported','conv':conv,'ffn':ffn,'framework':a.framework,'real_validation_chunks':16,'torch_imported':'torch' in sys.modules}),flush=True)
if __name__=='__main__':main()
