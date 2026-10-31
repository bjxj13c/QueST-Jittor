"""Fixed-token decoder traces for numeric diagnosis;not the acceptance gate."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from quest_repro.config import TokenizerConfig
def main():
    p=argparse.ArgumentParser();p.add_argument('--framework',choices=['torch','jittor'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    unit=ROOT/'outputs/selected_activation_holding_20261030/units/mish_gelu__holding';assets=unit/'assets';cfg=TokenizerConfig(**json.loads((assets/'config.json').read_text())['config'])
    with np.load(ROOT/'outputs/holding_completion_20261030/original_fixture_jittor.npz') as z:tokens=z['tokens'].copy()
    records={};sample=130
    if a.framework=='torch':
        import torch
        from quest_repro.tokenizer_torch import SkillVAE
        from quest_repro.training import load_npz_state
        model=SkillVAE(cfg).cuda().eval();load_npz_state(model,assets/'weights.npz');handles=[]
        for i,layer in enumerate(model.decoder.layers):
            for suffix,module in [('norm1',layer.norm1),('norm2',layer.norm2),('norm3',layer.norm3),('out',layer)]:
                name=f'layer{i}.{suffix}'
                def hook(_m,_args,out,name=name):records[name]=out.detach()[sample]
                handles.append(module.register_forward_hook(hook))
        with torch.no_grad():pred=model.decode_actions(torch.from_numpy(tokens.astype(np.int64)).cuda())
        arrays={k:v.cpu().numpy() for k,v in records.items()};arrays['prediction']=pred.cpu().numpy()
    else:
        import jittor as jt
        from quest_repro import tokenizer_jittor as t
        jt.flags.use_cuda=1;model=t.SkillVAE(cfg);t.strict_load_npz(model,assets/'weights.npz');model.eval();names={}
        for i,layer in enumerate(model.decoder.layers):
            names[id(layer)]=f'layer{i}.out'
            for suffix in ['norm1','norm2','norm3']:names[id(getattr(layer,suffix))]=f'layer{i}.{suffix}'
        def wrap(original):
            def execute(self,*args,**kwargs):
                value=original(self,*args,**kwargs)
                if id(self) in names:records[names[id(self)]]=value[sample]
                return value
            return execute
        t.LayerNorm.execute=wrap(t.LayerNorm.execute);t.DecoderLayer.execute=wrap(t.DecoderLayer.execute)
        with jt.no_grad():pred=model.decode_actions(jt.array(tokens).int32());prediction=pred.numpy()
        arrays={k:v.numpy() for k,v in records.items()};arrays['prediction']=prediction;assert 'torch' not in sys.modules
    np.savez_compressed(a.output,**arrays)
    print(json.dumps({'framework':a.framework,'sample':sample,'fields':list(arrays),'scope':'instrumented trace only;separate strict fixture required'}),flush=True)
if __name__=='__main__':main()
