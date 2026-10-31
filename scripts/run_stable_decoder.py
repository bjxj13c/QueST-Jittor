"""Candidate numeric repair:FP64 LayerNorm accumulation only in frozen decoder.

Weights,formula,FSQ,activations andFP32moduleinterfaces remain unchanged.
Not enabled unless a separate strict same-weight GPU check passes.
"""
import argparse
from pathlib import Path
import runpy
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from run_activation_variant import install
def stabilize():
    import jittor as jt
    from quest_repro import tokenizer_jittor as t
    original_init=t.SkillVAE.__init__;original_norm=t.LayerNorm.execute
    def init(self,*args,**kwargs):
        original_init(self,*args,**kwargs)
        for layer in self.decoder.layers:
            for name in ('norm1','norm2','norm3'):setattr(getattr(layer,name),'stable_decoder_accumulation',True)
    def normalize(self,value):
        if not getattr(self,'stable_decoder_accumulation',False) or self.is_training():return original_norm(self,value)
        x=value.float64();mean=x.mean(dim=-1,keepdims=True);centered=x-mean;variance=(centered*centered).mean(dim=-1,keepdims=True)
        result=centered/(variance+self.eps).sqrt()*self.weight.float64()+self.bias.float64()
        return result.float32()
    t.SkillVAE.__init__=init;t.LayerNorm.execute=normalize
    assert 'torch' not in sys.modules
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--framework',choices=['torch','jittor'],required=True);p.add_argument('--conv',choices=['mish'],required=True);p.add_argument('--ffn',choices=['gelu'],required=True);p.add_argument('script',type=Path);p.add_argument('arguments',nargs=argparse.REMAINDER);a=p.parse_args()
    install(a.framework,a.conv,a.ffn)
    if a.framework=='jittor':stabilize()
    target=a.script.resolve();assert target.is_relative_to(ROOT/'scripts');sys.argv=[str(target),*a.arguments];runpy.run_path(str(target),run_name='__main__')
