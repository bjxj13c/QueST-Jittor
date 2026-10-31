"""Process-local QueST activation ablation;never edits shared model sources.

Only Conv1dBlock's final activation and Transformer FFN activation are varied.
ResNet ReLU,attention,FSQ bound/rounding,and projection structure stay intact.
All train/export/fixture/eval calls must use this entry point.
"""
import argparse
import os
from pathlib import Path
import runpy
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]

def install(framework,conv,ffn):
    assert conv in ('mish','silu','gelu','relu') and ffn in ('gelu','silu','relu')
    os.environ['QUEST_ACTIVATION_CONV']=conv;os.environ['QUEST_ACTIVATION_FFN']=ffn
    if framework=='torch':
        import torch
        from quest_repro import tokenizer_torch as t
        from quest_repro import policy_torch as p
        def activation(x,name):
            if name=='mish':return torch.nn.functional.mish(x)
            if name=='silu':return torch.nn.functional.silu(x)
            if name=='gelu':return torch.nn.functional.gelu(x,approximate='none')
            return torch.nn.functional.relu(x)
        class ConvActivation(torch.nn.Module):
            def forward(self,value):return activation(value,conv)
        original=t.Conv1dBlock.__init__
        def init(self,*args,**kwargs):
            original(self,*args,**kwargs);self.block[4]=ConvActivation()
        t.Conv1dBlock.__init__=init
        def feedforward(value):return activation(value,ffn)
        t.exact_gelu=feedforward;p.exact_gelu=feedforward
    else:
        import jittor as jt
        from jittor import nn
        from quest_repro import tokenizer_jittor as t
        def activation(x,name):
            if name=='mish':return x*nn.softplus(x).tanh()
            if name=='silu':return x*jt.sigmoid(x)
            if name=='gelu':return nn.gelu(x)
            return nn.relu(x)
        class ConvActivation(jt.Module):
            def execute(self,value):return activation(value,conv)
        original=t.Conv1dBlock.__init__
        def init(self,*args,**kwargs):
            original(self,*args,**kwargs);self.block=nn.Sequential(*[self.block[i] for i in range(4)],ConvActivation())
        t.Conv1dBlock.__init__=init
        def encoder(self,value,mask=None):
            normalized=self.norm1(value);value=value+self.dropout1(self.self_attn(normalized,normalized,normalized,mask))
            return value+self.dropout2(self.linear2(self.dropout(activation(self.linear1(self.norm2(value)),ffn))))
        def decoder(self,value,memory,mask=None):
            normalized=self.norm1(value);value=value+self.dropout1(self.self_attn(normalized,normalized,normalized,mask))
            value=value+self.dropout2(self.multihead_attn(self.norm2(value),memory,memory))
            return value+self.dropout3(self.linear2(self.dropout(activation(self.linear1(self.norm3(value)),ffn))))
        t.EncoderLayer.execute=encoder;t.DecoderLayer.execute=decoder
        assert 'torch' not in sys.modules

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--framework',choices=['torch','jittor'],required=True)
    p.add_argument('--conv',choices=['mish','silu','gelu','relu'],required=True);p.add_argument('--ffn',choices=['gelu','silu','relu'],required=True)
    p.add_argument('script',type=Path);p.add_argument('arguments',nargs=argparse.REMAINDER);a=p.parse_args()
    script=a.script.resolve();assert script.is_relative_to(ROOT/'scripts')
    install(a.framework,a.conv,a.ffn);sys.argv=[str(script),*a.arguments];runpy.run_path(str(script),run_name='__main__')
