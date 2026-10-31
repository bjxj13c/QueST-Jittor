"""Isolated ImageNet RGB normalization for explicitly selected vision experiments.

Run EVERY train/fixture/rollout command through this entry point. Original
policy sources and checkpoints are untouched. NPZ keys remain unchanged.
"""
import argparse
import os
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
MEAN=(.485,.456,.406)
STD=(.229,.224,.225)

def install(framework):
    os.environ['QUEST_VISION_NORMALIZATION']='imagenet_v1'
    if framework=='torch':
        import torch
        from quest_repro.policy_torch import ResnetEncoder
        def forward(self,value):
            mean=value.new_tensor(MEAN).reshape(1,3,1,1)
            std=value.new_tensor(STD).reshape(1,3,1,1)
            return self.projection_layer(self.features((value-mean)/std))
        ResnetEncoder.forward=forward
    else:
        import jittor as jt
        from quest_repro.policy_jittor import ResnetEncoder
        def execute(self,value):
            mean=jt.array(MEAN).float32().reshape((1,3,1,1)).stop_grad()
            std=jt.array(STD).float32().reshape((1,3,1,1)).stop_grad()
            value=(value-mean)/std
            value=self.block_4(self.block_3(self.block_2(self.block_1(self.resnet18_base(value)))))
            return self.projection_layer(value)
        ResnetEncoder.execute=execute
        assert 'torch' not in sys.modules

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--framework',choices=['torch','jittor'],required=True)
    p.add_argument('script',type=Path);p.add_argument('arguments',nargs=argparse.REMAINDER);a=p.parse_args()
    script=a.script.resolve();assert script.is_relative_to(ROOT/'scripts')
    install(a.framework)
    sys.argv=[str(script),*a.arguments];runpy.run_path(str(script),run_name='__main__')
