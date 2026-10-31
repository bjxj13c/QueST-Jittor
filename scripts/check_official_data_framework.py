"""Separate-process framework tensor roundtrip on the same real action chunks."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from quest_repro.data import ActionChunkDataset

parser=argparse.ArgumentParser()
parser.add_argument('--run',type=Path,required=True)
parser.add_argument('--framework',choices=['torch','jittor'],required=True)
args=parser.parse_args()
dataset=ActionChunkDataset(args.run/'adapted',include_observations=True)
samples=[dataset[i] for i in (0,len(dataset)//2,len(dataset)-1)]
arrays={'actions':np.stack([x['actions'] for x in samples]),
        'images':np.stack([x['obs']['corner_rgb'] for x in samples]),
        'robot_states':np.stack([x['obs']['robot_states'] for x in samples]),
        'pad_mask':np.stack([x['pad_mask'] for x in samples]).astype(np.int32)}
if args.framework=='torch':
    import torch
    values={k:torch.from_numpy(v.copy()).numpy() for k,v in arrays.items()}
else:
    import jittor as jt
    assert 'torch' not in sys.modules
    values={k:jt.array(v).numpy() for k,v in arrays.items()}
for key in arrays:
    np.testing.assert_array_equal(values[key],arrays[key])
np.savez(args.run/f'batch_{args.framework}.npz',**values)
if args.framework=='jittor':
    with np.load(args.run/'batch_torch.npz') as reference:
        for key,value in values.items():
            np.testing.assert_array_equal(value,reference[key])
result={'framework':args.framework,'roundtrip_exact':True,
        'cross_framework_exact':args.framework=='jittor',
        'torch_imported':'torch' in sys.modules,
        'shapes':{k:list(v.shape) for k,v in values.items()}}
(args.run/f'batch_{args.framework}_check.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
