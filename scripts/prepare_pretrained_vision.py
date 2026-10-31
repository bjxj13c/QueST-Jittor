"""Export public ImageNet ResNet18 partial-backbone weights to shared NPZ.

This exporter uses Torch on CPU only; native Jittor consumes static arrays.
Non-backbone parameters are byte-identical to the existing random initializer.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--initial',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    if any(a.output.iterdir()):raise FileExistsError('Preserve existing export')
    a.cache.mkdir(parents=True,exist_ok=True)
    os.environ['TORCH_HOME']=str(a.cache.resolve())
    import torch
    from torchvision.models import resnet18,ResNet18_Weights
    torch.set_num_threads(2)
    weights=ResNet18_Weights.IMAGENET1K_V1
    # torchvision checks the published filename SHA256 prefix on download.
    reference=weights.get_state_dict(progress=True,check_hash=True)
    net=resnet18(weights=None);net.load_state_dict(reference,strict=True)
    # Legacy public checkpoint omits BN counters; torchvision supplies zeros.
    reference=net.state_dict()
    with np.load(a.initial,allow_pickle=False) as z:base={k:z[k].copy() for k in z.files}
    changed={};prefix='image_encoders.corner_rgb.'
    for name in base:
        if not name.startswith(prefix):continue
        suffix=name[len(prefix):];source=None
        if suffix.startswith('resnet18_base.0.'):source='conv1.'+suffix.split('.',2)[2]
        elif suffix.startswith('resnet18_base.1.'):source='bn1.'+suffix.split('.',2)[2]
        else:
            for b,s in [('block_1','layer1.0'),('block_2','layer1.1'),('block_3','layer2.0'),('block_4','layer2.1')]:
                if suffix.startswith(b+'.'):source=s+suffix[len(b):];break
        if source:
            value=reference[source].detach().cpu().numpy();assert value.shape==base[name].shape,(name,value.shape,base[name].shape)
            changed[name]=source;base[name]=value.astype(base[name].dtype,copy=True)
    assert len(changed)==60,len(changed)
    np.savez_compressed(a.output/'imagenet_initial.npz',**base)
    with np.load(a.initial) as original:
        for name in original.files:
            if name not in changed:np.testing.assert_array_equal(original[name],base[name])
    checkpoint=a.cache/'hub/checkpoints'/Path(weights.url).name
    m={'source_url':weights.url,'weights_enum':'ResNet18_Weights.IMAGENET1K_V1',
       'source_checkpoint_sha256':sha(checkpoint),'original_initial_sha256':sha(a.initial),
       'pretrained_initial_sha256':sha(a.output/'imagenet_initial.npz'),'mapped_keys':changed,
       'untouched_non_backbone_verified':True,'input':'uint8 RGB /255, then ImageNet mean/std;128x128;no ImageNet center crop',
       'scope':'stem+layer1+layer2 only;all layers fine-tuned;not VLM;not full ResNet18'}
    (a.output/'manifest.json').write_text(json.dumps(m,indent=2),encoding='utf-8');print(json.dumps(m,indent=2))

if __name__=='__main__':main()
