"""Frozen tokenizer reconstruction audit across datasets; no policy success claim."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from quest_repro.config import TokenizerConfig
from quest_repro.data import ActionChunkDataset,load_manifest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--framework',choices=['torch','jittor'],required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    config=TokenizerConfig(**json.loads(args.config.read_text())['config'])
    manifest=load_manifest(args.data_root,verify_hash=True)
    if args.framework=='torch':
        import torch
        from quest_repro.tokenizer_torch import SkillVAE
        from quest_repro.training import load_npz_state
        torch.set_num_threads(4)
        model=SkillVAE(config).cuda().eval()
        load_npz_state(model,args.weights)
        def predict(actions):
            with torch.no_grad():
                x=torch.from_numpy(actions).cuda()
                indices=model.get_indices(x)
                return model.decode_actions(indices).cpu().numpy(),indices.cpu().numpy()
    else:
        import jittor as jt
        from quest_repro.tokenizer_jittor import SkillVAE,strict_load_npz
        assert 'torch' not in sys.modules
        model=SkillVAE(config)
        strict_load_npz(model,args.weights)
        model.eval()
        def predict(actions):
            with jt.no_grad():
                details=model.forward_details(jt.array(actions))
                return details['prediction'].numpy(),details['indices'].numpy()
    train=ActionChunkDataset(args.data_root,'train')
    mean=np.concatenate(list(train.actions.values())).mean(axis=0)
    report={'framework':args.framework,'weights_sha256':hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        'dataset_sha256':manifest['sha256'],'batch_size':16,
        'scope':'frozen action autoencoder reconstruction; future expert actions are input; no policy rollout',
        'torch_imported':'torch' in sys.modules,'splits':{}}
    tick=time.monotonic()
    for split in ('train','validation'):
        dataset=ActionChunkDataset(args.data_root,split)
        predictions=[];targets=[];masks=[];tokens=[]
        for start in range(0,len(dataset),16):
            items=[dataset[i] for i in range(start,min(start+16,len(dataset)))]
            actions=np.stack([item['actions'] for item in items])
            mask=np.stack([item['pad_mask'] for item in items])
            prediction,indices=predict(actions)
            assert np.isfinite(prediction).all()
            predictions.append(prediction);targets.append(actions);masks.append(mask);tokens.append(indices)
        prediction=np.concatenate(predictions);target=np.concatenate(targets)
        mask=np.concatenate(masks);indices=np.concatenate(tokens)
        error=np.abs(prediction-target); valid=np.broadcast_to(mask,error.shape)
        counts=np.bincount(indices.reshape(-1),minlength=1000)
        complete=mask.all(axis=(1,2))
        metric={'episodes':len(dataset.episodes),'windows':len(dataset),
            'mae_all':float(error.mean(dtype=np.float64)),
            'mae_valid':float(error[valid].mean(dtype=np.float64)),
            'mae_complete_chunks':float(error[complete].mean(dtype=np.float64)),
            'complete_windows':int(complete.sum()),
            'mae_by_dimension':[(error[:,:,i]*mask[:,:,0]).sum(dtype=np.float64)/mask.sum() for i in range(4)],
            'zero_baseline_valid':float(np.abs(target)[valid].mean(dtype=np.float64)),
            'train_mean_baseline_valid':float(np.abs(target-mean)[valid].mean(dtype=np.float64)),
            'hold_first_action_baseline_valid':float(np.abs(target-target[:,:1])[valid].mean(dtype=np.float64)),
            'used_codes':int((counts>0).sum()),
            'window_weighting':'each start within episode,repeat-final-action padding;valid MAE excludes padding;complete-chunk MAE also reported'}
        report['splits'][split]=metric
        np.savez_compressed(args.output/f'{split}.npz',prediction=prediction,target=target,mask=mask,indices=indices,
            episode=np.array([x[0] for x in dataset.index]),start=np.array([x[1] for x in dataset.index]),code_counts=counts)
        print(split,json.dumps(metric),flush=True)
    report['elapsed_seconds']=time.monotonic()-tick
    (args.output/'metrics.json').write_text(json.dumps(report,indent=2))
    print('Finished frozen-tokenizer audit:',args.output,flush=True)


if __name__=='__main__':main()
