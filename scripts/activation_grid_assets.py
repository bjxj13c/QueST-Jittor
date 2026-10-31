"""Common fresh initialization and combo-specific static native assets/labels."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--case');a=p.parse_args();root=a.root.resolve();out=root/'outputs/activation_grid_20261029'
    import torch
    from quest_repro.config import PolicyConfig,TokenizerConfig
    from quest_repro.policy_torch import QueSTPolicy
    from quest_repro.tokenizer_torch import SkillVAE
    from quest_repro.training import load_npz_state
    torch.set_num_threads(2)
    if not a.case:
        import os
        assert 'QUEST_ACTIVATION_CONV' not in os.environ,'Generate shared init with original parameter schema'
        assets=out/'shared';assets.mkdir(exist_ok=False);torch.manual_seed(520500);tokenizer=SkillVAE(TokenizerConfig(attn_pdrop=0))
        torch.manual_seed(520501);policy=QueSTPolicy(PolicyConfig(tasks=5,dropout=0))
        for name,model in [('tokenizer',tokenizer),('policy',policy)]:
            np.savez_compressed(assets/f'initial_{name}.npz',**{k:v.detach().numpy() for k,v in model.state_dict().items()});(assets/f'{name}_config.json').write_text(json.dumps({'config':model.config.to_dict()},indent=2))
        source=root/'outputs/five_task_10pct_20261029/dataset/records/reach_split_v1.json';manifest=json.loads(source.read_text());assert len(manifest['train'])==450 and len(manifest['validation'])==100
        protocol=json.loads((root/'records/activation_grid_protocol_20261029.json').read_text())
        for name in protocol['cases']:
            data=out/'cases'/name/'dataset';(data/'records').mkdir(parents=True)
            m=dict(manifest);m['dataset']='../../../../five_task_10pct_20261029/dataset/'+manifest['dataset']
            assert (data/m['dataset']).resolve()==(root/'outputs/five_task_10pct_20261029/dataset'/manifest['dataset']).resolve()
            (data/'records/reach_split_v1.json').write_text(json.dumps(m,indent=2))
        (assets/'initialization_manifest.json').write_text(json.dumps({n:hashlib.sha256((assets/f'initial_{n}.npz').read_bytes()).hexdigest() for n in ('tokenizer','policy')},indent=2));return
    import os
    assert a.case==os.environ['QUEST_ACTIVATION_CONV']+'_'+os.environ['QUEST_ACTIVATION_FFN']
    case=out/'cases'/a.case;assets=case/'assets';assets.mkdir(exist_ok=False);source=case/'tokenizer_jittor'
    cfg=json.loads((source/'config.json').read_text())['config'];selected=json.loads((source/'best_validation.json').read_text())
    tokenizer=SkillVAE(TokenizerConfig(**cfg)).cuda().eval();load_npz_state(tokenizer,source/'best_weights.npz')
    np.savez_compressed(assets/'weights.npz',**{k:v.detach().cpu().numpy() for k,v in tokenizer.state_dict().items()})
    torch.save({'model':{k:v.detach().cpu() for k,v in tokenizer.state_dict().items()},'config':cfg,'step':selected['step'],'extra':{'activation_case':a.case,'native_source_sha256':hashlib.sha256((source/'best_weights.npz').read_bytes()).hexdigest()}},assets/'tokenizer_shared.pt')
    (assets/'config.json').write_text(json.dumps({'config':cfg,'activation_case':a.case,'source':'learned native tokenizer;static conversion shared by paired policies','selected_step':selected['step']},indent=2))
    from quest_repro.policy_data import PolicyDataset
    labels=assets/'canonical_labels';labels.mkdir()
    for split in ('train','validation'):
        data=PolicyDataset(case/'dataset',split,tokenizer,assets/'weights.npz',batch_size=16,device='cuda');np.savez_compressed(labels/(split+'.npz'),labels=data.labels)
    print('Combo-specific native tokenizer and canonical labels exported',flush=True)
if __name__=='__main__':main()
