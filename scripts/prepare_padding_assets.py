"""Static native-tokenizer export and canonical labels;Torch only for conversion."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--condition',choices=['repeat','masked','holding','initial'],required=True);a=p.parse_args();base=a.root/'outputs/padding_hold_20261029'
    import torch
    from quest_repro.config import PolicyConfig,TokenizerConfig
    from quest_repro.policy_torch import QueSTPolicy
    from quest_repro.tokenizer_torch import SkillVAE
    from quest_repro.training import load_npz_state
    from quest_repro.policy_data import PolicyDataset
    torch.set_num_threads(4)
    if a.condition=='initial':
        torch.manual_seed(510500);model=QueSTPolicy(PolicyConfig(tasks=1,dropout=0));np.savez_compressed(base/'initial_policy.npz',**{k:v.detach().numpy() for k,v in model.state_dict().items()});return
    folder=base/a.condition;assets=folder/'assets';assets.mkdir(exist_ok=False)
    source=folder/'tokenizer_jittor';weights=source/'best_weights.npz';cfg=json.loads((source/'config.json').read_text())['config'];model=SkillVAE(TokenizerConfig(**cfg)).cuda().eval();load_npz_state(model,weights)
    np.savez_compressed(assets/'weights.npz',**{k:v.detach().cpu().numpy() for k,v in model.state_dict().items()})
    torch.save({'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'config':cfg,'extra':{},'step':2000},assets/'tokenizer_shared.pt')
    (assets/'config.json').write_text(json.dumps({'config':cfg,'source':'native learned tokenizer converted statically;no Torch execution in Jittor'},indent=2))
    root=base/'collection'/('holding' if a.condition=='holding' else 'repeat');labels=assets/'canonical_labels';labels.mkdir()
    for split in ['train','validation']:
        d=PolicyDataset(root,split,model,assets/'weights.npz',batch_size=16,device='cuda');np.savez_compressed(labels/(split+'.npz'),labels=d.labels)
    print('Native tokenizer assets/canonical labels prepared',flush=True)
if __name__=='__main__':main()
