"""Independent Torch CPU initializers exported for both frameworks;no training."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
    import torch
    from quest_repro.policy_torch import QueSTPolicy
    from quest_repro.config import PolicyConfig
    torch.set_num_threads(2);out=root/'outputs/sampling_stability_20261029';protocol=json.loads((root/'records/sampling_stability_protocol_20261029.json').read_text())
    info={}
    for seed in protocol['training_seeds'][1:]:
        torch.manual_seed(seed);model=QueSTPolicy(PolicyConfig(tasks=5,dropout=0));path=out/f'initial_{seed}.npz';assert not path.exists()
        np.savez_compressed(path,**{k:v.detach().cpu().numpy() for k,v in model.state_dict().items()})
        info[str(seed)]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'torch_seed':seed,'shared_by_conditions_and_frameworks':True}
    assert len(set(v['sha256'] for v in info.values()))==2
    (out/'initialization_manifest.json').write_text(json.dumps(info,indent=2));print(json.dumps(info,indent=2))
if __name__=='__main__':main()
