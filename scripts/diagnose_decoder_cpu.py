"""Compare saved GPU actions with CPU FP32 and a diagnostic FP64 reference.

No trained weights,original tolerances,or saved GPU fixtures are modified.
FP64 preserves original FP32 FSQ codes and positional constants before upcast.
It is a diagnostic reference,not a substitute for the required GPU gate.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from run_activation_variant import install
from quest_repro.config import TokenizerConfig
from quest_repro.tokenizer_torch import SkillVAE
from quest_repro.training import load_npz_state

def main():
    torch.set_num_threads(2);install('torch','mish','gelu')
    base=ROOT/'reports/decoder_diagnostic_20261030/outputs/selected_activation_holding_20261030/units/mish_gelu__holding'
    assets=base/'assets';cfg=TokenizerConfig(**json.loads((assets/'config.json').read_text())['config']);model=SkillVAE(cfg).eval();load_npz_state(model,assets/'weights.npz')
    double=copy.deepcopy(model).double().eval()
    with np.load(base/'policy_fixture_torch.npz') as a,np.load(base/'policy_fixture_jittor.npz') as b:
        tokens=a['tokens'].copy();assert np.array_equal(tokens,b['tokens']);original={'torch_gpu':a['actions'].copy(),'jittor_gpu':b['actions'].copy()}
    cpu=[];reference=[]
    with torch.no_grad():
        for start in range(0,len(tokens),16):
            ids=torch.from_numpy(tokens[start:start+16].astype(np.int64));cpu.append(model.decode_actions(ids).numpy())
            raw_codes=model.vq._indices_to_codes(ids).double();memory=double.vq.project_out(raw_codes)
            queries=model.fixed_positional_emb(torch.zeros((len(ids),cfg.skill_block_size,cfg.decoder_dim))).double()
            mask=torch.nn.Transformer.generate_square_subsequent_mask(cfg.skill_block_size).double()
            hidden=double.decoder(queries,memory,tgt_mask=mask,tgt_is_causal=True);reference.append(double.action_head(hidden).numpy())
    cpu=np.concatenate(cpu);reference=np.concatenate(reference);target=original['jittor_gpu'];bad=np.abs(original['torch_gpu']-target)>(2e-5+2e-4*np.abs(target));indices=np.argwhere(bad);assert len(indices)==1;point=tuple(int(v) for v in indices[0])
    comparisons={}
    for name,array in {**original,'torch_cpu_fp32':cpu}.items():
        error=np.abs(array-reference);comparisons[name]={'max_error_vs_fp64':float(error.max()),'mean_error_vs_fp64':float(error.mean()),'at_original_failed_point':float(array[point]),'error_at_original_failed_point':float(error[point])}
    report={'status':'diagnostic_completed_cpu_only','original_gpu_gate_still_failed':True,'original_atol':2e-5,'original_rtol':2e-4,'all1600_tokens_exact':True,'failed_index':list(point),
        'original_absolute_difference':float(abs(original['torch_gpu'][point]-target[point])),'original_allowed_error':float(2e-5+2e-4*abs(target[point])),
        'fp64_at_failed_point':float(reference[point]),'comparisons':comparisons,'weights_sha256':hashlib.sha256((assets/'weights.npz').read_bytes()).hexdigest(),
        'scope':'same learned native tokenizer weights,fixed common tokens;decode stage only;no training or GPU execution',
        'limitations':['FP64 reference preserves CPU-computed FP32 positional constants;not a proof of exact GPU kernel origin','does not replace original GPU alignment gate','missing unit still requires approved compatible GPU evaluation after diagnosis']}
    np.savez_compressed(ROOT/'reports/decoder_cpu_reference_20261030.npz',tokens=tokens,cpu_fp32=cpu,cpu_fp64_reference=reference)
    (ROOT/'reports/DECODER_CPU_DIAGNOSIS_20261030.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
