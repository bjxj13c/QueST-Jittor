"""Local CPU recheck of the360supplementalrollouts and immutable model hashes."""
import argparse
import json
from pathlib import Path
import numpy as np
from verify_closeout_backups import ROOT,TARGET,verify_merge,digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--workflow',type=Path,required=True);a=p.parse_args()
    workflow=json.loads(a.workflow.read_text(encoding='utf-8'));assert workflow['status']=='completed';info=workflow['archive'];assert a.archive.stat().st_size==info['bytes']
    merged=verify_merge(a.archive,info['sha256']);out=TARGET/'outputs/holding_completion_precision_20261030';unit=TARGET/'outputs/selected_activation_holding_20261030/units/mish_gelu__holding'
    protocol=json.loads((TARGET/'records/holding_completion_inputs_20261030.json').read_text());comparison=json.loads((out/'strict_repaired_comparison.json').read_text());assert comparison['passed'] and comparison['atol']==2e-5 and comparison['rtol']==2e-4
    for rel,expected in protocol['file_hashes'].items():assert digest(unit/rel)==expected,rel
    for rel,expected in protocol['external_sources'].items():assert digest(TARGET/rel)==expected,rel
    with np.load(out/'repaired_fixture_torch.npz') as x,np.load(out/'repaired_fixture_jittor.npz') as y:
        for key in x.files:np.testing.assert_allclose(x[key],y[key],atol=2e-5,rtol=2e-4)
        np.testing.assert_array_equal(x['tokens'],y['tokens']);assert x['tokens'].size==1600
    report=json.loads((TARGET/'reports/SELECTED_HOLDING_RESULTS_COMPLETED_20261030.json').read_text());assert report['completion']['actual_added_rollouts']==360 and not report['failures'];counts={};results={}
    tasks=['reach-v2','reach-wall-v2','push-v2','door-open-v2','window-open-v2'];total=0
    for f in ('torch','jittor'):
        success_counts=[];standard_reach=None
        for task_id,task in enumerate(tasks):
            folder=unit/'dataset/outputs'/f'completion_mish_gelu_holding_{f}_{task}_20261030';s=json.loads((folder/'summary.json').read_text());assert len(s['episodes'])==30
            assert [e['seed'] for e in s['episodes']]==list(range(120000,120030));assert s['task_id']==task_id and s['horizon']==8 and s['top_k']==5 and s['sampling_seed']==130000
            expected=protocol['file_hashes']['policy_torch/best.pt' if f=='torch' else 'policy_jittor/best_weights.npz'];actual=s['policy_sha256' if f=='torch' else 'policy_weights_sha256'];assert actual==expected
            if f=='jittor':assert not s['torch_imported']
            for e in s['episodes']:
                with np.load(folder/f"seed_{e['seed']}.npz") as z:
                    actions=z['predicted_actions'];tokens=z['generated_tokens'];assert actions.shape==(e['steps'],4) and np.isfinite(actions).all();assert tokens.shape==((e['steps']+7)//8,8) and (tokens>=0).all() and (tokens<1000).all()
            success_counts.append(sum(bool(e['success']) for e in s['episodes']));total+=30
            if task_id==0:standard_reach=s['episodes']
        diag=out/f'holding_diagnostic_{f}';s=json.loads((diag/'summary.json').read_text());assert len(s['episodes'])==30 and s['seed_start']==120000 and s['sampling_seed']==130000
        assert [e['success'] for e in standard_reach]==[e['success_within150'] for e in s['episodes']]
        if f=='jittor':assert not s['torch_imported']
        held=0
        for e in s['episodes']:
            with np.load(diag/f"seed_{e['seed']}.npz") as z:
                assert len(z['actions'])==e['steps'] and np.isfinite(z['actions']).all();hits=np.flatnonzero(z['success']);first=int(hits[0])+1 if len(hits) else None;assert first==e['first_success_step']
                post=z['success'][first:first+60] if first is not None else np.array([],dtype=bool);assert len(post)==e['post_steps'];value=bool(len(post)==60 and post.all());assert value==e['held_all60'];held+=value
        total+=30;v=report['results']['mish_gelu']['holding'][f];assert sum(success_counts)==v['successes'] and success_counts==v['task_successes'];assert np.isclose(held/30,v['reach_hold_all60_rate'])
        results[f]={'successes':sum(success_counts),'episodes':150,'task_successes':success_counts,'held_all60':held,'holding_episodes':30};counts[f]=180
    assert total==360
    record={'status':'passed','archive_verification':merged,'actual_rollouts':total,'framework_rollouts':counts,'strict_original_tolerances_preserved':True,'token_ids_exact':True,'original_weights_and_data_hashes_unchanged':True,'numeric_implementation_patch':'frozen Jittor decoder LayerNorm FP64accumulation,FP32params/interfaces','old_failed_reports_preserved':True,'no_training':True,'results':results}
    (ROOT/'reports/HOLDING_COMPLETION_LOCAL_VERIFICATION_20261030.json').write_text(json.dumps(record,indent=2),encoding='utf-8');print(json.dumps(record,indent=2))
if __name__=='__main__':main()
