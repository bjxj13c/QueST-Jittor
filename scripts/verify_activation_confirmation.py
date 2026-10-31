"""Independent CPU evidence audit; use after archive and model downloads.

The evidence tree must retain paths relative to the remote project. No model
execution, remote calls, training, or statistical re-selection happens here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

SEEDS = (560500, 560600, 560700)
CASES = ('mish_gelu', 'mish_relu')
NAME = 'activation_confirmation_20261030'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def main(root, destination):
    report = read(root/'reports'/f'{NAME.upper()}.json')
    assert report['status']=='verified_complete' and not report['failures']
    assert report['actual_rollouts']==1800
    tasks = report['protocol']['tasks']
    assert report['protocol']['training_seeds']==list(SEEDS)
    assert report['protocol']['new_scene_seeds']==[150000,150029]
    total = 0
    values = {}
    for seed in SEEDS:
        snap = root/'outputs'/NAME/f'seed_{seed}'
        grid = snap/'outputs/activation_grid_20261029'
        streams = {'tokenizer':set(), 'policy':set()}
        initial = {k:sha(grid/'shared'/f'initial_{k}.npz') for k in streams}
        for case in CASES:
            folder = grid/'cases'/case
            assert read(folder/'workflow.json')['status']=='completed'
            # Whole-model floating checks use the original absolute+relative
            # criterion; integer codes have no floating tolerance.
            for stem in ('initial_probe','learned_tokenizer_probe','policy_fixture'):
                with np.load(folder/f'{stem}_torch.npz') as a, np.load(folder/f'{stem}_jittor.npz') as b:
                    assert set(a.files)==set(b.files)
                    for key in a.files:
                        if key in ('indices','tokens'):
                            np.testing.assert_array_equal(a[key],b[key])
                        else:
                            assert np.isfinite(a[key]).all() and np.isfinite(b[key]).all()
                            np.testing.assert_allclose(a[key],b[key],atol=2e-5,rtol=2e-4)
            freeze = read(folder/'evaluation_freeze.json')
            assert sha(folder/'assets/weights.npz')==freeze['tokenizer']
            for f in ('torch','jittor'):
                assert sha(folder/f'policy_{f}/best_weights.npz')==freeze['models'][f]
                for kind, budget in (('tokenizer',22180), ('policy',12000)):
                    base = folder/f'{kind}_{f}'
                    s,c = read(base/'summary.json'),read(base/'config.json')
                    assert s['status']=='completed' and s['updates']==budget
                    assert c['conv_activation']=='mish' and c['ffn_activation']==case.split('_')[1]
                    assert c['dataset_sha256']==report['protocol']['dataset_sha256']
                    assert c['initial_sha256' if kind=='tokenizer' else 'initial_policy_npz_sha256']==initial[kind]
                    assert not s.get('nonfinite_loss_or_gradient',False)
                    if kind=='tokenizer':assert s['finite']
                    if f=='jittor':assert not s['torch_imported']
                    rows = [json.loads(v) for v in (base/'gradients.jsonl').read_text().splitlines()]
                    assert [v['step'] for v in rows]==list(range(1,budget+1))
                    key = 'gradient_norm' if kind=='tokenizer' else 'preclip_norm'
                    assert np.isfinite([v[key] for v in rows]).all()
                    streams[kind].add(s['batch_stream_sha256'])
                observed = []
                for task_id, task in enumerate(tasks):
                    base = folder/'dataset/outputs'/f'activation_{case}_{f}_{task}'
                    s = read(base/'summary.json')
                    assert len(s['episodes'])==30 and s['task_id']==task_id
                    assert s['horizon']==8 and s['top_k']==5 and s['max_steps']==150 and s['sampling_seed']==160000
                    assert [v['seed'] for v in s['episodes']]==list(range(150000,150030))
                    expected = sha(folder/('policy_torch/best.pt' if f=='torch' else 'policy_jittor/best_weights.npz'))
                    assert s['policy_sha256' if f=='torch' else 'policy_weights_sha256']==expected
                    if f=='jittor':assert not s['torch_imported']
                    for e in s['episodes']:
                        with np.load(base/f"seed_{e['seed']}.npz") as z:
                            a,t = z['predicted_actions'],z['generated_tokens']
                            assert a.shape==(e['steps'],4) and np.isfinite(a).all()
                            assert t.shape==((e['steps']+7)//8,8) and (t>=0).all() and (t<1000).all()
                    observed.append([int(e['success']) for e in s['episodes']])
                    total += 30
                key = f'{seed}_{case}_{f}'
                np.testing.assert_array_equal(observed,report['outcomes'][key])
                values[key] = np.asarray(observed)
                r = next(v for v in report['rows'] if v['seed']==seed and v['case']==case and v['framework']==f)
                assert r['successes']==int(values[key].sum()) and r['task_successes']==values[key].sum(axis=1).tolist()
        assert all(len(v)==1 for v in streams.values())
    assert total==1800
    effects = {}
    for f in ('torch','jittor'):
        deltas=np.array([(values[f'{s}_mish_relu_{f}']-values[f'{s}_mish_gelu_{f}']).mean() for s in SEEDS])
        v=report['effects'][f]
        np.testing.assert_allclose(deltas,v['paired_seed_deltas'],atol=1e-12,rtol=0)
        half=4.3026527299*deltas.std(ddof=1)/np.sqrt(3)
        np.testing.assert_allclose([deltas.mean()-half,deltas.mean()+half],v['paired_seed_t95_df2'],atol=1e-12,rtol=0)
        effects[f]={'paired_seed_deltas':deltas.tolist(),'mean_delta':float(deltas.mean()),'paired_seed_t95_df2':v['paired_seed_t95_df2']}
    record={'status':'passed','actual_rollouts':total,'training_seeds':list(SEEDS),'all_raw_records_and_fixtures_checked':True,
            'original_strict_tolerances_preserved':True,'initializations_batches_budgets_verified':True,'downloaded_policy_hashes_verified':True,
            'paired_seed_statistics_verified':effects,'no_model_execution_or_training':True}
    destination.write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evidence-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    main(a.evidence_root.resolve(),a.output.resolve())
