"""Fresh paired-seed confirmation of Mish/ReLU versus Mish/GELU.

Reuses the audited grid worker in isolated source snapshots, not old checkpoints.
No edits to original experiments, no retries of failed statistical units.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import zipfile
import numpy as np

NAME = 'activation_confirmation_20261030'
SEEDS = (560500, 560600, 560700)
CASES = ('mish_gelu', 'mish_relu')


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def write(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_suffix('.tmp')
    q.write_text(json.dumps(v, indent=2), encoding='utf-8')
    q.replace(p)


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def module(snapshot):
    spec = importlib.util.spec_from_file_location('confirmation_worker', snapshot/'scripts/run_activation_grid.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def prepare(root, out, seed, protocol):
    snap = out/f'seed_{seed}'
    snap.mkdir(exist_ok=False)
    for name in ('records', 'reports', 'outputs'):
        (snap/name).mkdir()
    for name in ('src', 'cache', '.venv-torch-linux', '.venv-jittor-linux'):
        (snap/name).symlink_to(root/name, target_is_directory=True)
    (snap/'outputs/five_task_10pct_20261029').symlink_to(root/'outputs/five_task_10pct_20261029', target_is_directory=True)
    shutil.copytree(root/'scripts', snap/'scripts', ignore=shutil.ignore_patterns('__pycache__'))
    # Root-local snapshots preserve all old hardcoded relative data paths.
    runner = snap/'scripts/run_activation_grid.py'
    text = runner.read_text(encoding='utf-8')
    for a, b in [('100000', '150000'), ('100030', '150030'), ('110000', '160000')]:
        text = text.replace(a, b)
    runner.write_text(text, encoding='utf-8')
    assets = snap/'scripts/activation_grid_assets.py'
    text = assets.read_text(encoding='utf-8').replace('manual_seed(520500)', f'manual_seed({seed})').replace('manual_seed(520501)', f'manual_seed({seed+1})')
    assets.write_text(text, encoding='utf-8')
    # Reuse the same physical compilation cache, including the same cache key path.
    launcher = snap/'scripts/run_jittor_linux.py'
    text = launcher.read_text(encoding='utf-8')
    for value in ('cache/jittor_home', 'cache/tmp', 'cache/matplotlib'):
        text = text.replace(f'root / "{value}"', f'(root / "{value}").resolve()')
    launcher.write_text(text, encoding='utf-8')
    p = read(root/'records/activation_grid_protocol_20261029.json')
    p.update(cases=list(CASES), seed=seed, initialization=f'fresh paired tokenizer seed{seed},policy seed{seed+1}', scope='fresh confirmation; original selection seed excluded')
    p['evaluation'].update(scene_seeds=[150000,150029], sampling_seed=160000, total_rollouts=600)
    write(snap/'records/activation_grid_protocol_20261029.json', p)
    grid = snap/'outputs/activation_grid_20261029'
    grid.mkdir()
    env = module(snap).environment(snap)
    with (snap/'initialization.log').open('xb') as log:
        subprocess.run([str(root/'.venv-torch-linux/bin/python'), str(assets), '--root', str(snap)], cwd=snap, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
    files = list((snap/'scripts').glob('*.py')) + list((root/'src/quest_repro').glob('*.py'))
    freeze = {str(q.relative_to(snap) if q.is_relative_to(snap) else Path('src/quest_repro')/q.name): sha(q) for q in files}
    freeze.update({f'outputs/activation_grid_20261029/shared/initial_{k}.npz': sha(grid/'shared'/f'initial_{k}.npz') for k in ('tokenizer','policy')})
    freeze['records/activation_grid_protocol_20261029.json'] = sha(snap/'records/activation_grid_protocol_20261029.json')
    write(grid/'freeze.json', freeze)
    return snap


def summarize(root):
    out = root/'outputs'/NAME
    protocol = read(root/'records'/f'{NAME}_protocol.json')
    state = read(out/'workflow.json')
    rows, outcomes, errors = [], {}, []
    for seed in SEEDS:
        snap = out/f'seed_{seed}'
        g = module(snap)
        grid = snap/'outputs/activation_grid_20261029'
        for rel, h in read(grid/'freeze.json').items():
            assert sha(snap/rel) == h, rel
        for case in CASES:
            unit = f'{seed}_{case}'
            folder = grid/'cases'/case
            w = read(folder/'workflow.json') if (folder/'workflow.json').exists() else {'status':'not_started'}
            if w['status'] != 'completed':
                errors.append({'unit':unit, 'workflow':w})
                continue
            g.verify_case(snap, case)
            for f in ('torch','jittor'):
                values, task_counts = [], []
                for task in protocol['tasks']:
                    base = folder/'dataset/outputs'/f'activation_{case}_{f}_{task}'
                    s = read(base/'summary.json')
                    actual_hash = s['policy_sha256' if f == 'torch' else 'policy_weights_sha256']
                    assert actual_hash == sha(folder/('policy_torch/best.pt' if f == 'torch' else 'policy_jittor/best_weights.npz'))
                    for e in s['episodes']:
                        with np.load(base/f"seed_{e['seed']}.npz") as z:
                            a, t = z['predicted_actions'], z['generated_tokens']
                            assert a.shape == (e['steps'],4) and np.isfinite(a).all()
                            assert t.shape == ((e['steps']+7)//8,8) and (t >= 0).all() and (t < 1000).all()
                    v = [int(e['success']) for e in s['episodes']]
                    values.append(v)
                    task_counts.append(sum(v))
                outcomes[f'{seed}_{case}_{f}'] = values
                tok = read(folder/f'tokenizer_{f}/best_validation.json')
                ts = read(folder/f'tokenizer_{f}/summary.json')
                ps = read(folder/f'policy_{f}/summary.json')
                rows.append({'seed':seed, 'case':case, 'framework':f, 'successes':sum(task_counts), 'episodes':150, 'task_successes':task_counts,
                             'tokenizer_mae_valid':tok['mae_valid'], 'tokenizer_selected_step':tok['step'], 'policy_best_step':ps['best_step'],
                             'policy_validation_ce':ps['best_validation_ce'], 'tokenizer_training_seconds_contended':ts['elapsed_seconds'], 'policy_training_seconds_contended':ps['elapsed_seconds'],
                             'policy_gradient_max':ps['preclip_gradient_norm_max'], 'policy_clip_fraction':ps['gradient_clip_fraction'],
                             'tokenizer_curve':[json.loads(v) for v in (folder/f'tokenizer_{f}/metrics.jsonl').read_text().splitlines()],
                             'policy_curve':[json.loads(v) for v in (folder/f'policy_{f}/metrics.jsonl').read_text().splitlines()]})
    effects = {}
    if not errors:
        for kind in ('tokenizer','policy'):
            for seed in SEEDS:
                streams = {read(out/f'seed_{seed}/outputs/activation_grid_20261029/cases'/case/f'{kind}_{f}/summary.json')['batch_stream_sha256'] for case in CASES for f in ('torch','jittor')}
                assert len(streams) == 1, (seed,kind)
        for f in ('torch','jittor'):
            diffs = np.array([np.asarray(outcomes[f'{s}_mish_relu_{f}'])-np.asarray(outcomes[f'{s}_mish_gelu_{f}']) for s in SEEDS])
            seed_deltas = diffs.mean(axis=(1,2))
            # Independent training replicates, not 450 independent training runs.
            half = 4.3026527299 * seed_deltas.std(ddof=1)/np.sqrt(3)
            rng = np.random.default_rng(560800)
            draws = []
            for _ in range(10000):
                chosen = rng.integers(0,3,3)
                draws.append(np.mean([diffs[s,t,rng.integers(0,30,30)].mean() for s in chosen for t in range(5)]))
            effects[f] = {'paired_seed_deltas':seed_deltas.tolist(), 'mean_delta':float(seed_deltas.mean()), 'paired_seed_t95_df2':[float(seed_deltas.mean()-half),float(seed_deltas.mean()+half)],
                          'hierarchical_paired_bootstrap95':np.percentile(draws,[2.5,97.5]).tolist(), 'positive_seeds':int((seed_deltas>0).sum()),
                          'baseline_mean':float(np.mean([outcomes[f'{s}_mish_gelu_{f}'] for s in SEEDS])), 'candidate_mean':float(np.mean([outcomes[f'{s}_mish_relu_{f}'] for s in SEEDS]))}
    report = {'status':'verified_complete' if not errors else 'partial_with_failures', 'protocol':protocol, 'rows':rows, 'outcomes':outcomes, 'effects':effects, 'failures':errors,
              'actual_rollouts':len(rows)*150, 'limitations':['only three fresh training seeds; intervals have substantial uncertainty','five tasks, same 450/100 expert dataset; not full paper benchmark','winner selected before confirmation; original screening scores excluded','framework policies share their seed/case native trained tokenizer','different token vocabularies: absolute policy CE is not a success-rate comparison','no holding-data changes or decoder precision patch applied; original gates preserved','training times measured under contention, not isolated speed rankings']}
    target = root/'reports'/NAME.upper()
    write(target.with_suffix('.json'), report)
    lines = ['# Mish＋ReLU独立种子验证', '', '|种子|框架|Mish＋GELU成功/150|Mish＋ReLU成功/150|差值|', '|---|---|---:|---:|---:|']
    for seed in SEEDS:
        for f in ('torch','jittor'):
            a = next((r for r in rows if r['seed']==seed and r['framework']==f and r['case']=='mish_gelu'),None)
            b = next((r for r in rows if r['seed']==seed and r['framework']==f and r['case']=='mish_relu'),None)
            if a and b:lines.append(f"|{seed}|{f}|{a['successes']}|{b['successes']}|{(b['successes']-a['successes'])/150:+.2%}|")
    for f,v in effects.items():
        lines += ['', f"{f}：平均{v['baseline_mean']:.2%}→{v['candidate_mean']:.2%}，差值{v['mean_delta']:+.2%}；按训练种子的配对t区间为[{v['paired_seed_t95_df2'][0]:+.2%}, {v['paired_seed_t95_df2'][1]:+.2%}]。{v['positive_seeds']}/3个种子提高。"]
    lines += ['', '三种子与五任务仍有明显局限，不把场景次数当作独立训练次数。若区间跨零，则不能确认稳定提升；即使均为正，也不外推至完整论文基准。失败组保留，不筛掉负结果。', '', '重建误差、梯度、收敛曲线与并发训练耗时见JSON。']
    target.with_suffix('.md').write_text('\n'.join(lines), encoding='utf-8')
    return report


def main(root):
    import fcntl
    out = root/'outputs'/NAME
    out.mkdir(exist_ok=False)
    lock = (out/'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol = read(root/'records'/f'{NAME}_protocol.json')
    state = {'status':'preparing', 'started_utc':datetime.now(timezone.utc).isoformat(), 'units':{f'{s}_{c}':'queued' for s in SEEDS for c in CASES}, 'active':[], 'max_workers':4, 'no_shutdown':True}
    def save():write(out/'workflow.json',state)
    save()
    children = {}
    try:
        original = read(root/'records/activation_grid_protocol_20261029.json')
        manifest = read(root/'outputs/five_task_10pct_20261029/dataset/records/reach_split_v1.json')
        assert sha(root/'outputs/five_task_10pct_20261029/dataset'/manifest['dataset']) == original['dataset_sha256'] == protocol['dataset_sha256']
        assert len(manifest['train'])==450 and len(manifest['validation'])==100
        for seed in SEEDS:prepare(root,out,seed,protocol)
        pending = [(s,c) for s in SEEDS for c in CASES]
        started = time.monotonic()
        while pending or children:
            for unit, entry in list(children.items()):
                code = entry['process'].poll()
                if code is not None:
                    entry['log'].close()
                    state['units'][unit] = 'completed' if code == 0 else 'failed'
                    del children[unit]
            resource = module(out/f'seed_{SEEDS[0]}').gpu_resource()
            state.update(status='running', resource=resource, active=list(children), elapsed_seconds=time.monotonic()-started)
            # Reserve 0.6GiB for each new case plus1GiB for logs/cache; never delete evidence.
            can_start = resource['free_vram_mib']>=6144 and resource['disk_free_gib']>=1.6
            if pending and len(children)<4 and can_start:
                seed,case = pending.pop(0)
                snap = out/f'seed_{seed}'
                unit = f'{seed}_{case}'
                log = (snap/'outputs/activation_grid_20261029/cases'/case/'worker_launcher.log').open('xb')
                child = subprocess.Popen(['nice','-n','10',str(root/'.venv-torch-linux/bin/python'),str(snap/'scripts/run_activation_grid.py'),'--root',str(snap),'--case',case], cwd=snap,env=module(snap).environment(snap),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                children[unit] = {'process':child,'log':log}
                state['units'][unit]='running'
                state['active']=list(children)
            if pending and not children and not can_start:
                raise RuntimeError('Insufficient disk/VRAM to start remaining units; evidence preserved, no cleanup or rental')
            with (out/'resources.jsonl').open('a') as log:log.write(json.dumps({'unix':time.time(),**resource,'active':list(children)})+'\n')
            save()
            time.sleep(10)
        state['status']='auditing';save()
        report = summarize(root)
        state['status']='completed' if report['status']=='verified_complete' else 'completed_with_failures'
        state['finished_utc']=datetime.now(timezone.utc).isoformat();save()
        # Small evidence archive; trained weights remain on server, no multi-GB duplicate.
        archive=root/'reports'/f'{NAME}_evidence.zip'
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
            for p in out.rglob('*'):
                if p.is_file() and not p.is_symlink() and p.suffix in ('.json','.jsonl','.csv','.log','.py','.npz'):
                    if p.name.endswith('_weights.npz') or p.name.startswith('initial_') or p.name=='weights.npz' or 'canonical_labels' in p.parts:continue
                    z.write(p,p.relative_to(root))
            for p in [root/'records'/f'{NAME}_protocol.json',root/'scripts/validate_activation_winner.py',root/'reports'/f'{NAME.upper()}.json',root/'reports'/f'{NAME.upper()}.md']:z.write(p,p.relative_to(root))
        state['archive']={'path':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive)};save()
    except BaseException as e:
        state.update(status='failed',error=f'{type(e).__name__}: {e}',active=list(children));save()
        # Keep independently running workers and all artifacts for diagnosis.
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--summarize',action='store_true');a=p.parse_args()
    if a.summarize:summarize(a.root.resolve())
    else:main(a.root.resolve())
