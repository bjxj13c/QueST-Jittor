"""Run the pinned author's collector unchanged, restricted to Reach.

The upstream checkout stays read-only. Only the task list, sample count,
output directories and explicit random seed are overridden.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import runpy
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--seed', type=int, default=302810)
    parser.add_argument('--tasks',nargs='+',default=['reach-v2'])
    args = parser.parse_args()
    if not 1 <= args.episodes <= 100:
        parser.error('Use 1..100 episodes per explicit run')
    root = Path(__file__).resolve().parents[1]
    upstream = root / 'upstream/QueST'
    if (upstream / '.git').exists():
        commit = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    else:
        commit = (upstream / '.quest-upstream-commit').read_text().strip()
    if commit != 'cc25a835a00cf2d91df69704bdd757904d13c061':
        raise ValueError('Unexpected upstream commit')
    output = args.output.resolve()
    if not output.is_relative_to(root):
        raise ValueError('Output must stay within project')
    # Refuse to resume partial data or overwrite any prior run.
    output.mkdir(parents=True, exist_ok=False)
    os.environ['MUJOCO_GL'] = 'egl'
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    os.environ['WANDB_MODE'] = 'disabled'
    sys.path.insert(0, str(upstream))
    import numpy as np
    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Upstream frame_stack imports a helper added after Gymnasium 0.29.1.
    # Keep the pinned simulator/renderer version. For this Dict-of-Box space,
    # provide only its zero-array constructor; reset padding overwrites it.
    import importlib.util
    import types
    compatibility = 'none'
    if importlib.util.find_spec('gymnasium.wrappers.utils') is None:
        import gymnasium as gym
        helper = types.ModuleType('gymnasium.wrappers.utils')
        def create_zero_array(space):
            if isinstance(space, gym.spaces.Dict):
                return {key:create_zero_array(value) for key,value in space.spaces.items()}
            if isinstance(space, gym.spaces.Box):
                return np.zeros(space.shape, dtype=space.dtype)
            raise TypeError('Compatibility helper supports only Dict/Box')
        helper.create_zero_array = create_zero_array
        sys.modules[helper.__name__] = helper
        compatibility = 'Dict/Box zero initializer supplied; reset replaces padding'
    import quest.utils.metaworld_utils as mu

    # Preserve ML45_PRISE benchmark construction, tasks, expert, wrapper,
    # runner, Gaussian noise, stopping rules and upstream HDF5 writer.
    original_get_env_names = mu.get_env_names
    def reach_only(benchmark=None, mode=None):
        names = original_get_env_names(benchmark, mode)
        if benchmark == 'ML45_PRISE' and mode == 'train':
            if not set(args.tasks)<=set(names):raise ValueError('Requested task not in ML45_PRISE train split')
            return args.tasks
        return names
    mu.get_env_names = reach_only
    with initialize_config_dir(config_dir=str(upstream / 'config'), version_base=None):
        cfg = compose(config_name='collect_data', overrides=[
            f'task.demos_per_env={args.episodes}', f'seed={args.seed}',
            f'data_prefix={output.as_posix()}/raw',
            f'output_prefix={output.as_posix()}/videos',
        ])
    # Author's collection-only config has unused dataset interpolations that
    # require training config keys; preserve these without resolving them.
    OmegaConf.save(cfg, output / 'collection_config.yaml', resolve=False)
    metadata = {'upstream_commit': commit, 'seed': args.seed,
        'episodes_requested_per_task': args.episodes, 'task_filter': args.tasks,
        'scope': 'author-script expert collection; not learned-policy performance',
        'overrides': 'task list, demo count, seed, output directories only',
        'compatibility': compatibility,
        'gymnasium_version': __import__('gymnasium').__version__,
        'noise_std': float(cfg.task.demo_noise),
        'termination': 'upstream wrapper terminates on success; underlying environment truncates at its horizon',
        'source_hashes': {str(p.relative_to(upstream)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [upstream/'scripts/generate_metaworld_dataset.py',
                      upstream/'quest/utils/metaworld_utils.py',
                      upstream/'quest/env_runner/metaworld_runner.py']}}
    (output / 'provenance.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    module = runpy.run_path(str(upstream / 'scripts/generate_metaworld_dataset.py'), run_name='official_collector')
    module['main'].__wrapped__(cfg)
    for relative, expected in metadata['source_hashes'].items():
        assert hashlib.sha256((upstream/relative).read_bytes()).hexdigest() == expected
    print('Official Reach collection completed:', output, flush=True)


if __name__ == '__main__':
    main()
