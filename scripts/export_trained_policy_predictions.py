"""Compare learned native policy weights in separate Torch/Jittor processes.

Uses starts0/1 from each validation episode. No fitting, final seeds or rollout.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from quest_repro.config import PolicyConfig, TokenizerConfig
from quest_repro.data import ActionChunkDataset, load_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", choices=("torch", "jittor"), required=True)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--top-k',type=int,default=1)
    parser.add_argument("--policy-weights", type=Path, required=True)
    parser.add_argument("--policy-config", type=Path, required=True)
    parser.add_argument("--tokenizer-weights", type=Path, required=True)
    parser.add_argument("--tokenizer-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.output.exists():
        raise FileExistsError("Preserve previous prediction export")
    info = json.loads(args.policy_config.read_text(encoding="utf-8"))
    token_info = json.loads(args.tokenizer_config.read_text(encoding="utf-8"))
    if info["tokenizer_weights_sha256"] != hashlib.sha256(args.tokenizer_weights.read_bytes()).hexdigest():
        raise ValueError("Policy and tokenizer identity mismatch")
    manifest = load_manifest(root, verify_hash=True)
    data = ActionChunkDataset(root, "validation", include_observations=True)
    items = [data[data.index.index((episode, start))] for episode in data.episodes
             for start in (0, 1) if (episode, start) in data.index]
    images = np.stack([item["obs"]["corner_rgb"] for item in items])
    states = np.stack([item["obs"]["robot_states"] for item in items])
    tasks = np.asarray([item['task_id'] for item in items],dtype=np.int64)
    if args.framework == "torch":
        import torch
        from quest_repro.policy_torch import QueSTPolicy
        from quest_repro.tokenizer_torch import SkillVAE
        from quest_repro.training import load_npz_state
        torch.set_num_threads(4)
        model = QueSTPolicy(PolicyConfig(**info["config"])).cuda().eval()
        tokenizer = SkillVAE(TokenizerConfig(**token_info["config"])).cuda().eval()
        load_npz_state(model, args.policy_weights)
        load_npz_state(tokenizer, args.tokenizer_weights)
        with torch.no_grad():
            context = model.get_context(torch.from_numpy(images).cuda(), torch.from_numpy(states).cuda(), torch.from_numpy(tasks).cuda())
            if args.top_k==1:
                tokens = model.policy_prior.generate(context, top_k=1)
            else:
                tokens=model.policy_prior.generate_sampled(context,top_k=args.top_k,rng=np.random.default_rng(30000))
            actions = tokenizer.decode_actions(tokens)
        arrays = {"context": context.cpu().numpy(), "tokens": tokens.cpu().numpy(), "actions": actions.cpu().numpy()}
    else:
        import jittor as jt
        from quest_repro.policy_jittor import QueSTPolicy
        from quest_repro.tokenizer_jittor import SkillVAE, strict_load_npz
        model = QueSTPolicy(PolicyConfig(**info["config"]))
        tokenizer = SkillVAE(TokenizerConfig(**token_info["config"]))
        strict_load_npz(model, args.policy_weights)
        strict_load_npz(tokenizer, args.tokenizer_weights)
        model.eval()
        tokenizer.eval()
        with jt.no_grad():
            context = model.get_context(jt.array(images), jt.array(states), jt.array(tasks).int32())
            tokens = model.policy_prior.generate_sampled(context,top_k=args.top_k,rng=np.random.default_rng(30000))
            actions = tokenizer.decode_actions(tokens)
        arrays = {"context": context.numpy(), "tokens": tokens.numpy(), "actions": actions.numpy()}
        assert "torch" not in sys.modules
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    report = {"framework": args.framework, "dataset_sha256": manifest["sha256"],
              'top_k':args.top_k,'sampling':'common NumPy uniform draws for this fixture;not Torch deployment RNG equality',
              "samples": [{"episode": item["episode"], "start": int(item["start"])} for item in items],
              "scope": "same learned native weights on diverse validation inputs; not policy success or paired-training equality",
              "policy_weights_sha256": hashlib.sha256(args.policy_weights.read_bytes()).hexdigest()}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
