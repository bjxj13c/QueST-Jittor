"""Closed-loop Torch policy evaluation on held-out environment seeds.

Development seeds default to 2000+. Reserve seeds1000-1019 for a single final
paired Torch/Jittor evaluation after tuning; never optimize on that final set.
"""
import argparse
from collections import deque
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import torch

from quest_repro.config import PolicyConfig, TokenizerConfig
from quest_repro.policy_torch import QueSTPolicy
from quest_repro.policy_data import sha256
from quest_repro.tokenizer_torch import SkillVAE
from quest_repro.simulation import ReachSimulation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed-set", choices=("development", "final","experiment"), default="development")
    parser.add_argument('--seed-start',type=int)
    parser.add_argument('--env-name',default='reach-v2-goal-observable')
    parser.add_argument('--task-id',type=int,default=0)
    parser.add_argument('--per-episode-sampling',action='store_true')
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--sampling-seed", type=int, default=30000)
    parser.add_argument('--sampling-backend',choices=['torch','numpy'],default='torch')
    parser.add_argument("--videos", type=int, default=2)
    parser.add_argument("--observation-ablation", choices=("none", "zero-image", "zero-state", "zero-both"), default="none")
    args = parser.parse_args()
    root, device = args.root.resolve(), torch.device(args.device)
    torch.set_num_threads(4)
    torch.manual_seed(args.sampling_seed)
    sampling_rng=np.random.default_rng(args.sampling_seed)
    output = root / "outputs" / args.name
    if output.exists():
        raise FileExistsError("Preserve prior evaluation results")
    if args.episodes <= 0 or args.max_steps <= 0 or not 1 <= args.action_horizon <= 32:
        raise ValueError("Positive episodes and horizon 1..32 required")
    if args.seed_set == "final" and args.episodes != 20:
        raise ValueError("Final protocol uses all 20 reserved seeds")
    if args.seed_set=='experiment' and (args.seed_start is None or args.seed_start<4000):raise ValueError('New experiments require explicit new seed range>=4000')
    if args.seed_set == "final" and args.observation_ablation != "none":
        raise ValueError("Keep observation ablations on development seeds")
    for path in (args.policy, args.tokenizer):
        trusted_root=Path(__file__).resolve().parents[1] if args.seed_set=='experiment' else root
        if not path.resolve().is_relative_to(trusted_root):
            raise ValueError("Only load trusted locally generated project checkpoints")
    policy_state, tokenizer_state = [torch.load(p, map_location="cpu", weights_only=False)
                                     for p in (args.policy, args.tokenizer)]
    tokenizer_hash = sha256(args.tokenizer)
    if policy_state["extra"]["tokenizer_sha256"] != tokenizer_hash:
        raise ValueError("Policy targets used a different tokenizer checkpoint")
    policy = QueSTPolicy(PolicyConfig(**policy_state["config"])).to(device).eval()
    policy.load_state_dict(policy_state["model"], strict=True)
    tokenizer = SkillVAE(TokenizerConfig(**tokenizer_state["config"])).to(device).eval()
    tokenizer.load_state_dict(tokenizer_state["model"], strict=True)
    output.mkdir(parents=True)
    seed_start = args.seed_start if args.seed_set=='experiment' else (1000 if args.seed_set == "final" else 2000)
    records, all_latency, started = [], [], time.perf_counter()
    for i, seed in enumerate(range(seed_start, seed_start + args.episodes)):
        if args.per_episode_sampling:sampling_rng=np.random.default_rng(args.sampling_seed+seed)
        queue, trajectories, token_chunks = deque(), [], []
        success, returns, clips = False, 0.0, 0
        writer = None
        with ReachSimulation(seed, video=i < args.videos,env_name=args.env_name) as simulation:
            observation, _ = simulation.reset()
            if i < args.videos:
                import imageio.v2 as imageio
                writer = imageio.get_writer(str(output / f"seed_{seed}.mp4"), fps=20,
                    codec="libx264", quality=7, macro_block_size=16)
            try:
                for step in range(args.max_steps):
                    observation['task_id'][:]=args.task_id
                    if writer is not None:
                        writer.append_data(simulation.video_frame())
                    if not queue:
                        tensors = {key: torch.from_numpy(value).to(device) for key, value in observation.items()}
                        if args.observation_ablation in ("zero-image", "zero-both"):
                            tensors["images"] = torch.zeros_like(tensors["images"])
                        if args.observation_ablation in ("zero-state", "zero-both"):
                            tensors["robot_states"] = torch.zeros_like(tensors["robot_states"])
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        tick = time.perf_counter()
                        with torch.no_grad():
                            context = policy.get_context(**tensors)
                            if args.sampling_backend=='numpy':
                                indices=policy.policy_prior.generate_sampled(context,top_k=args.top_k,rng=sampling_rng)
                            else:
                                indices = policy.policy_prior.generate(context, top_k=args.top_k)
                            actions = tokenizer.decode_actions(indices)[0].cpu().numpy()
                        all_latency.append((time.perf_counter() - tick) * 1000)
                        if not np.isfinite(actions).all():
                            raise FloatingPointError("Nonfinite policy actions")
                        token_chunks.append(indices.cpu().numpy()[0])
                        queue.extend(actions[:args.action_horizon])
                    action = queue.popleft()
                    clips += int(np.any(np.abs(action) > 1))
                    trajectories.append(action.copy())
                    observation, reward, terminated, truncated, info = simulation.step(action)
                    returns += float(reward)
                    success = success or bool(info.get("success", False))
                    if success or terminated or truncated:
                        break
                if writer is not None:
                    writer.append_data(simulation.video_frame())
            finally:
                if writer is not None:
                    writer.close()
        np.savez_compressed(output / f"seed_{seed}.npz", predicted_actions=np.asarray(trajectories),
                            generated_tokens=np.asarray(token_chunks))
        row = {"seed": seed, "steps": step + 1, "success": success, "return": returns,
               "actions_requiring_clip": clips}
        records.append(row)
        with (output / "episodes.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    n = len(records)
    proportion = sum(r["success"] for r in records) / n
    z = 1.959963984540054
    center = (proportion + z*z/(2*n)) / (1 + z*z/n)
    half = z * math.sqrt(proportion*(1-proportion)/n + z*z/(4*n*n)) / (1 + z*z/n)
    report = {"scope": "learned-policy closed-loop evaluation", "framework": "torch", "device": str(device),
        'environment':args.env_name,'task_id':args.task_id,'per_episode_sampling':args.per_episode_sampling,
        "torch": torch.__version__, "platform": platform.platform(), "mujoco": importlib.metadata.version("mujoco"),
        "metaworld": importlib.metadata.version("metaworld"), "seed_set": args.seed_set,
        "seeds": [r["seed"] for r in records], "episodes": records,
        "success_rate": sum(r["success"] for r in records) / len(records), "horizon": args.action_horizon,
        "success_rate_wilson_95_interval": [center-half, center+half],
        "observation_ablation": args.observation_ablation,
        "top_k": args.top_k, "sampling_seed": args.sampling_seed, "sampling_backend":args.sampling_backend,"max_steps": args.max_steps,
        "policy_sha256": sha256(args.policy), "tokenizer_sha256": tokenizer_hash,
        "median_decision_ms": float(np.median(all_latency)), "p95_decision_ms": float(np.percentile(all_latency, 95)),
        "performance_scope": "includes host transfer; first JIT/BLAS warmup not excluded; excludes rendering",
        "elapsed_seconds": time.perf_counter() - started, "created_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": ["single reach task", "small demonstration dataset", "not paper benchmark equivalence",
                        "greedy top-1 default differs from official stochastic top-5",
                        "development seeds may be used for debugging; final seeds are reserved"]}
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"success_rate": report["success_rate"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
