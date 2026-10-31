"""Native Jittor closed-loop evaluator; Linux runtime verification pending.

Consumes weights/config emitted by train_policy_jittor.py. No Torch import.
Use development seeds first. Only run final seeds after freezing the protocol.
"""
import argparse
from collections import deque
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import jittor as jt
import numpy as np

from quest_repro.config import PolicyConfig, TokenizerConfig
from quest_repro.policy_jittor import QueSTPolicy
from quest_repro.tokenizer_jittor import SkillVAE, strict_load_npz
from quest_repro.policy_data_jittor import sha256
from quest_repro.simulation import ReachSimulation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--policy-weights", type=Path, required=True)
    parser.add_argument("--policy-config", type=Path, required=True)
    parser.add_argument("--tokenizer-weights", type=Path, required=True)
    parser.add_argument("--tokenizer-config", type=Path, required=True)
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
    parser.add_argument("--videos", type=int, default=2)
    parser.add_argument('--top-k',type=int,default=1)
    parser.add_argument('--sampling-seed',type=int,default=30000)
    args = parser.parse_args()
    if args.seed_set=='experiment' and (args.seed_start is None or args.seed_start<4000):raise ValueError('Explicit new experiment seeds>=4000required')
    if not 1<=args.top_k<=1000:raise ValueError('Invalid top-k')
    rng=np.random.default_rng(args.sampling_seed)
    if args.episodes <= 0 or not 1 <= args.action_horizon <= 32 or args.max_steps <= 0:
        raise ValueError("Invalid episode/horizon/step count")
    if args.seed_set == "final" and args.episodes != 20:
        raise ValueError("Final protocol uses all 20 reserved seeds")
    output = args.root.resolve() / "outputs" / args.name
    if output.exists():
        raise FileExistsError("Preserve previous evaluation")
    policy_info = json.loads(args.policy_config.read_text(encoding="utf-8"))
    tokenizer_info = json.loads(args.tokenizer_config.read_text(encoding="utf-8"))
    if policy_info.get("tokenizer_weights_sha256") != sha256(args.tokenizer_weights):
        raise ValueError("Requires native policy training metadata linking this exact tokenizer NPZ")
    jt.flags.use_cuda = int(args.device == "cuda")
    policy = QueSTPolicy(PolicyConfig(**policy_info["config"]))
    tokenizer = SkillVAE(TokenizerConfig(**tokenizer_info["config"]))
    strict_load_npz(policy, args.policy_weights)
    strict_load_npz(tokenizer, args.tokenizer_weights)
    policy.eval()
    tokenizer.eval()
    output.mkdir(parents=True)
    records, latency, started = [], [], time.perf_counter()
    seed_start = args.seed_start if args.seed_set=='experiment' else (1000 if args.seed_set == "final" else 2000)
    for index, seed in enumerate(range(seed_start, seed_start + args.episodes)):
        if args.per_episode_sampling:rng=np.random.default_rng(args.sampling_seed+seed)
        queue, actions_record, token_record = deque(), [], []
        success, returns, clips, writer = False, 0.0, 0, None
        with ReachSimulation(seed, video=index < args.videos,env_name=args.env_name) as simulation:
            observation, _ = simulation.reset()
            if index < args.videos:
                import imageio.v2 as imageio
                writer = imageio.get_writer(str(output / f"seed_{seed}.mp4"), fps=20, codec="libx264", quality=7)
            try:
                for step in range(args.max_steps):
                    observation['task_id'][:]=args.task_id
                    if writer is not None:
                        writer.append_data(simulation.video_frame())
                    if not queue:
                        jt.sync_all(True)
                        tick = time.perf_counter()
                        with jt.no_grad():
                            context = policy.get_context(jt.array(observation["images"]),
                                jt.array(observation["robot_states"]), jt.array(observation["task_id"]).int32())
                            indices = policy.policy_prior.generate_sampled(context,top_k=args.top_k,rng=rng)
                            actions = tokenizer.decode_actions(indices).numpy()[0]
                        latency.append((time.perf_counter() - tick) * 1000)
                        if not np.isfinite(actions).all():
                            raise FloatingPointError("Nonfinite native policy actions")
                        token_record.append(indices.numpy()[0])
                        queue.extend(actions[:args.action_horizon])
                    action = queue.popleft()
                    clips += int(np.any(np.abs(action) > 1))
                    actions_record.append(action.copy())
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
        np.savez_compressed(output / f"seed_{seed}.npz", predicted_actions=np.asarray(actions_record),
                            generated_tokens=np.asarray(token_record))
        row = {"seed": seed, "steps": step + 1, "success": success, "return": returns, "actions_requiring_clip": clips}
        records.append(row)
        with (output / "episodes.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    assert "torch" not in sys.modules
    report = {"scope": "native learned-policy closed-loop evaluation", "framework": "jittor", "version": jt.__version__,
              'environment':args.env_name,'task_id':args.task_id,'per_episode_sampling':args.per_episode_sampling,
              "device": args.device, "seed_set": args.seed_set, "episodes": records,
              "success_rate": sum(r["success"] for r in records) / len(records), "top_k": args.top_k,
              "sampling_seed":args.sampling_seed,"sampling_rng":"NumPy PCG64 inverse CDF; differs from Torch multinomial RNG",
              "horizon": args.action_horizon, "max_steps": args.max_steps,
              "policy_weights_sha256": sha256(args.policy_weights), "tokenizer_weights_sha256": sha256(args.tokenizer_weights),
              "median_decision_ms": float(np.median(latency)), "p95_decision_ms": float(np.percentile(latency, 95)),
              "performance_scope": "includes host transfer/cold start, excludes rendering; not standardized benchmark",
              "elapsed_seconds": time.perf_counter() - started, "created_utc": datetime.now(timezone.utc).isoformat(),
              "torch_imported": False, "limitations": ["single task, small-data protocol", "not paper benchmark equivalence"]}
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"success_rate": report["success_rate"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
