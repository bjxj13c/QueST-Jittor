"""Collect exact-version reach expert data; never overwrite an existing file."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from quest_repro.simulation import ReachSimulation, set_render_environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True,
                        help="Use a separate root for regenerated data; preserve the frozen original")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=150)
    args = parser.parse_args()
    if args.episodes <= 1 or args.seed_start < 0 or args.max_steps <= 0:
        raise ValueError("At least two episodes and a nonnegative seed required")
    set_render_environment()
    from metaworld.policies import SawyerReachV2Policy
    output = args.root.resolve() / "data/reach_30_v1.hdf5"
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "x") as file:
        file.attrs.update(env_name="reach-v2-goal-observable", purpose="regenerated expert data",
                          max_steps=args.max_steps, expert_noise_std=0.0)
        for i in range(args.episodes):
            seed, expert = args.seed_start + i, SawyerReachV2Policy()
            records = {key: [] for key in ("images", "observations", "robot_states", "actions",
                                           "rewards", "success", "terminated", "truncated")}
            with ReachSimulation(seed) as simulation:
                observation, _ = simulation.reset()
                for _ in range(args.max_steps):
                    action = np.clip(expert.get_action(simulation.observation), -1, 1).astype(np.float32)
                    records["images"].append(observation["images"][0, 0].transpose(1, 2, 0).copy())
                    records["observations"].append(simulation.observation.copy())
                    records["robot_states"].append(observation["robot_states"][0, 0].copy())
                    records["actions"].append(action.copy())
                    observation, reward, terminated, truncated, info = simulation.step(action)
                    success = bool(info.get("success", False))
                    for key, value in (("rewards", reward), ("success", success),
                                       ("terminated", terminated), ("truncated", truncated)):
                        records[key].append(value)
                    if success or terminated or truncated:
                        break
            group = file.create_group(f"demo_{i}")
            group.attrs.update(seed=seed, episode_success=bool(any(records["success"])))
            for key, value in records.items():
                group.create_dataset(key, data=np.asarray(value), compression="gzip")
            print(json.dumps({"episode": i, "seed": seed, "steps": len(records["actions"]),
                              "success": bool(group.attrs["episode_success"])}), flush=True)
    print("Saved:", output)


if __name__ == "__main__":
    main()
