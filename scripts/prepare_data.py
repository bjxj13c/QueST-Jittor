"""Validate and freeze regenerated data in a separate root; no Torch required."""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from quest_repro.data import ActionChunkDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=20261027)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()
    root = args.root.resolve()
    path, records = root / "data/reach_30_v1.hdf5", root / "records"
    records.mkdir(parents=True, exist_ok=True)
    manifest_path = records / "reach_split_v1.json"
    stats_path = records / "reach_train_statistics_v1.json"
    if manifest_path.exists() or stats_path.exists():
        raise FileExistsError("Preserve the frozen split/statistics; use a new data root")
    if not 0 < args.validation_fraction < 1:
        raise ValueError("Validation fraction must be between zero and one")
    with h5py.File(path, "r") as file:
        episodes = sorted(file, key=lambda key: int(key.split("_")[-1]))
        if len(episodes) < 2:
            raise ValueError("At least two episodes required")
        for episode in episodes:
            group = file[episode]
            length = len(group["actions"])
            if length < 1:
                raise ValueError("Empty episode")
            for key in ("images", "observations", "robot_states", "actions", "rewards", "success", "terminated", "truncated"):
                if len(group[key]) != length:
                    raise ValueError(f"Length mismatch: {episode}/{key}")
            if group["images"].shape[1:] != (128, 128, 3) or group["images"].dtype != np.uint8:
                raise ValueError("Expected raw 128x128 RGB uint8")
            actions, privileged, states = group["actions"][:], group["observations"][:], group["robot_states"][:]
            if actions.shape[1:] != (4,) or privileged.shape[1:] != (39,) or states.shape[1:] != (8,):
                raise ValueError("Invalid action/state shapes")
            if not np.isfinite(actions).all() or np.max(np.abs(actions)) > 1:
                raise ValueError("Invalid controls")
            np.testing.assert_allclose(states, np.concatenate((privileged[:, :4], privileged[:, 18:22]), axis=1), atol=1e-7)
        shuffled = np.random.default_rng(args.split_seed).permutation(episodes).tolist()
        val_count = min(len(episodes) - 1, max(1, round(len(episodes) * args.validation_fraction)))
        train, validation = shuffled[:-val_count], shuffled[-val_count:]
        statistics = {"source": "training episodes only", "applied": False}
        for key in ("actions", "robot_states"):
            values = np.concatenate([file[episode][key][:] for episode in train])
            statistics[key] = {name: getattr(values, name)(axis=0).tolist() for name in ("mean", "std", "min", "max")}
    manifest = {"dataset": "data/reach_30_v1.hdf5", "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "split_seed": args.split_seed, "chunk_length": 32, "stride": 1,
                "padding": "repeat final action within same episode", "train": train, "validation": validation,
                "collection": {"env_name": "reach-v2-goal-observable", "camera": "corner2",
                    "camera_position_override_index": 2, "camera_position_override": [0.75, 0.075, 0.7],
                    "image_vertical_flip": True, "rgb_size": [128, 128], "expert_noise_std": 0.0},
                "evaluation_seeds_reserved": list(range(1000, 1020)),
                "scope": "regenerated dataset, not byte-identical to original frozen dataset"}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    stats_path.write_text(json.dumps(statistics, indent=2), encoding="utf-8")
    print(json.dumps({"root": str(root), "train": len(train), "validation": len(validation),
                      "train_windows": len(ActionChunkDataset(root)),
                      "validation_windows": len(ActionChunkDataset(root, "validation")), "sha256": manifest["sha256"]}, indent=2))


if __name__ == "__main__":
    main()
