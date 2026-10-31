"""Episode-bounded NumPy data interface, shared by Torch and Jittor loaders.

Policy images use uint8 CHW [0,255]. A model must normalize exactly once.
The previous day-one /255 adapter is preserved in provenance, not reused here.
"""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def load_manifest(root, verify_hash=False):
    root = Path(root)
    manifest = json.loads((root / "records/reach_split_v1.json").read_text(encoding="utf-8"))
    # 按完整轨迹划分，检查训练与验证集合没有同一条轨迹。
    if set(manifest["train"]) & set(manifest["validation"]):
        raise ValueError("Train/validation episode overlap")
    if verify_hash:
        digest = hashlib.sha256()
        with (root / manifest["dataset"]).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != manifest["sha256"]:
            raise ValueError("Dataset SHA256 differs from frozen split manifest")
    return manifest


class ActionChunkDataset:
    def __init__(self, root, split="train", include_observations=False, complete_only=False):
        if split not in ("train", "validation"):
            raise ValueError("split must be train or validation")
        self.root = Path(root)
        manifest = load_manifest(root)
        self.path = self.root / manifest["dataset"]
        self.chunk_length = manifest["chunk_length"]
        self.episodes = manifest[split]
        self.task_ids=manifest.get('episode_task_ids',{})
        self.include_observations = include_observations
        self.actions, self.index = {}, []
        with h5py.File(self.path, "r") as file:
            for episode in self.episodes:
                actions = file[episode]["actions"][:].astype(np.float32)
                self.actions[episode] = actions
                count = max(0, len(actions) - self.chunk_length + 1) if complete_only else len(actions)
                self.index.extend((episode, start) for start in range(count))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        episode, start = self.index[index]
        actions = self.actions[episode][start:start + self.chunk_length].copy()
        valid = len(actions)
        # 仅在当前轨迹内部截取动作，末尾不足32步时重复最后一个动作。
        if valid < self.chunk_length:
            actions = np.pad(actions, ((0, self.chunk_length - valid), (0, 0)), mode="edge")
        # pad_mask标记真实动作位置；是否将补齐位置计入损失由训练配置决定。
        result = {"actions": actions, "pad_mask": (np.arange(self.chunk_length) < valid)[:, None],
                  "episode": episode, "start": np.int64(start), "task_id": np.int64(self.task_ids.get(episode,0))}
        if self.include_observations:
            with h5py.File(self.path, "r") as file:
                group = file[episode]
                image = group["images"][start].transpose(2, 0, 1).copy()
                state = group["robot_states"][start].astype(np.float32)
            result["obs"] = {"corner_rgb": image[None], "robot_states": state[None]}
        return result


def sample_distinct_episode_batch(dataset, size=8, seed=0):
    """Prefer complete windows from different episodes for an honest overfit smoke check."""
    rng = np.random.default_rng(seed)
    episodes = rng.choice(dataset.episodes, size=min(size, len(dataset.episodes)), replace=False)
    lookup = {pair: i for i, pair in enumerate(dataset.index)}
    items = []
    for episode in episodes:
        max_start = max(0, len(dataset.actions[episode]) - dataset.chunk_length)
        start = int(rng.integers(max_start + 1))
        items.append(dataset[lookup[(episode, start)]])
    return {"actions": np.stack([item["actions"] for item in items]),
            "pad_mask": np.stack([item["pad_mask"] for item in items]),
            "episodes": [item["episode"] for item in items],
            "starts": [int(item["start"]) for item in items]}
