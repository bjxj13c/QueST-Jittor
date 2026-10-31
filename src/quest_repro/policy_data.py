"""Real observation/label cache. Cache identity includes tokenizer and data hashes."""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ActionChunkDataset, load_manifest


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


class PolicyDataset:
    def __init__(self, root, split, tokenizer, tokenizer_path, batch_size=64, device="cpu"):
        root = Path(root)
        manifest = load_manifest(root, verify_hash=True)
        self.chunks = ActionChunkDataset(root, split)
        self.images, self.states = {}, {}
        with h5py.File(root / manifest["dataset"], "r") as file:
            for episode in self.chunks.episodes:
                self.images[episode] = file[episode]["images"][:]
                self.states[episode] = file[episode]["robot_states"][:].astype(np.float32)
        identity = {"tokenizer_sha256": sha256(tokenizer_path), "dataset_sha256": manifest["sha256"],
                    "split": split, "episodes": self.chunks.episodes, "windows": len(self.chunks),
                    "label_batch_size": batch_size}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        self.cache_path = root / "cache/policy_labels" / f"{split}_{key}.npz"
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        if self.cache_path.exists():
            with np.load(self.cache_path, allow_pickle=False) as cache:
                if json.loads(str(cache["identity"])) != identity:
                    raise ValueError("Label cache identity mismatch")
                self.labels = cache["labels"].copy()
        else:
            tokenizer.eval()
            labels = []
            with torch.no_grad():
                for batch in DataLoader(self.chunks, batch_size=batch_size, shuffle=False):
                    labels.append(tokenizer.get_indices(batch["actions"].to(device)).cpu().numpy().astype(np.int64))
            self.labels = np.concatenate(labels)
            np.savez_compressed(self.cache_path, labels=self.labels, identity=np.asarray(json.dumps(identity)))
        if self.labels.shape != (len(self.chunks), 8):
            raise ValueError("Expected eight tokens per 32-step chunk")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, index):
        result = self.chunks[index]
        episode, start = result["episode"], result["start"]
        result.update(images=self.images[episode][start].transpose(2, 0, 1).copy()[None],
                      robot_states=self.states[episode][start][None], targets=self.labels[index])
        return result
