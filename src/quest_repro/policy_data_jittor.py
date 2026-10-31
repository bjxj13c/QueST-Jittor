"""Native Jittor labels and NumPy observations; no Torch dependency."""
import hashlib
import json
from pathlib import Path

import h5py
import jittor as jt
import numpy as np

from .data import ActionChunkDataset, load_manifest


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class PolicyDataset:
    def __init__(self, root, split, tokenizer, tokenizer_weights, batch_size=64):
        root = Path(root)
        manifest = load_manifest(root, verify_hash=True)
        self.chunks = ActionChunkDataset(root, split)
        self.images, self.states = {}, {}
        with h5py.File(root / manifest["dataset"], "r") as file:
            for episode in self.chunks.episodes:
                self.images[episode] = file[episode]["images"][:]
                self.states[episode] = file[episode]["robot_states"][:].astype(np.float32)
        # 标签缓存绑定数据和分词器权重；更换任意一项都不能复用旧标签。
        identity = {"framework": "jittor", "tokenizer_sha256": sha256(tokenizer_weights),
                    "dataset_sha256": manifest["sha256"], "split": split, "episodes": self.chunks.episodes,
                    "label_batch_size": batch_size}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        cache_path = root / "cache/policy_labels" / f"{split}_{key}.npz"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            with np.load(cache_path, allow_pickle=False) as cache:
                if json.loads(str(cache["identity"])) != identity:
                    raise ValueError("Jittor label cache identity mismatch")
                self.labels = cache["labels"].copy()
        else:
            labels = []
            # 固定分词器，用专家动作生成8个token标签；此处不更新分词器。
            tokenizer.eval()
            with jt.no_grad():
                for start in range(0, len(self.chunks), batch_size):
                    actions = np.stack([self.chunks[i]["actions"] for i in
                                        range(start, min(start + batch_size, len(self.chunks)))])
                    labels.append(tokenizer.get_indices(jt.array(actions)).numpy().astype(np.int32))
            self.labels = np.concatenate(labels)
            np.savez_compressed(cache_path, labels=self.labels, identity=np.asarray(json.dumps(identity)))
        if self.labels.shape != (len(self.chunks), 8):
            raise ValueError("Expected eight token labels per chunk")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, index):
        sample = self.chunks[index]
        episode, start = sample["episode"], int(sample["start"])
        sample.update(images=self.images[episode][start].transpose(2, 0, 1).copy()[None],
                      robot_states=self.states[episode][start][None], targets=self.labels[index])
        return sample


def batch_from_indices(dataset, indices):
    items = [dataset[int(index)] for index in indices]
    return {key: np.stack([item[key] for item in items]) for key in
            ("images", "robot_states", "task_id", "targets", "actions", "pad_mask")}
