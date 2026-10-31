"""Framework-independent epoch permutations bound to the frozen split."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .data import ActionChunkDataset, load_manifest


def split_fingerprint(root):
    manifest = load_manifest(root, verify_hash=True)
    keys = ("sha256", "train", "validation", "chunk_length", "stride", "padding")
    return hashlib.sha256(json.dumps({k: manifest[k] for k in keys}, sort_keys=True).encode()).hexdigest()


def create_plan(root, output, epochs, batch_size, seed):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Keep the original plan")
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("Positive epoch and batch size required")
    count = len(ActionChunkDataset(root))
    # 先生成与框架无关的样本顺序，让配对训练读取相同批次。
    rng = np.random.default_rng(seed)
    metadata = {"split_fingerprint": split_fingerprint(root), "windows": count,
                "epochs": epochs, "batch_size": batch_size, "seed": seed,
                "rng": "numpy.default_rng PCG64", "order": "one permutation of train windows per epoch"}
    arrays = {f"epoch_{epoch}": rng.permutation(count).astype(np.int64) for epoch in range(1, epochs + 1)}
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, metadata=np.asarray(json.dumps(metadata)), **arrays)
    return metadata


def load_plan(path, root, epochs, batch_size):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        # 加载前核对数据划分，防止将旧批次计划用于另一份数据。
        if metadata["split_fingerprint"] != split_fingerprint(root):
            raise ValueError("Batch plan belongs to another dataset/split")
        if metadata["epochs"] != epochs or metadata["batch_size"] != batch_size:
            raise ValueError("Batch plan epochs or batch size mismatch")
        count = len(ActionChunkDataset(root))
        if metadata["windows"] != count:
            raise ValueError("Batch plan window count mismatch")
        if set(archive.files) != {"metadata", *[f"epoch_{i}" for i in range(1, epochs + 1)]}:
            raise ValueError("Unexpected batch plan fields")
        orders = [archive[f"epoch_{i}"].copy() for i in range(1, epochs + 1)]
    for order in orders:
        if order.dtype.kind not in "iu" or order.shape != (count,) or not np.array_equal(np.sort(order), np.arange(count)):
            raise ValueError("Invalid epoch permutation")
    return orders, hashlib.sha256(Path(path).read_bytes()).hexdigest()
