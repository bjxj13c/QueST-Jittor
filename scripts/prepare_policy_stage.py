"""Bridge native tokenizer weights or independently export policy target labels.

Torch and Jittor execute in separate processes. Native mode never imports Torch.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

from quest_repro.config import TokenizerConfig
from quest_repro.data import load_manifest


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--framework", choices=("torch", "jittor"), required=True)
    parser.add_argument("--tokenizer-weights", type=Path, required=True)
    parser.add_argument("--tokenizer-config", type=Path, required=True)
    parser.add_argument("--tokenizer-checkpoint", type=Path)
    parser.add_argument("--prepare-checkpoint", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    for source in (args.tokenizer_weights, args.tokenizer_config):
        if not source.resolve().is_relative_to(root):
            raise ValueError("Use trusted project tokenizer assets")
    if args.output.exists():
        raise FileExistsError("Preserve previous preparation")
    args.output.mkdir(parents=True)
    info = json.loads(args.tokenizer_config.read_text(encoding="utf-8"))
    config = TokenizerConfig(**info["config"])
    manifest = load_manifest(root, verify_hash=True)
    report = {"framework": args.framework, "tokenizer_native_npz_sha256": digest(args.tokenizer_weights),
              "dataset_sha256": manifest["sha256"], "config": config.to_dict(), "label_batch_size": args.batch_size}
    if args.framework == "torch":
        import torch
        from quest_repro.tokenizer_torch import SkillVAE
        from quest_repro.training import load_npz_state
        from quest_repro.policy_data import PolicyDataset
        torch.set_num_threads(4)
        model = SkillVAE(config).cuda().eval()
        load_npz_state(model, args.tokenizer_weights)
        if args.prepare_checkpoint:
            model.cpu()
            torch.save({"model": model.state_dict(), "config": config.to_dict(),
                "extra": {"native_tokenizer_npz_sha256": report["tokenizer_native_npz_sha256"]},
                "scope": "static native NPZ to Torch container; no training or optimizer state"}, args.output / "tokenizer_shared.pt")
            shutil.copy2(args.tokenizer_weights, args.output / "weights.npz")
            (args.output / "metadata.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(report, indent=2), flush=True)
            return
        if args.tokenizer_checkpoint is None:
            raise ValueError("Torch label cache must be keyed to its exact checkpoint")
        def dataset(split):
            return PolicyDataset(root, split, model, args.tokenizer_checkpoint, batch_size=args.batch_size, device="cuda")
    else:
        if args.prepare_checkpoint:
            raise ValueError("Checkpoint container export is a separate Torch process")
        from quest_repro.tokenizer_jittor import SkillVAE, strict_load_npz
        from quest_repro.policy_data_jittor import PolicyDataset
        model = SkillVAE(config)
        strict_load_npz(model, args.tokenizer_weights)
        model.eval()
        def dataset(split):
            return PolicyDataset(root, split, model, args.tokenizer_weights, batch_size=args.batch_size)
    report["splits"] = {}
    for split in ("train", "validation"):
        data = dataset(split)
        episode_ids = np.asarray([int(episode.split("_")[-1]) for episode, _ in data.chunks.index], dtype=np.int32)
        starts = np.asarray([start for _, start in data.chunks.index], dtype=np.int32)
        np.savez_compressed(args.output / (split + ".npz"), labels=data.labels, episode_id=episode_ids, start=starts)
        report["splits"][split] = {"windows": len(data), "label_shape": list(data.labels.shape)}
    if args.framework == "jittor":
        assert "torch" not in sys.modules
        report["torch_imported"] = False
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
