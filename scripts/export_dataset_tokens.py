"""Export all frozen-split tokenizer outputs for comprehensive parity checks.

Same NPZ weights, same real episode-bounded windows, no shuffle or dropout.
No Torch import in the native Jittor branch. Does not train on validation data.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

from quest_repro.config import TokenizerConfig
from quest_repro.data import ActionChunkDataset, load_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", choices=("torch", "jittor"), required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Keep prior exports")
    args.output.mkdir(parents=True)
    info = json.loads((args.fixture / "metadata.json").read_text(encoding="utf-8"))
    config = TokenizerConfig(**info["config"])
    manifest = load_manifest(args.root, verify_hash=True)
    if manifest["sha256"] != info["dataset_sha256"]:
        raise ValueError("Fixture/data identity mismatch")
    if args.framework == "torch":
        import torch
        from quest_repro.tokenizer_torch import SkillVAE
        torch.set_num_threads(4)
        model = SkillVAE(config).to(args.device).eval()
        with np.load(args.fixture / "weights.npz", allow_pickle=False) as weights:
            model.load_state_dict({name: torch.from_numpy(weights[name].copy()).to(args.device) for name in weights.files}, strict=True)
        def predict(actions):
            with torch.no_grad():
                codes, indices = model.vq(model.encode(torch.from_numpy(actions).to(args.device)))
                return model.decode(codes).cpu().numpy(), indices.cpu().numpy()
    else:
        import jittor as jt
        from quest_repro.tokenizer_jittor import SkillVAE, strict_load_npz
        jt.flags.use_cuda = int(args.device == "cuda")
        model = SkillVAE(config)
        strict_load_npz(model, args.fixture / "weights.npz")
        model.eval()
        def predict(actions):
            with jt.no_grad():
                codes, indices = model.vq(model.encode(jt.array(actions)))
                return model.decode(codes).numpy(), indices.numpy()
    started, metrics = time.perf_counter(), {}
    for split in ("train", "validation"):
        dataset = ActionChunkDataset(args.root, split)
        arrays = {key: [] for key in ("indices", "prediction", "actions", "pad_mask", "episode_id", "start")}
        for start in range(0, len(dataset), args.batch_size):
            items = [dataset[index] for index in range(start, min(start + args.batch_size, len(dataset)))]
            actions = np.stack([item["actions"] for item in items])
            prediction, indices = predict(actions)
            values = {"prediction": prediction, "indices": indices, "actions": actions,
                      "pad_mask": np.stack([item["pad_mask"] for item in items]),
                      "episode_id": np.asarray([int(item["episode"].split("_")[-1]) for item in items], np.int32),
                      "start": np.asarray([item["start"] for item in items], np.int32)}
            for key in arrays:
                arrays[key].append(values[key])
        arrays = {key: np.concatenate(parts) for key, parts in arrays.items()}
        np.savez_compressed(args.output / f"{split}.npz", **arrays)
        error = np.abs(arrays["prediction"] - arrays["actions"])
        mask = np.broadcast_to(arrays["pad_mask"], error.shape)
        metrics[split] = {"windows": len(dataset), "mae_all": float(error.mean(dtype=np.float64)),
                          "mae_valid": float(error[mask].mean(dtype=np.float64)),
                          "unique_tokens": int(len(np.unique(arrays["indices"])))}
    if args.framework == "jittor":
        assert "torch" not in sys.modules
    report = {"framework": args.framework, "device": args.device, "dataset_sha256": manifest["sha256"],
              "metrics": metrics, "elapsed_seconds": time.perf_counter() - started,
              "scope": "frozen-weight full-corpus export for parity; no optimization or online success"}
    (args.output / "metadata.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
