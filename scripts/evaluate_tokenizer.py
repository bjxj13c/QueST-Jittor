"""Evaluate a trusted local tokenizer checkpoint on the full frozen validation split."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from quest_repro.config import TokenizerConfig
from quest_repro.data import ActionChunkDataset, load_manifest
from quest_repro.tokenizer_torch import SkillVAE
from train_tokenizer import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    root, checkpoint_path = args.root.resolve(), args.checkpoint.resolve()
    if not checkpoint_path.is_relative_to(root):
        raise ValueError("Only load this project's trusted local checkpoints")
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    manifest = load_manifest(root, verify_hash=True)
    if checkpoint["extra"].get("dataset_sha256") != manifest["sha256"]:
        raise ValueError("Checkpoint dataset hash mismatch")
    model = SkillVAE(TokenizerConfig(**checkpoint["config"])).to(args.device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    dataset = ActionChunkDataset(root, "validation")
    train = ActionChunkDataset(root, "train")
    mean = torch.tensor(np.concatenate(list(train.actions.values())).mean(axis=0), device=args.device)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        generator=torch.Generator().manual_seed(42))
    metrics = evaluate(model, loader, torch.device(args.device), mean)
    output = checkpoint_path.parent / f"evaluation_{checkpoint_path.stem}"
    if output.exists():
        raise FileExistsError("Evaluation directory already exists")
    output.mkdir()
    report = {"checkpoint": str(checkpoint_path.relative_to(root)), "framework": "torch",
              "device": args.device, "validation_episodes": dataset.episodes,
              "validation_windows": len(dataset), "dataset_sha256": manifest["sha256"],
              "metrics": metrics, "scope": "action reconstruction; future demonstration actions are inputs, not policy rollout"}
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.environ.setdefault("MPLCONFIGDIR", str(root / "cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected = [next(i for i, (name, start) in enumerate(dataset.index) if name == episode and start == 0)
                for episode in dataset.episodes[:3]]
    figure, axes = plt.subplots(3, 4, figsize=(13, 7.5), sharex=True)
    arrays = {}
    with torch.no_grad():
        for row, index in enumerate(selected):
            item = dataset[index]
            target = torch.from_numpy(item["actions"])[None].to(args.device)
            prediction = model(target)[0][0].cpu().numpy()
            arrays[item["episode"] + "_target"] = item["actions"]
            arrays[item["episode"] + "_prediction"] = prediction
            for dimension in range(4):
                axis = axes[row, dimension]
                axis.plot(item["actions"][:, dimension], label="demonstration", linewidth=1.6)
                axis.plot(prediction[:, dimension], label="reconstruction", linewidth=1.4)
                axis.set_title(f"{item['episode']}, action dim {dimension}", fontsize=9)
                axis.set_xlabel("Future time index")
                axis.set_ylabel("Control value")
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Held-out action reconstruction (not robot policy execution)")
    figure.tight_layout()
    figure.savefig(output / "validation_reconstructions.png", dpi=170)
    figure.savefig(output / "validation_reconstructions.pdf")
    plt.close(figure)
    np.savez_compressed(output / "examples.npz", **arrays)
    print(json.dumps({"output": str(output), "mae_all": metrics["mae_all"],
                      "mae_valid": metrics["mae_valid"], "zero_baseline": metrics["zero_action_mae"],
                      "mean_baseline": metrics["train_mean_action_mae"]}, indent=2))


if __name__ == "__main__":
    main()
