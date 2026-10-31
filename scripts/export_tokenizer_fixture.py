"""Export real-data Torch weights, intermediates, gradients and one AdamW update."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from quest_repro.config import TokenizerConfig
from quest_repro.data import ActionChunkDataset, load_manifest, sample_distinct_episode_batch
from quest_repro.tokenizer_torch import SkillVAE
from quest_repro.training import optimizer_groups, seed_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--name", default="tokenizer_tiny_seed123")
    parser.add_argument("--arch", choices=("full", "tiny"), default="tiny")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "outputs/fixtures" / args.name
    if output.exists():
        raise FileExistsError("Do not overwrite a numerical reference fixture")
    output.mkdir(parents=True)
    torch.set_num_threads(4)
    seed_all(args.seed)
    config = TokenizerConfig(attn_pdrop=0.0) if args.arch == "full" else TokenizerConfig.tiny()
    if args.checkpoint:
        checkpoint_path = args.checkpoint.resolve()
        if not checkpoint_path.is_relative_to(root):
            raise ValueError("Only load trusted locally generated project checkpoints")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = TokenizerConfig(**checkpoint["config"])
    model = SkillVAE(config).eval()
    if args.checkpoint:
        model.load_state_dict(checkpoint["model"], strict=True)
    batch = sample_distinct_episode_batch(ActionChunkDataset(root), args.batch_size, args.seed)
    actions = torch.from_numpy(batch["actions"]).requires_grad_()
    weights = {name: value.detach().numpy().copy() for name, value in model.state_dict().items()}
    features = {"input": actions.detach().numpy().copy()}
    handles = []
    for label, module in [("action_projection", model.action_proj), ("convolution", model.conv_block),
                          ("encoder", model.encoder), ("fsq_project_in", model.vq.project_in),
                          ("decoder", model.decoder)]:
        def capture(_module, _args, result, label=label):
            features[label] = result.detach().numpy().copy()
        handles.append(module.register_forward_hook(capture))
    prediction, _, _, _, codes = model(actions)
    for handle in handles:
        handle.remove()
    encoded = model.encode(actions)
    projected = model.vq.project_in(encoded)
    bounded = model.vq.bound(projected)
    half_integer_margin = (bounded - (bounded.floor() + 0.5)).abs().min().item()
    indices = model.get_indices(actions)
    loss = (prediction - actions).abs().mean()
    loss.backward()
    features.update(prediction=prediction.detach().numpy(), codes=codes.detach().numpy(),
                    indices=indices.detach().numpy(), loss=np.asarray(loss.item(), dtype=np.float32),
                    input_gradient=actions.grad.detach().numpy())
    gradients = {name: parameter.grad.detach().numpy().copy() for name, parameter in model.named_parameters()}
    optimizer = torch.optim.AdamW(optimizer_groups(model, 1e-4), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)
    optimizer.step()
    updated = {name: parameter.detach().numpy().copy() for name, parameter in model.named_parameters()}
    np.savez_compressed(output / "weights.npz", **weights)
    np.savez_compressed(output / "features.npz", **features)
    np.savez_compressed(output / "gradients.npz", **gradients)
    np.savez_compressed(output / "updated_weights.npz", **updated)
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), "framework": "torch",
                "torch": torch.__version__, "device": "cpu", "mode": "eval, gradients enabled",
                "config": config.to_dict(), "seed": args.seed,
                "episodes": batch["episodes"], "starts": batch["starts"],
                "dataset_sha256": load_manifest(root)["sha256"],
                "fsq_half_integer_margin": half_integer_margin,
                "optimizer": {"name": "AdamW", "lr": 1e-4, "betas": [0.9, 0.999], "eps": 1e-8,
                              "weight_decay": 1e-4, "norm_and_bias_decay": 0.0, "gradient_clip": None},
                "limitations": "Numerical reference fixture; not a training or online-control result"}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"directory": str(output), "loss": loss.item(),
                      "parameters_with_gradients": len(gradients), "half_integer_margin": half_integer_margin}, indent=2))


if __name__ == "__main__":
    main()
