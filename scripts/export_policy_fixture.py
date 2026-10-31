"""Real RGB/state/target fixture for native policy parity; no environment rollout."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from quest_repro.config import TokenizerConfig, PolicyConfig
from quest_repro.policy_torch import QueSTPolicy
from quest_repro.policy_data import PolicyDataset, sha256
from quest_repro.tokenizer_torch import SkillVAE
from quest_repro.training import seed_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--name", default="policy_tiny_seed456")
    parser.add_argument("--arch", choices=("full", "tiny"), default="tiny")
    parser.add_argument("--seed", type=int, default=456)
    args = parser.parse_args()
    torch.set_num_threads(4)
    seed_all(args.seed)
    root, tokenizer_path = args.root.resolve(), args.tokenizer.resolve()
    if not tokenizer_path.is_relative_to(root):
        raise ValueError("Use a trusted project checkpoint")
    output = root / "outputs/fixtures" / args.name
    if output.exists():
        raise FileExistsError("Preserve the original fixture")
    output.mkdir(parents=True)
    checkpoint = torch.load(tokenizer_path, map_location="cpu", weights_only=False)
    tokenizer = SkillVAE(TokenizerConfig(**checkpoint["config"])).eval()
    tokenizer.load_state_dict(checkpoint["model"], strict=True)
    dataset = PolicyDataset(root, "train", tokenizer, tokenizer_path)
    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
    config = PolicyConfig(dropout=0.0) if args.arch == "full" else PolicyConfig.tiny()
    model = QueSTPolicy(config).eval()
    weights = {name: value.detach().numpy().copy() for name, value in model.state_dict().items()}
    images, state, tasks, targets = [batch[k] for k in ("images", "robot_states", "task_id", "targets")]
    state = state.requires_grad_()
    context = model.get_context(images, state, tasks)
    logits, loss = model(images, state, tasks, targets)
    loss.backward()
    features = {"images": images.numpy(), "robot_states": state.detach().numpy(), "task_id": tasks.numpy(),
        "targets": targets.numpy(), "context": context.detach().numpy(), "logits": logits.detach().numpy(),
        "loss": np.asarray(loss.item(), np.float32), "robot_state_gradient": state.grad.numpy()}
    gradients = {name: parameter.grad.detach().numpy().copy() for name, parameter in model.named_parameters()
                 if parameter.grad is not None}
    unused = [name for name, parameter in model.named_parameters() if parameter.grad is None]
    np.savez_compressed(output / "weights.npz", **weights)
    np.savez_compressed(output / "features.npz", **features)
    np.savez_compressed(output / "gradients.npz", **gradients)
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), "framework": "torch", "version": torch.__version__,
        "device": "cpu", "mode": "eval, parameter gradients enabled", "config": config.to_dict(),
        "seed": args.seed, "episodes": batch["episode"], "starts": batch["start"].tolist(),
        "tokenizer_sha256": sha256(tokenizer_path), "loss": loss.item(),
        "parameters_without_gradient": unused,
        "scope": "numeric reference, no Jittor execution or online success implied"}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
