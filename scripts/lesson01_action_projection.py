"""Read a real action chunk and inspect its trained Linear(4, 256) on CPU.

Learning exercise only: no optimizer, parameter updates, simulator, or Jittor
import. Reference weights and intermediate outputs come from the saved run.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from quest_repro.data import ActionChunkDataset, load_manifest


def main():
    root = Path(__file__).resolve().parents[1]
    fixture = root / "outputs/fixtures/tokenizer_full_trained"
    metadata = json.loads((fixture / "metadata.json").read_text(encoding="utf-8"))
    manifest = load_manifest(root, verify_hash=True)
    if metadata["dataset_sha256"] != manifest["sha256"]:
        raise ValueError("Fixture and dataset do not match")
    episode, start = metadata["episodes"][0], int(metadata["starts"][0])
    dataset = ActionChunkDataset(root, "train")
    item = dataset[dataset.index.index((episode, start))]
    actions = torch.from_numpy(item["actions"]).unsqueeze(0)
    with np.load(fixture / "weights.npz", allow_pickle=False) as state:
        weight = torch.from_numpy(state["action_proj.weight"].copy())
        bias = torch.from_numpy(state["action_proj.bias"].copy())
    torch.set_num_threads(2)
    with torch.no_grad():
        projected = F.linear(actions, weight, bias)
    with np.load(fixture / "features.npz", allow_pickle=False) as expected:
        np.testing.assert_array_equal(actions.numpy()[0], expected["input"][0])
        np.testing.assert_allclose(projected.numpy()[0], expected["action_projection"][0], atol=2e-6, rtol=1e-5)
    print("Lesson 01: real action chunk -> action projection")
    print(f"Source episode: {episode}; start index: {start} (zero-based)")
    print("Device:", actions.device)
    print("First 3 actions [x-control, y-control, z-control, gripper-control]:")
    print(actions[0, :3].numpy().round(4))
    print("Input shape [batch, time, controls]:", list(actions.shape))
    print("Weight shape [output features, input features]:", list(weight.shape))
    print("Bias shape:", list(bias.shape))
    print("Operation: output = input @ weight.T + bias")
    print("Output shape [batch, time, learned features]:", list(projected.shape))
    print("Layer parameters:", weight.numel() + bias.numel())
    print("Same learned projection is applied to all 32 time steps.")
    print("Saved PyTorch reference: matched within declared float tolerances.")
    print("Scope: CPU projection inspection; no training or Jittor execution.")


if __name__ == "__main__":
    main()
