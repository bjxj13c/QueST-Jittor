"""Inspect trained temporal-convolution shapes on a saved real CPU example.

Starts with lesson01's saved projection output. Loads the convolution weights
from the same tokenizer fixture. No optimizer or parameter updates.
"""
from pathlib import Path

import numpy as np
import torch

from quest_repro.tokenizer_torch import ResidualTemporalBlock


def main():
    root = Path(__file__).resolve().parents[1]
    fixture = root / "outputs/fixtures/tokenizer_full_trained"
    torch.set_num_threads(2)
    with np.load(fixture / "features.npz", allow_pickle=False) as reference:
        projected = torch.from_numpy(reference["action_projection"][:1].copy())
        expected = reference["convolution"][:1].copy()
    model = ResidualTemporalBlock(256, factor=4, causal=True).eval()
    prefix = "conv_block."
    with np.load(fixture / "weights.npz", allow_pickle=False) as state:
        weights = {name[len(prefix):]: torch.from_numpy(state[name].copy())
                   for name in state.files if name.startswith(prefix)}
    model.load_state_dict(weights, strict=True)
    traces, handles = [], []
    for number, block in enumerate(model.blocks, start=1):
        def capture(module, inputs, output, number=number):
            convolution = module.block[0]
            traces.append((number, convolution.kernel_size, convolution.stride,
                           list(inputs[0].shape), list(output.shape)))
        handles.append(block.register_forward_hook(capture))
    with torch.no_grad():
        output = model(projected)
    for handle in handles:
        handle.remove()
    np.testing.assert_allclose(output.numpy(), expected, atol=2e-5, rtol=2e-4)
    print("Lesson 02: trained temporal convolution, CPU only")
    print("Input [batch, time, features]:", list(projected.shape))
    print("Reorder for Conv1d [batch, channels, time]:", list(projected.transpose(1, 2).shape))
    for number, kernel, stride, before, after in traces:
        print(f"Block {number}: kernel={kernel}, stride={stride}; {before} -> {after}")
    print("Final reorder [batch, compressed positions, features]:", list(output.shape))
    print("Time lengths: 32 -> 16 -> 8 -> 8")
    print("Reordering axes changes layout; convolution with stride reduces length.")
    print("These 8 positions are float features; FSQ token indices come later.")
    print("Saved PyTorch reference: matched within declared float tolerances.")
    print("Scope: trained convolution inspection; no training or Jittor execution.")


if __name__ == "__main__":
    main()
