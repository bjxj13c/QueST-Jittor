"""Inspect real position encoding and tokenizer Encoder shapes on CPU.

Uses trained fixture weights. Q/K/V shapes are inspected with the first real
attention layer's projections; Encoder outputs use the project's actual model.
No optimizer, backward pass, parameter updates, GPU, or Jittor execution.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from quest_repro.config import TokenizerConfig
from quest_repro.tokenizer_torch import SkillVAE
from quest_repro.training import load_npz_state


def main():
    root = Path(__file__).resolve().parents[1]
    fixture = root / "outputs/fixtures/tokenizer_full_trained"
    metadata = json.loads((fixture / "metadata.json").read_text(encoding="utf-8"))
    config = TokenizerConfig(**metadata["config"])
    torch.set_num_threads(2)
    model = SkillVAE(config).eval()
    load_npz_state(model, fixture / "weights.npz")
    with np.load(fixture / "features.npz", allow_pickle=False) as reference:
        convolution = torch.from_numpy(reference["convolution"][:1].copy())
        expected = reference["encoder"][:1].copy()
    traces, handles = [], []
    for number, layer in enumerate(model.encoder.layers, start=1):
        def capture(module, inputs, output, number=number):
            traces.append((number, list(output.shape)))
        handles.append(layer.register_forward_hook(capture))
    with torch.no_grad():
        position = model.add_positional_emb.penc(convolution)
        positioned = model.add_positional_emb(convolution)
        first = model.encoder.layers[0]
        normalized = first.norm1(positioned)
        packed = F.linear(normalized, first.self_attn.in_proj_weight, first.self_attn.in_proj_bias)
        q, k, v = packed.chunk(3, dim=-1)
        batch, length, width = q.shape
        heads, head_dim = config.encoder_heads, width // config.encoder_heads
        split = lambda value: value.reshape(batch, length, heads, head_dim).transpose(1, 2)
        qh, kh, vh = [split(value) for value in (q, k, v)]
        scores = (qh @ kh.transpose(-2, -1)) / (head_dim ** 0.5)
        if config.use_causal_encoder:
            mask = torch.nn.Transformer.generate_square_subsequent_mask(length)
            attention = (scores + mask).softmax(-1)
            encoded = model.encoder(positioned, mask=mask, is_causal=True)
        else:
            attention = scores.softmax(-1)
            encoded = model.encoder(positioned)
    for handle in handles:
        handle.remove()
    actual = encoded.numpy()
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-4)
    print("Lesson 03: positional encoding and trained Transformer Encoder, CPU only")
    print("Convolution output:", list(convolution.shape))
    print("Position encoding:", list(position.shape))
    print("After addition:", list(positioned.shape))
    print("First position's first 6 positional values:", position[0, 0, :6].numpy().round(4))
    print("Second position's first 6 positional values:", position[0, 1, :6].numpy().round(4))
    print("First-layer Q, K, V (each):", list(q.shape))
    print("After splitting heads, Q, K, V (each):", list(qh.shape))
    print("Attention weights:", list(attention.shape))
    print("Weighted V per head:", list((attention @ vh).shape))
    print("FFN feature dimensions:", width, "->", first.linear1.out_features, "->", width)
    for number, shape in traces:
        print(f"Encoder layer {number} output: {shape}")
    print("Final Encoder output:", list(encoded.shape))
    print("Max absolute difference from saved reference:", float(np.abs(actual - expected).max()))
    print("The 8 compressed positions remain; their features have been updated.")
    print("Scope: saved real example, eval forward only; no training or Jittor verification.")


if __name__ == "__main__":
    main()
