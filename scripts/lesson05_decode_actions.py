"""Inspect trained QueST decoding from real token indices on local CPU.

Uses the same saved example as lessons 01-04. No training or simulation.
"""
import json
from pathlib import Path

import numpy as np
import torch

from quest_repro.config import TokenizerConfig
from quest_repro.tokenizer_torch import SkillVAE
from quest_repro.training import load_npz_state


def main():
    root = Path(__file__).resolve().parents[1]
    fixture = root / "outputs/fixtures/tokenizer_full_trained"
    metadata = json.loads((fixture / "metadata.json").read_text(encoding="utf-8"))
    torch.set_num_threads(2)
    model = SkillVAE(TokenizerConfig(**metadata["config"])).eval()
    load_npz_state(model, fixture / "weights.npz")
    with np.load(fixture / "features.npz", allow_pickle=False) as reference:
        indices = torch.from_numpy(reference["indices"][:1].copy())
        original = reference["input"][:1].copy()
        expected_prediction = reference["prediction"][:1].copy()
        expected_decoder = reference["decoder"][:1].copy()

    layer_outputs, cross_inputs, handles = [], [], []
    for number, layer in enumerate(model.decoder.layers, start=1):
        def capture(module, inputs, output, number=number):
            layer_outputs.append((number, output.detach().clone()))
        handles.append(layer.register_forward_hook(capture))

    def capture_cross(module, inputs):
        cross_inputs.append(tuple(list(value.shape) for value in inputs[:3]))

    handles.append(model.decoder.layers[0].multihead_attn.register_forward_pre_hook(capture_cross))
    try:
        with torch.no_grad():
            latent = model.vq._indices_to_codes(indices)
            memory = model.vq.indices_to_codes(indices)
            queries = model.fixed_positional_emb(memory.new_zeros(
                (memory.shape[0], model.skill_block_size, model.decoder_dim)))
            prediction = model.decode_actions(indices)
    finally:
        for handle in handles:
            handle.remove()

    np.testing.assert_allclose(layer_outputs[-1][1].numpy(), expected_decoder, atol=2e-5, rtol=2e-4)
    np.testing.assert_allclose(prediction.numpy(), expected_prediction, atol=2e-5, rtol=2e-4)
    print("Lesson 05: trained Decoder, real saved example, CPU only")
    print("Token indices:", indices[0].tolist())
    print("Index shape:", list(indices.shape))
    print("Recovered normalized FSQ coordinates:", list(latent.shape))
    print("Projected memory (4 -> 256):", list(memory.shape))
    print("Fixed positional queries for 32 output steps:", list(queries.shape))
    q_shape, k_shape, v_shape = cross_inputs[0]
    print("First-layer cross-attention inputs BEFORE head splitting:")
    print("  Q input from decoder:", q_shape)
    print("  K input from token memory:", k_shape)
    print("  V input from token memory:", v_shape)
    print("After learned Q/K/V projections and 4-head splitting:")
    print("  Q: [1, 4, 32, 64]; K and V (each): [1, 4, 8, 64]")
    print("  Cross-attention weights: [1, 4, 32, 8]")
    for number, output in layer_outputs:
        print(f"Decoder layer {number} output:", list(output.shape))
    print("Action head (256 -> 4):", list(prediction.shape))
    print("Original first 3 actions:")
    print(original[0, :3].round(4))
    print("Reconstructed first 3 actions:")
    print(prediction[0, :3].numpy().round(4))
    print("Max absolute difference from saved prediction:",
          float(np.abs(prediction.numpy() - expected_prediction).max()))
    print("The 32 output positions come from 32 fixed queries; no token repetition.")
    print("All decoder positions are computed in one forward call, with a causal self-attention mask.")
    print("Scope: one saved-example reconstruction; no training, online success, or Jittor verification.")


if __name__ == "__main__":
    main()
