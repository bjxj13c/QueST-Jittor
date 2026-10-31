"""Inspect FSQ on one saved real action chunk using trained Torch weights.

CPU eval only: no training, backward pass, or Jittor execution.
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
        encoded = torch.from_numpy(reference["encoder"][:1].copy())
        expected_projection = reference["fsq_project_in"][:1].copy()
        expected_codes = reference["codes"][:1].copy()
        expected_indices = reference["indices"][:1].copy()

    fsq = model.vq
    with torch.no_grad():
        projected = fsq.project_in(encoded)
        bounded = fsq.bound(projected)
        rounded = bounded.round()
        normalized = rounded / (fsq._levels // 2)
        digits = rounded.to(torch.int32) + fsq._levels // 2
        manual_indices = (digits * fsq._basis).sum(dim=-1).to(torch.int32)
        codes, indices = fsq(encoded)
        recovered = fsq.indices_to_codes(indices)

    np.testing.assert_allclose(projected.numpy(), expected_projection, atol=2e-5, rtol=2e-4)
    np.testing.assert_allclose(codes.numpy(), expected_codes, atol=2e-5, rtol=2e-4)
    np.testing.assert_array_equal(indices.numpy(), expected_indices)
    torch.testing.assert_close(manual_indices, indices, atol=0, rtol=0)
    torch.testing.assert_close(recovered, codes, atol=0, rtol=0)
    print("Lesson 04: trained FSQ on a saved real example, CPU only")
    print("Encoder features:", list(encoded.shape))
    print("FSQ projection (256 -> 4):", list(projected.shape))
    print("Levels:", fsq._levels.tolist())
    print("Total possible tokens:", fsq.codebook_size)
    print("First position, projected values:", projected[0, 0].numpy().round(4))
    print("First position, bounded values:", bounded[0, 0].numpy().round(4))
    print("First position, rounded values:", rounded[0, 0].tolist())
    print("First position, normalized values:", normalized[0, 0].tolist())
    print("First position, nonnegative digits:", digits[0, 0].tolist())
    print("Index basis:", fsq._basis.tolist())
    terms = [f"{digit}*{base}" for digit, base in zip(digits[0, 0].tolist(), fsq._basis.tolist())]
    print("First token calculation:", " + ".join(terms), "=", indices[0, 0].item())
    print("All 8 token indices:", indices[0].tolist())
    print("Token index shape:", list(indices.shape))
    print("Quantized features after projection (4 -> 256):", list(codes.shape))
    print("Index -> quantized features: exact match")
    print("Saved PyTorch reference: floats within tolerances; integer indices exact")
    print("The 4 FSQ coordinates are learned latent coordinates, not robot controls.")
    print("Scope: eval forward only; no training or Jittor verification.")


if __name__ == "__main__":
    main()
