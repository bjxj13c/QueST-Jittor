from pathlib import Path
import pytest
import torch
from positional_encodings.torch_encodings import PositionalEncoding1D as OriginalEncoding
from vector_quantize_pytorch import FSQ as OriginalFSQ

from quest_repro.config import TokenizerConfig
from quest_repro.reference import build_upstream_tokenizer, load_upstream_module
from quest_repro.tokenizer_torch import FSQ, PositionalEncoding1D, SkillVAE, round_ste

ROOT = Path(__file__).resolve().parents[1]


def test_round_ste_keeps_unit_gradient_at_quantization_boundaries():
    value = torch.tensor([-2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5], requires_grad=True)
    result = round_ste(value)
    torch.testing.assert_close(result, value.round(), rtol=0, atol=0)
    result.sum().backward()
    torch.testing.assert_close(value.grad, torch.ones_like(value), rtol=0, atol=0)


def test_fsq_exhaustive_all_1000_indices():
    reference, candidate = OriginalFSQ(levels=[8, 5, 5, 5], dim=4), FSQ(4, [8, 5, 5, 5])
    indices = torch.arange(1000).reshape(10, 100)
    expected, actual = reference.indices_to_codes(indices), candidate.indices_to_codes(indices)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(candidate.codes_to_indices(actual).long(), indices, rtol=0, atol=0)
    assert actual[..., 0].min() == -1 and actual[..., 0].max() == 0.75


def test_projected_fsq_outputs_and_gradients_match_dependency():
    torch.manual_seed(42)
    reference, candidate = OriginalFSQ(levels=[8, 5, 5, 5], dim=64), FSQ(64, [8, 5, 5, 5])
    candidate.load_state_dict(reference.state_dict(), strict=True)
    x = torch.randn(3, 8, 64, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    expected, expected_indices = reference(x)
    actual, actual_indices = candidate(y)
    torch.testing.assert_close(actual_indices, expected_indices, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(y.grad, x.grad, rtol=1e-6, atol=1e-7)
    for name, parameter in candidate.named_parameters():
        torch.testing.assert_close(parameter.grad, dict(reference.named_parameters())[name].grad, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("channels", [7, 64, 256])
def test_sinusoidal_encoding_order_and_values(channels):
    x = torch.zeros(2, 32, channels)
    torch.testing.assert_close(PositionalEncoding1D(channels)(x), OriginalEncoding(channels)(x), rtol=0, atol=0)


@pytest.mark.parametrize("factor", [1, 2, 4, 8])
def test_causal_convolution_length_and_values(factor):
    original = load_upstream_module(ROOT)
    reference = original.ResidualTemporalBlock(64, 64,
        kernel_size=[3, 2] if factor == 1 else [5] + [3] * (factor.bit_length() - 1),
        stride=[1, 1] if factor == 1 else [2] * (factor.bit_length() - 1) + [1], causal=True)
    candidate = SkillVAE(TokenizerConfig.tiny()).conv_block
    if factor != 4:
        from quest_repro.tokenizer_torch import ResidualTemporalBlock
        candidate = ResidualTemporalBlock(64, factor)
    candidate.load_state_dict(reference.state_dict(), strict=True)
    x = torch.randn(2, 32, 64)
    torch.testing.assert_close(candidate(x), reference(x), rtol=0, atol=0)
    assert candidate(x).shape == (2, 32 // factor, 64)


@pytest.mark.parametrize("full_size", [False, True])
def test_full_tokenizer_forward_and_backward_match_pinned_quest(full_size):
    config = TokenizerConfig(attn_pdrop=0.0) if full_size else TokenizerConfig.tiny()
    reference, candidate = build_upstream_tokenizer(ROOT, config), SkillVAE(config)
    candidate.load_state_dict(reference.state_dict(), strict=True)
    reference.eval()
    candidate.eval()
    x = torch.randn(2, 32, 4, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    expected, actual = reference(x), candidate(y)
    for i in (0, 1, 2, 3, 4):
        torch.testing.assert_close(actual[i], expected[i], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(candidate.get_indices(y), reference.get_indices(x), rtol=0, atol=0)
    expected_loss = (expected[0] - x).abs().mean()
    actual_loss = (actual[0] - y).abs().mean()
    expected_loss.backward()
    actual_loss.backward()
    torch.testing.assert_close(y.grad, x.grad, rtol=1e-5, atol=1e-6)
    reference_params = dict(reference.named_parameters())
    for name, parameter in candidate.named_parameters():
        assert parameter.grad is not None, name
        torch.testing.assert_close(parameter.grad, reference_params[name].grad, rtol=1e-5, atol=1e-6)
