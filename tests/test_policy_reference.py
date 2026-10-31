from pathlib import Path
import pytest
import torch
from torch import nn
import torchvision
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer

from quest_repro.policy_torch import PolicyConfig, QueSTPolicy, ResnetEncoder, SkillGPT
from quest_repro.reference import load_selected_definitions

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(4)


@pytest.mark.parametrize("full", [False, True])
def test_skill_gpt_outputs_and_gradients_match_pinned_source(full):
    config = PolicyConfig(dropout=0.0) if full else PolicyConfig.tiny()
    module = load_selected_definitions(ROOT, "quest/algos/quest_modules/skill_gpt.py",
        ("SkillGPT", "top_k_sampling"), {"torch": torch, "nn": nn,
        "PositionalEncoding1D": PositionalEncoding1D, "Summer": Summer})
    reference = module.SkillGPT(4, 1000, 1000, 8, config.layers, config.heads,
        config.embed_dim, 0.0, 0.0, 5, 1.0, "cpu").eval()
    candidate = SkillGPT(config).eval()
    candidate.load_state_dict(reference.state_dict(), strict=True)
    indices = torch.randint(0, 1000, (2, 8))
    left = torch.randn(2, 2, config.embed_dim, requires_grad=True)
    right = left.detach().clone().requires_grad_()
    expected, actual = reference(indices, left), candidate(indices, right)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected.square().mean().backward()
    actual.square().mean().backward()
    torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
    for name, parameter in candidate.named_parameters():
        torch.testing.assert_close(parameter.grad, dict(reference.named_parameters())[name].grad, rtol=0, atol=0)


def test_visual_encoder_matches_exact_upstream_selected_classes():
    module = load_selected_definitions(ROOT, "quest/algos/utils/rgb_modules.py",
        ("SpatialSoftmax", "SpatialProjection", "ResnetEncoder"),
        {"torch": torch, "nn": nn, "F": torch.nn.functional, "torchvision": torchvision})
    reference = module.ResnetEncoder((3, 128, 128), 256, pretrained=False,
                                     remove_layer_num=4, language_fusion="none").eval()
    candidate = ResnetEncoder().eval()
    candidate.load_state_dict(reference.state_dict(), strict=True)
    inputs = torch.rand(2, 3, 128, 128)
    torch.testing.assert_close(candidate(inputs), reference(inputs), rtol=0, atol=0)


def test_policy_input_contract_and_autoregressive_teacher_forcing():
    model = QueSTPolicy(PolicyConfig.tiny()).eval()
    images = torch.randint(0, 256, (2, 1, 3, 128, 128), dtype=torch.uint8)
    state, tasks = torch.randn(2, 1, 8), torch.zeros(2, dtype=torch.long)
    target = torch.randint(0, 1000, (2, 8))
    logits, loss = model(images, state, tasks, target)
    assert logits.shape == (2, 8, 1000) and torch.isfinite(loss)
    with pytest.raises(TypeError):
        model(images.float().div(255), state, tasks, target)
    with pytest.raises(ValueError):
        model.get_context(images, torch.randn(2, 1, 39), tasks)
    # Changing a later teacher-forced token cannot affect an earlier output.
    altered = target.clone()
    altered[:, 4:] = (altered[:, 4:] + 1) % 1000
    changed = model(images, state, tasks, altered)[0]
    torch.testing.assert_close(logits[:, :5], changed[:, :5], rtol=0, atol=0)
